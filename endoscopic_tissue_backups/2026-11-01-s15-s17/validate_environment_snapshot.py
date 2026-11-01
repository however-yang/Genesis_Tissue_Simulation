"""Real native/Python snapshot acceptance, never a prefix-replay substitute."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from validation_support import ROOT, environment, command_sequence, exact_values, replay
from validate_native_snapshot import NativeSnapshotAPI
from endoscopic_tissue.training.data import atomic_json
from endoscopic_tissue.training.snapshot import EnvSnapshot, SnapshotBackend
from endoscopic_tissue.training.vision import SynchronizedVision


def distribution(values):
    return {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
            'max': float(np.max(values)), 'min': float(np.min(values))}


def gpu_memory():
    return int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).strip().splitlines()[0])*1024**2


def smoke(out, scene, vision_enabled):
    record = {'passed': False, 'scene': scene, 'vision': vision_enabled}
    vision = None
    try:
        with environment(scene=scene, preparation='cut_ready') as env:
            initial, context = env.reset(0)
            atomic_json(out/'parent-context.json', context)
            record['preparation_quality_failures'] = sum(not r['quality']['checked_metrics_passed'] for r in env.preparation)
            record['preparation_scope_failures'] = sum(not r['quality']['within_declared_local_scope'] for r in env.preparation)
            if vision_enabled:
                vision = SynchronizedVision(env, {'width': 640, 'height': 480})
                vision.capture(env, initial)
            native_before = asdict(env.observed)
            state_arrays = {k: v.copy() for k, v in initial.arrays.items()}
            topology = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in initial.topology.items()}
            snapshot = env.snapshot()
            snapshot.save(out/'cut-ready.lcssnap')
            record['metadata'] = snapshot.metadata
            record['compressed_bytes'] = snapshot.size_bytes
            loaded = EnvSnapshot.load(out/'cut-ready.lcssnap', dll_sha256=context['dll_sha256'],
                physics_config_sha256=snapshot.metadata['physics_config_sha256'])
            assert loaded.sha256 == snapshot.sha256
            incise = next(s for s in env.scene.stages if s.name == 'incise_1')
            prior = next(s.command for s in env.scene.stages if s.name == 'blade_approach')
            sequence = command_sequence([incise], prior)
            first = env.replay_command(sequence[0][2])
            assert first.after is not None
            expected_nodes = first.after.arrays['positions_m'].copy()
            restored, branch = env.restore(loaded, branch_label='round-trip')
            record['restore_wall_ms'] = branch['restore_wall_time_ms']
            atomic_json(out/'branch-context.json', branch)
            assert SnapshotBackend(env.native).save() == snapshot.native_bytes(), 'T1 native saved buffers differ'
            record['T1_native_exact'] = True
            exact_values(native_before, asdict(env.native.observe()))
            for name in ('positions_m', 'estimated_velocity_m_s', 'grip_flags'):
                exact_values(state_arrays[name], restored.arrays[name])
            exact_values(topology, restored.topology)
            record['T2_observation_exact'] = True
            if vision:
                assert restored.metadata['vision']['native_tick'] == snapshot.metadata['native_tick']
                assert restored.metadata['vision']['step_id'] == 0
                assert restored.arrays['rgb'] is not initial.arrays['rgb']
                # Rendering is required afresh; report image difference rather
                # than silently assuming the GPU renderer is bit deterministic.
                record['vision_rgb_max_abs_difference'] = int(np.abs(
                    state_arrays['rgb'].astype(np.int16)-restored.arrays['rgb']).max())
                record['vision_depth_max_abs_difference_m'] = float(np.abs(
                    state_arrays['depth_pinhole_m']-restored.arrays['depth_pinhole_m']).max())
                record['vision_rerender_ms'] = branch['restore_vision_ms']
            continued = env.replay_command(sequence[0][2])
            assert continued.after is not None
            error_mm = float(np.linalg.norm(expected_nodes-continued.after.arrays['positions_m'], axis=1).max()*1000)
            record['T3_max_node_difference_mm'] = error_mm
            assert error_mm < .01, 'T3 one-step continuation exceeded 0.01 mm'
            branches = []
            from dataclasses import replace
            for i, offset in enumerate(((.03, 0, 0), (0, .03, 0), (0, 0, .03))):
                _, branch = env.restore(snapshot, branch_label=f'different-action-{i}')
                branches.append(branch['episode_id'])
                command = replace(prior, hinge=tuple(np.asarray(prior.hinge)+offset))
                for _ in range(3):
                    assert env.replay_command(command).after is not None
            env.restore(snapshot)
            assert len(set(branches)) == 3
            assert SnapshotBackend(env.native).save() == snapshot.native_bytes()
            record['T5_distinct_branches_no_pollution'] = True
            first_cut = replay(env, sequence)
            assert first_cut['outcome']['new_material_edges'] > 0, 'T6 test failed to create an actual material cut'
            env.restore(snapshot, branch_label='cut-again')
            exact_values(topology, env.state.topology)
            assert env.quality.previous_neighbors is env.observed.neighbors
            recut = replay(env, sequence)
            assert recut['outcome']['new_material_edges'] > 0, 'T6 restored tissue could not be cut again'
            assert recut['quality']['healed_edges'] == 0, 'T6 monitor mistook restored topology for healing'
            record['T6'] = {'first_cut': {k: v for k, v in first_cut.items() if k != 'rows'},
                            'repeated_cut': {k: v for k, v in recut.items() if k != 'rows'},
                            'restored_topology_exact': True}
            atomic_json(out/'first-cut-steps.json', first_cut['rows'])
            atomic_json(out/'repeated-cut-steps.json', recut['rows'])
            if vision:
                vision.close()
                vision = None
            if not vision_enabled:
                env.close()
                env.reset(1, scene_config=replace(env.config.scene, preparation='settled'))
                env.restore(snapshot, branch_label='new-native-session')
                assert SnapshotBackend(env.native).save() == snapshot.native_bytes()
                exact_values(native_before, asdict(env.native.observe()))
                record['new_native_session_exact'] = True
            record['snapshot_mechanism_passed'] = True
            record['passed'] = (record['preparation_quality_failures'] == record['preparation_scope_failures'] == 0 and
                record['restore_wall_ms'] < 11500 and
                all(v['quality']['quality_failure_steps'] == v['quality']['scope_failure_steps'] == 0
                    for v in (first_cut, recut)))
            record['performance_target'] = {'restore_below_preparation_11500ms': record['restore_wall_ms'] < 11500,
                'restore_at_most_1000ms': record['restore_wall_ms'] <= 1000,
                'snapshot_at_most_200MB': snapshot.size_bytes <= 200000000}
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        if vision:
            vision.close()
        atomic_json(out/'result.json', record)
    return record


def resources(out, scene, count):
    record = {'passed': False, 'count': count, 'scene': scene}
    rows = []
    try:
        with environment(scene=scene, preparation='cut_ready') as env:
            _, context = env.reset(0)
            atomic_json(out/'parent-context.json', context)
            record['preparation_failure_steps'] = sum(not r['quality']['checked_metrics_passed'] or
                not r['quality']['within_declared_local_scope'] for r in env.preparation)
            api = NativeSnapshotAPI(env.native)
            original_inventory = api.inventory()
            saved_times, restored_times, samples, sizes = [], [], [], []
            process = psutil.Process()
            for i in range(count):
                started = time.perf_counter()
                snapshot = env.snapshot()
                if i == 0:
                    record['metadata'] = snapshot.metadata
                saved_times.append((time.perf_counter()-started)*1000)
                sizes.append(snapshot.size_bytes)
                started = time.perf_counter()
                env.restore(snapshot, branch_label='resource-cycle')
                restored_times.append((time.perf_counter()-started)*1000)
                del snapshot
                if i % 50 == 0 or i == count-1:
                    gc.collect()
                    inventory = api.inventory()
                    assert inventory == original_inventory, 'Native allocation inventory grew during save/load'
                    samples.append({'cycle': i+1, 'rss_bytes': process.memory_info().rss,
                                    'gpu_used_bytes': gpu_memory(), 'native_bytes': inventory['bytes'],
                                    'native_allocations': len(inventory['allocations'])})
                    atomic_json(out/'resource-progress.json', {'count': i+1, 'samples': samples,
                        'save_ms': distribution(saved_times), 'restore_ms': distribution(restored_times)})
                    print('snapshot cycles', i+1, flush=True)
            tail = [s for s in samples if s['cycle'] >= 200]
            if len(tail) < 3:
                raise ValueError('Resource acceptance requires at least 300 cycles; T8 requires 1000')
            rss_slope = float(np.polyfit([s['cycle'] for s in tail], [s['rss_bytes'] for s in tail], 1)[0])
            rss_growth = tail[-1]['rss_bytes']-tail[0]['rss_bytes']
            gpu_growth = tail[-1]['gpu_used_bytes']-tail[0]['gpu_used_bytes']
            record.update(samples=samples, save_ms=distribution(saved_times), restore_ms=distribution(restored_times),
                snapshot_bytes=distribution(sizes), rss_tail_slope_bytes_per_cycle=rss_slope,
                rss_tail_growth_bytes=rss_growth, gpu_tail_growth_bytes=gpu_growth,
                memory_criteria='after 200 warmup cycles: RSS slope <= 1 KiB/cycle and net <= 8 MiB; GPU <= 64 MiB reporting granularity; native allocation inventory exactly constant',
                native_inventory_exact=True,
                passed=count >= 1000 and rss_slope <= 1024 and rss_growth <= 8*1024**2 and gpu_growth <= 64*1024**2
                    and record['preparation_failure_steps'] == 0 and max(restored_times) < 11500)
            record['performance_target'] = {'restore_below_preparation_11500ms': max(restored_times) < 11500,
                'restore_p95_at_most_1000ms': np.percentile(restored_times, 95) <= 1000,
                'snapshot_at_most_200MB': max(sizes) <= 200000000}
            record['performance_target'] = {k: bool(v) for k, v in record['performance_target'].items()}
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        atomic_json(out/'result.json', record)
    return record


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--scene', choices=['wall-traction', 'flap'], default='wall-traction')
    parser.add_argument('--resources', type=int, default=0)
    parser.add_argument('--vision', action='store_true')
    args = parser.parse_args()
    out = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/snapshot'/args.name
    out.mkdir(parents=True, exist_ok=False)
    result = resources(out, args.scene, args.resources) if args.resources else smoke(out, args.scene, args.vision)
    print(json.dumps({k: v for k, v in result.items() if k not in ('samples', 'metadata', 'T6')}), flush=True)
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
