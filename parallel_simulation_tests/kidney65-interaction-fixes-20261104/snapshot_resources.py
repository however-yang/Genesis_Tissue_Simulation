"""Isolated S15 resource screen/acceptance, server A100 GPU1 only.

No solver/config defaults change. A 20-cycle screen is explicitly NOT T8;
only 1000 cycles can satisfy the frozen resource acceptance criteria.
"""
import argparse
import ctypes as ct
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import psutil


def distribution(values):
    return dict(mean=float(np.mean(values)), p95=float(np.percentile(values, 95)),
                max=float(np.max(values)), min=float(np.min(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--cycles', type=int, choices=(20, 300, 1000), default=20)
    args = parser.parse_args()
    if os.name == 'nt':
        parser.error('Run all GPU tests on server A100 physical GPU1')
    if not all(v.replace('-', '').replace('_', '').isalnum() for v in (args.library, args.name)):
        raise ValueError('Invalid isolated run/library label')
    # The existing kidney API smoke configures precisely the frozen scene,
    # adapter, native library parameters and imports without starting a session.
    import training_api_smoke as base
    from endoscopic_tissue.native_backend import NativeTissue
    from endoscopic_tissue.training.snapshot import SnapshotBackend
    from endoscopic_tissue.training.data import atomic_json
    root = Path(__file__).resolve().parent
    output = root / 'outputs' / args.name
    output.mkdir(exist_ok=False)
    (output / 'driver.py').write_bytes(Path(__file__).read_bytes())
    (output / 'scene-helper.py').write_bytes((root / 'training_api_smoke.py').read_bytes())
    controls = root / 'controls' / args.name
    controls.mkdir(exist_ok=False)
    guard = base.GPUReservation(base.GPU, controls)
    env = None
    original_step = NativeTissue.step

    def guarded_step(native, command):
        guard({os.getpid()})
        return original_step(native, command)

    NativeTissue.step = guarded_step
    record = dict(passed=False, functional_passed=False, required_cycles=1000,
                  requested_cycles=args.cycles, gpu_uuid=base.GPU,
                  scope='S15 T8 kidney65 cut_ready snapshot resources only; no policy training',
                  full_surface_checks=True, default_replaced=False)
    samples, save_ms, restore_ms, sizes = [], [], [], []
    try:
        library = root / 'binaries' / (args.library + '.so')
        digest = hashlib.sha256(library.read_bytes()).hexdigest()
        assert digest == json.loads((root / 'results' / ('build-' + args.library + '.json')).read_text())['sha256']
        record['library_sha256'] = digest
        native = ct.CDLL(str(library))
        for name, values in {'LCS_SetCoreOptimizations': (43264,), 'LCS_SetBatchQueryMode': (3,),
                             'LCS_SetQueryThreads': (64,), 'LCS_SetNativeSchedule': (512, 1024),
                             'LCS_SetCoarseBoundCache': (1,)}.items():
            fn = getattr(native, name)
            fn.argtypes, fn.restype = [ct.c_int] * len(values), ct.c_int
            assert fn(*values) == 0
        scene = base.run.SCISSORS_SCENES['kidney']
        targets = np.array([base.command_to_action(stage.command) for stage in scene.stages])
        lo, hi = targets.min(axis=0), targets.max(axis=0)
        padding = np.array([.001] * 3 + [0] + [.001] * 3 + [.01] * 3 + [0])
        lo -= padding
        hi += padding
        lo[[3, 10]], hi[[3, 10]] = 0, 1
        bounds = base.ActionBounds(tuple(lo), tuple(hi), tuple([.02] * 3 + [2] + [.02] * 3 + [.2] * 3 + [2]))
        config = base.EnvironmentConfig(str(library), digest, bounds,
            scene=base.SceneConfig('kidney', 'cut_ready'), max_steps=6,
            full_surface_checks=True, gpu_topology_checks=True)
        env = base.KidneyEnvironment(config)
        state, context = env.reset(65)
        assert context['material_node_count'] == 65688
        atomic_json(output / 'context.json', context)
        record['preparation_failure_steps'] = sum(
            not row['quality']['checked_metrics_passed'] or not row['quality']['within_declared_local_scope']
            for row in env.preparation)
        assert record['preparation_failure_steps'] == 0
        inventory_fn = native.LCS_StateInventory
        inventory_fn.argtypes, inventory_fn.restype = [ct.c_void_p, ct.c_size_t], ct.c_int

        def inventory():
            buffer = ct.create_string_buffer(131072)
            assert inventory_fn(buffer, len(buffer)) == 0
            return json.loads(buffer.value)

        original_inventory = inventory()
        atomic_json(output / 'allocation-inventory.json', original_inventory)
        backend = SnapshotBackend(env.native)
        observed = env.native.observe()
        process = psutil.Process()
        for cycle in range(1, args.cycles + 1):
            guard({os.getpid()})
            started = time.perf_counter()
            snapshot = env.snapshot()
            save_ms.append((time.perf_counter() - started) * 1000)
            sizes.append(snapshot.size_bytes)
            started = time.perf_counter()
            restored, _ = env.restore(snapshot, branch_label='resource-cycle')
            restore_ms.append((time.perf_counter() - started) * 1000)
            np.testing.assert_array_equal(restored.arrays['positions_m'], state.arrays['positions_m'])
            if cycle == 1:
                assert backend.save() == snapshot.native_bytes(), 'Native save/load/save differs'
                record['native_roundtrip_exact'] = True
                record['snapshot_raw_native_bytes'] = len(snapshot.native_bytes())
                record['snapshot_metadata'] = snapshot.metadata
            del snapshot
            if cycle == 1 or cycle % 50 == 0 or cycle == args.cycles:
                current = env.native.observe()
                np.testing.assert_array_equal(current.nodes, observed.nodes)
                np.testing.assert_array_equal(current.neighbors, observed.neighbors)
                np.testing.assert_array_equal(current.jaw_load_n, observed.jaw_load_n)
                assert inventory() == original_inventory, 'Native allocation inventory grew/changed'
                del current
                gc.collect()
                gpu_mib = int(subprocess.check_output(['nvidia-smi', '-i', base.GPU,
                    '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
                samples.append(dict(cycle=cycle, rss_bytes=process.memory_info().rss,
                                    gpu_used_bytes=gpu_mib * 1024**2))
                atomic_json(output / 'progress.json', dict(completed_cycles=cycle,
                    samples=samples, save_ms=distribution(save_ms), restore_ms=distribution(restore_ms)))
                print('snapshot cycles', cycle, 'restore mean ms', round(float(np.mean(restore_ms)), 3), flush=True)
        record.update(functional_passed=True, native_inventory_exact=True, completed_cycles=args.cycles,
                      save_ms=distribution(save_ms), restore_ms=distribution(restore_ms),
                      snapshot_bytes=distribution(sizes), samples=samples)
        tail = [row for row in samples if row['cycle'] >= 200]
        if len(tail) >= 3:
            slope = float(np.polyfit([s['cycle'] for s in tail], [s['rss_bytes'] for s in tail], 1)[0])
            growth = tail[-1]['rss_bytes'] - tail[0]['rss_bytes']
            gpu_growth = tail[-1]['gpu_used_bytes'] - tail[0]['gpu_used_bytes']
            record.update(rss_tail_slope_bytes_per_cycle=slope, rss_tail_growth_bytes=growth,
                          gpu_tail_growth_bytes=gpu_growth)
            record['passed'] = bool(args.cycles >= 1000 and slope <= 1024 and growth <= 8*1024**2
                                    and gpu_growth <= 64*1024**2 and max(restore_ms) < 11500)
        record['performance_target'] = dict(restore_below_preparation_11500ms=max(restore_ms) < 11500,
            restore_p95_at_most_1000ms=float(np.percentile(restore_ms, 95)) <= 1000,
            snapshot_at_most_200MB=max(sizes) <= 200000000)
        return 0 if record['functional_passed'] and (args.cycles < 1000 or record['passed']) else 2
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        NativeTissue.step = original_step
        if env is not None:
            env.close()
        guard.close()
        atomic_json(output / 'result.json', record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
