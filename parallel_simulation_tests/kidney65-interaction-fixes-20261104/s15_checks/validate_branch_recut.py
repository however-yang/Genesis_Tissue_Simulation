"""GPU1-only S15 T5/T6 with all five branch videos and exact native rollback."""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

from bootstrap import GPU, ROOT
import numpy as np
from validation_capture import PairedCapture
from validation_support import command_sequence, environment, exact_values, replay, row, quality_summary
from endoscopic_tissue.native_backend import NativeTissue
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.training.data import atomic_json
from endoscopic_tissue.training.kidney_snapshot import SnapshotBackend
from endoscopic_tissue.training.snapshot import EnvSnapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--library', default='r33-packed-mg')
    args = parser.parse_args()
    if os.name == 'nt' or not all(v.replace('-', '').isalnum() for v in (args.name,args.library)):
        parser.error('Use simple labels and run on server A100 GPU1')
    out = ROOT/'outputs'/args.name
    out.mkdir(exist_ok=False)
    inputs = out/'driver-inputs'
    inputs.mkdir()
    for path in (Path(__file__), Path(__file__).with_name('validation_support.py'),
                 Path(__file__).with_name('validation_capture.py'),
                 ROOT/'python/endoscopic_tissue/training/env.py',
                 ROOT/'python/endoscopic_tissue/training/kidney_snapshot.py'):
        (inputs/path.name).write_bytes(path.read_bytes())
    controls = ROOT/'controls'/args.name
    controls.mkdir(exist_ok=False)
    guard = GPUReservation(GPU, controls)
    original_step, original_check = NativeTissue.step, NativeTissue.check
    record = dict(passed=False, gpu_uuid=GPU, full_surface_checks=True, branches={},
                  scope='One T5/T6 wall-traction test; no T4/T7 distribution claim',
                  input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()})
    capture = None

    def guarded(native, command):
        guard({os.getpid()})
        return original_step(native, command)

    def measured(native, name, *values):
        start = time.perf_counter()
        value = original_check(native, name, *values)
        if name == 'LCS_Step':
            native._s15_native_ms = (time.perf_counter()-start)*1000
        return value

    NativeTissue.step, NativeTissue.check = guarded, measured
    try:
        from endoscopic_tissue.training.egl_device import bind
        bind(GPU)
        names = ('original','shift-x','shift-y','shift-z','restored')
        capture = PairedCapture(out, 'wall-traction', args.name, arms=names)
        with environment(dll=args.library, preparation='cut_ready') as env:
            initial, parent = env.reset(0)
            atomic_json(out/'parent-context.json', parent)
            record['dll_sha256'] = parent['dll_sha256']
            assert all(r['quality']['checked_metrics_passed'] and r['quality']['within_declared_local_scope']
                       for r in env.preparation)
            before = asdict(env.observed)
            topology = initial.topology
            monitor = copy.deepcopy(capture.monitor.__dict__)
            snapshot = env.snapshot()
            snapshot.save(out/'cut-ready.lcssnap')
            record['snapshot_sha256'] = snapshot.sha256
            cached = EnvSnapshot.load(out/'cut-ready.lcssnap', dll_sha256=parent['dll_sha256'],
                                      physics_config_sha256=snapshot.metadata['physics_config_sha256'])
            assert cached.sha256 == snapshot.sha256
            expected = snapshot.native_bytes()
            incise = next(s for s in env.scene.stages if s.name == 'incise_1')
            prior = next(s.command for s in env.scene.stages if s.name == 'blade_approach')
            sequence = command_sequence([incise], prior)

            def select(arm, frames):
                capture.arm = arm
                capture.expected_end_steps[arm] = env.native.tick + frames

            def restore(arm):
                state, context = env.restore(cached, branch_label=arm)
                exact_values(before, asdict(env.native.observe()))
                exact_values(topology, state.topology)
                assert env.quality.previous_neighbors is env.observed.neighbors
                capture.monitor.__dict__.clear()
                capture.monitor.__dict__.update(copy.deepcopy(monitor))
                atomic_json(out/(arm+'-context.json'), context)
                return context

            def cut(arm):
                select(arm, len(sequence))
                report = replay(env, sequence, before_step=lambda label: setattr(capture,'label',label))
                atomic_json(out/(arm+'-steps.json'), report.pop('rows'))
                record[arm] = report
                assert report['outcome']['new_material_edges'] > 0
                assert all(report['quality'][k] == 0 for k in ('quality_failure_steps','scope_failure_steps','healed_edges'))

            cut('original')
            for name, axis in zip(names[1:4], np.eye(3)):
                context = restore(name)
                select(name, 3)
                capture.label = name
                command = replace(prior, hinge=tuple(np.asarray(prior.hinge)+.03*axis), scissors_cutting=False)
                rows = [row(env, env.replay_command(command), name, i+1) for i in range(3)]
                quality = quality_summary(rows)
                atomic_json(out/(name+'-steps.json'), rows)
                record['branches'][name] = dict(episode_id=context['episode_id'], quality=quality)
                assert all(quality[k] == 0 for k in ('quality_failure_steps','scope_failure_steps','healed_edges'))
            ids = {v['episode_id'] for v in record['branches'].values()}
            assert len(ids) == 3 and parent['episode_id'] not in ids
            branch = restore('restored')
            assert branch['episode_id'] not in ids|{parent['episode_id']}
            assert memoryview(SnapshotBackend(env.native).save_buffer()) == memoryview(expected)
            record['T5_distinct_branches_native_rollback_exact'] = True
            cut('restored')
            record['T6_topology_restored_recut_no_healing'] = True
            record['restore_ms'] = branch['restore_wall_time_ms']
            record['compressed_snapshot_bytes'] = snapshot.size_bytes
        capture.close()
        capture = None
        for arm in names:
            report = json.loads((out/arm/'report.json').read_text())
            assert report['contiguous_native_frames'] and report['completed_requested_continuation']
            rows = [json.loads(line) for line in (out/arm/'trajectory.jsonl').read_text().splitlines()]
            values = [r['native_ms'] for r in rows]
            assert values and all(v is not None and np.isfinite(v) and v>0 for v in values)
            record.setdefault('native_ms', {})[arm] = dict(mean=float(np.mean(values)),
                p95=float(np.percentile(values,95)), frames=len(values))
        record['passed'] = record['restore_ms'] < 11500
        return 0 if record['passed'] else 2
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        NativeTissue.step, NativeTissue.check = original_step, original_check
        try:
            if capture is not None:
                capture.close()
        finally:
            guard.close()
            atomic_json(out/'result.json', record)
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
