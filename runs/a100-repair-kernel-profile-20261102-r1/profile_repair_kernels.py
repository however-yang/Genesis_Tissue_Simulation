"""Opt-in CUDA activity trace of the existing same-library surface repair.

Extra repair calls are diagnostic only, do not advance simulation time, and
are never placed in the default frame loop. Trace times are not normal FPS.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--library-dir', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == 'GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'
    root = args.release.resolve(strict=True)
    sys.path[:0] = [str(root), str(root/'endoscopic_tissue_backups/2026-11-01-s15-s17')]
    import torch
    from threadpoolctl import threadpool_limits
    from validation_support import environment
    from endoscopic_tissue.training.provenance import json_hash, sha256
    from endoscopic_tissue.training.snapshot import EnvSnapshot, SnapshotBackend, binding
    from endoscopic_tissue.instrument_geometry import load_scissors, scissors_body_depths
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(diagnostic_only=True, simulation_time_advanced=False, default_replaced=False,
                  performance_accepted=False, completed=False, snapshot=str(args.snapshot),
                  worker_sha256=sha256(__file__), torch_version=torch.__version__)
    try:
        torch.cuda.init()
        assert torch.cuda.device_count() == 1
        record['device'] = torch.cuda.get_device_name(0)
        with threadpool_limits(limits=1, user_api='blas'), environment(dll=str(args.library_dir)) as env:
            env.reset(0)
            snapshot = EnvSnapshot.load(args.snapshot, dll_sha256=env.context['dll_sha256'],
                                       physics_config_sha256=json_hash(binding(env)))
            env.restore(snapshot, branch_label='repair-kernel-diagnostic')
            assert SnapshotBackend(env.native).save() == snapshot.native_bytes()
            record.update(snapshot_sha256=snapshot.sha256, dll_sha256=env.context['dll_sha256'],
                          native_import_exact=True, native_tick=env.native.tick)
            geometry = load_scissors()
            def measure():
                observed, surface = env.native.observe(), env.native.surface()
                pose = observed.scissors
                body, wrist = scissors_body_depths(surface.vertices, surface.triangles, geometry,
                                                   pose.hinge, pose.axis, pose.normal, pose.gap)
                return dict(body_overlap_mm=float(body*10), wrist_overlap_mm=float(wrist*10),
                            material_quality=env.native.material_quality().tolist())
            record['initial'] = measure()
            # Build this restored state's graph before the kernel trace. This
            # warmup is also a repair; record its effect instead of hiding it.
            start = time.perf_counter()
            env.native.check('LCS_RefineContactSurface', env.settings.contact_refine_passes)
            torch.cuda.synchronize()
            record['warmup_wall_ms'] = (time.perf_counter()-start)*1000
            record['after_warmup'] = measure()
            times = []
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=False, with_stack=False) as profiler:
                for _ in range(3):
                    start = time.perf_counter()
                    with torch.profiler.record_function('native_surface_repair'):
                        env.native.check('LCS_RefineContactSurface', env.settings.contact_refine_passes)
                        torch.cuda.synchronize()
                    times.append((time.perf_counter()-start)*1000)
            trace_path = args.output/'cuda-trace.json'
            profiler.export_chrome_trace(str(trace_path))
            record.update(profiled_calls=3, profiled_wall_ms=times, after_profile=measure())
            trace = json.loads(trace_path.read_text())
            groups = defaultdict(list)
            for event in trace.get('traceEvents', []):
                if event.get('cat') == 'kernel' and event.get('ph') == 'X':
                    groups[event['name']].append(float(event['dur'])/1000)
            record['kernels'] = sorted([dict(name=name, launches=len(values), total_ms=sum(values),
                max_ms=max(values), mean_ms=sum(values)/len(values)) for name, values in groups.items()],
                key=lambda row: -row['total_ms'])
            record['native_kernels_observed'] = any('k_' in name for name in groups)
            record['trace_sha256'] = sha256(trace_path)
            record['completed'] = True
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        with (args.output/'result.json').open('x') as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps({k:v for k,v in record.items() if k != 'kernels'}), flush=True)


if __name__ == '__main__':
    main()
