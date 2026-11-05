"""Reproduce the nominal S16 tear regression on A100 GPU1 with full video."""
import argparse
import ctypes
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from bootstrap import GPU, ROOT
import numpy as np
from validation_capture import PairedCapture
from validation_support import WallCutMeasure, environment, row, quality_summary
from endoscopic_tissue.native_backend import NativeTissue
from endoscopic_tissue.scenario import interpolate, stage_start
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.training.data import atomic_json
from endoscopic_tissue.training.physics import PhysicsOverrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--library', default='r33-packed-mg')
    parser.add_argument('--ratio', type=float, choices=(0.,1.3,1.5,2.), required=True)
    args = parser.parse_args()
    if os.name == 'nt' or not all(v.replace('-', '').isalnum() for v in (args.name,args.library)):
        parser.error('Use simple labels and run on server A100 GPU1')
    out = ROOT/'outputs'/args.name
    out.mkdir(exist_ok=False)
    (out/'driver.py').write_bytes(Path(__file__).read_bytes())
    controls = ROOT/'controls'/args.name
    controls.mkdir(exist_ok=False)
    guard = GPUReservation(GPU, controls)
    check, step = NativeTissue.check, NativeTissue.step
    capture = None
    record = dict(passed=False, ratio=args.ratio, gpu_uuid=GPU, snips={},
        scope='S16 nominal fixture only; compare against ratio 0 before any dose scan',
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    rows = []
    terminal_reasons = {'non_finite_state','invalid_material_tetrahedra',
                        'workspace_displacement_exceeded','estimated_speed_limit_exceeded'}

    def guarded(native, command):
        guard({os.getpid()})
        return step(native, command)

    def measured(native, name, *values):
        start = time.perf_counter()
        value = check(native, name, *values)
        if name == 'LCS_Step':
            native._s15_native_ms = (time.perf_counter()-start)*1000
        return value

    NativeTissue.step, NativeTissue.check = guarded, measured
    try:
        from endoscopic_tissue.training.egl_device import bind
        bind(GPU)
        capture = PairedCapture(out, 'wall-traction', args.name, arms=('original',))
        overrides = PhysicsOverrides(tear_stretch_ratio=args.ratio)
        with environment(dll=args.library, physics=overrides, telemetry=True) as env:
            _, context = env.reset(0)
            atomic_json(out/'context.json', context)
            atomic_json(out/'preparation.json', env.preparation)
            record['dll_sha256'] = context['dll_sha256']
            record['effective_overrides'] = asdict(overrides)
            record['preparation_quality'] = quality_summary(env.preparation)
            read_work = getattr(env.native.dll,'LCS_ReadShellWork',None)
            if read_work is not None:
                read_work.argtypes, read_work.restype = [ctypes.c_void_p], ctypes.c_int
            capture.expected_end_step = sum(s.steps for s in env.scene.stages)
            measure = WallCutMeasure(env)
            requested = env.scene.stages[0].command
            for stage in env.scene.stages[1:]:
                capture.label = stage.name
                start = stage_start(stage.name, requested, env.observed)
                before = measure.sample(env)
                first_cut = None
                for i in range(stage.steps):
                    requested = interpolate(start, stage.command, (i+1)/(stage.steps*stage.ramp))
                    result = env.replay_command(requested)
                    observed = row(env, result, stage.name, i+1)
                    rows.append(observed)
                    if stage.name.startswith('incise_') and first_cut is None:
                        if observed['failure_physics']['new_material_blade_edges'] > 0:
                            first_cut = i+1
                    quality = observed['quality']
                    if terminal_reasons.intersection(quality['numerical_failure_reasons']) or quality['scope_failure_reasons']:
                        raise RuntimeError(f'Unchanged terminal quality gate at {stage.name} step {i+1}')
                if stage.name.startswith('incise_'):
                    record['snips'][stage.name] = measure.outcome(before, measure.sample(env), first_cut)
                if read_work is not None:
                    work = (ctypes.c_ulonglong*3)()
                    assert read_work(work) == 0
                    record.setdefault('shell_work_by_stage',{})[stage.name] = dict(
                        deferred=int(work[0]), propagated=int(work[1]), fixed_point_passes=int(work[2]))
            record['completed'] = True
        quality = quality_summary(rows)
        record['quality'] = quality
        record['passed'] = (
            all(v['first_new_material_cut_step'] is not None for v in record['snips'].values())
            and len(record['snips']) == 3
            and all(quality[k] == record['preparation_quality'][k] == 0 for k in
                    ('quality_failure_steps','scope_failure_steps','healed_edges'))
            and all(not r['failure_physics']['new_tear_edges'] and
                    not r['failure_physics']['anchors']['new_broken_support_corners'] for r in rows)
            and all(not (r.get('failure_physics') or {}).get('new_tear_edges',0) for r in env.preparation))
        if len(capture.rows['original']) != capture.expected_end_step:
            record['passed'] = False
            raise RuntimeError('Nominal tear run did not record every preparation and interaction step')
        return 0 if record['passed'] else 2
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        NativeTissue.step, NativeTissue.check = step, check
        try:
            if capture is not None:
                capture.close()
                values = [r['native_ms'] for r in capture.rows['original']]
                if values and all(v is not None and np.isfinite(v) and v>0 for v in values):
                    record['native_ms'] = dict(mean=float(np.mean(values)), p95=float(np.percentile(values,95)))
        finally:
            guard.close()
            record['new_tear_edges_after_preparation'] = sum((r.get('failure_physics') or {}).get('new_tear_edges',0) for r in rows)
            atomic_json(out/'steps.json', rows)
            atomic_json(out/'result.json', record)
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
