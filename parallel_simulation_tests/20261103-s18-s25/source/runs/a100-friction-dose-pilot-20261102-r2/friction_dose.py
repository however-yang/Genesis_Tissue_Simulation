"""S16d fresh pure-friction doses; zero auxiliary points is required, not failure.

Use --pilot for one +z, 4 mm trial from each required preparation state.
The full curve is two preparations x five directions x nine doses x ten runs.
Initial acquisition failures are retained as GRASP-01, never turned into anchors.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--library-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--capture', action='store_true')
    args = p.parse_args()
    root = args.release.resolve(strict=True)
    library = args.library_dir.resolve(strict=True)
    sys.path[:0] = [str(root), str(root/'endoscopic_tissue_backups/2026-11-01-s15-s17')]
    from threadpoolctl import threadpool_limits
    from validation_support import environment, command_sequence, row, quality_summary
    from validation_guard import stop_reasons
    from endoscopic_tissue.scenario import Command, Stage
    from endoscopic_tissue.training.env import ResetFailure
    from endoscopic_tissue.training.data import atomic_json
    from endoscopic_tissue.training.provenance import sha256
    from functional_worker import selected_gpu_memory, GPU
    from single_capture import NativeCapture
    selected_gpu_memory()
    args.output.mkdir(parents=True, exist_ok=False)
    directions = {'+x': (1,0,0), '-x': (-1,0,0), '+y': (0,1,0), '+z': (0,0,1), '-z': (0,0,-1)}
    curve = [('+z', 4., 0)] if args.pilot else [
        (direction, float(dose), repeat) for direction in directions
        for dose in np.arange(0, 4.01, .5) for repeat in range(10)]
    record = dict(schema='s16d_friction_dose_v1', pilot=args.pilot, gpu_uuid=GPU,
        library_sha256=sha256(library/'LiverCudaSim.dll'), worker_sha256=sha256(__file__),
        required_trials=2*len(curve), full_required_trials=900, trials=[], completed=False,
        captured=args.capture, timing_scope='includes complete capture work' if args.capture else 'uncaptured training step',
        full_surface_checks=True, fresh_preparations=True, default_replaced=False,
        candidate_adopted=False, started=time.time())
    try:
        with threadpool_limits(limits=1, user_api='blas'):
            for preparation in ['held', 'cut_ready']:
                for direction, dose, repeat in curve:
                    label=f'{preparation}-{direction.replace("+","plus").replace("-","minus")}-{dose:.1f}-r{repeat:02}'
                    folder=args.output/label
                    folder.mkdir(exist_ok=False)
                    trial=dict(case=label, preparation=preparation, direction=direction,
                        dose_mm=dose, repeat=repeat, acquisition_passed=False, completed_requested_steps=False)
                    env=environment(dll=str(library), preparation=preparation, telemetry=False, contact=1)
                    # Stop invalid states using exactly the existing quality/scope limits.
                    env.config=replace(env.config, diagnostic_continue=False)
                    rows=[]
                    capture=NativeCapture(folder, 'wall-traction', label) if args.capture else None
                    try:
                        try:
                            env.reset(repeat)
                        except ResetFailure as error:
                            diagnostic=env.last_reset_diagnostic or error.diagnostic
                            atomic_json(folder/'reset-failure.json', diagnostic)
                            if env.last_reset_failure_arrays:
                                np.savez_compressed(folder/'reset-failure.npz', **env.last_reset_failure_arrays)
                            trial.update(failure='GRASP-01' if diagnostic.get('failure')=='hold_not_acquired'
                                         else 'preparation_quality_or_scope_failure',
                                         failure_details=diagnostic.get('failure'))
                            continue
                        assert env.settings.contact_model == 1 and env.observed.grip_points == 0
                        if capture:
                            capture.expected=env.native.tick+120
                        atomic_json(folder/'context.json', env.context)
                        atomic_json(folder/'preparation.json', env.preparation)
                        first_grasp=env.state.metadata['grasp']
                        assert first_grasp['friction_hold_evidence'] and not first_grasp['assisted']
                        trial.update(acquisition_passed=True, initial_grasp=first_grasp,
                                     first_geometric_hold_loss=None, first_friction_evidence_loss=None)
                        # held ends at lift; cut_ready ends at blade_approach. Use
                        # the actual final request, not an approach teleport.
                        start=Command(**env.preparation[-1]['requested_command_native'])
                        target=replace(start, hinge=tuple(np.asarray(start.hinge)+np.asarray(directions[direction])*dose*.1),
                                       scissors_cutting=False)
                        start=replace(start, scissors_cutting=False)
                        hinge0=env.observed.hinge.copy()
                        sequence=command_sequence([Stage('dose_ramp',90,target),Stage('dose_hold',30,target)],start)
                        for stage,index,command in sequence:
                            if capture:
                                capture.label=stage
                            result=env.replay_command(command)
                            if result.after is None:
                                trial['terminal_without_state']=dict(reason=result.reason, stage=stage, stage_step=index)
                                break
                            item=row(env,result,stage,index)
                            assert item['grasp']['anchor_points'] == 0 and not item['grasp']['assisted']
                            item['requested_displacement_mm']=((np.asarray(command.hinge)-start.hinge)*10).tolist()
                            item['actual_hinge_displacement_mm']=((env.observed.hinge-hinge0)*10).tolist()
                            event={key:item[key] for key in ['stage','stage_step','native_tick','requested_displacement_mm',
                                                            'actual_hinge_displacement_mm','grasp']}
                            if trial['first_geometric_hold_loss'] is None and not item['grasp']['geometric_loaded_hold']:
                                trial['first_geometric_hold_loss']=event
                            if trial['first_friction_evidence_loss'] is None and not item['grasp']['friction_hold_evidence']:
                                trial['first_friction_evidence_loss']=event
                            rows.append(item)
                            reasons=stop_reasons(item['quality'])
                            if reasons or not item['quality']['checked_metrics_passed']:
                                reasons=reasons or item['quality']['numerical_failure_reasons']
                                trial['terminated_at_existing_limit']=reasons
                                np.savez_compressed(folder/'terminal-state.npz',nodes=env.observed.nodes,
                                    neighbors=env.observed.neighbors,rest=env.grid.positions,material=env.grid.inside)
                                break
                        trial.update(completed_requested_steps=len(rows)==120, quality=quality_summary(rows),
                                     final_grasp=rows[-1]['grasp'] if rows else first_grasp)
                    finally:
                        env.close()
                        if capture:
                            capture.close()
                        if rows:
                            atomic_json(folder/'steps.json',rows)
                        atomic_json(folder/'result.json',trial)
                        record['trials'].append(trial)
                        atomic_json(args.output/'result.json',record)
                        print(label,trial['acquisition_passed'],trial['completed_requested_steps'],flush=True)
        record['completed']=len(record['trials'])==record['required_trials']
        record['acquisition_failures']=sum(not t['acquisition_passed'] for t in record['trials'])
        record['quality_and_scope_passed']=all(t['completed_requested_steps'] and
            t['quality']['quality_failure_steps']==t['quality']['scope_failure_steps']==0 for t in record['trials'])
        record['pilot_passed']=bool(args.pilot and record['completed'] and record['quality_and_scope_passed'])
        record['full_curve_passed']=bool(not args.pilot and record['completed'] and record['quality_and_scope_passed'])
        record['semantics']='Sliding/loss of hold under increasing dose is measured, not a solver failure. '
        record['semantics']+='Acquisition, quality and scope failures remain rejected. No auxiliary anchors or altered friction.'
        return 0 if record['quality_and_scope_passed'] else 2
    except BaseException as error:
        record['error']=repr(error)
        raise
    finally:
        record['wall_s']=time.time()-record['started']
        atomic_json(args.output/'result.json',record)


if __name__ == '__main__':
    raise SystemExit(main())
