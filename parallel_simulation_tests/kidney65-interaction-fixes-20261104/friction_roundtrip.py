"""Opt-in controller test; old and new friction witnesses remain separate.

No monitor writes native positions, forces, contacts, material, or anchors.
The return-to-bite controller uses material IDs, so this is a diagnostic scripted
controller and does not demonstrate a vision-only policy.
"""
from dataclasses import asdict, replace
import numpy as np


def run(env, start, direction, dose_mm, output, *, regrasp_lower_mm=0., return_nominal_bite=False):
    from endoscopic_tissue.scenario import interpolate
    from endoscopic_tissue.training.actions import command_to_action
    from endoscopic_tissue.training.data import atomic_json
    from endoscopic_tissue.training.grasp import GraspMonitor

    rows = []
    initial = env.state.arrays['positions_m']
    old_ids = np.array(env.state.metadata['grasp']['witness_ids'], dtype=np.int64)
    assert len(old_ids) and env.settings.contact_model == 1
    old_center = initial[old_ids].mean(axis=0)
    new_monitor = None
    requested = start

    def tick(command, phase):
        nonlocal requested
        env._pilot_phase = phase
        step = env.step(command_to_action(command))
        assert not step.action['clipped_coordinates'] and not step.action['rate_limited_coordinates']
        assert step.after is not None, step.reason
        q = step.after.metadata['quality']
        assert q['checked_metrics_passed'] and q['within_declared_local_scope'], q
        assert env.observed.grip_points == 0
        fresh = new_monitor.assess(step.after) if new_monitor is not None else None
        row = dict(tick=env.native.tick, phase=phase, native_command=asdict(command),
            old_grasp=step.after.metadata['grasp'], new_grasp=fresh,
            old_witness_center_m=step.after.arrays['positions_m'][old_ids].mean(axis=0).tolist(),
            hinge_m=step.after.metadata['tools']['grasper_hinge_m'],
            quality=q, timings_ms=step.timings_ms)
        rows.append(row);requested = command
        atomic_json(output/'roundtrip-progress.json', dict(frames=len(rows),phase=phase,tick=env.native.tick))
        return row

    def stage(target, frames, ramp_frames, phase):
        origin = requested
        for i in range(frames):tick(interpolate(origin,target,min(1,(i+1)/ramp_frames)),phase)

    report = dict(passed=False, scope='One scripted pure-friction lift/open/withdraw/regrasp/relift; not S16d',
        new_monitor_is_read_only=True, persistent_old_witnesses_retained=True,
        return_controller_uses_material_ids=True,
        training_adapter_automatic_witness_reacquisition_implemented=False)
    try:
        lifted = replace(start,hinge=tuple(np.asarray(start.hinge)+np.asarray(direction)*dose_mm*.1))
        stage(lifted,120,90,'first_lift')
        first = rows[-120:]
        report['first_lift'] = dict(all_friction_evidence=all(r['old_grasp']['friction_hold_evidence'] for r in first),
            material_delta_mm=((np.array(first[-1]['old_witness_center_m'])-old_center)*1000).tolist(),
            minimum_retention=min(r['old_grasp']['witness_retained_fraction'] for r in first))
        assert report['first_lift']['all_friction_evidence'], 'First small lift lost strict friction evidence'
        opened = replace(lifted,closure=0.)
        stage(opened,120,60,'open_and_settle')
        report['open_control'] = dict(last10_no_friction_evidence=all(not r['old_grasp']['friction_hold_evidence'] for r in rows[-10:]),
            material_return_delta_mm=((np.array(rows[-1]['old_witness_center_m'])-
                np.array(first[-1]['old_witness_center_m']))*1000).tolist())
        assert report['open_control']['last10_no_friction_evidence'], 'Opening did not release strict hold evidence'
        axis = np.asarray(env.settings.jaw_axis)
        withdrawn = replace(opened,hinge=tuple(np.asarray(opened.hinge)-axis*.4+np.array([0,.2,0])))
        stage(withdrawn,60,60,'open_withdraw')
        material_delta = env.state.arrays['positions_m'][old_ids].mean(axis=0)-old_center
        approach = replace(start,closure=0.,hinge=tuple(np.asarray(start.hinge)+material_delta*100-
            np.array([0,regrasp_lower_mm*.1,0])))
        if return_nominal_bite:
            # The old witnesses stay stretched and move away from the first
            # bite. Align from outside along the original insertion direction
            # before re-entering the known thicker tissue with open jaws.
            approach = replace(start, closure=0., hinge=tuple(np.asarray(start.hinge)-
                np.array([0,regrasp_lower_mm*.1,0])))
            pre_bite = replace(approach, hinge=tuple(np.asarray(approach.hinge)-axis*.6))
            stage(pre_bite,60,60,'nominal_bite_align_open')
        report['tracked_return_delta_mm'] = (material_delta*1000).tolist()
        report['additional_return_lower_mm'] = regrasp_lower_mm
        report['return_nominal_bite'] = return_nominal_bite
        stage(approach,60,60,'tracked_return_open')
        # Preserve the original monitor and all its old-material slip records.
        # This independent read-only monitor certifies a NEW bite only after
        # the explicit open control above. No reset during uncommanded slip.
        new_monitor = GraspMonitor(env.settings,env.grid.inside)
        closed = replace(approach,closure=1.)
        stage(closed,90,90,'new_bite_close')
        stable = 0
        for _ in range(180):
            r = tick(closed,'new_bite_settle')
            stable = stable+1 if r['new_grasp']['friction_hold_evidence'] else 0
            if stable >= 5:break
        report['new_bite_consecutive_five_steps'] = stable >= 5
        assert stable >= 5, 'New bite did not acquire five consecutive strict friction checks'
        new_ids = new_monitor.ids.copy()
        before = env.state.arrays['positions_m'][new_ids].mean(axis=0)
        relift = replace(closed,hinge=tuple(np.asarray(closed.hinge)+np.asarray(direction)*.2))
        stage(relift,90,60,'second_lift_2mm')
        second = rows[-90:]
        report['second_lift'] = dict(all_friction_evidence=all(r['new_grasp']['friction_hold_evidence'] for r in second),
            material_delta_mm=((env.state.arrays['positions_m'][new_ids].mean(axis=0)-before)*1000).tolist(),
            minimum_retention=min(r['new_grasp']['witness_retained_fraction'] for r in second),
            first_evidence_loss_tick=next((r['tick'] for r in second if not r['new_grasp']['friction_hold_evidence']),None),
            final_new_grasp=second[-1]['new_grasp'], final_persistent_old_grasp=second[-1]['old_grasp'])
        assert report['second_lift']['all_friction_evidence'], 'Second lift lost strict friction evidence'
        report['passed'] = True
        return report
    except Exception as error:
        report['error'] = repr(error)
        return report
    finally:
        report['post_preparation_frames'] = len(rows)
        atomic_json(output/'roundtrip-steps.json',rows)
        atomic_json(output/'roundtrip-result.json',report)
