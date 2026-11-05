"""Minimal unchanged-R33 replay of the saved failed regrasp fork on GPU1."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'s15_checks'))
from bootstrap import GPU
from validation_support import environment,command_sequence
from endoscopic_tissue.native_backend import NativeTissue
from endoscopic_tissue.training.actions import action_to_command
from endoscopic_tissue.training.kidney_snapshot import EnvSnapshot
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.training.data import atomic_json
import numpy as np
from state_contacts import blade_contacts,deepest_body_sample


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True)
    p.add_argument('--source',default='s15-v1-formal20-01-regrasp-02')
    a=p.parse_args()
    if os.name=='nt':p.error('GPU tests run on server A100 GPU1')
    if not all(v.replace('-','').isalnum() for v in (a.name,a.source)):p.error('Use simple labels')
    out=ROOT/'outputs'/a.name;out.mkdir(exist_ok=False)
    for filename in ('replay_regrasp_overlap.py','state_contacts.py'):(out/filename).write_bytes((ROOT/filename).read_bytes())
    original=json.loads((ROOT/'outputs'/a.source/'result.json').read_text())
    source=ROOT/'outputs'/a.source/'fork.lcssnap'
    snapshot=EnvSnapshot.load(source,dll_sha256=original['dll_sha256'],
        physics_config_sha256=original['snapshot_metadata']['physics_config_sha256'])
    controls=ROOT/'controls'/a.name;controls.mkdir(exist_ok=False)
    guard=GPUReservation(GPU,controls);env=view=writer=None
    record=dict(passed=False,diagnostic_reproduced=False,gpu_uuid=GPU,
        default_replaced=False,source_run=a.source,source_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Complete saved-fork continuation only; original full-prefix video is retained in the source run',
        timing_scope='LCS_Step only; first-failure archive/CPU inspection is outside native timing')
    check=NativeTissue.check;step_native=NativeTissue.step;native_ms=[];rows=[]
    def checked(self,name,*values):
        start=time.perf_counter();value=check(self,name,*values)
        if name=='LCS_Step':native_ms.append((time.perf_counter()-start)*1000)
        return value
    def guarded(self,command):guard({os.getpid()});return step_native(self,command)
    NativeTissue.check,NativeTissue.step=checked,guarded
    try:
        from endoscopic_tissue.training.egl_device import bind
        bind(GPU)
        env=environment(dll='r33-packed-mg',preparation='held')
        env.reset(original['repeat']);state,context=env.restore(snapshot,branch_label=a.name)
        assert env.native.tick==original['fork_native_tick']
        atomic_json(out/'context.json',context)
        # replay_command bypasses the policy limiter and the old adapter leaves
        # previous_action at reset's target. Use its explicitly recorded last
        # diagnostic request, not that stale limiter member, to reproduce the
        # original continuation. This does not change the restored state.
        last_request=np.asarray(state.metadata['previous_target_si'])
        record['diagnostic_limiter_target_differs_from_last_request']=not np.array_equal(last_request,env.previous_action)
        prior=replace(action_to_command(last_request,cutting_enabled=False),scissors_cutting=False)
        stages=[replace(s,command=replace(s.command,hinge=prior.hinge)) for s in env.scene.stages
                if s.name in ('reposition_2','incise_2')]
        sequence=command_sequence(stages,prior)
        import cv2
        from endoscopic_tissue.rendering import EndoscopicView
        cv2.setNumThreads(1)
        view=EndoscopicView(env.settings,1280,960,False,supersample=1,camera=env.scene.shot,
            gpu_postprocess=True,gpu_aligned_depth=True,fused_mesh=True,persistent_soup=True)
        writer=cv2.VideoWriter(str(out/'interactions.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,960))
        assert writer.isOpened()
        def frame():
            view.update(env.last_surface,env.observed);image,_=view.render()
            image=image.copy()  # EGL readback may expose a non-contiguous view.
            cv2.putText(image,f'Saved-fork continuation | native tick {env.native.tick}',(12,24),
                cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            writer.write(image)
        frame()
        for label,index,command in sequence:
            offset=len(native_ms);result=env.replay_command(command)
            assert result.after is not None,result.reason
            quality=result.after.metadata['quality']
            rows.append(dict(tick=env.native.tick,stage=label,stage_step=index,command=asdict(command),
                             quality=quality,native_ms=sum(native_ms[offset:])))
            frame()
            if not quality['checked_metrics_passed'] and 'first_failure_tick' not in record:
                record['first_failure_tick']=env.native.tick
                cpu=deepest_body_sample(env.last_surface,env.native.scissors_geometry,env.observed.scissors)
                expected=quality['metrics']['blade_body_overlap_m']*100
                np.testing.assert_allclose(max(0,cpu['depth_cm']),expected,rtol=3e-6,atol=1e-7)
                record['independent_cpu_body_witness']=cpu
                saved=env.snapshot();saved.save(out/'first-overlap.lcssnap')
                contacts,points,gaps,checks=blade_contacts(saved.native_bytes(),env.observed)
                body=np.flatnonzero(contacts['kind']==cpu['kind'])
                distances=np.linalg.norm(points[body]-np.asarray(cpu['point_cm']),axis=1)
                nearest=body[np.argsort(distances)[:8]]
                record['contact_layout_checks']=checks
                record['nearest_same_body_contacts']=[dict(kind=int(contacts[i]['kind']),count=int(contacts[i]['count']),
                    point_cm=points[i].tolist(),normal=contacts[i]['normal'].tolist(),plane_gap_mm=float(gaps[i]*10),
                    distance_to_deepest_sample_mm=float(np.linalg.norm(points[i]-np.asarray(cpu['point_cm']))*10)) for i in nearest]
                np.savez_compressed(out/'first-failure-arrays.npz',positions=env.observed.nodes,
                    vertices=env.last_surface.vertices,triangles=env.last_surface.triangles,
                    scissors_hinge=env.observed.scissors.hinge,scissors_axis=env.observed.scissors.axis,
                    scissors_normal=env.observed.scissors.normal,scissors_gap=env.observed.scissors.gap,
                    backs=env.native.scissors_geometry.backs)
                atomic_json(out/'first-failure.json',record)
        record.update(diagnostic_reproduced=any('tool_body_overlap_exceeded' in r['quality']['numerical_failure_reasons'] for r in rows),
            quality_failure_steps=sum(not r['quality']['checked_metrics_passed'] for r in rows),
            peak_body_overlap_mm=max(r['quality']['metrics']['blade_body_overlap_m'] for r in rows)*1000,
            completed_continuation=True,steps=len(rows),video_frames=len(rows)+1)
        return 0 if record['diagnostic_reproduced'] else 2
    except BaseException as error:
        record['error']=repr(error);raise
    finally:
        if env is not None:env.close()
        if writer is not None:writer.release()
        if view is not None:view.close()
        NativeTissue.check,NativeTissue.step=check,step_native;guard.close()
        atomic_json(out/'steps.json',rows);atomic_json(out/'result.json',record)
        print(json.dumps(record),flush=True)


if __name__=='__main__':raise SystemExit(main())
