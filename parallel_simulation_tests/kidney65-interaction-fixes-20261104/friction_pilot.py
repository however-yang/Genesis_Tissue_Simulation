"""Strict zero-anchor friction acquisition/dose/release probe on A100 GPU1.

The frozen training adapter and its force/quality limits are used unchanged.
Both preparation failures and legitimate sliding are retained as evidence.
"""
import argparse
import ctypes
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
GPU='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'
SOURCE=ROOT.parents[1]/'parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/python-gpu-check'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',required=True);p.add_argument('--name',required=True)
    p.add_argument('--preparation',choices=('held','cut_ready'),default='held')
    p.add_argument('--task',choices=('dose','release'),default='dose')
    p.add_argument('--direction',choices=('plusx','minusx','plusy','plusz','minusz'),default='plusz')
    p.add_argument('--dose-mm',type=float,default=4.);p.add_argument('--mu',type=float)
    p.add_argument('--bite-inset-mm',type=float,default=0.,help='Explicit control-only variant: shift bite and subsequent grasper targets in -x; no physical parameter changes')
    p.add_argument('--settle-steps',type=int,choices=(0,30,60,120),default=0,help='Additional closed stationary steps at the end of the original lift ramp')
    p.add_argument('--adaptive-grasp',action='store_true',help='Opt-in isolated preparation: require five consecutive unchanged friction-hold checks, at most 180 stationary steps')
    p.add_argument('--open-jaw-preparation',action='store_true',help='Negative control: keep preparation jaws open; no held state should be accepted')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--no-video',action='store_true')
    a=p.parse_args()
    if os.name=='nt':p.error('All GPU tests must execute on server A100 GPU1')
    if not all(v.replace('-','').replace('_','').isalnum() for v in (a.library,a.name)):raise ValueError('Invalid label')
    if not 0<=a.dose_mm<=4:raise ValueError('Dose outside frozen 0-4 mm sweep')
    if not 0<=a.bite_inset_mm<=1:raise ValueError('Bite inset outside diagnostic 0-1 mm range')
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU,PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    sys.path.insert(0,str(SOURCE))
    import numpy as np
    from endoscopic_tissue.native_backend import NativeTissue
    from endoscopic_tissue.scenario import SCISSORS_SCENES,Command,interpolate
    if a.adaptive_grasp:
        if a.settle_steps or a.bite_inset_mm:raise ValueError('Test the adaptive protocol independently of position/fixed-duration variants')
        from endoscopic_tissue import training
        if 'endoscopic_tissue.training.env' in sys.modules:raise RuntimeError('Frozen adapter was already imported; refuse an ineffective overlay')
        training.__path__.insert(0,str(ROOT/'python/endoscopic_tissue/training'))
    from endoscopic_tissue.training.env import EnvironmentConfig,NativeEnvironment,SceneConfig,ResetFailure
    from endoscopic_tissue.training.actions import ActionBounds,command_to_action
    from endoscopic_tissue.training.physics import PhysicsOverrides
    from endoscopic_tissue.training.acceptance_queue import GPUReservation
    from endoscopic_tissue.training.data import atomic_json
    output=ROOT/'outputs'/a.name;output.mkdir(exist_ok=False)
    (output/'driver.py').write_bytes(Path(__file__).read_bytes())
    controls=ROOT/'controls'/a.name;controls.mkdir(exist_ok=False)
    guard=GPUReservation(GPU,controls)
    env=view=writer=None;rows=[];native_calls=[]
    record=dict(passed=False,completed=False,scope='One strict wall-traction pilot; not the 900-trial S16d matrix',
                gpu_uuid=GPU,arguments=vars(a),contact_model=1,default_replaced=False,full_surface_checks=True,
                timing_scope='LCS_Step wall time only; video and QA excluded; initial zero-gravity setup excluded from frame means')
    original_step,original_check=NativeTissue.step,NativeTissue.check
    def checked(self,name,*values):
        start=time.perf_counter();result=original_check(self,name,*values)
        if name=='LCS_Step':native_calls.append((time.perf_counter()-start)*1000)
        return result
    scene=SCISSORS_SCENES['wall-traction']
    if a.bite_inset_mm or a.settle_steps or a.open_jaw_preparation:
        stages=[];inside=False
        for stage in scene.stages:
            inside|=stage.name=='bite'
            command=stage.command
            if inside:command=replace(command,hinge=tuple(np.asarray(command.hinge)-np.array([a.bite_inset_mm*.1,0,0])))
            if a.open_jaw_preparation:command=replace(command,closure=0.)
            changed=replace(stage,command=command)
            if stage.name=='lift' and a.settle_steps:
                changed=replace(changed,steps=stage.steps+a.settle_steps,ramp=stage.steps*stage.ramp/(stage.steps+a.settle_steps))
            stages.append(changed)
        scene=replace(scene,stages=tuple(stages))
        # The adapter/geometry alias this same dictionary. Keep settings and
        # acceptance thresholds frozen; the full changed command log is saved.
        SCISSORS_SCENES['wall-traction']=scene
        record['control_preparation_variant']=dict(bite_and_later_x_offset_mm=-a.bite_inset_mm,stationary_closed_settle_steps=a.settle_steps,physical_parameters_changed=False)
        if a.open_jaw_preparation:record['control_preparation_variant']['jaws_always_open']=True
    endpoints=[];n=0
    for stage in scene.stages:
        n+=stage.steps;endpoints.append((n,stage.name))
    preparation_end=next(end for end,label in endpoints if label==('lift' if a.preparation=='held' else 'blade_approach'))
    def stepped(self,command):
        nonlocal view,writer
        guard({os.getpid()});start=len(native_calls)
        observed=original_step(self,command)
        assert observed.grip_points==0, 'Pure-friction test acquired auxiliary anchors'
        stage=next((label for end,label in endpoints if self.tick<=end),'post_preparation')
        if a.adaptive_grasp and self.tick>preparation_end:stage='friction_acquisition_settle'
        if getattr(env,'_pilot_phase',None):stage=env._pilot_phase
        rows.append(dict(tick=self.tick,stage=stage,native_ms=sum(native_calls[start:]),
                         command=asdict(command),jaw_load_n=list(observed.jaw_load_n),
                         gap_mm=float(observed.jaw_gap*10),hinge_mm=(observed.hinge*10).tolist(),anchors=observed.grip_points))
        if not a.no_video:
            import cv2
            if view is None:
                from endoscopic_tissue.rendering import EndoscopicView
                view=EndoscopicView(self.settings,1280,960,False,supersample=1,camera=scene.shot,
                    gpu_postprocess=True,gpu_aligned_depth=True,fused_mesh=True,persistent_soup=True)
                writer=cv2.VideoWriter(str(output/'interactions.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),1/self.settings.dt,(1280,960))
                if not writer.isOpened():raise RuntimeError('Could not open complete interaction video')
            view.update(self.surface(),observed);frame,_=view.render();writer.write(frame)
        return observed
    NativeTissue.check,NativeTissue.step=checked,stepped
    try:
        library=ROOT/'binaries'/(a.library+'.so');digest=hashlib.sha256(library.read_bytes()).hexdigest()
        assert digest==json.loads((ROOT/'results'/('build-'+a.library+'.json')).read_text())['sha256']
        record['library_sha256']=digest
        native=ctypes.CDLL(str(library))
        for name,values in {'LCS_SetCoreOptimizations':(43264,),'LCS_SetBatchQueryMode':(3,),
             'LCS_SetQueryThreads':(64,),'LCS_SetNativeSchedule':(512,1024),'LCS_SetCoarseBoundCache':(1,)}.items():
            fn=getattr(native,name);fn.argtypes=[ctypes.c_int]*len(values);fn.restype=ctypes.c_int;assert fn(*values)==0
        if not a.no_video:
            from endoscopic_tissue.training.egl_device import bind
            record['egl_binding']=bind(GPU)
        center=command_to_action(scene.stages[0].command);lo,hi=center-.2,center+.2
        lo[[3,10]]=0;hi[[3,10]]=1
        physics=None if a.mu is None else PhysicsOverrides(grasp_friction=a.mu)
        acquisition=dict(acquisition_max_steps=180,acquisition_stable_steps=5) if a.adaptive_grasp else {}
        config=EnvironmentConfig(str(library),digest,ActionBounds(tuple(lo),tuple(hi),(10.,)*11),
            SceneConfig('wall-traction',a.preparation,contact_model=1,physics=physics,**acquisition),
            max_steps=120,cutting_enabled=False,full_surface_checks=True,gpu_topology_checks=True)
        env=NativeEnvironment(config)
        import inspect
        adapter_source=Path(inspect.getfile(NativeEnvironment)).read_bytes()
        record['training_adapter_sha256']=hashlib.sha256(adapter_source).hexdigest()
        (output/'training-adapter.py').write_bytes(adapter_source)
        try:
            state,context=env.reset(a.seed)
        except ResetFailure as error:
            record['failure']='GRASP-01' if error.diagnostic.get('failure')=='hold_not_acquired' else 'preparation_quality_or_scope'
            record['reset_diagnostic']=env.last_reset_diagnostic or error.diagnostic
            if a.open_jaw_preparation:record['negative_control_passed']=record['failure']=='GRASP-01'
            if env.last_reset_failure_arrays:np.savez_compressed(output/'reset-failure.npz',**env.last_reset_failure_arrays)
            return 2
        if a.open_jaw_preparation:
            record.update(negative_control_passed=False,failure='Open-jaw preparation incorrectly accepted')
            return 2
        atomic_json(output/'context.json',context);atomic_json(output/'preparation.json',env.preparation)
        record['initial_grasp']=state.metadata['grasp'];assert record['initial_grasp']['friction_hold_evidence']
        witness_ids=np.asarray(record['initial_grasp']['witness_ids'],dtype=np.int64)
        assert len(witness_ids)>0 and np.all((witness_ids>=0)&(witness_ids<len(state.arrays['positions_m'])))
        witness_start=state.arrays['positions_m'][witness_ids].mean(axis=0)
        grasper_start=np.asarray(state.metadata['tools']['grasper_hinge_m'])
        if a.adaptive_grasp:
            stable=env.preparation[-5:]
            assert all(r['stage']=='friction_acquisition_settle' and
                       np.all(np.asarray(r['jaw_load_n'])>.05) and r['grip_points']==0 and
                       r['grasp']['witness_retained_fraction']>=.5 for r in stable)
            record['consecutive_five_steps_verified']=True
            record['acquisition_stationary_steps']=sum(r['stage']=='friction_acquisition_settle' for r in env.preparation)
        record['preparation_last10_jaw_load_range_n']={
            'min':np.min([r['jaw_load_n'] for r in env.preparation[-10:]],axis=0).tolist(),
            'max':np.max([r['jaw_load_n'] for r in env.preparation[-10:]],axis=0).tolist()}
        start=replace(Command(**env.preparation[-1]['requested_command_native']),scissors_cutting=False)
        directions=dict(plusx=(1,0,0),minusx=(-1,0,0),plusy=(0,1,0),plusz=(0,0,1),minusz=(0,0,-1))
        target=(replace(start,closure=0) if a.task=='release' else
                replace(start,hinge=tuple(np.asarray(start.hinge)+np.asarray(directions[a.direction])*(a.dose_mm*.1))))
        results=[]
        for i in range(120):
            env._pilot_phase='dose_ramp' if i<90 else 'dose_hold'
            requested=interpolate(start,target,min(1,(i+1)/90));step=env.step(command_to_action(requested))
            assert not step.action['clipped_coordinates'] and not step.action['rate_limited_coordinates']
            if step.after is None:record['failure']=step.reason;break
            metadata=step.after.metadata;quality=metadata['quality'];grasp=metadata['grasp']
            witness_delta=step.after.arrays['positions_m'][witness_ids].mean(axis=0)-witness_start
            grasper_delta=np.asarray(step.after.metadata['tools']['grasper_hinge_m'])-grasper_start
            results.append(dict(tick=env.native.tick,quality=quality,grasp=grasp,timings_ms=step.timings_ms,reason=step.reason,
                witness_mean_displacement_m=witness_delta.tolist(),grasper_displacement_m=grasper_delta.tolist()))
            if not quality['checked_metrics_passed'] or not quality['within_declared_local_scope']:
                record['failure']='quality_or_scope_limit';break
        atomic_json(output/'dose-steps.json',results)
        record.update(completed=len(results)==120,final_grasp=results[-1]['grasp'] if results else record['initial_grasp'])
        record['hold_evidence_at_end']=record['final_grasp']['friction_hold_evidence']
        if results:
            record['witness_mean_displacement_mm']=(np.asarray(results[-1]['witness_mean_displacement_m'])*1000).tolist()
            record['grasper_displacement_mm']=(np.asarray(results[-1]['grasper_displacement_m'])*1000).tolist()
            slips=[r['grasp'].get('witness_tangential_slip_m') for r in results]
            slips=[v for v in slips if v is not None and np.isfinite(v)]
            retained=[r['grasp'].get('witness_retained_fraction') for r in results]
            retained=[v for v in retained if v is not None and np.isfinite(v)]
            record['max_witness_tangential_slip_mm']=max(slips)*1000 if slips else None
            record['minimum_witness_retained_fraction']=min(retained) if retained else None
            record['first_loss_of_friction_evidence_tick']=next((r['tick'] for r in results if not r['grasp']['friction_hold_evidence']),None)
            record['first_majority_witness_exit_tick']=next((r['tick'] for r in results if r['grasp'].get('witness_retained_fraction') is not None and r['grasp']['witness_retained_fraction']<.5),None)
            initial_lift=record['initial_grasp'].get('witness_mean_lift_m')
            final_lift=record['final_grasp'].get('witness_mean_lift_m')
            record['witness_mean_lift_change_mm']=(final_lift-initial_lift)*1000 if initial_lift is not None and final_lift is not None and np.isfinite(initial_lift) and np.isfinite(final_lift) else None
        record['passed']=record['completed'] and not record.get('failure')
        if a.task=='release' and record['final_grasp']['friction_hold_evidence']:
            record.update(passed=False,failure='Open-jaw negative control retained friction evidence')
        return 0 if record['passed'] else 2
    except BaseException as error:
        record['error']=repr(error);raise
    finally:
        NativeTissue.check,NativeTissue.step=original_check,original_step
        if env is not None:env.close()
        if writer is not None:writer.release()
        if view is not None:view.close()
        guard.close()
        record['frames']=len(rows);record['all_anchor_counts_zero']=all(r['anchors']==0 for r in rows)
        if rows:
            values=np.array([r['native_ms'] for r in rows]);record['native_ms']=dict(mean=float(values.mean()),p95=float(np.percentile(values,95)),max=float(values.max()))
        record['sliding_is_not_a_solver_failure']=True
        atomic_json(output/'frame-timing.json',rows);atomic_json(output/'result.json',record)
        print(json.dumps({k:v for k,v in record.items() if k!='reset_diagnostic'}),flush=True)


if __name__=='__main__':raise SystemExit(main())
