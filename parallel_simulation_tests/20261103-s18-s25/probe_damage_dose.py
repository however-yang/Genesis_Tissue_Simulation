"""S19 tear=2 positive control with actual termination; controlled 4 mm pull without cutting; not calibration."""
import argparse,json,os,sys,traceback
from pathlib import Path
from dataclasses import replace
import numpy as np
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',required=True);p.add_argument('--direction',choices=['+y','+x','-z'],required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
sys.path.insert(0,str(root/'source'))
from threadpoolctl import threadpool_limits
from endoscopic_tissue.training.env import NativeEnvironment,EnvironmentConfig,SceneConfig
from endoscopic_tissue.training.physics import PhysicsOverrides
from endoscopic_tissue.training.collect import local_bounds
from endoscopic_tissue.training.data import EpisodeWriter,EpisodeReader
from endoscopic_tissue.training.actions import command_to_action
from endoscopic_tissue.training.provenance import sha256
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.scenario import Stage,stage_start,interpolate
r={'completed':False,'passed':False,'scope':'one tear=2 4 mm controlled pull, not calibration','events':[]};env=guard=writer=None
try:
 guard=GPUReservation(a.gpu,a.output)
 dll=root/'native-s25/build/lib/libLiverCudaSim.so';scene=SceneConfig(preparation='cut_ready',physics=PhysicsOverrides(tear_stretch_ratio=2.))
 with threadpool_limits(limits=1):
  env=NativeEnvironment(EnvironmentConfig(str(dll),sha256(dll),local_bounds(scene),scene,cutting_enabled=False,max_steps=120,
      outcome_semantics='physical_outcome_v1',fast_topology_checks=True))
  state,context=env.reset(0);writer=EpisodeWriter(a.output/'episode',context,state,env.static_arrays())
  requested=next(s.command for s in env.scene.stages if s.name=='blade_approach')
  direction={'+y':(0,1,0),'+x':(1,0,0),'-z':(0,0,-1)}[a.direction]
  target=replace(requested,hinge=tuple(np.asarray(requested.hinge)+np.asarray(direction)*.4),scissors_cutting=False)
  r.update(direction=a.direction,dose_mm=4,ramp_steps=90,hold_steps=30);stop=False
  for stage in [Stage('dose_ramp',90,target),Stage('dose_hold',30,target)]:
   start=stage_start(stage.name,requested,env.observed)
   for i in range(stage.steps):
    guard({os.getpid()});requested=interpolate(start,stage.command,(i+1)/(stage.steps*stage.ramp))
    # Private diagnostic driver preserves exact scripted flags while exercising
    # the ordinary terminating pipeline (diagnostic_continue is False).
    t=env._advance(requested,{'policy':command_to_action(requested).tolist(),'nominal_control_test':True})
    writer.append(t)
    if t.after:r['events'].extend(t.after.metadata.get('events',[]))
    if t.terminated or t.truncated:
     r.update(reason=t.reason,valid_outcome=t.valid_outcome,stage=stage.name,step=env.step_id,
              quality=t.after.metadata['quality'] if t.after else None,damage=t.after.metadata.get('failure_physics') if t.after else None)
     r['passed']=bool(t.reason=='damage:tear_separation' and t.valid_outcome and t.after is not None);stop=True;break
   (a.output/'heartbeat.json').write_text(json.dumps({'stage':stage.name,'step':env.step_id,'stop':stop}))
   if stop:break
  r['final_damage']=env.state.metadata.get('failure_physics');r['final_grasp']=env.state.metadata.get('grasp')
  writer.close(r.get('reason') or 'script_complete');writer=None
  reader=EpisodeReader(a.output/'episode');r['reader_outcomes']=reader.outcome_counts();r['completed']=True
except BaseException as error:r.update(error=repr(error),traceback=traceback.format_exc())
finally:
 if writer is not None:writer.close('probe_failure')
 if env is not None:env.close()
 if guard is not None:guard.close()
 (a.output/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps(r,indent=2))
if not r['passed']:raise SystemExit(2)
