"""S18 geometry pilot: strict preparation then nominal continuation, early rejection."""
import argparse,json,os,sys,traceback
from pathlib import Path
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',required=True);p.add_argument('--support-mm',type=float,default=4.);p.add_argument('--contact',type=int,choices=[1,2],required=True);p.add_argument('--scene',default='wall-traction',choices=['wall','wall-traction','flap','kidney']);p.add_argument('--collision',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(a.output/'cache'))
sys.path.insert(0,str(root/'source'))
import numpy as np
from threadpoolctl import threadpool_limits
from endoscopic_tissue.training.env import NativeEnvironment,EnvironmentConfig,SceneConfig
from endoscopic_tissue.training.geometry import WallGeometry
from endoscopic_tissue.training.collect import local_bounds
from endoscopic_tissue.training.provenance import sha256
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.scenario import stage_start,interpolate
r={'completed':False,'passed':False,'scope':'single geometry pilot, not C1-C5 acceptance','support_mm':a.support_mm,'contact_model':a.contact,'steps':[]};env=guard=None
try:
 guard=GPUReservation(a.gpu,a.output)
 dll=root/'native-s25/build/lib/libLiverCudaSim.so';scene=SceneConfig(scene=a.scene,preparation='settled' if a.scene=='wall' else 'cut_ready',contact_model=a.contact,geometry=WallGeometry(support_length_m=a.support_mm*.001) if a.support_mm else None)
 with threadpool_limits(limits=1):
  env=NativeEnvironment(EnvironmentConfig(str(dll),sha256(dll),local_bounds(scene),scene,max_steps=2000,cutting_enabled=True,outcome_semantics='physical_outcome_v1',diagnostic_continue=True,fast_topology_checks=True,instrument_collision_checks=a.collision))
  state,context=env.reset(seed=0);r['context']=context;r['preparation_passed']=True;print('PREPARED',flush=True)
  start_name={'settled':'settle','cut_ready':'blade_approach'}[scene.preparation]
  start_index=next(i for i,s in enumerate(env.scene.stages) if s.name==start_name)
  requested=env.scene.stages[start_index].command;passed=True
  for stage in env.scene.stages[start_index+1:]:
   start=stage_start(stage.name,requested,env.observed)
   for i in range(stage.steps):
    guard({os.getpid()});requested=interpolate(start,stage.command,(i+1)/(stage.steps*stage.ramp))
    t=env.replay_command(requested)
    r['steps'].append(dict(stage=stage.name,step=env.step_id,quality=t.after.metadata['quality'] if t.after else None,
                           outcome=t.after.metadata.get('outcome') if t.after else None,reason=t.reason))
    if t.after is None or env._invalid(t.after.metadata['quality']):passed=False;break
   (a.output/'heartbeat.json').write_text(json.dumps({'stage':stage.name,'steps':env.step_id,'valid':passed}))
   if not passed:break
  r['completed']=True;r['passed']=passed
except BaseException as error:
 r.update(error=repr(error),traceback=traceback.format_exc(),reset_diagnostic=env.last_reset_diagnostic if env else None)
finally:
 if env is not None:
  if env.observed is not None:
   s=env.last_surface
   np.savez_compressed(a.output/'final-state.npz',nodes_cm=env.observed.nodes,neighbors=env.observed.neighbors,
     vertices_cm=s.vertices if s else [],faces=s.triangles if s else [],anchors=s.anchors if s else [])
  env.close()
 if guard is not None:guard.close()
 (a.output/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in r.items() if k not in ('steps','context','reset_diagnostic')},indent=2))
if not r['passed']:raise SystemExit(2)
