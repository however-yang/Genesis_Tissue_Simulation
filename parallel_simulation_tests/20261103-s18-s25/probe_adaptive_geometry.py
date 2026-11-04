"""S18 preparation feasibility probe: follow settled material, no physical retuning.

Diagnostic commands only; not a released reset path or C1-C5 acceptance.
"""
import argparse,json,os,sys,traceback
from pathlib import Path
from dataclasses import replace,asdict
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',required=True);p.add_argument('--contact',type=int,choices=[1,2],required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
sys.path.insert(0,str(root/'source'))
import numpy as np
from threadpoolctl import threadpool_limits
from endoscopic_tissue.training.env import NativeEnvironment,EnvironmentConfig,SceneConfig
from endoscopic_tissue.training.geometry import WallGeometry
from endoscopic_tissue.training.collect import local_bounds
from endoscopic_tissue.training.provenance import sha256
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.scenario import stage_start,interpolate
r={'completed':False,'passed':False,'scope':'one adaptive preparation feasibility probe, not C1-C5 or integrated reset','contact_model':a.contact,'stages':[]};env=guard=None
try:
 guard=GPUReservation(a.gpu,a.output)
 dll=root/'native-s25/build/lib/libLiverCudaSim.so';geometry=WallGeometry(support_length_m=.008,height_m=.009,thickness_m=.005)
 scene=SceneConfig(preparation='settled',contact_model=a.contact,geometry=geometry)
 with threadpool_limits(limits=1):
  env=NativeEnvironment(EnvironmentConfig(str(dll),sha256(dll),local_bounds(scene),scene,max_steps=2000,cutting_enabled=True,
      outcome_semantics='physical_outcome_v1',diagnostic_continue=True,fast_topology_checks=True))
  state,context=env.reset(0);r['context']=context;r['geometry']=asdict(geometry)
  high=np.asarray(geometry.tab.high);rest=env.grid.positions
  selected=(env.grid.inside!=0)&(rest[:,0]>=high[0]-.4)&(rest[:,1]>=high[1]-.2)
  if not selected.any():raise ValueError('No prescribed material tip witnesses')
  shift=(env.observed.nodes[selected]-rest[selected]).mean(axis=0)
  r['tip_material_ids']=np.flatnonzero(selected).tolist();r['settled_tip_translation_m']=(shift*.01).tolist()
  requested=env.scene.stages[0].command;passed=True;acquired=False
  for stage in env.scene.stages[1:]:
   target=replace(stage.command,hinge=tuple(np.asarray(stage.command.hinge)+shift))
   start=stage_start(stage.name,requested,env.observed)
   for i in range(stage.steps):
    guard({os.getpid()});requested=interpolate(start,target,(i+1)/(stage.steps*stage.ramp))
    t=env.replay_command(requested)
    if t.after is None or env._invalid(t.after.metadata['quality']):
     r['first_failure']={'stage':stage.name,'step':env.step_id,'reason':t.reason,'quality':t.after.metadata['quality'] if t.after else None};passed=False;break
   row={'stage':stage.name,'step':env.step_id,'quality_passed':passed,'grasp':env.state.metadata['grasp']};r['stages'].append(row)
   if stage.name=='lift':
    grasp=env.state.metadata['grasp'];acquired=bool(env.observed.grip_points>0 if a.contact==2 else grasp['friction_hold_evidence'])
    r['hold_acquired']=acquired
    if not acquired:r['first_failure']={'stage':stage.name,'reason':'hold_not_acquired'};passed=False
   (a.output/'heartbeat.json').write_text(json.dumps(row))
   if not passed:break
  r['completed']=True;r['passed']=passed and acquired
except BaseException as error:r.update(error=repr(error),traceback=traceback.format_exc())
finally:
 if env is not None:
  if env.observed is not None:
   s=env.last_surface;np.savez_compressed(a.output/'final-state.npz',nodes_cm=env.observed.nodes,neighbors=env.observed.neighbors,vertices_cm=s.vertices if s else [],faces=s.triangles if s else [],anchors=s.anchors if s else [])
  env.close()
 if guard is not None:guard.close()
 (a.output/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in r.items() if k not in ('context','stages','tip_material_ids')},indent=2))
if not r['passed']:raise SystemExit(2)
