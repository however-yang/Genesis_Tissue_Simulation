"""S18 preparation feasibility probe: follow settled material for bite, then lift toward original geometric target; no physical retuning.

Diagnostic commands only; not a released reset path or C1-C5 acceptance.
"""
import argparse,json,os,sys,traceback
from pathlib import Path
from dataclasses import replace,asdict
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',required=True);p.add_argument('--contact',type=int,choices=[2],default=2);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
sys.path.insert(0,str(root/'source'))
import numpy as np
import hashlib
from types import SimpleNamespace
from threadpoolctl import threadpool_limits
from endoscopic_tissue.training.env import NativeEnvironment,EnvironmentConfig,SceneConfig
from endoscopic_tissue.training.geometry import WallGeometry
from endoscopic_tissue.training.collect import local_bounds
from endoscopic_tissue.training.provenance import sha256
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.scenario import Stage,Sequence,stage_start,interpolate
from endoscopic_tissue.training.snapshot import SnapshotBackend
r={'completed':False,'passed':False,'scope':'one assisted geometry screening: 0/4 mm added pull, not registered C1, no repetitions or CI, not calibrated failure dose','contact_model':a.contact,'stages':[]};env=guard=None
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
   if stage.name=='incise_2':break
   target=replace(stage.command,hinge=tuple(np.asarray(stage.command.hinge)+(shift if stage.name in ('approach','bite','close') else 0)))
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
  
  if not (passed and acquired):raise RuntimeError('Preparation failed; no dose branches executed')
  baseline_command=requested
  snapshot=env.snapshot()
  original_hash=hashlib.sha256(snapshot.native_bytes()).hexdigest()
  r['snapshot_roundtrips']=[]
  for repeat in range(5):
   env.restore(snapshot,branch_label='geometry-restore-'+str(repeat))
   exact=hashlib.sha256(SnapshotBackend(env.native).save()).hexdigest()==original_hash
   r['snapshot_roundtrips'].append(exact)
   if not exact:raise RuntimeError('New geometry native snapshot mismatch')
  seq=Sequence(SimpleNamespace(neighbors=env.grid.neighbors,nodes=env.grid.positions),env.settings,env.scene,env.grid.positions,env.grid.inside)
  ids,column,ncols=seq.column_links(env.scene.tab.low,env.scene.tab.high)
  rest=env.grid.positions[ids];across=[k for k in range(3) if k!=env.scene.plane_axis]
  keys=np.unique(np.round(rest[:,across],4),axis=0)
  def progress():
   cuts=seq.column_cuts(env.observed.neighbors,env.scene.tab.low,env.scene.tab.high)
   xs=np.sort(np.unique(keys[:,0]))[::-1];full=[bool(np.all(cuts[keys[:,0]==x]>0)) for x in xs]
   prefix=next((i for i,cut in enumerate(full) if not cut),len(full))
   return dict(severed_columns=int(np.count_nonzero(cuts)),total_columns=len(cuts),x_from_tip_cm=xs.tolist(),fully_severed_across_thickness=full,continuous_prefix_mm=float(prefix*1.))
  r['diagnostic_column_definition']='original Sequence y-link cuts in tab box; for each descending rest-x require all z columns severed, 1 mm spacing; not asserted identical to external T0 implementation'
  r['branches']=[]
  incision=next(s for s in env.scene.stages if s.name=='incise_2')
  for label,delta in [('zero',(0,0,0)),('tangent4',(-.4,0,0)),('normal4',(0,.4,0))]:
   env.restore(snapshot,branch_label=label);before_progress=progress();requested=baseline_command
   target=replace(requested,hinge=tuple(np.asarray(requested.hinge)+delta),scissors_cutting=False)
   cut_target=replace(incision.command,hinge=target.hinge)
   branch=dict(label=label,added_pull_mm=(np.asarray(delta)*10).tolist(),before=before_progress,events=[],valid=True)
   for stage in [Stage('dose_ramp',90,target),Stage('dose_hold',30,target),Stage('incise_2',incision.steps,cut_target,ramp=incision.ramp)]:
    start=stage_start(stage.name,requested,env.observed)
    for i in range(stage.steps):
     guard({os.getpid()});requested=interpolate(start,stage.command,(i+1)/(stage.steps*stage.ramp));t=env.replay_command(requested)
     if t.after is None or env._invalid(t.after.metadata['quality']):
      branch.update(valid=False,failure_stage=stage.name,quality=t.after.metadata['quality'] if t.after else None);break
     branch['events'].extend(t.after.metadata.get('events',[]))
    if not branch['valid']:break
   branch.update(after=progress(),grasper_gap_m=env.state.metadata['tools']['grasper_gap_m'],final_grasp=env.state.metadata['grasp'])
   branch['continuous_increment_mm']=branch['after']['continuous_prefix_mm']-branch['before']['continuous_prefix_mm']
   r['branches'].append(branch)
   (a.output/'heartbeat.json').write_text(json.dumps(dict(branch=label,step=env.step_id)))
  r['completed']=True;r['passed']=False
  r['pilot_all_branches_valid']=all(b['valid'] for b in r['branches'])
  r['acceptance']='screening only; no statistical coupling certification'

except BaseException as error:r.update(error=repr(error),traceback=traceback.format_exc())
finally:
 if env is not None:
  if env.observed is not None:
   s=env.last_surface;np.savez_compressed(a.output/'final-state.npz',nodes_cm=env.observed.nodes,neighbors=env.observed.neighbors,vertices_cm=s.vertices if s else [],faces=s.triangles if s else [],anchors=s.anchors if s else [])
  env.close()
 if guard is not None:guard.close()
 (a.output/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in r.items() if k not in ('context','stages','tip_material_ids')},indent=2))
if not r['passed']:raise SystemExit(2)
