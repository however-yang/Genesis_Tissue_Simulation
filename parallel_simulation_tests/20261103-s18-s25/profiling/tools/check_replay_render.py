"""Same-snapshot A/A and A/B rendering replay; do not invent a physics tolerance."""
import argparse,json,os,sys,time,traceback,hashlib
from pathlib import Path
r=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
os.sched_setaffinity(0,[7,19,20,25]);os.environ.update(CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',XDG_CACHE_HOME=str(a.output/'cache'),TMPDIR=str(r/'tmp'))
sys.path.insert(0,str(r/'candidate-v2-source'))
import numpy as np
from threadpoolctl import threadpool_limits
from endoscopic_tissue.training.env import NativeEnvironment,EnvironmentConfig,SceneConfig
from endoscopic_tissue.training.collect import local_bounds
from endoscopic_tissue.training.provenance import sha256
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.training.snapshot import SnapshotBackend
from endoscopic_tissue.training.egl_device import bind
from endoscopic_tissue.training.render_layout import SurfaceLayoutCache
from endoscopic_tissue.rendering import EndoscopicView
result=dict(completed=False,physical_checks_passed=False,branches=[],comparisons=[],scope='40 identical close actions per branch from one snapshot; changes renderer only, not full physical-domain acceptance');env=view=post=guard=None;arrays={}
try:
 guard=GPUReservation(os.environ['CUDA_VISIBLE_DEVICES'],a.output);bind(os.environ['CUDA_VISIBLE_DEVICES'])
 dll=r.parent/'native-s25/build/lib/libLiverCudaSim.so';scene=SceneConfig(preparation='cut_ready')
 with threadpool_limits(limits=1):
  import cv2
  cv2.setNumThreads(4)
  env=NativeEnvironment(EnvironmentConfig(str(dll),sha256(dll),local_bounds(scene),scene,max_steps=100,cutting_enabled=True,full_surface_checks=True,outcome_semantics='physical_outcome_v1',instrument_collision_checks=True,fast_topology_checks=True))
  state,context=env.reset(7);view=EndoscopicView(env.settings,1280,960,supersample=1,kidney=env.scene.shape,camera=env.scene.shot,gpu_postprocess=True,cache_surface_layout=True);view.update(env.last_surface,env.observed);view.render(depth=True);post=view._gpu_post
  snap=env.snapshot();backend=SnapshotBackend(env.native);action=env.previous_action.copy();action[10]=1
  for branch,optimized in [('A1',False),('B1',True),('A2',False),('B2',True)]:
   state,context=env.restore(snap);view._gpu_post=post if optimized else None;view._layout_cache=SurfaceLayoutCache() if optimized else None
   record=dict(branch=branch,optimized=optimized,quality_passed=True,frames=[],render_does_not_mutate_native=[]);trajectory=[]
   for i in range(40):
    guard({os.getpid()})
    if i in (0,39):before=hashlib.sha256(backend.save()).hexdigest()
    view.update(env.last_surface,env.observed);view.render(depth=True)
    if i in (0,39):record['render_does_not_mutate_native'].append(hashlib.sha256(backend.save()).hexdigest()==before)
    transition=env.step(action)
    if transition.after is None or transition.terminated or transition.truncated:raise RuntimeError('Replay rejected: '+str(transition.reason))
    quality=transition.after.metadata['quality'];record['quality_passed'] &= quality['checked_metrics_passed'] and quality['within_declared_local_scope'];record['frames'].append(dict(step=i,metrics=quality['metrics']));trajectory.append(env.observed.nodes.copy())
   arrays[branch]=np.array(trajectory);record['cut_edges_total']=sum(x['metrics']['new_cut_edges'] for x in record['frames']);result['branches'].append(record)
  for x,y in [('A1','A2'),('B1','B2'),('A1','B1'),('A2','B2')]:
   delta=np.abs(arrays[x]-arrays[y]);result['comparisons'].append(dict(pair=[x,y],bit_exact=bool(np.array_equal(arrays[x],arrays[y])),max_node_difference_m=float(delta.max())*.01,mean_node_difference_m=float(delta.mean())*.01))
  result.update(completed=True,physical_checks_passed=all(x['quality_passed'] for x in result['branches']),render_nonmutation_passed=all(all(x['render_does_not_mutate_native']) for x in result['branches']),trajectory_equivalence_certified=all(x['bit_exact'] for x in result['comparisons']))
except BaseException as e:result.update(error=repr(e),traceback=traceback.format_exc())
finally:
 if view:
  if post:view._gpu_post=post
  view.close()
 if env:env.close()
 if guard:guard.close()
 (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='branches'},indent=2))
if not result.get('completed') or not result.get('physical_checks_passed') or not result.get('render_nonmutation_passed'):raise SystemExit(2)
