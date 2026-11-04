import argparse,os,sys,time,json,hashlib,traceback
from pathlib import Path
p=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--dll',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--parallel-frame',action='store_true');parser.add_argument('--render-nogil',action='store_true');parser.add_argument('--direct-rgb',action='store_true');parser.add_argument('--gpu-topology',action='store_true');parser.add_argument('--static-visuals',action='store_true');parser.add_argument('--persistent-soup',action='store_true');a=parser.parse_args();a.output.mkdir(exist_ok=False)
os.sched_setaffinity(0,[7,19,20,25]);os.environ.update(CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(a.output/'cache'),TMPDIR=str(p/'tmp'));sys.path.insert(0,str(a.source))
import numpy as np,cv2
from threadpoolctl import threadpool_limits
from endoscopic_tissue.training.env import NativeEnvironment,EnvironmentConfig,SceneConfig
from endoscopic_tissue.training.vision import SynchronizedVision,align_depth_to_endoscope
from endoscopic_tissue.training.collect import local_bounds
from endoscopic_tissue.training.provenance import sha256
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.training.snapshot import SnapshotBackend
r=dict(passed=False,scope='actual current-frame RGBD, control, physical checks, archived-state and ownership regression; not complete physics-domain approval',frames=[],pairs=[]);env=vision=guard=None
try:
 guard=GPUReservation(os.environ['CUDA_VISIBLE_DEVICES'],a.output);scene=SceneConfig(preparation='cut_ready')
 config=EnvironmentConfig(str(a.dll),sha256(a.dll),local_bounds(scene),scene,cutting_enabled=True,max_steps=100,full_surface_checks=True,outcome_semantics='physical_outcome_v1',instrument_collision_checks=True,fast_topology_checks=True,**(dict(gpu_topology_checks=True) if a.gpu_topology else {}))
 with threadpool_limits(limits=1):
  env=NativeEnvironment(config);env.reset(7);vision=SynchronizedVision(env,dict(width=1280,height=960,opencv_threads=4,gpu_postprocess=True,gpu_aligned_depth=True,fused_mesh=True,direct_rgb=a.direct_rgb,static_visuals=a.static_visuals,persistent_soup=a.persistent_soup,render_nogil=a.render_nogil,parallel_same_frame=a.parallel_frame));vision.capture(env,env.state)
  action=env.previous_action.copy();action[10]=1.;retained=[];actual_cut=False
  for step in range(32):
   guard({os.getpid()});t=env.step(action)
   if t.after is None or t.terminated or t.truncated:raise RuntimeError('Real interaction regression rejected '+str(t.reason))
   vision.capture(env,t.after);state=t.after
   assert state.metadata['vision']['native_tick']==env.native.tick and state.metadata['vision']['step_id']==env.step_id
   actual_cut|=any(e['kind']=='cut_onset' for e in state.metadata.get('events',[]))
   r['frames'].append(dict(step_id=env.step_id,native_tick=env.native.tick,quality_passed=state.metadata['quality']['checked_metrics_passed'],new_cut_edges=state.metadata['quality']['metrics']['new_cut_edges']))
   if step not in (0,12,19,31):continue
   saved={k:state.arrays[k] for k in ['rgb','depth_pinhole_m','depth_rgb_aligned_m','depth_valid']};hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in saved.items()};retained.append((saved,hashes))
   archive_before=hashlib.sha256(SnapshotBackend(env.native).save()).hexdigest() if step in (19,31) else None
   view=vision.view;post=view._gpu_post;view._gpu_post=None;view.fused_mesh=False;persist=getattr(view,'persistent_soup',False);view.persistent_soup=False;static=getattr(view,'static_visuals',False);view.static_visuals=False;schedule=getattr(view,'_render_scheduling',None)
   if schedule:schedule.set_enabled(False)
   try:
    view.update(env.last_surface,env.observed);bgr,raw=view.render(depth=True);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);aligned,valid=align_depth_to_endoscope(raw,view.lens_map,view.vignette,view.camera.near,view.camera.far)
   finally:
    view._gpu_post=post;view.fused_mesh=True;view.persistent_soup=persist;view.static_visuals=static
    if schedule:schedule.set_enabled(True)
   pair=dict(step=step,rgb_bit_exact=bool(np.array_equal(rgb,saved['rgb'])),raw_depth_error_m=float(np.max(np.abs(raw-saved['depth_pinhole_m']))),aligned_depth_error_m=float(np.max(np.abs(aligned-saved['depth_rgb_aligned_m']))),validity_bit_exact=bool(np.array_equal(valid,saved['depth_valid'])))
   if archive_before is not None:pair['native_archive_unchanged']=archive_before==hashlib.sha256(SnapshotBackend(env.native).save()).hexdigest();assert pair['native_archive_unchanged']
   if archive_before is not None and a.gpu_topology:
    from endoscopic_tissue.training.gpu_topology import topology_statistics
    from endoscopic_tissue.training.topology_fast import topology_statistics_fast
    assert topology_statistics(env.native,env.last_surface)==topology_statistics_fast(env.last_surface)
    pair['gpu_topology_archive_unchanged']=archive_before==hashlib.sha256(SnapshotBackend(env.native).save()).hexdigest();assert pair['gpu_topology_archive_unchanged']
   assert pair['rgb_bit_exact'] and pair['validity_bit_exact'] and pair['raw_depth_error_m']<=1e-6 and pair['aligned_depth_error_m']<=1e-6,pair
   r['pairs'].append(pair)
  for arrays,hashes in retained:
   for k,arr in arrays.items():assert hashlib.sha256(arr.tobytes()).hexdigest()==hashes[k],'retained frame overwritten'
  r.update(passed=True,actual_cut_onset=actual_cut,retained_observations_unchanged=True)
  if not actual_cut:raise RuntimeError('Actual cut onset was not exercised')
except BaseException as e:r.update(passed=False,error=repr(e),traceback=traceback.format_exc())
finally:
 if vision:vision.close()
 if env:env.close()
 if guard:guard.close()
 (a.output/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if not r['passed']:raise SystemExit(2)
