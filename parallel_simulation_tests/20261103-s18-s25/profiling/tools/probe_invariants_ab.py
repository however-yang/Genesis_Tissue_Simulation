"""Small real-CUDA S19/S20/S25 probe; records rejection rather than assuming acceptance."""
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',required=True)
p.add_argument('--preparation',default='settled');p.add_argument('--cut',action='store_true');p.add_argument('--randomized',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
profile_root=Path(__file__).resolve().parents[1]
root=profile_root.parent
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,XDG_CACHE_HOME=str(a.output/'cache'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
sys.path.insert(0,str(profile_root/'candidate-v4-source'))
os.sched_setaffinity(0,[7,19,20,25])
os.environ['TMPDIR']=str(profile_root/'tmp')
import numpy as np
from threadpoolctl import threadpool_limits
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.training.env import NativeEnvironment,EnvironmentConfig,SceneConfig
from endoscopic_tissue.training.collect import local_bounds
from endoscopic_tissue.training.provenance import sha256
from endoscopic_tissue.training.pool_session import WorkerSession
from endoscopic_tissue.training.snapshot import SnapshotBackend
from endoscopic_tissue.training.vision import SynchronizedVision
from endoscopic_tissue.training.policies import observation,recorded_observation
from endoscopic_tissue.training.data import EpisodeReader
from endoscopic_tissue.grid import TO_GENESIS
result={'schema':'s18_s25_gpu_smoke_v1','completed':False,'passed':False,'checks':{},'scope':'single-state actual CUDA readback and five pool-session restores; not C1-C5, multiprocess or 20Hz acceptance'}
guard=env=session=vision=None
try:
 guard=GPUReservation(a.gpu,a.output)
 dll=root/'native-s25/build/lib/libLiverCudaSim.so';scene=SceneConfig(preparation=a.preparation)
 config=EnvironmentConfig(str(dll),sha256(dll),local_bounds(scene),scene,max_steps=200,cutting_enabled=True,outcome_semantics='physical_outcome_v1',fast_topology_checks=a.cut)
 with threadpool_limits(limits=1):
  env=NativeEnvironment(config)
  session=WorkerSession(env,a.output/'episodes',record=True,observation_contract='tool_kinematics_v1',
     privileged_fields=('positions_m','topology'),max_snapshot_bytes=2*1024**3,max_snapshot_capture_bytes=3*1024**3,max_response_bytes=64*1024**2)
  r=session.execute('reset',{'seed':7});result['context']=r['context'];print('RESET',flush=True)
  if a.cut:
   action=env.previous_action.copy();action[10]=1.;cut=False
   for i in range(150):
    guard({os.getpid()});transition=session.execute('step',{'action':action})
    if transition['terminated']:raise RuntimeError('Cut probe rejected: '+str(transition['reason']))
    if any(e['kind']=='cut_onset' for e in env.state.metadata['events']):cut=True;break
   result['checks']['actual_cut_onset']=cut
   if not cut:raise RuntimeError('No cut onset in 150 actions')
  initial=env.observed.nodes.copy();base=hashlib.sha256(SnapshotBackend(env.native).save()).hexdigest()
  snap=session.execute('snapshot',{'hot':True});result['snapshot_metadata']=snap['metadata'];checks=[]
  for i in range(5):
   guard({os.getpid()});out=session.execute('restore',{'snapshot_id':snap['snapshot_id'],'branch_label':'real-gpu-'+str(i)})
   native_nodes=env.native.observe().nodes
   checks.append(dict(branch=i,position_max_abs_cm=float(np.max(np.abs(native_nodes-initial))),
      native_archive_exact=hashlib.sha256(SnapshotBackend(env.native).save()).hexdigest()==base,
      lineage=out['context'].get('lineage'),context_branch=out['context'].get('branch_label')))
  result['checks']['five_restores']=checks
  session.close();session=None
  import cProfile,pstats
  profiler=cProfile.Profile();profiler.enable()
  for _ in range(5):
   transition=env.step(env.previous_action)
   if transition.after is None or transition.terminated:raise RuntimeError('Profile step rejected')
  profiler.disable();profiler.dump_stats(str(a.output/'quality-profile.pstats'))
  with (a.output/'quality-profile.txt').open('w') as stream:pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(35)
  visual={'width':1280,'height':960,'opencv_threads':4,'gpu_aligned_depth':True,'gpu_postprocess':True,'privileged_labels':['segmentation','surface_supports','pixel_supports']}
  if a.randomized:visual['randomization']={'camera_translation_m':.001,'camera_target_m':.001,'fov_delta_deg':2.,'lamp_gain_delta':.1,'roughness_scale_delta':.1,'rgb_noise_std':1.,'depth_noise_std_m':.00001}
  vision=SynchronizedVision(env,visual);result['vision_manifest']=vision.manifest
  base=hashlib.sha256(SnapshotBackend(env.native).save()).hexdigest()
  print('VISION_READY',flush=True)
  duration=vision.capture(env,env.state)
  result['checks']['visual_label_quality']=env.state.metadata.get('visual_label_quality')
  result['checks']['visual_capture_ms']=duration
  result['checks']['labels_do_not_mutate_native_archive']=hashlib.sha256(SnapshotBackend(env.native).save()).hexdigest()==base
  state=env.state;valid=state.arrays['privileged_pixel_valid'];ids=state.arrays['privileged_pixel_node_ids'][valid];weights=state.arrays['privileged_pixel_weights'][valid]
  points=(state.arrays['positions_m'][np.maximum(ids,0)]*weights[...,None]).sum(axis=1)+state.arrays['privileged_pixel_offset_m'][valid]
  world=points@TO_GENESIS.T;extr=state.arrays['camera_world_to_cv'];cv=world@extr[:3,:3].T+extr[:3,3]
  intrinsics=vision.static_arrays['camera_intrinsics'];pixel=cv@intrinsics.T;pixel=pixel[:,:2]/pixel[:,2,None]
  import cv2
  xy,_=cv2.convertMaps(*vision.view.lens_map,cv2.CV_32FC2)
  yy,xx=np.indices(valid.shape,dtype=np.float32)
  lookup=np.stack((xx,valid.shape[0]-1-yy),axis=-1)
  sampled=cv2.remap(lookup,xy,None,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)[valid]+.5
  error=np.linalg.norm(pixel-sampled,axis=1)
  result['checks']['pixel_reprojection_error_px']={'count':len(error),'max':float(error.max(initial=0)), 'p95':float(np.percentile(error,95)) if len(error) else None}
  depth=state.arrays['depth_rgb_aligned_m'][valid]
  delta=np.abs(cv[:,2]-depth)
  result['checks']['material_to_sensor_depth_error_m']={'max':float(delta.max(initial=0)),'p95':float(np.percentile(delta,95)) if len(delta) else None,
      'meaning':'existing multisample-resolved sensor depth versus single-center truth; not identical at silhouettes'}
  result['checks']['material_to_render_depth_max_error_m']=float(np.max(np.abs(cv[:,2]-state.arrays['privileged_pixel_center_depth_m'][valid]),initial=0))
  import cv2
  cv2.imwrite(str(a.output/'rgb.png'),cv2.cvtColor(state.arrays['rgb'],cv2.COLOR_RGB2BGR))
  cv2.imwrite(str(a.output/'segmentation.png'),state.arrays['privileged_segmentation']*70)
  cv2.imwrite(str(a.output/'support-validity.png'),valid.astype(np.uint8)*255)
  np.savez_compressed(a.output/'surface-checkpoint.npz',vertices_cm=env.last_surface.vertices,faces=env.last_surface.triangles,
      anchors=env.last_surface.anchors,nodes_m=state.arrays['positions_m'])
  before=state.arrays['rgb'].copy();warm=vision.capture(env,state)
  result['checks']['warm_visual_capture_ms']=warm
  from endoscopic_tissue.training.visual_labels import compose_pixel_supports
  import importlib.util
  spec=importlib.util.spec_from_file_location('baseline_label_reference',profile_root/'baseline-source/endoscopic_tissue/training/visual_labels.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
  from endoscopic_tissue.training.visual_labels import surface_supports
  bindings=surface_supports(env.native,env.last_surface)
  triangle,bary,center=vision.pixel_pass.render(env.last_surface)
  ref_pixel=ref.compose_pixel_supports(triangle,bary,bindings);fast_pixel=compose_pixel_supports(triangle,bary,bindings)
  result['checks']['real_pixel_composition_bit_exact']=all(np.array_equal(ref_pixel[k],fast_pixel[k]) for k in ref_pixel)
  assert result['checks']['real_pixel_composition_bit_exact']
  result['checks']['repeated_render_rgb_max_difference']=int(np.max(np.abs(state.arrays['rgb'].astype(int)-before.astype(int))))
  # Same physical state and same GL scene: restore the original CPU postprocess,
  # alignment and original dense label composer; no physics step between A/B.
  gpu_arrays=dict(state.arrays);gpu_quality=dict(state.metadata['visual_label_quality'])
  from endoscopic_tissue.training import visual_labels as label_module
  fast_compose=label_module.compose_pixel_supports;post=vision.view._gpu_post
  label_module.compose_pixel_supports=ref.compose_pixel_supports
  vision.view._gpu_post=None;vision.gpu_aligned_depth=False
  cpu_capture_ms=vision.capture(env,state)
  comparisons={}
  for key in gpu_arrays:
   if key not in ('rgb','depth_pinhole_m','depth_rgb_aligned_m','depth_valid') and not key.startswith('privileged_'):continue
   left,right=gpu_arrays[key],state.arrays[key]
   exact=bool(np.array_equal(left,right,equal_nan=True));delta=float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64)),initial=0))
   # Only sensor depths may differ by float32 rounding. Material labels and masks must be exact.
   tolerant=key in ('depth_pinhole_m','depth_rgb_aligned_m','privileged_pixel_sensor_depth_delta_m')
   comparisons[key]=dict(bit_exact=exact,max_abs_difference=delta,passed=exact or (tolerant and delta<=1e-6))
  result['checks']['same_state_cpu_gpu']=dict(arrays=comparisons,cpu_label_quality=dict(state.metadata['visual_label_quality']),gpu_label_quality=gpu_quality,cpu_capture_ms=cpu_capture_ms,passed=all(x['passed'] for x in comparisons.values()))
  result['checks']['cpu_capture_native_archive_exact']=hashlib.sha256(SnapshotBackend(env.native).save()).hexdigest()==base
  label_module.compose_pixel_supports=fast_compose;vision.view._gpu_post=post;vision.gpu_aligned_depth=True
  result['checks']['force_free_observation']=set(observation(state,env.previous_action,contract='tool_kinematics_v1')['tools'])
  result['checks']['force_free_observation']=sorted(result['checks']['force_free_observation'])
  vision.close();vision=None
  q=result['checks']['visual_label_quality']
  result['passed']=bool(all(x['native_archive_exact'] and x['position_max_abs_cm']==0 for x in checks)
     and q['tissue_pixels']>0 and q['valid_material_pixels']/q['tissue_pixels']>.99
     and q['visibility_disagreement_pixels']==0 and len(error)>0 and error.max()<.01
     and result['checks']['material_to_render_depth_max_error_m']<1e-5
     and result['checks']['labels_do_not_mutate_native_archive']
     and result['checks']['repeated_render_rgb_max_difference']==0
     and result['checks']['same_state_cpu_gpu']['passed']
     and result['checks']['cpu_capture_native_archive_exact'])
  result['completed']=True
except BaseException as error:
 result['error']=repr(error);result['traceback']=traceback.format_exc()
finally:
 if vision is not None:vision.close()
 if session is not None:session.close()
 if env is not None:env.close()
 if guard is not None:guard.close()
 (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2),flush=True)
if not result.get('passed'):raise SystemExit(2)
