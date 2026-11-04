"""Measured full frames plus optional hierarchical CPU/NVTX traces.

No physical parameters or quality thresholds are changed. Profiling runs and
unprofiled acceptance runs are separate artifacts. Native/CUDA/GL child times
are inclusive and must not be added to their parents.
"""
import argparse,contextlib,ctypes,cProfile,json,os,pstats,sys,threading,time,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--gpu',default='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa');p.add_argument('--dll',type=Path)
p.add_argument('--compact-blocks',type=int,choices=[0,8,16,32,64],default=0);p.add_argument('--bound-cache',action='store_true');p.add_argument('--query-threads',type=int,choices=[32,64,128],default=64);p.add_argument('--assembly-blocks',type=int,choices=[64,128,256,512],default=512);p.add_argument('--manifold-blocks',type=int,choices=[64,128,256,512,1024],default=1024);p.add_argument('--workload',action='store_true');p.add_argument('--query-mode',type=int,choices=[0,1],default=0);p.add_argument('--direct-rgb',action='store_true');p.add_argument('--opencv-threads',type=int,default=4);p.add_argument('--topology-ab',action='store_true');p.add_argument('--gpu-topology',action='store_true');p.add_argument('--static-visuals',action='store_true');p.add_argument('--persistent-soup',action='store_true');p.add_argument('--render-nogil',action='store_true');p.add_argument('--pbo',action='store_true');p.add_argument('--validate-native',action='store_true');p.add_argument('--parallel-frame',action='store_true');p.add_argument('--fused-mesh',action='store_true');p.add_argument('--gpu-aligned-depth',action='store_true');p.add_argument('--cache-surface-layout',action='store_true');p.add_argument('--vision',action='store_true');p.add_argument('--gpu-postprocess',action='store_true');p.add_argument('--profile',action='store_true');p.add_argument('--start',type=int,default=470);p.add_argument('--count',type=int,default=60)
p.add_argument('--max-frames',type=int);p.add_argument('--labels',action='store_true');p.add_argument('--no-render',action='store_true')
p.add_argument('--render-warmup',type=int,default=-1);p.add_argument('--cpus');p.add_argument('--save-state',action='store_true');p.add_argument('--repeat-render',type=int,default=0)
a=p.parse_args()
if a.cpus:os.sched_setaffinity(0,[int(x) for x in a.cpus.split(',')])
a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[3]
a.dll=a.dll or root/'native-s25/build/lib/libLiverCudaSim.so'
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',XDG_CACHE_HOME=str(a.output/'cache'),TMPDIR=str(root/'profiling/tmp'))
sys.path.insert(0,str(a.source.resolve()))
import numpy as np
from threadpoolctl import threadpool_limits
from endoscopic_tissue.training.env import NativeEnvironment,EnvironmentConfig,SceneConfig
from endoscopic_tissue.training.collect import local_bounds
from endoscopic_tissue.training.provenance import sha256
from endoscopic_tissue.training.policies import observation
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.training.egl_device import bind
from endoscopic_tissue.scenario import stage_start,interpolate

class Tracer:
 def __init__(self):
  self.enabled=False;self.events=[];self.origin=time.perf_counter_ns();self.local=threading.local()
  self.nvtx=ctypes.CDLL(str(Path(sys.executable).parent.parent/'lib/libnvToolsExt.so.1'))
  self.nvtx.nvtxRangePushA.argtypes=[ctypes.c_char_p];self.nvtx.nvtxRangePop.argtypes=[]
 @contextlib.contextmanager
 def span(self,name):
  if not self.enabled:yield;return
  if not hasattr(self.local,'stack'):self.local.stack=[]
  stack=self.local.stack
  start=time.perf_counter_ns();cpu=time.thread_time_ns();child=[0,0];stack.append(child)
  self.nvtx.nvtxRangePushA(name.encode())
  try:yield
  finally:
   self.nvtx.nvtxRangePop();elapsed=time.perf_counter_ns()-start;cpu_elapsed=time.thread_time_ns()-cpu
   stack.pop()
   if stack:stack[-1][0]+=elapsed;stack[-1][1]+=cpu_elapsed
   self.events.append(dict(name=name,ph='X',pid=os.getpid(),tid=threading.get_native_id(),ts=(start-self.origin)/1000,dur=elapsed/1000,
    args=dict(thread_cpu_us=cpu_elapsed/1000,exclusive_wall_us=(elapsed-child[0])/1000,exclusive_cpu_us=(cpu_elapsed-child[1])/1000)))
 def wrap(self,obj,attr,name=None):
  original=getattr(obj,attr)
  def wrapped(*args,**kw):
   label=(name(*args,**kw) if callable(name) else name) or attr
   with self.span(label):return original(*args,**kw)
  setattr(obj,attr,wrapped)
 def export(self,path):
  path.write_text(json.dumps({'traceEvents':self.events,'displayTimeUnit':'ms','metadata':{'origin_perf_ns':self.origin,'time_meaning':'wall-clock envelopes; thread_cpu_us is on-CPU time including driver spin; not disjoint CPU/GPU utilization'}},separators=(',',':')))
  stats={}
  for e in self.events:
   s=stats.setdefault(e['name'],dict(calls=0,inclusive_wall_ms=0,exclusive_wall_ms=0,thread_cpu_ms=0,exclusive_cpu_ms=0));s['calls']+=1
   for dest,v in [('inclusive_wall_ms',e['dur']),('exclusive_wall_ms',e['args']['exclusive_wall_us']),('thread_cpu_ms',e['args']['thread_cpu_us']),('exclusive_cpu_ms',e['args']['exclusive_cpu_us'])]:s[dest]+=v/1000
  path.with_name('cpu-spans.json').write_text(json.dumps(stats,indent=2))
tracer=Tracer();profile=cProfile.Profile();env=view=guard=vision=None;stop=threading.Event();monitor_errors=[];monitor=None
result=dict(validation_requested=a.validate_native,schema='profile_full_frame_v1',source=str(a.source),dll=str(a.dll),dll_sha256=sha256(a.dll),gpu=a.gpu,profiled=a.profile,
 render=dict(width=1280,height=960,supersample=1,fresh_rgbd_each_step=not a.no_render),labels=a.labels,training_vision=a.vision,cache_surface_layout=a.cache_surface_layout,gpu_postprocess=a.gpu_postprocess,gpu_aligned_depth=a.gpu_aligned_depth,parallel_frame=a.parallel_frame,fused_mesh=a.fused_mesh,pbo=a.pbo,render_nogil=a.render_nogil,persistent_soup=a.persistent_soup,static_visuals=a.static_visuals,gpu_topology=a.gpu_topology,topology_validation=a.topology_ab,direct_rgb=a.direct_rgb,opencv_threads=a.opencv_threads,compact_blocks=a.compact_blocks,bound_cache=a.bound_cache,query_mode=a.query_mode,query_threads=a.query_threads,assembly_blocks=a.assembly_blocks,manifold_blocks=a.manifold_blocks,workload_profiled=a.workload,frames=[],completed=False,quality_passed=False,
 cpu_affinity=sorted(os.sched_getaffinity(0)), note='CPU-returned RGB-D includes native readback and OpenGL readback; display/transport and disk recording excluded; labels separately flagged')
cudart=ctypes.CDLL(str(Path(sys.executable).parent.parent/'lib/libcudart.so'));profiler_active=False

def supervise():
 while not stop.is_set():
  try:guard({os.getpid()})
  except BaseException as e:monitor_errors.append(e);return
  stop.wait(2)
try:
 guard=GPUReservation(a.gpu,a.output);monitor=threading.Thread(target=supervise,daemon=True);monitor.start()
 result['binding']=bind(a.gpu)
 native_control=ctypes.CDLL(str(a.dll))
 if hasattr(native_control,'LCS_SetBatchQueryMode'):
  native_control.LCS_SetBatchQueryMode.argtypes=[ctypes.c_int];native_control.LCS_SetBatchQueryMode.restype=ctypes.c_int
  assert native_control.LCS_SetBatchQueryMode(a.query_mode)==0
 elif a.query_mode:raise RuntimeError('Requested batch mode absent from native library')
 if hasattr(native_control,'LCS_SetNativeSchedule'):
  native_control.LCS_SetNativeSchedule.argtypes=[ctypes.c_int,ctypes.c_int];native_control.LCS_SetNativeSchedule.restype=ctypes.c_int
  assert native_control.LCS_SetNativeSchedule(a.assembly_blocks,a.manifold_blocks)==0
 elif (a.assembly_blocks,a.manifold_blocks)!=(512,1024):raise RuntimeError('Requested launch grid absent from native library')
 if hasattr(native_control,'LCS_SetQueryThreads'):
  native_control.LCS_SetQueryThreads.argtypes=[ctypes.c_int];native_control.LCS_SetQueryThreads.restype=ctypes.c_int
  assert native_control.LCS_SetQueryThreads(a.query_threads)==0
 elif a.query_threads!=64:raise RuntimeError('Requested query block absent from native library')
 if hasattr(native_control,'LCS_SetQueryCompactBlocks'):
  native_control.LCS_SetQueryCompactBlocks.argtypes=[ctypes.c_int];native_control.LCS_SetQueryCompactBlocks.restype=ctypes.c_int;assert native_control.LCS_SetQueryCompactBlocks(a.compact_blocks)==0
 elif a.compact_blocks:raise RuntimeError('Requested query prefilter absent from library')
 native_control.LCS_SetCoarseBoundCache.argtypes=[ctypes.c_int];native_control.LCS_SetCoarseBoundCache.restype=ctypes.c_int;assert native_control.LCS_SetCoarseBoundCache(int(a.bound_cache))==0
 with threadpool_limits(limits=1):
  scene=SceneConfig(preparation='settled')
  env=NativeEnvironment(EnvironmentConfig(str(a.dll),sha256(a.dll),local_bounds(scene),scene,cutting_enabled=True,max_steps=5000,
   full_surface_checks=True,diagnostic_continue=True,outcome_semantics='physical_outcome_v1',instrument_collision_checks=True,fast_topology_checks=True,**(dict(gpu_topology_checks=True) if a.gpu_topology else {})))
  state,context=env.reset(0);(a.output/'context.json').write_text(json.dumps(context,indent=2))
  import cv2
  cv2.setNumThreads(a.opencv_threads)
  if not a.no_render:
   from endoscopic_tissue.rendering import EndoscopicView
   if a.labels or a.vision:
    from endoscopic_tissue.training.vision import SynchronizedVision
    vision=SynchronizedVision(env,dict(width=1280,height=960,opencv_threads=a.opencv_threads,gpu_postprocess=a.gpu_postprocess,cache_surface_layout=a.cache_surface_layout,gpu_aligned_depth=a.gpu_aligned_depth,**(dict(fused_mesh=True) if a.fused_mesh else {}),**(dict(parallel_same_frame=True) if a.parallel_frame else {}),**(dict(pbo_readback=True) if a.pbo else {}),**(dict(render_nogil=True) if a.render_nogil else {}),**(dict(persistent_soup=True) if a.persistent_soup else {}),**(dict(static_visuals=True) if a.static_visuals else {}),**(dict(direct_rgb=True) if a.direct_rgb else {}),privileged_labels=['segmentation','surface_supports','pixel_supports'] if a.labels else []));view=vision.view
    vision.capture(env,env.state)
   else:
    view=EndoscopicView(env.settings,1280,960,supersample=1,kidney=env.scene.shape,camera=env.scene.shot,**(dict(fused_mesh=True) if a.fused_mesh else {}),**(dict(gpu_postprocess=a.gpu_postprocess,cache_surface_layout=True) if a.cache_surface_layout else ({'gpu_postprocess':True} if a.gpu_postprocess else {})));view.update(env.last_surface,env.observed);view.render(depth=True)
  if a.profile:
   from endoscopic_tissue.training import quality,instrument_collision
   for obj,attr,name in [(env.native,'step','native.step'),(env.native,'observe','native.observe'),(env.native,'surface','native.surface'),(env.native,'material_quality','native.material_quality'),
    (env,'_assess','quality.all'),(env.quality,'assess','quality.assess'),(env,'_snapshot','state.snapshot'),(env.grasp,'assess','grasp.assess'),
    (quality,'scissors_body_depths','quality.tissue_tool_overlap'),(instrument_collision,'instrument_distance','quality.instrument_distance'),(env.damage,'assess','damage.assess')]:tracer.wrap(obj,attr,name)
   if hasattr(env,'_assess_cpu'):tracer.wrap(env,'_assess_cpu','quality.cpu_worker')
   if vision and hasattr(vision,'prepare_frame'):tracer.wrap(vision,'prepare_frame','vision.prepare_current')
   tracer.wrap(env.native,'check',lambda name,*args:'api.'+name)
   original_topology=env.quality.topology_function
   def topology(surface):
    with tracer.span('quality.topology'):return original_topology(surface)
   env.quality.topology_function=topology
   if view:
    for obj,attr,name in [(view,'update','render.update_mesh'),(view,'render','render.rgbd'),(view.camera,'render','render.camera'),
     (view.camera._rasterizer,'render_camera','render.rasterizer'),(view.camera._rasterizer,'update_scene','render.update_scene')]:tracer.wrap(obj,attr,name)
    if getattr(view,'_gpu_post',None) is not None:tracer.wrap(view._gpu_post,'read','render.gpu_post')
    from genesis.ext.pyrender.renderer import Renderer
    for attr in ['_forward_pass','_read_main_framebuffer','_update_context']:
     if hasattr(Renderer,attr):tracer.wrap(Renderer,attr,'gl.'+attr)
    for name in ['remap','cvtColor','multiply']:tracer.wrap(cv2,name,'cv.'+name)
    if vision:
     from endoscopic_tissue.training import vision as vision_module
     tracer.wrap(vision_module,'align_depth_to_endoscope','vision.align_depth')
     from endoscopic_tissue.training import visual_labels,pixel_labels
     for name in ['segmentation','surface_supports','compose_pixel_supports']:tracer.wrap(visual_labels,name,'labels.'+name)
     tracer.wrap(pixel_labels.PixelLabelPass,'render','labels.float_gl_pass')
  result['measurement_start_unix_s']=time.time()
  requested=env.scene.stages[0].command;valid=True;finished=True;frame_id=0
  for stage in env.scene.stages[1:]:
   start=stage_start(stage.name,requested,env.observed)
   for i in range(stage.steps):
    if monitor_errors:raise monitor_errors[0]
    if a.validate_native and frame_id==a.start:
     native_control.LCS_BatchQueryValidate.argtypes=[ctypes.c_int];native_control.LCS_BatchQueryValidate.restype=ctypes.c_int;assert native_control.LCS_BatchQueryValidate(1)==0
    if a.profile and frame_id==a.start:
     assert cudart.cudaProfilerStart()==0;profiler_active=True;tracer.enabled=True;profile.enable()
    begin=time.perf_counter_ns();cpu=time.thread_time_ns()
    with tracer.span('frame'):
     with tracer.span('control'):
      requested=interpolate(start,stage.command,(i+1)/(stage.steps*stage.ramp))
     with tracer.span('environment.step'):t=env.replay_command(requested)
     if t.after is None:raise RuntimeError(t.reason)
     if a.topology_ab:
      from endoscopic_tissue.training.topology_fast import topology_statistics_fast
      expected=topology_statistics_fast(env.last_surface);actual=t.after.metadata['quality']['metrics']['surface_topology']
      if expected!=actual:raise RuntimeError('Current-frame CPU/GPU topology mismatch: '+str((expected,actual)))
      result['topology_frames_compared']=result.get('topology_frames_compared',0)+1
     if not vision:
      with tracer.span('policy.observation'):observation(t.after,env.previous_action,contract='tool_kinematics_v1')
     rendering=time.perf_counter_ns()
     if a.profile and a.render_warmup>=0 and frame_id<a.start-a.render_warmup:pass
     elif vision:
      with tracer.span('vision.all'):vision.capture(env,t.after)
     elif view:
      view.update(env.last_surface,env.observed);rgb,depth=view.render(depth=True)
     render_ms=(time.perf_counter_ns()-rendering)/1e6
     if vision:
      with tracer.span('policy.observation'):observation(t.after,env.previous_action,contract='tool_kinematics_v1')
    elapsed=(time.perf_counter_ns()-begin)/1e6;cpu_ms=(time.thread_time_ns()-cpu)/1e6
    q=t.after.metadata['quality'];result['frames'].append(dict(frame_id=frame_id,native_tick=env.native.tick,stage=stage.name,step=env.step_id,frame_ms=elapsed,thread_cpu_ms=cpu_ms,render_ms=render_ms,timings_ms=t.timings_ms,quality=q))
    if a.profile and frame_id+1==a.start+a.count:
     profile.disable();tracer.enabled=False;assert cudart.cudaProfilerStop()==0;profiler_active=False
    if a.validate_native and frame_id+1==a.start+a.count:
     values=(ctypes.c_int*32)();native_control.LCS_FullContactValidationRead.argtypes=[ctypes.POINTER(ctypes.c_int)];native_control.LCS_FullContactValidationRead.restype=ctypes.c_int;assert native_control.LCS_FullContactValidationRead(values)==0;assert native_control.LCS_BatchQueryValidate(0)==0
     if a.bound_cache:
      stats=(ctypes.c_ulonglong*8)();native_control.LCS_CoarseBoundCacheStats.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)];native_control.LCS_CoarseBoundCacheStats.restype=ctypes.c_int;assert native_control.LCS_CoarseBoundCacheStats(stats)==0
      result['coarse_bound_validation']=dict(accumulate_terms=stats[0],accumulate_target_differences=stats[1],accumulate_value_differences=stats[2],bound_terms=stats[3],bound_target_differences=stats[4],bound_scale_differences=stats[5],active_bound_outputs=stats[6],nontrivial_bound_scales=stats[7],passed=stats[0]>0 and stats[3]>0 and stats[6]>0 and stats[7]>0 and stats[1]+stats[2]+stats[4]+stats[5]==0);assert result['coarse_bound_validation']['passed']
     result['native_validation']=dict(paired_batches=values[0],mismatched_batches=values[1],positive_contact_records=values[2],captured_batches=values[3],equivalent_query_calls=values[4],record_diffs=values[5],degree_diffs=values[6],activity_diffs=values[7],validity_diffs=values[8],group_order_diffs=values[9],fallbacks=values[10],manifold_batches=values[16],manifold_mismatches=values[17],manifold_records=values[18],captured_manifold_batches=values[19],manifold_degree_diffs=values[20],manifold_record_diffs=values[21],quadrature_diffs=values[22],manifold_mapping_diffs=values[23],manifold_skips=values[24],first_mismatch_entry=values[11],first_mismatch_word=values[12],reference_word=values[13],candidate_word=values[14],passed=values[0]>0 and values[1]==0 and values[2]>0 and values[3]>0 and sum(values[k] for k in range(5,11))==0 and values[16]>0 and values[18]>0 and values[19]>0 and values[17]==0 and sum(values[k] for k in range(20,25))==0)
     if not result['native_validation']['passed']:raise RuntimeError('Captured/native contact query equivalence validation failed')
    if a.workload and frame_id%10==0:
     values=(ctypes.c_int*9)();native_control.LCS_ReadNativeWorkload.argtypes=[ctypes.POINTER(ctypes.c_int)];native_control.LCS_ReadNativeWorkload.restype=ctypes.c_int
     assert native_control.LCS_ReadNativeWorkload(values)==0
     result.setdefault('workload_samples',[]).append(dict(frame_id=frame_id,stage=stage.name,nodes=values[0],tets=values[1],surface_triangles=values[2],assembly_blocks=values[3],manifold_blocks=values[4],grasper_manifold=values[5],grasper_anchors=values[6],blade_manifold=values[7],blade_anchors=values[8]))
    frame_id+=1
    if frame_id%50==0:(a.output/'heartbeat.json').write_text(json.dumps(dict(frame=frame_id,stage=stage.name,frame_ms=elapsed)))
    if not q['checked_metrics_passed'] or not q['within_declared_local_scope']:valid=False;finished=False;break
    if a.max_frames and frame_id>=a.max_frames:finished=False;break
   if not valid or not finished:break
  result['measurement_end_unix_s']=time.time()
  result.update(completed=finished,quality_passed=valid)
  if not valid:
   from endoscopic_tissue.training.snapshot import encode_python,SnapshotBackend
   (a.output/'failure-render-state.bin').write_bytes(encode_python(dict(surface=env.last_surface,observed=env.observed)))
   if SnapshotBackend(env.native).size()<2*1024**3:
    env.snapshot().save(a.output/'failure.envsnap');result['failure_snapshot']='failure.envsnap' 
  if a.save_state:
   from dataclasses import asdict
   from endoscopic_tissue.training.snapshot import encode_python
   (a.output/'render-state.bin').write_bytes(encode_python(dict(surface=env.last_surface,observed=env.observed)))
   np.savez_compressed(a.output/'render-output.npz',rgb=rgb,depth=depth)
except BaseException as e:result.update(error=repr(e),traceback=traceback.format_exc())
finally:
 if a.validate_native and a.bound_cache and env is not None and 'coarse_bound_validation' not in result:
  try:
   v=(ctypes.c_ulonglong*8)();native_control.LCS_CoarseBoundCacheStats.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)];native_control.LCS_CoarseBoundCacheStats.restype=ctypes.c_int;code=native_control.LCS_CoarseBoundCacheStats(v)
   result['interrupted_coarse_bound_validation']=dict(read_returncode=code,counters=list(v) if code==0 else None,window_completed=False,passed=False)
  except BaseException as e:result['interrupted_coarse_bound_validation']=dict(error=repr(e),window_completed=False,passed=False)
 if a.validate_native and env is not None and 'native_validation' not in result:
  try:
   v=(ctypes.c_int*32)();native_control.LCS_FullContactValidationRead.argtypes=[ctypes.POINTER(ctypes.c_int)];native_control.LCS_FullContactValidationRead.restype=ctypes.c_int;code=native_control.LCS_FullContactValidationRead(v)
   result['interrupted_native_validation']=dict(read_returncode=code,counters=list(v) if code==0 else None,requested_start=a.start,requested_count=a.count,window_completed=False,passed=False)
  except BaseException as e:result['interrupted_native_validation']=dict(error=repr(e),window_completed=False,passed=False)
 if profiler_active:cudart.cudaProfilerStop()
 profile.disable();tracer.enabled=False
 if a.profile:
  profile.dump_stats(str(a.output/'python.pstats'))
  with (a.output/'python-profile.txt').open('w') as stream:pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(80)
  tracer.export(a.output/'cpu-timeline.json')
 if view and getattr(view,'_layout_cache',None) is not None:
  cache=view._layout_cache;result['layout_cache_statistics']=dict(hits=cache.hits,misses=cache.misses)
 if vision:vision.close()
 elif view:view.close()
 if env:env.close()
 stop.set()
 if monitor:monitor.join(timeout=5)
 if guard:guard.close()
 if result['frames']:
  times=np.array([f['frame_ms'] for f in result['frames']]);result['performance']=dict(frames=len(times),mean_ms=float(times.mean()),p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),p99_ms=float(np.percentile(times,99)),max_ms=float(times.max()),fps=float(1000/times.mean()),over_50ms=int((times>50).sum()),rolling20_min_hz=float(20000/np.convolve(times,np.ones(20),'valid').max()) if len(times)>=20 else None)
 (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2),flush=True)
if 'error' in result or not result['quality_passed']:raise SystemExit(2)
