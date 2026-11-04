"""Complete, unique physical + rendered frames; no repeated display frames count."""
import argparse,hashlib,json,os,sys,time,threading
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--dll',type=Path,required=True)
p.add_argument('--sha256',required=True);p.add_argument('--gpu',required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--scene',choices=['wall','wall-traction','flap','kidney'],default='wall-traction')
p.add_argument('--features',action='store_true');p.add_argument('--max-frames',type=int);p.add_argument('--fast-topology',action='store_true');p.add_argument('--support-mm',type=float)
p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=960)
p.add_argument('--opencv-threads',type=int,default=1)
p.add_argument('--supersample',type=int,choices=[1,2],default=1)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',XDG_CACHE_HOME=str(a.output/'cache'))
# Reuse the tested independent supervisor's resource guard.
sys.path.insert(0,'/srv/disk2/hryang/genesis-exec/parallel_simulation_tests/20261103-scaleup/continuation')
from acceptance_queue import GPUReservation
from threadpoolctl import threadpool_limits
import numpy as np
sys.path.insert(0,str(a.source.resolve()))
from endoscopic_tissue.training.env import NativeEnvironment,EnvironmentConfig,SceneConfig
from endoscopic_tissue.training.collect import local_bounds
from endoscopic_tissue.training.geometry import WallGeometry
from endoscopic_tissue.training.policies import observation
from endoscopic_tissue.training.provenance import sha256
from endoscopic_tissue.scenario import interpolate,stage_start
scene=SceneConfig(scene=a.scene,preparation='settled',geometry=WallGeometry(support_length_m=a.support_mm*.001) if a.support_mm else None)
options=dict(outcome_semantics='physical_outcome_v1',instrument_collision_checks=True) if a.features else {}
if a.fast_topology:options['fast_topology_checks']=True
config=EnvironmentConfig(str(a.dll),a.sha256,local_bounds(scene),scene,cutting_enabled=True,max_steps=5000,
 full_surface_checks=True,diagnostic_continue=True,**options)
result=dict(schema='interactive_frame_probe_v1',scope='one accepted physics step and one fresh RGB-D render per frame; display/encoding not timed',
 source=str(a.source.resolve()),adapter_sha256=sha256(a.source/'endoscopic_tissue/training/env.py'),dll_sha256=a.sha256,gpu=a.gpu,
 scene=a.scene,features=a.features,render=dict(width=a.width,height=a.height,supersample=a.supersample,depth=True,render_every=1),
 fast_topology_checks=a.fast_topology,supervision='separate monitoring thread, not timed as frame work',frames=[],completed=False,quality_passed=False,initialization_excluded=True)
guard=view=env=monitor=None
monitor_stop=threading.Event();monitor_errors=[]
def supervise():
 while not monitor_stop.is_set():
  try:guard({os.getpid()})
  except BaseException as error:monitor_errors.append(error);return
  monitor_stop.wait(2)
try:
 import importlib.util
 helper_root=Path(__file__).resolve().parent/'source/endoscopic_tissue/training'
 def helper(name):
  spec=importlib.util.spec_from_file_location('benchmark_'+name,helper_root/(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
 binding=helper('egl_device').bind(a.gpu);result['render_device_binding']=binding
 sampler=helper('resources').gpu_sample
 initial=sampler(a.gpu)
 if initial['processes'] or initial['used_mib']>512 or initial['gpu_percent']>5:raise RuntimeError('Selected GPU not idle, including graphics contexts')
 guard=GPUReservation(a.gpu,a.output);guard.sample=sampler
 monitor=threading.Thread(target=supervise,daemon=True);monitor.start()
 with threadpool_limits(limits=1):
  env=NativeEnvironment(config);state,context=env.reset(seed=0)
  (a.output/'context.json').write_text(json.dumps(context,indent=2))
  import cv2
  cv2.setNumThreads(a.opencv_threads);result['opencv_threads']=cv2.getNumThreads()
  from endoscopic_tissue.rendering import EndoscopicView
  view=EndoscopicView(env.settings,a.width,a.height,supersample=a.supersample,kidney=env.scene.shape,camera=env.scene.shot)
  view.update(env.last_surface,env.observed);view.render(depth=True)
  offscreen=view.camera._rasterizer._renderer
  offscreen.make_current()
  try:
   from OpenGL.GL import glGetString,GL_VENDOR,GL_RENDERER,GL_VERSION
   result['opengl']={k:glGetString(v).decode() for k,v in [('vendor',GL_VENDOR),('renderer',GL_RENDERER),('version',GL_VERSION)]}
  finally:offscreen.make_uncurrent()
  print(json.dumps({'phase':'render_ready','opengl':result['opengl']}),flush=True)
  probe_limit=False
  requested=env.scene.stages[0].command;valid=True
  for stage in env.scene.stages[1:]:
   start=stage_start(stage.name,requested,env.observed)
   for i in range(stage.steps):
    began=time.perf_counter()
    if monitor_errors:raise monitor_errors[0]
    requested=interpolate(start,stage.command,(i+1)/(stage.steps*stage.ramp))
    transition=env.replay_command(requested)
    if transition.after is None:raise RuntimeError(transition.reason)
    q=transition.after.metadata['quality']
    obs_start=time.perf_counter()
    obs=observation(transition.after,env.previous_action,**({'contract':'tool_kinematics_v1'} if a.features else {}))
    observe_ms=(time.perf_counter()-obs_start)*1000
    render_start=time.perf_counter();view.update(env.last_surface,env.observed);rgb,depth=view.render(depth=True)
    render_ms=(time.perf_counter()-render_start)*1000;frame_ms=(time.perf_counter()-began)*1000
    result['frames'].append(dict(stage=stage.name,step=env.step_id,native_tick=env.native.tick,frame_ms=frame_ms,
      render_ms=render_ms,observation_ms=observe_ms,quality=q,timings_ms=transition.timings_ms))
    if len(result['frames'])%100==0:
     (a.output/'heartbeat.json').write_text(json.dumps({'frames':len(result['frames']),'stage':stage.name,'last_frame_ms':frame_ms}))
    if a.max_frames and len(result['frames'])>=a.max_frames:probe_limit=True;break
    if not q['checked_metrics_passed'] or not q['within_declared_local_scope']:
     np.savez_compressed(a.output/'first-failure.npz',positions_m=transition.after.arrays['positions_m'],
                        neighbors=transition.after.topology['neighbors'],rgb=rgb,depth_m=depth)
     valid=False;break
   if not valid or probe_limit:break
  result['probe_limit_reached']=probe_limit
  result['completed']=valid and not probe_limit;result['quality_passed']=valid
except BaseException as error:
 result['error']=repr(error)
finally:
 if view is not None:view.close()
 if env is not None:env.close()
 monitor_stop.set()
 if monitor is not None:monitor.join(timeout=10)
 if guard is not None:guard.close()
 frames=np.array([r['frame_ms'] for r in result['frames']])
 if len(frames):
  result['performance']=dict(frames=len(frames),mean_ms=float(frames.mean()),p95_ms=float(np.percentile(frames,95)),
   p99_ms=float(np.percentile(frames,99)),max_ms=float(frames.max()),unique_frames_hz=float(1000/frames.mean()),
   frames_over_50ms=int(np.count_nonzero(frames>50)),rolling_20_frame_min_hz=float(20000/np.convolve(frames,np.ones(20),'valid').max()) if len(frames)>=20 else None)
 (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2))
if not result['completed'] and not result.get('probe_limit_reached'):raise SystemExit(2)
