import argparse,json,os,sys,time,traceback
from pathlib import Path
r=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--state',type=Path,default=r/'results/baseline-profile/render-state.bin');p.add_argument('--context',type=Path,default=r/'results/baseline-profile/context.json');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
os.sched_setaffinity(0,[7,19,20,25]);os.environ.update(CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(a.output/'cache'),TMPDIR=str(r/'tmp'))
sys.path.insert(0,str(r/'candidate-source'))
import numpy as np
import cv2
from threadpoolctl import threadpool_limits
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.training.egl_device import bind
from endoscopic_tissue.training.snapshot import decode_python
from endoscopic_tissue.native_backend import Settings
from endoscopic_tissue.scenario import SCISSORS_SCENES
from endoscopic_tissue.rendering import EndoscopicView
from endoscopic_tissue.training.gpu_postprocess import GPUPostprocess
from endoscopic_tissue.training.render_layout import SurfaceLayoutCache
from dataclasses import replace
view=guard=post=None;result={'passed':False,'scope':'same frozen cut-state and synthetic geometry variants; update+render+readback timed; no physical step benchmark'}
try:
 guard=GPUReservation(os.environ['CUDA_VISIBLE_DEVICES'],a.output);bind(os.environ['CUDA_VISIBLE_DEVICES'])
 state=decode_python(a.state.read_bytes());context=json.loads(a.context.read_text());settings=Settings(**context['settings'])
 scene=SCISSORS_SCENES['wall-traction']
 with threadpool_limits(limits=1):
  cv2.setNumThreads(4);view=EndoscopicView(settings,1280,960,supersample=1,kidney=scene.shape,camera=scene.shot)
  view.update(state['surface'],state['observed'])
  for _ in range(4):reference,reference_depth=view.render(depth=True)
  post=GPUPostprocess(view);view._gpu_post=post
  for _ in range(4):actual,actual_depth=view.render(depth=True)
  diff=np.abs(actual.astype(np.int16)-reference.astype(np.int16));depth_diff=np.abs(actual_depth-reference_depth)
  result.update(rgb_max_abs=int(diff.max()),rgb_differing_values=int((diff!=0).sum()),depth_max_abs_m=float(depth_diff.max()),depth_p95_abs_m=float(np.percentile(depth_diff,95)),depth_nonfinite=int((~np.isfinite(actual_depth)).sum()))
  cv2.imwrite(str(a.output/'reference.png'),reference);cv2.imwrite(str(a.output/'gpu.png'),actual)
  result['variants']=[]
  original=state['surface']
  vertices=original.vertices.copy();vertices[:,2]+=.002*np.sin(vertices[:,0])
  anchors=original.anchors.copy();anchors[:,0]+=.0001
  triangles=original.triangles.copy();triangles[[0,1]]=triangles[[1,0]]
  variants=[original,replace(original,vertices=vertices),replace(original,anchors=anchors),replace(original,triangles=triangles)]
  for i,surface in enumerate(variants):
   view._gpu_post=None;view._layout_cache=None;view.update(surface,state['observed']);ref,ref_depth=view.render(depth=True)
   view._gpu_post=post;view._layout_cache=SurfaceLayoutCache();view.update(surface,state['observed']);view.update(surface,state['observed']);out,out_depth=view.render(depth=True)
   delta=np.max(np.abs(out_depth-ref_depth));assert np.array_equal(ref,out),(i,'rgb');assert delta<=1e-6,(i,delta)
   result['variants'].append(dict(index=i,rgb_bit_exact=True,depth_max_abs_m=float(delta)))
  view._layout_cache=None;view.update(original,state['observed']);view._gpu_post=None;reference,reference_depth=view.render(depth=True)
  result['measurements']=[]
  for cycle in range(6):
   for mode in ('baseline','gpu','gpu_cached','gpu_cached','gpu','baseline'):
    guard({os.getpid()});view._gpu_post=post if mode!='baseline' else None
    view._layout_cache=SurfaceLayoutCache() if mode=='gpu_cached' and view._layout_cache is None else (view._layout_cache if mode=='gpu_cached' else None)
    view.update(original,state['observed'])
    start=time.perf_counter_ns();view.update(original,state['observed']);frame,depth=view.render(depth=True);ms=(time.perf_counter_ns()-start)/1e6
    result['measurements'].append(dict(cycle=cycle,mode=mode,ms=ms))
    assert np.array_equal(frame,reference),'RGB differs in replay'
    assert np.max(np.abs(depth-reference_depth))<=1e-6,'Depth tolerance exceeded'
  result['timings']={}
  for mode in ('baseline','gpu','gpu_cached'):
   values=[v['ms'] for v in result['measurements'] if v['mode']==mode];result['timings'][mode]=dict(mean_ms=float(np.mean(values)),p50_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))
  result['passed']=result['rgb_max_abs']==0 and result['depth_max_abs_m']<=1e-6 and result['depth_nonfinite']==0
except BaseException as error:result.update(error=repr(error),traceback=traceback.format_exc())
finally:
 if view is not None:
  if post is not None:view._gpu_post=post
  view.close()
 if guard:guard.close()
 (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if not result['passed']:raise SystemExit(2)
