import argparse,os,sys,time,json,traceback
from pathlib import Path
p=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.mkdir(exist_ok=False)
os.sched_setaffinity(0,[7,19,20,25]);os.environ.update(CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',XDG_CACHE_HOME=str(a.output/'cache'),TMPDIR=str(p/'tmp'));sys.path.insert(0,str(p/'candidate-r2-source'))
import numpy as np,cv2
from dataclasses import replace
from threadpoolctl import threadpool_limits
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.training.egl_device import bind
from endoscopic_tissue.training.snapshot import decode_python
from endoscopic_tissue.native_backend import Settings
from endoscopic_tissue.scenario import SCISSORS_SCENES
from endoscopic_tissue.rendering import EndoscopicView
view=guard=None;result=dict(passed=False,variants=[],measurements=[])
try:
 guard=GPUReservation(os.environ['CUDA_VISIBLE_DEVICES'],a.output);bind(os.environ['CUDA_VISIBLE_DEVICES']);cv2.setNumThreads(4)
 state=decode_python((p.parent/'results/baseline-profile/render-state.bin').read_bytes());settings=Settings(**json.loads((p.parent/'results/baseline-profile/context.json').read_text())['settings']);scene=SCISSORS_SCENES['wall-traction']
 with threadpool_limits(limits=1):
  view=EndoscopicView(settings,1280,960,supersample=1,kidney=scene.shape,camera=scene.shot,gpu_postprocess=True,gpu_aligned_depth=True,fused_mesh=True,pbo_readback=True)
  saved_pbo=view._gpu_post.pbo
  original=state['surface'];vertices=original.vertices.copy();vertices[:,2]+=.002*np.sin(vertices[:,0]);anchors=original.anchors.copy();anchors[:,0]+=.0001
  order=np.arange(len(original.triangles)-1,-1,-1);ids=original.triangles[order].ravel()
  # Reconstruct a fresh soup with reordered triangles and all attributes moved together.
  reorder=replace(original,vertices=original.vertices[ids],normals=original.normals[ids],anchors=original.anchors[ids])
  variants=[original,replace(original,vertices=vertices),replace(original,anchors=anchors),reorder]
  for i,surface in enumerate(variants):
   guard({os.getpid()});view._gpu_post.pbo=0;view.update(surface,state['observed']);ref,ref_raw=view.render(depth=True);ref_depth=view._gpu_post.aligned_depth.copy();ref_valid=view._gpu_post.aligned_valid.copy()
   view._gpu_post.pbo=saved_pbo;view.update(surface,state['observed']);actual,raw=view.render(depth=True)
   diff=int(np.max(np.abs(actual.astype(int)-ref.astype(int))));depth=float(np.max(np.abs(raw-ref_raw)))
   assert diff==0 and depth==0 and np.array_equal(ref_depth,view._gpu_post.aligned_depth) and np.array_equal(ref_valid,view._gpu_post.aligned_valid),(i,diff,depth)
   result['variants'].append(dict(index=i,rgb_bit_exact=True,raw_depth_bit_exact=True,aligned_depth_bit_exact=True,validity_bit_exact=True))
  for _ in range(4):view.update(original,state['observed']);view.render(depth=True)
  for cycle in range(6):
   for fused in (False,True,True,False):
    guard({os.getpid()});view._gpu_post.pbo=saved_pbo if fused else 0;t=time.perf_counter_ns();view.update(original,state['observed']);rgb,raw=view.render(depth=True);elapsed=(time.perf_counter_ns()-t)/1e6
    result['measurements'].append(dict(cycle=cycle,fused=fused,ms=elapsed))
  result['timings']={}
  for fused in (False,True):
   values=[x['ms'] for x in result['measurements'] if x['fused']==fused];result['timings']['pbo' if fused else 'synchronous_readback']=dict(mean_ms=float(np.mean(values)),p50_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))
  result['passed']=True
except BaseException as e:result.update(error=repr(e),traceback=traceback.format_exc())
finally:
 if view:
  view._gpu_post.pbo=saved_pbo
  view.close()
 if guard:guard.close()
 (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='measurements'},indent=2))
if not result['passed']:raise SystemExit(2)
