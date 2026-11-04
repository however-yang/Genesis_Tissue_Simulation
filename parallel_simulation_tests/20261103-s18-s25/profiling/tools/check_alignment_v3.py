import argparse,json,os,sys,time,traceback
from pathlib import Path
r=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
os.sched_setaffinity(0,[7,19,20,25]);os.environ.update(CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',XDG_CACHE_HOME=str(a.output/'cache'),TMPDIR=str(r/'tmp'))
sys.path.insert(0,str(r/'candidate-v3-source'))
import numpy as np
import cv2
from threadpoolctl import threadpool_limits
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.training.egl_device import bind
from endoscopic_tissue.training.snapshot import decode_python
from endoscopic_tissue.training.vision import align_depth_to_endoscope
from endoscopic_tissue.native_backend import Settings
from endoscopic_tissue.scenario import SCISSORS_SCENES
from endoscopic_tissue.rendering import EndoscopicView
result=dict(passed=False,measurements=[],scope='fixed real cut-state render+raw/aligned depth+validity, all CPU readbacks included; no physics step');guard=view=post=None
try:
 guard=GPUReservation(os.environ['CUDA_VISIBLE_DEVICES'],a.output);bind(os.environ['CUDA_VISIBLE_DEVICES'])
 state=decode_python((r/'results/baseline-profile/render-state.bin').read_bytes());settings=Settings(**json.loads((r/'results/baseline-profile/context.json').read_text())['settings']);scene=SCISSORS_SCENES['wall-traction']
 with threadpool_limits(limits=1):
  cv2.setNumThreads(4);view=EndoscopicView(settings,1280,960,supersample=1,kidney=scene.shape,camera=scene.shot,gpu_postprocess=True,cache_surface_layout=True,gpu_aligned_depth=True);post=view._gpu_post;view.update(state['surface'],state['observed'])
  view._gpu_post=None
  for _ in range(4):reference,raw=view.render(depth=True)
  reference_depth,reference_valid=align_depth_to_endoscope(raw,view.lens_map,view.vignette,view.camera.near,view.camera.far)
  view._gpu_post=post
  for _ in range(4):actual,raw_gpu=view.render(depth=True)
  result.update(rgb_bit_exact=bool(np.array_equal(actual,reference)),raw_depth_max_error_m=float(np.max(np.abs(raw_gpu-raw))),aligned_depth_max_error_m=float(np.max(np.abs(post.aligned_depth-reference_depth))),validity_differences=int(np.count_nonzero(post.aligned_valid!=reference_valid)))
  assert result['rgb_bit_exact'] and result['raw_depth_max_error_m']<=1e-6 and result['aligned_depth_max_error_m']<=1e-6 and result['validity_differences']==0,result
  for cycle in range(8):
   for mode in ('cpu_alignment','gpu_alignment','gpu_alignment','cpu_alignment'):
    guard({os.getpid()});post.aligned_enabled=mode=='gpu_alignment'
    start=time.perf_counter_ns();frame,raw_depth=view.render(depth=True)
    if post.aligned_enabled:depth,valid=post.aligned_depth,post.aligned_valid
    else:depth,valid=align_depth_to_endoscope(raw_depth,view.lens_map,view.vignette,view.camera.near,view.camera.far)
    elapsed=(time.perf_counter_ns()-start)/1e6
    assert np.array_equal(frame,reference) and np.array_equal(valid,reference_valid) and np.max(np.abs(depth-reference_depth))<=1e-6
    result['measurements'].append(dict(cycle=cycle,mode=mode,ms=elapsed))
  result['timings']={}
  for mode in ('cpu_alignment','gpu_alignment'):
   v=[x['ms'] for x in result['measurements'] if x['mode']==mode];result['timings'][mode]=dict(mean_ms=float(np.mean(v)),p50_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)))
  result['passed']=True
except BaseException as e:result.update(error=repr(e),traceback=traceback.format_exc())
finally:
 if view:
  if post:view._gpu_post=post
  view.close()
 if guard:guard.close()
 (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='measurements'},indent=2))
if not result['passed']:raise SystemExit(2)
