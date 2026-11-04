import argparse,ctypes,os,sys,json,time,traceback
from pathlib import Path
p=Path(__file__).resolve().parents[1];os.sched_setaffinity(0,[7,19,20,25]);os.environ.update(CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');sys.path.insert(0,str(p/'python-gpu-check'))
import numpy as np
from endoscopic_tissue import instrument_geometry as g
from endoscopic_tissue.training.gpu_body_check import parameters
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.training.snapshot import decode_python
from endoscopic_tissue.scenario import scissors_frame
parser=argparse.ArgumentParser();parser.add_argument('--dll',type=Path,default=p/'native-readonly-v8/build/lib/libLiverCudaSim.so');parser.add_argument('--output',type=Path,default=p/'results/gpu-body-v8-synthetic');a=parser.parse_args();out=a.output;out.mkdir(exist_ok=False);r=dict(passed=False,cases=[],max_abs_error_native_cm=0.);guard=None
try:
 guard=GPUReservation(os.environ['CUDA_VISIBLE_DEVICES'],out);dll=ctypes.CDLL(str(a.dll.resolve()));fun=dll.LCS_TestBodyDepth;fun.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];fun.restype=ctypes.c_int;geo=g.load_scissors();rng=np.random.default_rng(28111);cases=[]
 for i in range(240):
  hinge=rng.uniform(-2,2,3).astype(np.float32);axis,normal=[np.asarray(v,dtype=np.float32) for v in scissors_frame(*rng.uniform(-45,45,3))];gap=float(rng.uniform(0,geo.max_gap));vertices=(hinge+rng.normal(size=(120,3))*rng.choice([.01,.1,1.,3.,20.])).astype(np.float32)
  if i%3==0:vertices[:9]=hinge+np.array([[-4.,0,0],[-4,1e-8,0],[-4,0,1e-8]]*3,dtype=np.float32)
  cases.append((vertices,np.arange(120,dtype=np.int32).reshape(-1,3),geo,hinge,axis,normal,gap))
 data=decode_python((p/'results/query-v4-mode10-full-validation/failure-render-state.bin').read_bytes());surf=data['surface'];obs=data['observed'];surf=surf if hasattr(surf,'vertices') else type('Surface',(),surf);obs=obs if hasattr(obs,'scissors') else type('Observed',(),obs);pose=obs.scissors;pose=pose if hasattr(pose,'hinge') else type('Pose',(),pose);cases.append((surf.vertices,surf.triangles,geo,pose.hinge,pose.axis,pose.normal,float(pose.gap)))
 supported=positive=0
 for i,args in enumerate(cases):
  guard({os.getpid()});v,tri,geometry,hinge,axis,normal,gap=args;assert np.array_equal(tri.ravel(),np.arange(len(v)));packed,flags=parameters(geometry,hinge,axis,normal,gap);actual=np.empty(2,np.float64);reference=g.scissors_body_depths(*args);code=fun(len(v),v.ctypes.data,packed.ctypes.data,flags,actual.ctypes.data)
  if code==1:r['cases'].append(dict(case=i,compatibility_fallback=True));continue
  assert code==0,code;error=float(np.max(np.abs(actual-reference)));r['max_abs_error_native_cm']=max(r['max_abs_error_native_cm'],error);assert error<=2e-5,(i,actual,reference,error)
  supported+=1;positive+=int(max(reference)>0);r['cases'].append(dict(case=i,error_native_cm=error,reference=list(map(float,reference)),actual=actual.tolist()))
 assert supported>200 and positive>100
 r.update(passed=True,supported=supported,positive_reference_cases=positive,scope='GPU current-soup diagnostic sample/regression only, no physics forces modified')
except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc())
finally:
 if guard:guard.close()
 (out/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='cases'},indent=2))
if not r['passed']:raise SystemExit(2)
