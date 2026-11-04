import os,sys,ctypes,json,time,argparse,traceback
from pathlib import Path
p=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--dll',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.mkdir(exist_ok=False)
os.sched_setaffinity(0,[7,19,20,25]);os.environ.update(CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(p/'tmp'));sys.path.insert(0,str(p/'candidate-r6-source'))
import numpy as np
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from endoscopic_tissue.training.gpu_topology import decode
from endoscopic_tissue.training.topology_fast import topology_statistics_fast
from endoscopic_tissue.native_backend import Surface
from endoscopic_tissue.training.snapshot import decode_python
r=dict(passed=False,cases=[],timings={});guard=None
try:
 guard=GPUReservation(os.environ['CUDA_VISIBLE_DEVICES'],a.output);dll=ctypes.CDLL(str(a.dll));fn=dll.LCS_TestExactTopology;fn.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*4;fn.restype=ctypes.c_int
 def gpu(s):
  pos=np.ascontiguousarray(s.vertices[s.triangles].reshape(-1,3),np.float32);aux=np.ascontiguousarray(s.anchors[s.triangles].reshape(-1,4),np.float32);out=np.empty(9,np.int32);bounds=np.empty(6,np.float32)
  status=fn(len(pos),pos.ctypes.data,aux.ctypes.data,out.ctypes.data,bounds.ctypes.data)
  if status:raise RuntimeError('GPU topology test ABI '+str(status))
  assert not out[7:].any(),out
  return decode(out,bounds)
 rng=np.random.default_rng(88);cases=[]
 def surface(points,faces,wall=False):
  v=np.asarray(points,np.float32)[np.asarray(faces,np.int32)].reshape(-1,3);aux=np.zeros((len(v),4),np.float32);aux[:,3]=wall
  return Surface(v,np.zeros_like(v),np.arange(len(v),dtype=np.int32).reshape(-1,3),aux)
 points=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1]],np.float32)
 for faces in [[[0,1,2]],[[0,1,2],[0,2,3]],[[0,1,2],[0,3,2]],[[0,1,2],[0,1,2]],[[0,1,2],[0,1,3],[0,1,4]],[[0,0,1]],[[0,1,2],[2,1,0]]]:cases.append(surface(points,faces,True))
 for scale in [np.nextafter(np.float32(1e-4),np.float32(0)),np.float32(1e-4),np.nextafter(np.float32(1e-4),np.float32(np.inf))]:cases.append(surface([[0,0,0],[1e-4,0,0],[0,scale,0]],[[0,1,2]],False))
 for i in range(100):
  pts=rng.uniform(-.05,.05,(80,3)).astype(np.float32)
  if i%3==0:pts[20:40]=pts[:20]+rng.uniform(-.000004,.000004,(20,3)).astype(np.float32)
  if i%5==0:pts[:,2]=0
  faces=rng.integers(0,len(pts),(100,3),dtype=np.int32)
  if i%2==0:faces[10:15]=faces[:5]
  s=surface(pts,faces);aux=s.anchors.copy();aux[:,3]=rng.integers(0,2,len(aux));cases.append(Surface(s.vertices,s.normals,s.triangles,aux))
 state=decode_python((p.parent/'results/baseline-profile/render-state.bin').read_bytes());cases.append(state['surface'])
 for i,s in enumerate(cases):
  if i%10==0:guard({os.getpid()})
  expected=topology_statistics_fast(s);actual=gpu(s)
  assert expected==actual,(i,expected,actual)
  r['cases'].append(dict(index=i,triangles=len(s.triangles),exact=True,reference=expected))
 for name,fnc in [('cpu_reference',lambda:topology_statistics_fast(state['surface'])),('gpu_test_with_upload',lambda:gpu(state['surface']))]:
  values=[]
  for _ in range(20):
   t=time.perf_counter_ns();fnc();values.append((time.perf_counter_ns()-t)/1e6)
  r['timings'][name]=dict(mean_ms=float(np.mean(values)),p95_ms=float(np.percentile(values,95)))
 r['passed']=True
except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc())
finally:
 if guard:guard.close()
 (a.output/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='cases'},indent=2))
if not r['passed']:raise SystemExit(2)
