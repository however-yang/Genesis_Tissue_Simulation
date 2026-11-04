import os,ctypes,json,sys,traceback
from pathlib import Path
p=Path(__file__).resolve().parents[1];os.environ.update(CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TMPDIR=str(p/'tmp'));os.sched_setaffinity(0,[7,19,20,25]);sys.path.insert(0,str(p.parent/'aggressive-20261104/final-source'))
from endoscopic_tissue.training.acceptance_queue import GPUReservation
out=p/'results/query-order-stride-probes';out.mkdir(exist_ok=False);guard=None;r=dict(passed=False,cases=[])
try:
 guard=GPUReservation(os.environ['CUDA_VISIBLE_DEVICES'],out);dll=ctypes.CDLL(str(p/'native-all-queries-v2/build/lib/libLiverCudaSim.so'));fun=dll.LCS_QueryOrderProbe;fun.argtypes=[ctypes.c_int,ctypes.c_uint,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.POINTER(ctypes.c_int)];fun.restype=ctypes.c_int
 cases=[('grasp_ordered',3,0x88,[0,3,1,4,2,5],[0,3,1,4,2,5],True),('grasp_within_kind_permutation',3,0x88,[0,3,1,4,2,5],[3,0,4,1,5,2],True),('grasp_cross_kind_permutation',3,0x88,[0,3,1,4,2,5],[0,1,3,4,2,5],False),('six_slot_ordered',6,209,[1,7,2,8,3,9],[7,1,8,2,9,3],True),('six_slot_cross_kind_permutation',6,209,[1,7,2,8,3,9],[1,2,7,8,3,9],False)]
 for name,stride,packed,a,b,expected in cases:
  aa=(ctypes.c_int*len(a))(*a);bb=(ctypes.c_int*len(b))(*b);stats=(ctypes.c_int*32)();assert fun(stride,packed,3,aa,bb,len(a),stats)==0;actual=stats[7]==0 and stats[9]==0;item=dict(name=name,expected_accept=expected,accepted=actual,activity_differences=stats[7],group_order_differences=stats[9]);r['cases'].append(item);assert actual==expected,item;assert stats[7]==0
 guard({os.getpid()});r['passed']=True
except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc())
finally:
 if guard:guard.close()
 (out/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if not r['passed']:raise SystemExit(2)
