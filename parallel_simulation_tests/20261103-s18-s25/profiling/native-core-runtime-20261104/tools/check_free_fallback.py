"""Exercise both original fallback stages on a freely moving unpinned block.

Toy geometry only for branch coverage; not a cutting-domain or FPS benchmark.
"""
import os,json,sys,ctypes,traceback,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--name',default='free-component-fallback-checked');args=parser.parse_args();assert args.name and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.name)
OUT=ROOT/'results'/args.name;OUT.mkdir(exist_ok=False)
os.sched_setaffinity(0,[7,19,20,25]);os.environ.update(CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(OUT/'cache'))
baseline=json.loads((ROOT/'baseline.json').read_text());sys.path.insert(0,baseline['borrowed_python_source'])
import numpy as np
from dataclasses import replace
from endoscopic_tissue.grid import build_grid
from endoscopic_tissue.native_backend import NativeTissue,Command
from endoscopic_tissue.scenario import scissors_settings
from endoscopic_tissue.training.acceptance_queue import GPUReservation
result=dict(passed=False,scope='small unpinned free block; fallback coverage, not FPS and not post-fracture acceptance');guard=native=None
try:
 guard=GPUReservation(os.environ['CUDA_VISIBLE_DEVICES'],OUT)
 dll=ctypes.CDLL(str(ROOT/'native/binaries/runtime-core.so'))
 dll.LCS_SetCoreOptimizations.argtypes=[ctypes.c_int];dll.LCS_SetCoreOptimizations.restype=ctypes.c_int
 dll.LCS_ReadRuntimeModeGateStats.argtypes=[ctypes.POINTER(ctypes.c_ulonglong),ctypes.POINTER(ctypes.c_ulonglong),ctypes.c_int];dll.LCS_ReadRuntimeModeGateStats.restype=ctypes.c_int
 def stats():
  h=(ctypes.c_ulonglong*8)();p=(ctypes.c_ulonglong*8)();assert dll.LCS_ReadRuntimeModeGateStats(h,p,8)==0;return np.array(list(h),np.int64)
 grid=build_grid(.1,size=(1.,.8,1.),density=1050);grid=replace(grid,pinned=np.zeros_like(grid.pinned))
 settings=scissors_settings();command=Command(hinge=(20.,20.,20.),blade_center=(30.,20.,20.),blade_enabled=False)
 result.update(nodes=len(grid.positions),active_nodes=int(np.count_nonzero(grid.active)),pinned_nodes=int(np.count_nonzero(grid.pinned)),dt_s=settings.dt,substeps=settings.substeps,contact_refine_passes=settings.contact_refine_passes)
 trajectories={}
 for mode in [10496,43264]:
  assert dll.LCS_SetCoreOptimizations(mode)==0
  for name,args in [('LCS_SetCoarseBoundCache',[1]),('LCS_SetBatchQueryMode',[3]),('LCS_SetQueryThreads',[64]),('LCS_SetNativeSchedule',[512,1024])]:
   fn=getattr(dll,name);fn.argtypes=[ctypes.c_int]*len(args);fn.restype=ctypes.c_int;assert fn(*args)==0
  before=stats();native=NativeTissue(grid,settings,str(ROOT/'native/binaries/runtime-core.so'),command)
  quality=[];positions=[]
  for i in range(20):
   guard({os.getpid()});observed=native.step(command);assert np.isfinite(observed.nodes).all();positions.append(observed.nodes.copy());quality.append(native.material_quality().tolist())
  delta=(stats()-before).tolist();native.close();native=None
  trajectories[mode]=np.stack(positions)
  result[str(mode)]=dict(host_counter_delta=delta,quality=quality)
  if mode==43264:assert delta[0]>0 and delta[1]==0 and delta[2]==0 and delta[3]==delta[0] and delta[4]==delta[0],delta
 a=trajectories[10496];b=trajectories[43264];error=float(np.max(np.abs(a-b)));motion=float(np.max(np.abs(b[-1]-b[0])))
 result.update(max_coordinate_difference_native_cm=error,motion_native_cm=motion)
 assert motion>.001,'The free block must actually move'
 # This is an additional trajectory check, not an exact-determinism claim.
 assert error<=1e-4,(error,'fallback-on/off differs by more than 1 micrometre')
 assert all(q[1]==0 and q[0]>=.1499 for mode in ['10496','43264'] for q in result[mode]['quality'])
 np.savez_compressed(OUT/'positions.npz',gate_off=a,gate_on=b)
 result['passed']=True
except BaseException as e:result.update(error=repr(e),traceback=traceback.format_exc())
finally:
 if native:native.close()
 if guard:guard.close()
 (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['10496','43264']},indent=2))
if not result['passed']:raise SystemExit(2)
