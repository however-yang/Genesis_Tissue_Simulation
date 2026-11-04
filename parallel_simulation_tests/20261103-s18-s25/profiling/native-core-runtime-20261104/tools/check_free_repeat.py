"""Separate physical/fallback coverage from strict cross-run determinism."""
import os,json,sys,ctypes,traceback,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--name',default='free-component-repeat-control');args=parser.parse_args();assert args.name and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.name)
OUT=ROOT/'results'/args.name;OUT.mkdir(exist_ok=False)
os.sched_setaffinity(0,[7,19,20,25]);os.environ.update(CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(OUT/'cache'))
base=json.loads((ROOT/'baseline.json').read_text());sys.path.insert(0,base['borrowed_python_source'])
import numpy as np
from dataclasses import replace
from endoscopic_tissue.grid import build_grid
from endoscopic_tissue.native_backend import NativeTissue,Command
from endoscopic_tissue.scenario import scissors_settings
from endoscopic_tissue.training.acceptance_queue import GPUReservation
result=dict(passed=False,scope='small free-block control; not FPS or real post-fracture acceptance',strict_trajectory_threshold_native_cm=1e-4);native=guard=None
trajectories={}
try:
 guard=GPUReservation(os.environ['CUDA_VISIBLE_DEVICES'],OUT)
 grid=build_grid(.1,size=(1.,.8,1.),density=1050);grid=replace(grid,pinned=np.zeros_like(grid.pinned));settings=scissors_settings();command=Command(hinge=(20.,20.,20.),blade_center=(30.,20.,20.),blade_enabled=False)
 result['cases']={}
 candidate=str(ROOT/'native/binaries/runtime-core.so')
 for name,library,mode in [('off1',candidate,10496),('off2',candidate,10496),('on1',candidate,43264),('on2',candidate,43264),('v8_1',base['baseline_library'],0),('v8_2',base['baseline_library'],0)]:
  dll=ctypes.CDLL(library);dll.LCS_SetCoreOptimizations.argtypes=[ctypes.c_int];dll.LCS_SetCoreOptimizations.restype=ctypes.c_int;assert dll.LCS_SetCoreOptimizations(mode)==0
  def stats():
   if not hasattr(dll,'LCS_ReadRuntimeModeGateStats'):return np.zeros(8,np.int64)
   h=(ctypes.c_ulonglong*8)();p=(ctypes.c_ulonglong*8)();fn=dll.LCS_ReadRuntimeModeGateStats;fn.argtypes=[ctypes.POINTER(ctypes.c_ulonglong),ctypes.POINTER(ctypes.c_ulonglong),ctypes.c_int];fn.restype=ctypes.c_int;assert fn(h,p,8)==0;return np.array(list(h),np.int64)
  for symbol,args in [('LCS_SetCoarseBoundCache',[1]),('LCS_SetBatchQueryMode',[3]),('LCS_SetQueryThreads',[64]),('LCS_SetNativeSchedule',[512,1024])]:
   fn=getattr(dll,symbol);fn.argtypes=[ctypes.c_int]*len(args);fn.restype=ctypes.c_int;assert fn(*args)==0
  before=stats();native=NativeTissue(grid,settings,library,command);positions=[];quality=[]
  for i in range(20):
   guard({os.getpid()});observed=native.step(command);assert np.isfinite(observed.nodes).all();positions.append(observed.nodes.copy());quality.append(native.material_quality().tolist())
  delta=(stats()-before).tolist();native.close();native=None;trajectories[name]=np.stack(positions)
  result['cases'][name]=dict(library=library,mode=mode,host_delta=delta,min_volume_ratio=min(x[0] for x in quality),max_invalid_tets=max(x[1] for x in quality),quality=quality)
  if mode==43264:assert delta[0]>0 and delta[1]==delta[2]==0 and delta[3]==delta[4]==delta[0],delta
 np.savez_compressed(OUT/'positions.npz',**trajectories)
 result['comparisons']={}
 for a,b in [('v8_1','v8_2'),('off1','off2'),('on1','on2'),('off1','on1')]:
  delta=trajectories[a]-trajectories[b];by_frame=np.abs(delta).max(axis=(1,2));result['comparisons'][a+'-'+b]=dict(max_coordinate_error_native_cm=float(by_frame.max()),per_frame_coordinate_error_native_cm=by_frame.tolist(),max_com_error_native_cm=float(np.abs(trajectories[a].mean(axis=1)-trajectories[b].mean(axis=1)).max()),within_one_micrometre=bool(by_frame.max()<=1e-4))
 result['physical_quality_passed']=all(x['min_volume_ratio']>=.1499 and x['max_invalid_tets']==0 for x in result['cases'].values())
 result['fallback_coverage_passed']=True
 result['strict_cross_run_determinism_passed']=all(x['within_one_micrometre'] for x in result['comparisons'].values())
 result['passed']=result['physical_quality_passed'] and result['strict_cross_run_determinism_passed']
except BaseException as e:result.update(error=repr(e),traceback=traceback.format_exc())
finally:
 if native:native.close()
 if guard:guard.close()
 (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 summary={k:v for k,v in result.items() if k not in ['cases','comparisons']};summary['comparisons']={k:{key:value for key,value in row.items() if not key.startswith('per_frame')} for k,row in result.get('comparisons',{}).items()};print(json.dumps(summary,indent=2))
if not result['passed']:raise SystemExit(2)
