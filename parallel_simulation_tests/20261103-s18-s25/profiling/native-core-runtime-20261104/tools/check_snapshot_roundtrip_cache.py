"""Real cut-state snapshot/restore with GPU checks; does not assert future determinism."""
import argparse,ctypes,os,sys,json,hashlib,traceback
from pathlib import Path
p=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--dll',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--compact-blocks',type=int,default=0);parser.add_argument('--bound-cache',action='store_true');parser.add_argument('--query-mode',type=int,default=1);parser.add_argument('--query-threads',type=int,default=64);parser.add_argument('--assembly-blocks',type=int,default=128);parser.add_argument('--manifold-blocks',type=int,default=1024);parser.add_argument('--exact-zero-coarse',action='store_true');parser.add_argument('--gpu-topology',action='store_true');parser.add_argument('--core-optimizations',type=int,default=0);parser.add_argument('--gpu-body-check',action='store_true');parser.add_argument('--batch-observation',action='store_true');parser.add_argument('--validate-observation',action='store_true');a=parser.parse_args();a.output.mkdir(exist_ok=False)
os.sched_setaffinity(0,[7,19,20,25]);os.environ.update(CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',XDG_CACHE_HOME=str(a.output/'cache'));sys.path.insert(0,str(a.source))
if a.batch_observation:
 import batch_observation as batched_readback
 batched_readback.install(validate=a.validate_observation)
import runtime_binding
runtime_binding.install()
from endoscopic_tissue.training.env import NativeEnvironment,EnvironmentConfig,SceneConfig
from endoscopic_tissue.training.collect import local_bounds
from endoscopic_tissue.training.provenance import sha256
from endoscopic_tissue.training.snapshot import SnapshotBackend,source_binding
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from threadpoolctl import threadpool_limits
result=dict(passed=False,future_trajectory_determinism_claimed=False);env=guard=None
try:
 guard=GPUReservation(os.environ['CUDA_VISIBLE_DEVICES'],a.output);scene=SceneConfig(preparation='cut_ready')
 native=ctypes.CDLL(str(a.dll))
 native.LCS_SetCoreOptimizations.argtypes=[ctypes.c_int];native.LCS_SetCoreOptimizations.restype=ctypes.c_int;assert native.LCS_SetCoreOptimizations(a.core_optimizations)==0
 for symbol,args in [('LCS_SetQueryCompactBlocks',[a.compact_blocks]),('LCS_SetCoarseBoundCache',[int(a.bound_cache)]),('LCS_SetBatchQueryMode',[a.query_mode]),('LCS_SetNativeSchedule',[a.assembly_blocks,a.manifold_blocks]),('LCS_SetQueryThreads',[a.query_threads]),('LCS_SetExactZeroCoarse',[int(a.exact_zero_coarse)])]:
  if hasattr(native,symbol):
   fun=getattr(native,symbol);fun.argtypes=[ctypes.c_int]*len(args);fun.restype=ctypes.c_int;assert fun(*args)==0
 with threadpool_limits(limits=1):
  env=NativeEnvironment(EnvironmentConfig(str(a.dll),sha256(a.dll),local_bounds(scene),scene,cutting_enabled=True,max_steps=100,full_surface_checks=True,outcome_semantics='physical_outcome_v1',instrument_collision_checks=True,fast_topology_checks=True,**(dict(gpu_topology_checks=True) if a.gpu_topology else {})))
  env.reset(7);
  if a.gpu_body_check:
   from endoscopic_tissue.training.gpu_body_check import GPUScissorsBodyCheck
   env.quality.body_depth_function=GPUScissorsBodyCheck(env,validate=True)
   initial_backend=SnapshotBackend(env.native);initial_hash=hashlib.sha256(initial_backend.save()).hexdigest();pose=env.observed.scissors
   env.quality.body_depth_function(env.last_surface.vertices,env.last_surface.triangles,env.native.scissors_geometry,pose.hinge,pose.axis,pose.normal,pose.gap)
   result['first_gpu_body_allocation_archive_unchanged']=initial_hash==hashlib.sha256(initial_backend.save()).hexdigest();assert result['first_gpu_body_allocation_archive_unchanged']
  action=env.previous_action.copy();action[10]=1.;cut=False
  for _ in range(20):
   guard({os.getpid()});t=env.step(action);assert t.after is not None and not t.terminated and not t.truncated,t.reason;cut|=any(e['kind']=='cut_onset' for e in t.after.metadata.get('events',[]))
  assert cut,'Actual cut not exercised';saved=env.snapshot();backend=SnapshotBackend(env.native);before=hashlib.sha256(backend.save()).hexdigest();tick=env.native.tick
  if a.gpu_topology:
   from endoscopic_tissue.training.gpu_topology import topology_statistics
   from endoscopic_tissue.training.topology_fast import topology_statistics_fast
   for _ in range(3):assert topology_statistics(env.native,env.last_surface)==topology_statistics_fast(env.last_surface)
   result['repeated_diagnostic_archive_unchanged']=before==hashlib.sha256(backend.save()).hexdigest();assert result['repeated_diagnostic_archive_unchanged']
  if a.gpu_body_check:
   pose=env.observed.scissors
   for _ in range(3):env.quality.body_depth_function(env.last_surface.vertices,env.last_surface.triangles,env.native.scissors_geometry,pose.hinge,pose.axis,pose.normal,pose.gap)
   result['repeated_gpu_body_archive_unchanged']=before==hashlib.sha256(backend.save()).hexdigest();assert result['repeated_gpu_body_archive_unchanged']
  for _ in range(4):
   guard({os.getpid()});t=env.step(action);assert t.after is not None and not t.terminated and not t.truncated,t.reason
  if a.gpu_body_check:
   callback=env.quality.body_depth_function;current_hash=hashlib.sha256(backend.save()).hexdigest();del env.quality.body_depth_function
   try:
    try:env.restore(saved,branch_label='incompatible-diagnostic-mode')
    except ValueError:result['diagnostic_mode_mismatch_rejected']=True
    else:raise AssertionError('GPU diagnostic mode mismatch accepted')
   finally:env.quality.body_depth_function=callback
   assert result['diagnostic_mode_mismatch_rejected'] and current_hash==hashlib.sha256(backend.save()).hexdigest()
  if hasattr(native,'LCS_GetRuntimeCoreMode'):
   from endoscopic_tissue.training import snapshot as snapshot_module
   original_binding=snapshot_module.binding
   current_hash=hashlib.sha256(backend.save()).hexdigest()
   def wrong_mode_binding(instance):
    data=original_binding(instance);data['native_runtime_core_mode']=a.core_optimizations^256;return data
   snapshot_module.binding=wrong_mode_binding
   try:
    try:env.restore(saved,branch_label='incompatible-runtime-core-mode')
    except ValueError:result['runtime_core_mode_mismatch_rejected']=True
    else:raise AssertionError('Native core mode mismatch accepted')
   finally:snapshot_module.binding=original_binding
   assert current_hash==hashlib.sha256(backend.save()).hexdigest()
   native.LCS_GetRuntimeCoreMode.argtypes=[];native.LCS_GetRuntimeCoreMode.restype=ctypes.c_int
   native.LCS_SetCoreOptimizations.argtypes=[ctypes.c_int];native.LCS_SetCoreOptimizations.restype=ctypes.c_int
   result['live_core_mode_change_rejected']=native.LCS_SetCoreOptimizations(a.core_optimizations^256)!=0
   assert result['live_core_mode_change_rejected'] and native.LCS_GetRuntimeCoreMode()==a.core_optimizations
  restored=env.restore(saved,branch_label='performance-private-workspace-roundtrip');after=hashlib.sha256(backend.save()).hexdigest();result['native_archive_restored_bit_exact']=before==after;assert result['native_archive_restored_bit_exact']
  assert env.native.tick==tick
  if a.gpu_topology:
   assert topology_statistics(env.native,env.last_surface)==topology_statistics_fast(env.last_surface)
   result['post_restore_diagnostic_archive_unchanged']=after==hashlib.sha256(backend.save()).hexdigest();assert result['post_restore_diagnostic_archive_unchanged']
  t=env.step(action);assert t.after is not None and not t.terminated and not t.truncated,t.reason
  result.update(passed=True,actual_cut_onset=cut,post_restore_step_checked_metrics_passed=t.after.metadata['quality']['checked_metrics_passed'],python_helper_binding=source_binding(),dll_sha256=sha256(a.dll))
except BaseException as e:result.update(error=repr(e),traceback=traceback.format_exc())
finally:
 if a.batch_observation:result['batched_observation_stats']=dict(batched_readback.stats)
 if env and (a.core_optimizations&2048) and hasattr(native,'LCS_ReadRuntimeGripStats'):
  diagnostic=(ctypes.c_int*8)();fn=native.LCS_ReadRuntimeGripStats;fn.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int];fn.restype=ctypes.c_int;code=fn(diagnostic,8)
  result['grip_feedback_validation']=dict(returncode=code,counters=list(diagnostic),passed=code==0 and diagnostic[1]==0)
  if code!=0 or diagnostic[1]!=0:result.update(error='Read-only grip feedback differs from original',passed=False)
 if env and (a.core_optimizations&32768) and hasattr(native,'LCS_ReadRuntimeModeGateStats'):
  host=(ctypes.c_ulonglong*8)();proof=(ctypes.c_ulonglong*8)();fn=native.LCS_ReadRuntimeModeGateStats;fn.argtypes=[ctypes.POINTER(ctypes.c_ulonglong),ctypes.POINTER(ctypes.c_ulonglong),ctypes.c_int];fn.restype=ctypes.c_int;code=fn(host,proof,8)
  result['zero_mode_gate_stats']=dict(returncode=code,host=list(host),proof=list(proof),reference_coordinates_unchanged=code==0 and proof[1]==0 and proof[4]==0,reference_proof_executed=proof[0]+proof[3]>0)
  if code!=0 or proof[1]!=0 or proof[4]!=0:result.update(error='Original disabled mode changed positions; shortcut rejected',passed=False)
 if env:env.close()
 if guard:guard.close()
 (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if not result['passed']:raise SystemExit(2)
