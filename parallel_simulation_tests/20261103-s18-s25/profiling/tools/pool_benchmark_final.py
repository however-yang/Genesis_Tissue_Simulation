"""Actual asynchronous pool: fresh full RGB-D + observation copies + pipe transfer.
The initial experiment keeps the existing one-core-per-worker scheduling intact.
"""
import argparse,ctypes,json,os,sys,time,traceback
from pathlib import Path
r=Path(__file__).resolve().parents[1]
# Spawn children re-import this module; only select source here, never create CUDA.
source=Path(os.environ.get('PROFILE_POOL_SOURCE',str(r/'candidate-source')))
sys.path.insert(0,str(source))
from endoscopic_tissue.training import parallel
original_serve=parallel._serve

class TimedConnection:
 def __init__(self,connection,directory):
  self.c=connection;self.log=(directory/'worker-spans.jsonl').open('w',buffering=1);self.nvtx=ctypes.CDLL(str(Path(sys.executable).parent.parent/'lib/libnvToolsExt.so.1'));self.command=None
 def recv(self):
  message=self.c.recv();self.command=message['command'];self.start=time.perf_counter_ns();self.cpu=time.thread_time_ns();self.request=message['request']
  self.nvtx.nvtxRangePushA(('worker.'+self.command).encode());return message
 def send(self,message):
  if self.command is None:return self.c.send(message)
  ready=time.perf_counter_ns();self.nvtx.nvtxRangePushA(b'worker.pipe_send');self.c.send(message);self.nvtx.nvtxRangePop();end=time.perf_counter_ns();self.nvtx.nvtxRangePop()
  self.log.write(json.dumps(dict(pid=os.getpid(),request=self.request,command=self.command,start_ns=self.start,ready_ns=ready,end_ns=end,thread_cpu_ns=time.thread_time_ns()-self.cpu))+'\n');self.command=None
 def close(self):self.c.close();self.log.close()

def timed_serve(connection,config,release,directory,gpu_uuid,cpus):
 return original_serve(TimedConnection(connection,directory),config,release,directory,gpu_uuid,cpus)


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--steps',type=int,default=60);p.add_argument('--vision',action='store_true');p.add_argument('--optimized',action='store_true');p.add_argument('--gpu-aligned-depth',action='store_true');a=p.parse_args()
 os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NSYS_NVTX_PROFILER_REGISTER_ONLY='0',TMPDIR=str(r/'tmp'))
 os.sched_setaffinity(0,[7,19,20,25]);parallel._serve=timed_serve
 import numpy as np
 from dataclasses import asdict
 from endoscopic_tissue.training.env import EnvironmentConfig,SceneConfig
 from endoscopic_tissue.training.collect import local_bounds
 from endoscopic_tissue.training.provenance import sha256
 from endoscopic_tissue.training.actions import command_to_action
 from endoscopic_tissue.scenario import SCISSORS_SCENES
 from endoscopic_tissue.training.pool_session import payload_bytes
 dll=r.parent/'native-s25/build/lib/libLiverCudaSim.so';scene=SceneConfig(preparation='cut_ready')
 ec=EnvironmentConfig(str(dll),sha256(dll),local_bounds(scene),scene,cutting_enabled=True,max_steps=300,
  full_surface_checks=True,outcome_semantics='physical_outcome_v1',instrument_collision_checks=True,fast_topology_checks=True)
 config=dict(asdict(ec),record=False,observation_contract='tool_kinematics_v1',max_response_bytes=64*1024**2)
 if a.vision:config['vision']=dict(width=1280,height=960,opencv_threads=4,**(dict(gpu_postprocess=True) if a.optimized else {}))
 if a.gpu_aligned_depth:
  if not a.vision or not a.optimized:raise ValueError('Aligned GPU depth needs optimized vision')
  config['vision']['gpu_aligned_depth']=True
 nvtx=ctypes.CDLL(str(Path(sys.executable).parent.parent/'lib/libnvToolsExt.so.1'))
 result=dict(workers=a.workers,vision=a.vision,optimized=a.optimized,gpu_aligned_depth=a.gpu_aligned_depth,source=str(source),dll_sha256=sha256(dll),rows=[],passed=False,scope='cut-ready preparation, 8 warmup holds, 60-step incision ramp; not complete episode acceptance')
 pool=None
 try:
  pool=parallel.ParallelNativePool([config for _ in range(a.workers)],release=source,output=a.output,gpu_uuids=['GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'],cpus=[7,19],startup_timeout=180)
  original_sample=parallel.gpu_sample
  def sampled(uuid):
   row=original_sample(uuid)
   with (a.output/'gpu-monitor.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   return row
  parallel.gpu_sample=sampled
  for w in range(a.workers):pool.send(w,'reset',seed=w)
  initial={}
  while pool.pending:
   for msg in pool.recv_ready(300):
    if not msg['ok']:raise RuntimeError(str(msg))
    initial[msg['worker_id']]=np.array(msg['initial_action']);assert msg['quality']['checked_metrics_passed']
  for _ in range(8):
   for w in range(a.workers):pool.send(w,'step',action=initial[w].tolist())
   while pool.pending:
    for msg in pool.recv_ready():
     if not msg['ok'] or msg['terminated'] or msg['truncated']:raise RuntimeError(str({k:v for k,v in msg.items() if k!='observation'}))
  target=command_to_action(next(s for s in SCISSORS_SCENES['wall-traction'].stages if s.name=='incise_1').command)
  count=[0]*a.workers;issued=[0]*a.workers;started=time.perf_counter_ns();result['measure_start_ns']=started;result['measure_start_unix_s']=time.time();nvtx.nvtxRangePushA(b'pool-measure')
  def submit(w):
   issued[w]+=1;action=initial[w]+(target-initial[w])*min(1,issued[w]/60)
   pool.send(w,'step',action=action.tolist())
  for w in range(a.workers):submit(w)
  while pool.pending:
   for msg in pool.recv_ready():
    w=msg['worker_id'];count[w]+=1
    if not msg['ok']:raise RuntimeError(str(msg))
    q=msg['quality'];obs=msg['observation'];row={k:msg[k] for k in ('worker_id','request','roundtrip_ms','worker_elapsed_ms','timings_ms','terminated','truncated','reason')}
    row.update(received_ns=time.perf_counter_ns(),quality=q,payload_bytes=payload_bytes(obs));result['rows'].append(row)
    if obs is None or not q['checked_metrics_passed'] or not q['within_declared_local_scope'] or msg['terminated'] or msg['truncated']:raise RuntimeError('Physical/control validation stopped pool: '+str(row))
    if a.vision:
     assert obs['rgb'].shape==(960,1280,3) and obs['depth_m'].shape==(960,1280) and not obs['vision']['dropped']
     assert np.isfinite(obs['depth_m']).all()
    if count[w]<a.steps:submit(w)
  result['measure_end_ns']=time.perf_counter_ns();result['measure_end_unix_s']=time.time();nvtx.nvtxRangePop();elapsed=(result['measure_end_ns']-started)/1e9
  times=[x['roundtrip_ms'] for x in result['rows']];result.update(passed=True,performance=dict(wall_s=elapsed,total_steps=sum(count),total_steps_s=sum(count)/elapsed,per_worker_steps_s=[n/elapsed for n in count],roundtrip_mean_ms=float(np.mean(times)),roundtrip_p50_ms=float(np.median(times)),roundtrip_p95_ms=float(np.percentile(times,95)),worker_mean_ms=float(np.mean([x['worker_elapsed_ms'] for x in result['rows']]))))
 except BaseException as e:result.update(error=repr(e),traceback=traceback.format_exc())
 finally:
  if pool:pool.close()
  a.output.mkdir(parents=True,exist_ok=True);(a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
 if not result['passed']:raise SystemExit(2)
if __name__=='__main__':main()
