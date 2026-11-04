"""Two-worker S20 CUDA integration smoke, never a throughput/FPS benchmark."""
import argparse,json,os,sys,traceback
from pathlib import Path
from dataclasses import asdict
import numpy as np
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'source'))
from endoscopic_tissue.training.parallel import ParallelNativePool
from endoscopic_tissue.training.env import EnvironmentConfig,SceneConfig
from endoscopic_tissue.training.collect import local_bounds
from endoscopic_tissue.training.physics import PhysicsOverrides
from endoscopic_tissue.training.provenance import sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dll',type=Path,default=root/'native-s25/build/lib/libLiverCudaSim.so');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);r={'completed':False,'scope':'two real workers, typed reset, five cached Python-state restores, lineage and isolation; native byte equality checked separately','checks':{}}
 pool=None
 def call(worker,command,**message):
  pool.send(worker,command,**message);reply=pool.recv_ready(180)[0]
  if not reply.get('ok'):raise RuntimeError(str(reply))
  return reply
 try:
  dll=a.dll.resolve();scene=SceneConfig(preparation='settled')
  cfg=asdict(EnvironmentConfig(str(dll),sha256(dll),local_bounds(scene),scene,max_steps=20,outcome_semantics='physical_outcome_v1'))
  cfg.update(record=True,observation_contract='tool_kinematics_v1',privileged_fields=['positions_m','topology'],
     max_snapshot_bytes=2*1024**3,max_snapshot_capture_bytes=3*1024**3,max_response_bytes=64*1024**2)
  cpus=sorted(os.sched_getaffinity(0))[:2]
  pool=ParallelNativePool([cfg,cfg],release=root/'source',output=a.output/'pool',gpu_uuids=[a.gpu],cpus=cpus)
  resets=[]
  for i in range(2):
   chosen=asdict(SceneConfig(preparation='settled',contact_model=i+1,physics=PhysicsOverrides(young_pa=3000+3000*i)))
   resets.append(call(i,'reset',scene_config=chosen,seed=i))
  r['contexts']=[x['context'] for x in resets]
  r['checks']['distinct_context_hashes']=resets[0]['context']['context_sha256']!=resets[1]['context']['context_sha256']
  stable=resets[1]['privileged']['positions_m'].copy()
  snap=call(0,'snapshot',hot=True);r['snapshot']=snap
  branches=[]
  for i in range(5):
   moved=call(0,'step',action=resets[0]['initial_action'])
   out=call(0,'restore',snapshot_id=snap['snapshot_id'],branch_label='pool-repeat-'+str(i))
   branches.append(dict(index=i,position_equal=bool(np.array_equal(out['privileged']['positions_m'],resets[0]['privileged']['positions_m'])),
      context=out['context']))
  r['checks']['five_restores']=branches
  # Worker 1's first step must still be its own step 1; worker 0 reset/snapshot work cannot advance it.
  out=call(1,'step',action=resets[1]['initial_action'])
  r['checks']['other_worker_first_step']=out['observation']['step_id']
  r['checks']['other_worker_valid_outcome']=out['valid_outcome']
  pool.send(1,'restore',snapshot_id=snap['snapshot_id'],branch_label='must-reject');foreign=pool.recv_ready()[0]
  r['checks']['foreign_snapshot_rejected']=not foreign['ok'] and not foreign['requires_reset']
  call(0,'drop_snapshot',snapshot_id=snap['snapshot_id'])
  r['completed']=True
 except BaseException as error:r['error']=repr(error);r['traceback']=traceback.format_exc()
 finally:
  if pool is not None:pool.close()
  # snapshot response contains only JSON primitives, contexts too.
  (a.output/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps(r,indent=2))
 if not r['completed']:raise SystemExit(2)
if __name__=='__main__':main()
