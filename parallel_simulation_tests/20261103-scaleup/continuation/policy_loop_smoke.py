"""Two-process sampling -> small CPU imitation update -> fresh rollout evaluation.

This validates data/control plumbing for tiny assisted-held tool motions. It is
not a trained cutting policy, PPO benchmark, world model or domain approval.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
RELEASE=PROJECT/'releases/0643e10991cf'
LIBRARY=PROJECT/'builds/0643e10991cf-contact-groups31-sm80/lib/libLiverCudaSim.so'
GPU='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'
sys.path[:0]=[str(PROJECT/'parallel_simulation_tests/20261103-a100'),str(RELEASE)]
from parallel_env import ParallelNativePool
from endoscopic_tissue.scenario import SCISSORS_SCENES
from endoscopic_tissue.training.actions import command_to_action
from endoscopic_tissue.training.data import EpisodeReader


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def features(obs,origin,target):
 t=obs['tools'];x=float(t['grasper_hinge_m'][0]);loads=t['jaw_load_n']
 return np.array([(target-x)*1000,(x-origin)*1000,float(loads[0])/5,float(loads[1])/5],np.float32)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261103)
 from threadpoolctl import threadpool_limits
 threadpool_limits(limits=1)
 policy=torch.nn.Sequential(torch.nn.Linear(4,16),torch.nn.Tanh(),torch.nn.Linear(16,1),torch.nn.Tanh())
 center=command_to_action(SCISSORS_SCENES['wall-traction'].stages[0].command);low,high=center-.2,center+.2;low[[3,10]]=0;high[[3,10]]=1
 configs=[dict(dll_path=str(LIBRARY),dll_sha256=digest(LIBRARY),bounds=dict(lower=low.tolist(),upper=high.tolist(),max_rate=[.01]*11),scene=dict(scene='wall-traction',preparation='held',physics=dict(young_pa=young)),max_steps=100,cutting_enabled=False,full_surface_checks=True,record=True) for young in (SCISSORS_SCENES['wall-traction'].young_pa,)*2]
 record=dict(completed=False,passed=False,scope='Pipeline smoke only: two assisted-held environments, 20 micrometre maximum x target increments, +/-0.2mm target envelope, no cutting, CPU imitation update then fresh evaluation. Not an RL/cutting performance or domain certification.',training_domain_approved=False,source_sha256=digest(__file__),native_sha256=digest(LIBRARY),policy_device='cpu',simulation_gpu=GPU,events=[],episodes={},policy_versions={},quality_rejections=[])
 training_x=[];training_y=[];events=(out/'policy-events.jsonl').open('x');start=time.time()
 try:
  with ParallelNativePool(configs,release=RELEASE,output=out/'sampler',gpu_uuids=[GPU],cpus=[28,31]) as pool:
   for phase,count in [('collect',32),('reevaluate',16)]:
    if phase=='reevaluate':
     x=torch.from_numpy(np.stack(training_x));y=torch.tensor(training_y,dtype=torch.float32).reshape(-1,1)
     before=float(torch.mean((policy(x)-y)**2).item());opt=torch.optim.Adam(policy.parameters(),lr=.015)
     for _ in range(250):
      opt.zero_grad();loss=torch.mean((policy(x)-y)**2);loss.backward();opt.step()
     after=float(torch.mean((policy(x)-y)**2).item());assert np.isfinite(after) and after<before
     weights=out/'policy-v1.npz';np.savez(weights,**{k:v.detach().cpu().numpy() for k,v in policy.state_dict().items()})
     # Read back numeric-only checkpoint and verify exact CPU inference.
     restored=torch.nn.Sequential(torch.nn.Linear(4,16),torch.nn.Tanh(),torch.nn.Linear(16,1),torch.nn.Tanh())
     with np.load(weights,allow_pickle=False) as data:restored.load_state_dict({k:torch.from_numpy(data[k].copy()) for k in data.files})
     with torch.no_grad():assert torch.equal(policy(x),restored(x))
     policy=restored.eval();record['training']=dict(samples=len(training_x),optimizer_steps=250,loss_before=before,loss_after=after,checkpoint_sha256=digest(weights),checkpoint_inference_exact=True)
     version='learned-v1-'+digest(weights)
    else:version='bounded-teacher-v0-'+digest(__file__)
    record['policy_versions'][phase]=version
    observations={};origins={};targets={};requested={};steps={};pending_features={};errors=[]
    for i in range(2):pool.send(i,'reset',seed=(0 if phase=='collect' else 100)+i)
    remaining=2
    while remaining:
     for r in pool.recv_ready(timeout=240):
      assert r['ok'],r;i=r['worker_id'];assert r['quality']['checked_metrics_passed'] and r['quality']['within_declared_local_scope']
      observations[i]=r['observation'];origins[i]=float(r['observation']['tools']['grasper_hinge_m'][0]);targets[i]=origins[i]+(.0001 if i==0 else -.0001);requested[i]=np.array(r['initial_action']);steps[i]=0
      record['episodes'][r['episode_id']]=dict(phase=phase,worker=i,policy_version=version,reset_lineage_id=r['context']['reset_lineage_id'],context_sha256=r['context']['context_sha256'],target_x_m=targets[i]);remaining-=1
    def send(i):
     feat=features(observations[i],origins[i],targets[i])
     if phase=='collect':control=float(np.tanh(feat[0]/.02))
     else:
      with torch.no_grad():control=float(policy(torch.from_numpy(feat)).item())
     action=requested[i].copy();action[0]=np.clip(action[0]+control*.00002,origins[i]-.0002,origins[i]+.0002)
     pending_features[i]=(feat,control,action.copy());pool.send(i,'step',action=action.tolist())
    for i in range(2):send(i)
    finished=set()
    while len(finished)<2:
     for r in pool.recv_ready(timeout=120):
      assert r['ok'],r;i=r['worker_id'];quality=r.get('quality')
      if (r['terminated'] or r['truncated'] or quality is None or not quality['checked_metrics_passed'] or not quality['within_declared_local_scope']):
       record['quality_rejections'].append(r);raise RuntimeError('Original quality/scope/termination rejected sample; no unsafe training sample retained')
      assert not r['action']['clipped_coordinates'] and not r['action']['rate_limited_coordinates']
      feat,control,action=pending_features[i]
      if phase=='collect':training_x.append(feat);training_y.append(control)
      steps[i]+=1;observations[i]=r['observation'];requested[i]=np.array(r['action']['executed_target_si'])
      error=(targets[i]-float(r['observation']['tools']['grasper_hinge_m'][0]))*1000;errors.append(abs(error))
      event=dict(phase=phase,worker=i,episode_id=r['episode_id'],after_step_id=r['observation']['step_id'],policy_version=version,requested_target_si=action.tolist(),executed_target_si=requested[i].tolist(),target_x_m=targets[i],error_mm=error,roundtrip_ms=r['roundtrip_ms'],quality_passed=True)
      events.write(json.dumps(event)+'\n');events.flush();record['events'].append(event)
      if steps[i]<count:send(i)
      else:finished.add(i)
    record[phase]=dict(transitions=sum(steps.values()),mean_absolute_tool_target_error_mm=float(np.mean(errors)),last_error_each_mm={str(i):float((targets[i]-observations[i]['tools']['grasper_hinge_m'][0])*1000) for i in range(2)})
    (out/'progress.json').write_text(json.dumps(record,indent=2));print('PHASE',phase,'done',record[phase],flush=True)
  # Read persisted N transitions / N+1 states with existing integrity checks.
  windows=0;episodes=[]
  for directory in sorted((out/'sampler').glob('worker-*/episode-*')):
   reader=EpisodeReader(directory);n=sum(1 for _ in reader.windows(1,allow_unvalidated_scope=True));windows+=n
   context=json.loads((directory/'context.json').read_text());assert context['episode_id'] in record['episodes'];episodes.append(context['episode_id'])
  assert windows==96 and len(episodes)==4 and len(set(episodes))==4
  record.update(completed=True,passed=True,validated_recorded_windows=windows,independent_episode_count=4,wall_s=time.time()-start)
 finally:
  events.close();(out/'result.json').write_text(json.dumps(record,indent=2))
if __name__=='__main__':main()
