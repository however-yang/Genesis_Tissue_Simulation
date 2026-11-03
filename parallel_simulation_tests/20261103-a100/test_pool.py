"""GPU integration test: heterogeneous parameters, asynchronous actions, one-env reset, recorded data."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time
from benchmark import ROOT, RELEASE, LIBRARY, GPU
from parallel_env import ParallelNativePool


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='pool-integration-r0');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if not 2<=a.workers<=16:raise ValueError('Integration test requires 2..16 workers')
    sys.path.insert(0,str(RELEASE))
    from endoscopic_tissue.scenario import SCISSORS_SCENES
    from endoscopic_tissue.training.actions import command_to_action
    from endoscopic_tissue.training.data import EpisodeReader
    import numpy as np
    center=command_to_action(SCISSORS_SCENES['wall-traction'].stages[0].command)
    lower,upper=center-.2,center+.2;lower[[3,10]],upper[[3,10]]=0.,1.
    configs=[]
    for i in range(a.workers):
        young=(6000.,12000.)[i%2]
        configs.append({'dll_path':str(LIBRARY),'dll_sha256':hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
            'bounds':{'lower':lower.tolist(),'upper':upper.tolist(),'max_rate':[.01]*11},
            'scene':{'scene':'wall-traction','preparation':'settled','physics':{'young_pa':young}},
            'max_steps':100,'cutting_enabled':False,'full_surface_checks':True,'record':True})
    output=ROOT/'runs'/a.name;results=[];actions={};ids={};steps={};before_reset=None
    started=time.time()
    with ParallelNativePool(configs,release=RELEASE,output=output,gpu_uuids=[GPU],cpus=[24,4,25,22,8,35,31,14]) as pool:
        for i in range(a.workers):pool.send(i,'reset',seed=i)
        pending=a.workers
        while pending:
            for r in pool.recv_ready():
                assert r['ok'],r
                i=r['worker_id'];ids[i]=r['episode_id'];actions[i]=r['initial_action'];steps[i]=r['observation']['step_id']
                assert r['context']['settings']['young_pa']==configs[i]['scene']['physics']['young_pa']
                results.append({'op':'reset','worker':i,'episode_id':ids[i],'roundtrip_ms':r['roundtrip_ms'],
                                'young_pa':r['context']['settings']['young_pa']});pending-=1
        assert len(set(ids.values()))==a.workers
        # Different targets and independent step counts.
        for tick in range(8):
            for i in range(a.workers):
                action=actions[i].copy();action[0]+=(i*2-1)*.00005
                pool.send(i,'step',action=action)
                if tick==0 and i==0:
                    try:pool.send(i,'step',action=action)
                    except RuntimeError as error:assert 'outstanding response' in str(error)
                    else:raise AssertionError('Second in-flight action was not rejected')
            pending=a.workers
            while pending:
                for r in pool.recv_ready():
                    assert r['ok'] and not r['terminated'],r
                    i=r['worker_id'];assert r['episode_id']==ids[i]
                    assert r['observation']['step_id']==steps[i]+1;steps[i]+=1
                    assert r['quality']['checked_metrics_passed']
                    actions[i]=r['action']['executed_target_si']
                    results.append({'op':'step','worker':i,'roundtrip_ms':r['roundtrip_ms'],
                        'worker_elapsed_ms':r['worker_elapsed_ms'],'native_timings_ms':r['timings_ms'],
                        'response_bytes':len(pickle.dumps(r,protocol=5))});pending-=1
        old_ids=ids.copy();before_reset=steps[1]
        pool.send(0,'reset',seed=100)
        pool.send(1,'step',action=actions[1])
        pending=2
        while pending:
            for r in pool.recv_ready():
                assert r['ok'],r
                if r['worker_id']==0:
                    assert r['episode_id']!=old_ids[0] and r['observation']['step_id']==0
                    actions[0]=r['initial_action'];ids[0]=r['episode_id']
                else:
                    assert r['episode_id']==old_ids[1] and r['observation']['step_id']==before_reset+1
                results.append({'op':'independent_reset','worker':r['worker_id'],'episode_id':r['episode_id'],
                                'step':r['observation']['step_id']});pending-=1
        pool.send(0,'step',action=actions[0]);r=pool.recv_ready()[0]
        assert r['ok'] and not r['terminated']
    windows=0
    for directory in sorted(output.glob('worker-*/episode-*')):
        reader=EpisodeReader(directory)
        windows+=sum(1 for _ in reader.windows(1,allow_unvalidated_scope=True))
    assert windows==8*a.workers+2,windows
    summary={'passed':True,'scope':f'{a.workers} heterogeneous material environments; {windows} standard actions; independent reset; unchanged checks; bounded IPC; data reader hash validation',
             'complete_cutting_episode':False,'training_domain_approved':False,'recorded_valid_windows':windows,
             'wall_s':time.time()-started,'events':results}
    (output/'integration-result.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='events'}))

if __name__=='__main__':main()
