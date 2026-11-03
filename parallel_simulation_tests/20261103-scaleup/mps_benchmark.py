"""Start a private user MPS daemon for one benchmark and always request its shutdown."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from benchmark import GPU, ROOT, PYTHON, gpu_sample


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True)
    p.add_argument('--workers',type=int,required=True)
    p.add_argument('--pool-test',action='store_true')
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--cpus',default='24,4,25,22,8,35,31,14')
    p.add_argument('--thread-percentage',type=int,default=100)
    p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--max-gpu-mib',type=float,default=78000)
    p.add_argument('--attempt-capacity',action='store_true')
    p.add_argument('--stop-after',default='hold')
    a=p.parse_args()
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.name):raise ValueError('Unsafe run name')
    if not 1<=a.thread_percentage<=100:raise ValueError('Invalid MPS cap')
    before=gpu_sample(GPU)
    if before['processes'] or before['used_mib']>512 or before['gpu_percent']>5:raise RuntimeError('GPU not idle')
    # Never attach to, stop, or reconfigure an existing controller.
    existing=[p.pid for p in psutil.process_iter(['name']) if p.info['name']=='nvidia-cuda-mps-control']
    if existing:raise RuntimeError('Existing MPS controller found; leave it unchanged: '+str(existing))
    directory=ROOT/'mps'/a.name;directory.mkdir(parents=True,exist_ok=False)
    pipe=directory/'p';log=directory/'l';pipe.mkdir(mode=0o700);log.mkdir(mode=0o700)
    # Linux UNIX socket path must fit sun_path.
    if len(str(pipe))+20>=108:raise ValueError('MPS pipe path too long')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPU,CUDA_MPS_PIPE_DIRECTORY=str(pipe),
        CUDA_MPS_LOG_DIRECTORY=str(log),CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(a.thread_percentage))
    records=[];started=False;code=1
    def control(command):
        q=subprocess.run(['nvidia-cuda-mps-control'],input=command+'\n',env=env,text=True,capture_output=True,timeout=20)
        records.append({'command':command,'stdout':q.stdout,'stderr':q.stderr,'exit_code':q.returncode,'time':time.time()})
        return q
    try:
        q=subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,text=True,capture_output=True,timeout=20)
        records.append({'command':'start private daemon','stdout':q.stdout,'stderr':q.stderr,'exit_code':q.returncode})
        q.check_returncode();started=True
        control('get_server_list')
        command=[PYTHON,'-B','-u',str(ROOT/'benchmark.py'),'--name',a.name,'--workers',str(a.workers),
                 '--episodes',str(a.episodes),'--cpus',a.cpus,'--timeout',str(a.timeout),
                 '--max-gpu-mib',str(a.max_gpu_mib),'--stop-after',a.stop_after]
        if a.attempt_capacity:command.append('--attempt-capacity')
        if a.pool_test:
            command=[PYTHON,'-B','-u',str(ROOT/'test_pool.py'),'--name',a.name,'--workers',str(a.workers)]
        code=subprocess.call(command,env=env,cwd=ROOT)
        control('get_server_list')
    finally:
        if started:control('quit')
        (directory/'lifecycle.json').write_text(json.dumps(records,indent=2))
    raise SystemExit(code)

if __name__=='__main__':main()
