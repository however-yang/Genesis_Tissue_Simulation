"""One owned GPU worker at a time; wait for the user's GPU tasks, never stop them."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CONFIG=HERE/'gpu-queue.json'
LOGS=ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/gpu-queue'


def foreign_python():
    own=psutil.Process()
    ids={own.pid,*(p.pid for p in own.children(recursive=True))}
    result=[]
    for p in psutil.process_iter(['name','cmdline']):
        if p.pid in ids or 'python' not in (p.info['name'] or '').lower():continue
        scripts=[Path(a).name for a in (p.info['cmdline'] or ()) if a.lower().endswith('.py')]
        result.append({'pid':p.pid,'scripts':scripts})
    return result


def utilization():
    try:
        return float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],
            text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).strip().splitlines()[0])
    except (OSError,ValueError,subprocess.CalledProcessError):
        return None


def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    temp.replace(path)


def config():
    # Editing the queue does not stop an in-progress worker.
    while True:
        try:return json.loads(CONFIG.read_text(encoding='utf8'))
        except (json.JSONDecodeError,PermissionError):time.sleep(.5)


def main():
    LOGS.mkdir(parents=True,exist_ok=True)
    records=[]
    if (LOGS/'results.json').exists():records=json.loads((LOGS/'results.json').read_text(encoding='utf8'))
    own={'pid':os.getpid(),'created_utc':datetime.now(timezone.utc).isoformat(),'script':str(Path(__file__).resolve())}
    atomic(LOGS/'supervisor.json',own)
    last_status=0
    while True:
        plan=config()
        done={r['name'] for r in records}
        jobs=[j for j in plan['jobs'] if j['name'] not in done and j.get('enabled',True)]
        if not jobs:
            if plan.get('closed',False):return
            time.sleep(5);continue
        idle=0
        while idle<3:
            foreign=foreign_python();gpu=utilization()
            idle=idle+1 if not foreign and gpu is not None and gpu<=15 else 0
            if time.time()-last_status>=60:
                atomic(LOGS/'status.json',{'phase':'waiting_for_idle','foreign':foreign,'gpu_percent':gpu,'unix':time.time()})
                print('Awaiting idle GPU:',foreign,'utilization',gpu,flush=True);last_status=time.time()
            time.sleep(5)
        job=jobs[0]
        command=[sys.executable,'-X','utf8',*job['args']]
        started=time.time();interference=[];peak=0
        atomic(LOGS/'status.json',{'phase':'running','name':job['name'],'unix':started})
        print('Starting',job['name'],flush=True)
        with (LOGS/(job['name']+'.log')).open('w',encoding='utf8') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            while child.poll() is None:
                foreign=foreign_python()
                if foreign:interference.append({'unix':time.time(),'foreign':foreign})
                try:peak=max(peak,psutil.Process(child.pid).memory_info().rss)
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                time.sleep(3)
        records.append({'name':job['name'],'command':command,'start_unix':started,'wall_s':time.time()-started,
                        'returncode':child.returncode,'foreign_process_samples':interference,'peak_rss':peak})
        atomic(LOGS/'results.json',records)
        print('Finished',job['name'],'exit',child.returncode,'interference',len(interference),flush=True)


if __name__=='__main__':main()
