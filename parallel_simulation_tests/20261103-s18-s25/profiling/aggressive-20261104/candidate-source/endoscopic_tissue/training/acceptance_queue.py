"""Resume isolated acceptance jobs at immutable attempt boundaries.

Progress comes from declared child artifacts and stdout, never the supervisor's
own monitoring writes. Exit success and physical acceptance remain separate.
"""
import argparse
import fcntl
import glob
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import psutil

HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)

def progress_signature(patterns,exclude=()):
    result={};excluded={str(Path(p).resolve()) for p in exclude}
    for pattern in patterns:
        for name in glob.iglob(pattern,recursive=True):
            p=Path(name)
            if not p.is_file() or str(p.resolve()) in excluded:continue
            try:s=p.stat();result[str(p)]=(s.st_size,s.st_mtime_ns)
            except FileNotFoundError:pass
    return result

def owned_tree(proc,known):
    try:
        for p in [psutil.Process(proc.pid),*psutil.Process(proc.pid).children(recursive=True)]:
            known[p.pid]=p.create_time()
    except psutil.NoSuchProcess:pass
    return known

def living_owned(known):
    result=[]
    for pid,created in known.items():
        try:
            p=psutil.Process(pid)
            if p.create_time()==created and p.status()!=psutil.STATUS_ZOMBIE:result.append(p)
        except psutil.NoSuchProcess:pass
    return result

def stop_owned(proc,known,grace=5):
    owned_tree(proc,known)
    if proc.poll() is None:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    for p in living_owned(known):
        try:p.terminate()
        except psutil.NoSuchProcess:pass
    # Let Popen reap its own child so psutil cannot consume the exit status.
    try:proc.wait(timeout=max(grace,1))
    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=max(grace,1))
    _,alive=psutil.wait_procs([p for p in living_owned(known) if p.pid!=proc.pid],timeout=grace)
    for p in alive:
        try:
            if p.create_time()==known[p.pid]:p.kill()
        except psutil.NoSuchProcess:pass

def field(value,path):
    for name in path.split('.'):value=value[name]
    return value

class Runner:
    def __init__(self,directory,plan,*,resume=False,poll=.5,monitor=None):
        self.directory=Path(directory).resolve();self.directory.mkdir(parents=True,exist_ok=True)
        self.state_path=self.directory/'state.json';self.monitor=monitor;self.poll=poll
        self.lock=(self.directory/'queue.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.plan=plan;digest=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        if self.state_path.exists():
            if not resume:raise FileExistsError('Use --resume; previous attempts are immutable')
            self.state=json.loads(self.state_path.read_text())
            if self.state['plan_sha256']!=digest:raise ValueError('Resume requires the identical plan')
        else:self.state={'schema':1,'plan_sha256':digest,'started':time.time(),'jobs':{},'phase':'ready'}
        for filename,digest in plan.get('inputs_sha256',{}).items():
            if sha(filename)!=digest:raise ValueError('Pinned input changed: '+filename)
        ids=[j['id'] for j in plan['jobs']]
        if len(set(ids))!=len(ids) or any(not n or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in n) for n in ids):raise ValueError('Invalid unique job IDs')
        save(self.state_path,self.state)

    def run(self):
        try:
            for job in self.plan['jobs']:
                prior=self.state['jobs'].setdefault(job['id'],{'attempts':[]})
                if prior.get('execution_completed'):
                    for path,digest in prior.get('artifact_sha256',{}).items():
                        if sha(path)!=digest:raise ValueError('Completed evidence changed: '+path)
                    if prior.get('physical_passed') is False and job.get('stop_on_rejection',True):
                        self.state['phase']='acceptance_rejected';save(self.state_path,self.state);return 2
                    continue
                for old in prior['attempts']:
                    known_old={int(pid):created for pid,created in old.get('owned_processes',{}).items()}
                    if living_owned(known_old):raise RuntimeError('Previous attempt still has live owned processes; inspect before resume')
                attempt=self.directory/job['id']/f"attempt-{len(prior['attempts']):03d}";attempt.mkdir(parents=True,exist_ok=False)
                expand=lambda x:str(x).replace('{attempt}',str(attempt))
                command=[expand(x) for x in job['args']]
                row={'directory':str(attempt),'command':command,'started':time.time(),'status':'starting'}
                prior['attempts'].append(row);self.state.update(phase='running',current=job['id']);save(self.state_path,self.state)
                env=os.environ.copy();env.update(self.plan.get('env',{}));env.update({k:expand(v) for k,v in job.get('env',{}).items()})
                known={};proc=None
                try:
                    with (attempt/'stdout.log').open('xb') as log:
                        proc=subprocess.Popen(command,cwd=job.get('cwd',str(attempt)),env=env,stdin=subprocess.DEVNULL,
                                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        row['pid']=proc.pid;row['status']='running';owned_tree(proc,known);row['owned_processes']=known.copy()
                        patterns=[str(attempt/'stdout.log'),*[expand(x) for x in job.get('progress_globs',[])]]
                        excluded=[self.state_path,self.state_path.with_suffix('.json.tmp'),self.directory/'gpu-monitor.jsonl']
                        previous=progress_signature(patterns,excluded);last_progress=time.monotonic();started=last_progress
                        while proc.poll() is None:
                            owned_tree(proc,known)
                            current=progress_signature(patterns,excluded)
                            if current!=previous:
                                changed=[p for p,v in current.items() if previous.get(p)!=v]
                                previous=current;last_progress=time.monotonic();row['last_progress_at']=time.time();row['last_changed_paths']=changed[-12:]
                            if self.monitor:self.monitor(set(known))
                            if time.monotonic()-started>job.get('max_wall',3600):raise TimeoutError('wall_timeout')
                            if time.monotonic()-last_progress>job.get('max_idle',900):raise TimeoutError('idle_timeout')
                            row['owned_pids']=sorted(known);row['owned_processes']=known.copy();save(self.state_path,self.state);time.sleep(self.poll)
                        row.update(exit_code=proc.returncode,finished=time.time(),wall_s=time.time()-row['started'])
                    if living_owned(known):stop_owned(proc,known)
                    if proc.returncode not in job.get('allowed_exit_codes',[0]):raise RuntimeError('Unexpected exit code '+str(proc.returncode))
                    evidence={};physical=[]
                    for spec in job.get('results',[]):
                        path=Path(expand(spec['path']));value=json.loads(path.read_text())
                        for name,expected in spec.get('require',{}).items():
                            if field(value,name)!=expected:raise ValueError('Completion evidence mismatch: '+name)
                        if 'acceptance_field' in spec:physical.append(bool(field(value,spec['acceptance_field'])))
                        evidence[str(path)]=sha(path)
                    if not evidence:raise ValueError('Every completed job needs explicit result evidence')
                    prior.update(execution_completed=True,artifact_sha256=evidence,
                                 physical_passed=all(physical) if physical else None)
                    row['status']='completed';save(self.state_path,self.state)
                    print('COMPLETED',job['id'],'physical_passed',prior['physical_passed'],flush=True)
                    if physical and not all(physical) and job.get('stop_on_rejection',True):
                        self.state['phase']='acceptance_rejected';save(self.state_path,self.state);return 2
                except BaseException as error:
                    if proc is not None:stop_owned(proc,known)
                    row.update(status='interrupted',error=repr(error),finished=time.time(),exit_code=proc.returncode if proc else None)
                    self.state.update(phase='interrupted',error=repr(error));save(self.state_path,self.state)
                    raise
            self.state.update(phase='complete',finished=time.time());save(self.state_path,self.state);return 0
        finally:self.lock.close()

class GPUReservation:
    def __init__(self,gpu,directory):
        from .resources import gpu_sample
        self.sample=gpu_sample;self.gpu=gpu;self.directory=Path(directory).resolve();self.last=0
        self.lock=(HERE/('gpu-'+gpu+'.lock')).open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            initial=self.sample(gpu)
            if initial['processes'] or initial['used_mib']>512 or initial['gpu_percent']>5:raise RuntimeError('Selected GPU not idle')
            save(self.directory/'gpu-before.json',initial)
        except BaseException:
            self.lock.close()
            raise
    def __call__(self,owned):
        if time.monotonic()-self.last<2:return
        self.last=time.monotonic();sample=self.sample(self.gpu)
        with (self.directory/'gpu-monitor.jsonl').open('a') as stream:stream.write(json.dumps(sample)+'\n')
        if any(p['pid'] not in owned for p in sample['processes']):raise RuntimeError('Foreign GPU job appeared')
        if sample['used_mib']>78000 or sample['host_available_gib']<16 or sample['temp_c']>83:raise RuntimeError('Resource ceiling reached')
    def close(self):
        try:save(self.directory/'gpu-after.json',self.sample(self.gpu))
        finally:self.lock.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--gpu');a=p.parse_args()
    a.directory.mkdir(parents=True,exist_ok=True);guard=None
    try:
        if a.gpu:guard=GPUReservation(a.gpu,a.directory)
        return Runner(a.directory,json.loads(a.plan.read_text()),resume=a.resume,monitor=guard).run()
    finally:
        if guard:guard.close()
if __name__=='__main__':raise SystemExit(main())
