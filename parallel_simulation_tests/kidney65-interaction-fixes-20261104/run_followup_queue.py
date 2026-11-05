"""Serial, failure-stopping GPU1 diagnostics after the frozen S15 queue.

Uses stdlib only while waiting; never opens a GPU context in this coordinator.
No promotion, code installation, training launch, process search/termination,
or Git operation is performed. Only an owned timed-out child can be stopped.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
GPU = 'GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'


def jobs():
    common = ['--headless','--guard','--certify-envelope','--stress-tests',
              '--stress-route','tracked','--replay-repeats','20']
    rows = []
    for name, library, proof in [('r37-full-proof-01','r37-split-coarse',True),
                                 ('r33-paired-normal-02','r33-packed-mg',False),
                                 ('r37-paired-normal-01','r37-split-coarse',False)]:
        args = ['run_scene.py','--name',name,'--library',library]+common
        if proof:args += ['--coarse-proof']
        rows.append(dict(name=name,args=args,timeout=1800,
            checks={'report.json': {'status':'passed'},'training-stress.json': {'passed':True}}))
    for name, library, task, seed in [
        ('friction-kidney65-r23-roundtrip-01','r23-production','roundtrip',0),
        ('friction-kidney65-r33-roundtrip-01','r33-packed-mg','roundtrip',0),
        ('friction-kidney65-grasped-lower2-plusy4-seed1','r33-packed-mg','dose',1),
        ('friction-kidney65-grasped-lower2-plusy4-seed2','r33-packed-mg','dose',2)]:
        rows.append(dict(name=name,timeout=900,continue_after_failure=library=='r23-production',args=['friction_pilot.py','--name',name,
            '--library',library,'--scene','kidney65','--preparation','grasped',
            '--bite-lower-mm','2','--adaptive-grasp','--task',task,
            '--direction','plusy','--dose-mm','4','--seed',str(seed)],
            checks={'result.json':{'passed':True,'all_anchor_counts_zero':True}}))
    name='snapshot-r33-owned-resource20-02'
    rows.append(dict(name=name,timeout=900,args=['snapshot_resources.py','--name',name,
        '--library','r33-packed-mg','--cycles','20'],checks={'result.json':{'functional_passed':True}}))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--wait-for',default='s15-v1-formal20-01')
    p.add_argument('--allow-independent-after-predecessor-failure',action='store_true',
                   help='Explicit diagnostic continuation after the predecessor failure has been inspected; never clears its failure')
    a=p.parse_args()
    if os.name=='nt':p.error('All GPU tests run on server A100 GPU1')
    if not all(v.replace('-','').isalnum() for v in (a.name,a.wait_for)):p.error('Use simple labels')
    out=ROOT/'outputs'/a.name;out.mkdir(exist_ok=False)
    source_names=['run_scene.py','scene_config.py','training_stress.py','envelope_audit.py',
        'friction_pilot.py','friction_roundtrip.py','snapshot_resources.py','training_api_smoke.py',
        'kidney_training.py','python/endoscopic_tissue/training/env.py',
        'python/endoscopic_tissue/training/kidney_snapshot.py']
    source_names += ['binaries/'+v+'.so' for v in ('r23-production','r33-packed-mg','r37-split-coarse')]
    hashes={v:hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for v in source_names}
    (out/'driver.py').write_bytes(Path(__file__).read_bytes())
    state=dict(status='waiting_for_predecessor',gpu_uuid=GPU,jobs=[],input_sha256=hashes,
        all_training_and_frame_gates_passed=False,default_replaced=False,github_pushed=False)
    def save():
        temporary=out/'progress.tmp'
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        os.replace(temporary,out/'progress.json')
    save()
    try:
        while True:
            previous=json.loads((ROOT/'outputs'/a.wait_for/'progress.json').read_text())
            if previous['status'] not in ('running','waiting_for_gpu'):break
            time.sleep(10)
        state['predecessor']=dict(name=a.wait_for,status=previous['status'],passed=previous.get('passed'))
        if (previous['status']!='completed' or not previous.get('passed')) and not a.allow_independent_after_predecessor_failure:
            state['status']='blocked_by_predecessor_failure';save();return 2
        state['predecessor_failure_preserved']=not bool(previous.get('passed'))
        for job in jobs():
            for v,digest in hashes.items():
                if hashlib.sha256((ROOT/v).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError('Input changed; do not mix versions: '+v)
            while True:
                lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                    '--format=csv,noheader,nounits'],text=True).splitlines()
                busy=[v for v in lines if v.split(',')[0].strip()==GPU]
                if not busy:break
                state.update(status='waiting_for_gpu',gpu_processes=busy);save();time.sleep(10)
            state.pop('gpu_processes',None);state['status']='running'
            command=[sys.executable,'-X','utf8','-B',str(ROOT/job['args'][0])]+job['args'][1:]
            row=dict(name=job['name'],command=command,started_unix=time.time(),passed=False)
            state['jobs'].append(row);save()
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=GPU
            with (ROOT/'logs'/(job['name']+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT)
                row['owned_child_pid']=child.pid;save()
                try:code=child.wait(timeout=job['timeout'])
                except subprocess.TimeoutExpired:
                    child.kill();child.wait();row['owned_child_timeout']=True;code=-9
            row.update(returncode=code,finished_unix=time.time(),result_sha256={})
            ok=code==0
            for filename,checks in job['checks'].items():
                path=ROOT/'outputs'/job['name']/filename
                if not path.exists():ok=False;continue
                data=json.loads(path.read_text())
                row['result_sha256'][filename]=hashlib.sha256(path.read_bytes()).hexdigest()
                ok &= all(data.get(key)==expected for key,expected in checks.items())
            row['passed']=bool(ok);save();print(json.dumps(row),flush=True)
            if not ok and not job.get('continue_after_failure',False):
                state['status']='stopped_on_failed_test';save();return 2
        state['status']='completed';state['all_listed_jobs_passed']=all(r['passed'] for r in state['jobs']);save()
        return 0 if state['all_listed_jobs_passed'] else 2
    except BaseException as error:
        state.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':raise SystemExit(main())
