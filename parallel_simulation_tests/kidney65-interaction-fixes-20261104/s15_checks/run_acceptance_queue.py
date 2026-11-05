"""Serial S15 queue on A100 physical GPU1; wait for foreign users, stop on failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from bootstrap import ROOT,GPU
from endoscopic_tissue.training.data import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--library',default='r33-packed-mg')
    parser.add_argument('--repeats',type=int,choices=(1,20),default=1)
    parser.add_argument('--forks',nargs='+',choices=('before_contact','first_cut','squeezing','regrasp'),
                        default=['before_contact','first_cut','squeezing','regrasp'])
    a=parser.parse_args()
    if os.name=='nt':raise RuntimeError('GPU validations must run on server A100 GPU1')
    if not all(v.replace('-','').isalnum() for v in (a.name,a.library)):raise ValueError('Invalid label')
    out=ROOT/'outputs'/a.name;out.mkdir(exist_ok=False)
    files=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'python/endoscopic_tissue/training').glob('*.py'))
    files.append(ROOT/'binaries'/(a.library+'.so'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/'driver.py').write_bytes(Path(__file__).read_bytes())
    record=dict(status='running',passed=False,gpu_uuid=GPU,arguments=vars(a),cases=[],input_sha256=hashes,
        scope='S15 T4 independent fresh/restored pairs; one-repeat queues are preflights, never full acceptance')
    def save():atomic_json(out/'progress.json',record)
    save()
    try:
        for repeat in range(a.repeats):
            for fork in a.forks:
                for path,digest in hashes.items():
                    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                        raise RuntimeError('Validation input changed; refuse to mix source versions: '+path)
                while True:
                    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                                  '--format=csv,noheader,nounits'],text=True).splitlines()
                    busy=[r for r in rows if r.split(',')[0].strip()==GPU]
                    if not busy:break
                    record['waiting_for_gpu']=busy;save();time.sleep(10)
                record.pop('waiting_for_gpu',None)
                name=f'{a.name}-{fork.replace("_","-")}-{repeat:02d}'
                log=ROOT/'logs'/(name+'.log')
                command=[sys.executable,'-X','utf8','-B',str(Path(__file__).with_name('validate_snapshot_distribution.py')),
                    '--name',name,'--library',a.library,'--fork',fork,'--repeat',str(repeat),'--capture']
                item=dict(name=name,fork=fork,repeat=repeat,command=command,started_unix=time.time(),passed=False)
                record['cases'].append(item);save()
                with log.open('x') as stream:
                    worker=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,cwd=ROOT)
                    item['worker_pid']=worker.pid;save()
                    try:code=worker.wait(timeout=900)
                    except subprocess.TimeoutExpired:
                        # Only this queue's recorded child can be terminated.
                        worker.kill();worker.wait();item['timeout']=True;code=-9
                item.update(returncode=code,finished_unix=time.time())
                result_path=ROOT/'outputs'/name/'result.json'
                if result_path.exists():
                    result=json.loads(result_path.read_text())
                    item['passed']=bool(code==0 and result.get('passed'))
                    item['result_sha256']=hashlib.sha256(result_path.read_bytes()).hexdigest()
                    item['error']=result.get('error')
                save();print(json.dumps(item),flush=True)
                if not item['passed']:
                    record['status']='failed';save();return 2
        summaries={}
        if a.repeats==20:
            from validate_snapshot_distribution import summarize
            for fork in a.forks:
                paths=[ROOT/'outputs'/r['name']/'result.json' for r in record['cases'] if r['fork']==fork]
                summaries[fork]=summarize(paths)
            record['passed']=all(s['passed'] for s in summaries.values())
        record['summaries']=summaries
        record['preflight_passed']=all(r['passed'] for r in record['cases'])
        record['status']='completed';save()
        return 0 if record['preflight_passed'] and (a.repeats<20 or record['passed']) else 2
    except BaseException as error:
        record.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':raise SystemExit(main())
