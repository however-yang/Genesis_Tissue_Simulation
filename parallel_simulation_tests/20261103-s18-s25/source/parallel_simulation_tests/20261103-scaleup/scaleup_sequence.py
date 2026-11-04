"""Run the authorized 64-environment capacity probe only after 32 finishes/cleans up."""
import json
import subprocess
import time
from pathlib import Path
from benchmark import ROOT,PYTHON,GPU,gpu_sample

def main():
    deadline=time.monotonic()+4200
    while not (ROOT/'mps/m32a/lifecycle.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('32-worker batch did not finish')
        time.sleep(2)
    prior=json.loads((ROOT/'runs/m32a/summary.json').read_text())
    if not prior['complete']:raise RuntimeError('Inspect 32-worker failure before increasing capacity')
    cpus=','.join(map(str,json.loads((ROOT/'cpu-selection.json').read_text())['selected']))
    log=[]
    command=[PYTHON,'-B','-u',str(ROOT/'mps_benchmark.py'),'--name','m64a','--workers','64',
        '--cpus',cpus,'--attempt-capacity','--max-gpu-mib','80000','--stop-after','settle','--timeout','600']
    started=time.time();print('START actual 64-worker capacity probe',flush=True)
    code=subprocess.call(command,cwd=ROOT)
    log.append({'name':'m64a','command':command,'exit_code':code,'started':started,'finished':time.time()})
    (ROOT/'scaleup-sequence.json').write_text(json.dumps(log,indent=2))
    sample=gpu_sample(GPU);(ROOT/'after-64-gpu.json').write_text(json.dumps(sample,indent=2))
    if sample['processes'] or sample['used_mib']>512:raise RuntimeError('GPU not clear after capacity attempt; inspect before further tests')
    # A clean one-environment reset verifies the failed probe left the device usable.
    command=[PYTHON,'-B','-u',str(ROOT/'benchmark.py'),'--name','post64-reset','--workers','1','--episodes','2','--cpus',cpus,'--stop-after','settle','--timeout','300']
    started=time.time();code=subprocess.call(command,cwd=ROOT)
    log.append({'name':'post64-reset','command':command,'exit_code':code,'started':started,'finished':time.time()})
    (ROOT/'scaleup-sequence.json').write_text(json.dumps(log,indent=2))
    if code:raise RuntimeError('Post-capacity device/environment recovery check failed')
    print('Scaleup capacity attempts and recovery check finished',flush=True)
if __name__=='__main__':main()
