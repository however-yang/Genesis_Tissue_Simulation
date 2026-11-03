"""Sequential, guarded follow-up tests after mps-16-c16 (one GPU batch at a time)."""
import json
from pathlib import Path
import subprocess
import time
from benchmark import ROOT,PYTHON

def main():
    deadline=time.monotonic()+900
    while not (ROOT/'mps/mps-16-c16/lifecycle.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Previous MPS batch cleanup not recorded')
        time.sleep(2)
    s=json.loads((ROOT/'runs/mps-16-c16/summary.json').read_text())
    if not s['complete'] or s['passed_full_episodes']!=16:raise RuntimeError('Inspect previous batch before further GPU tests')
    plans=[
        ('serial-repeat',[PYTHON,'-B','-u','benchmark.py','--name','plain-1-repeat','--workers','1','--episodes','2','--cpus','24,4,25,22,8,35,31,14','--timeout','900'],0),
        ('profile',[PYTHON,'-B','-u','benchmark.py','--name','profile-1-r0','--workers','1','--profile','--stop-after','incise_1','--cpus','24,4,25,22,8,35,31,14','--timeout','600'],0),
        ('pool-integration',[PYTHON,'-B','-u','mps_benchmark.py','--name','mpspool','--workers','16','--pool-test','--timeout','600'],0),
        ('capacity-32',[PYTHON,'-B','-u','benchmark.py','--name','capacity-32','--workers','32','--stop-after','settle','--cpus','24,4,25,22,8,35,31,14'],1),
        ('capacity-64',[PYTHON,'-B','-u','benchmark.py','--name','capacity-64','--workers','64','--stop-after','settle','--cpus','24,4,25,22,8,35,31,14'],1)]
    log=[]
    for name,cmd,expected in plans:
        print('START',name,flush=True);started=time.time()
        code=subprocess.call(cmd,cwd=ROOT)
        log.append(dict(name=name,command=cmd,started=started,finished=time.time(),exit_code=code,expected_exit_code=expected))
        (ROOT/'followups.json').write_text(json.dumps(log,indent=2))
        if code!=expected:raise RuntimeError('Unexpected follow-up outcome: '+name)
        if name.startswith('capacity'):
            status=json.loads((ROOT/'runs'/name/'status.json').read_text())
            assert status['phase']=='rejected_preflight' and status['no_workers_launched']
        print('FINISH',name,flush=True)
if __name__=='__main__':main()
