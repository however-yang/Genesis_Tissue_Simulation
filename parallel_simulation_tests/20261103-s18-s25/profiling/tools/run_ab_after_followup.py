import json,time,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parents[1]
while True:
 q=p/'results/v3-followup-status.json'
 if q.exists() and len(json.loads(q.read_text()))==7:break
 time.sleep(5)
with (p/'logs/invariants-ab.txt').open('w') as f:
 r=subprocess.run([sys.executable,str(p/'tools/probe_invariants_ab.py'),'--output',str(p/'results/invariants-ab'),'--gpu','GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa','--preparation','cut_ready','--cut','--randomized'],stdout=f,stderr=subprocess.STDOUT)
(p/'results/invariants-ab-exit.json').write_text(json.dumps(dict(returncode=r.returncode)))
print('same-state label A/B finished',r.returncode,flush=True)
