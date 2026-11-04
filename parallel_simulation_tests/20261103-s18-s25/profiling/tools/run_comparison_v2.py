import hashlib,json,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parents[1];plan=json.loads((r/'comparison-v2-plan.json').read_text());status=[]
for job in plan['jobs']:
 for path,digest in plan['inputs_sha256'].items():
  assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
 start=time.time()
 print('START',job['name'],flush=True)
 with (r/'logs'/(job['name']+'.txt')).open('w') as log:
  completed=subprocess.run(job['argv'],stdout=log,stderr=subprocess.STDOUT)
 status.append(dict(name=job['name'],returncode=completed.returncode,start_unix_s=start,end_unix_s=time.time()))
 (r/'results/comparison-v2-status.json').write_text(json.dumps(status,indent=2));print('END',status[-1],flush=True)
 if completed.returncode:break
