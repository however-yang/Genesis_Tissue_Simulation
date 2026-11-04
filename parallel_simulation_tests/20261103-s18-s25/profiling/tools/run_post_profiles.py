import json,os,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parents[1]
# The unprofiled comparison owns the GPU first. Never overlap benchmark processes.
while True:
 p=r/'results/comparison-v2-retry-status.json'
 rows=json.loads(p.read_text()) if p.exists() else []
 if rows and rows[-1]['returncode']:
  raise SystemExit('Preceding comparison stopped; inspect before resuming GPU work')
 if len(rows)==7:break
 time.sleep(5)
status=[]
for job in json.loads((r/'post-profile-plan.json').read_text())['jobs']:
 print('START',job['name'],flush=True);env=os.environ.copy();env.update(job['env']);env.update(TMPDIR=str(r/'tmp'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 start=time.time()
 with (r/'logs'/(job['name']+'.txt')).open('w') as log:
  proc=subprocess.run(job['argv'],env=env,stdout=log,stderr=subprocess.STDOUT)
 status.append(dict(name=job['name'],returncode=proc.returncode,start_unix_s=start,end_unix_s=time.time()))
 (r/'results/post-profile-status.json').write_text(json.dumps(status,indent=2));print('END',status[-1],flush=True)
 # Expected scientific failures remain in their result files. Resource failures stop all new GPU jobs.
 result=r/'results'/job['name']/'result.json'
 if result.exists():
  error=json.loads(result.read_text()).get('error','')
  if any(x in error for x in ['Foreign GPU','not idle','Resource guard','Resource ceiling']):break
