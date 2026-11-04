import hashlib,json,os,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parents[1];previous=json.loads((r/'post-profile-plan.json').read_text())['jobs']
while True:
 p=r/'results/post-profile-status.json';rows=json.loads(p.read_text()) if p.exists() else []
 if len(rows)==len(previous):break
 if rows:
  result=r/'results'/rows[-1]['name']/'result.json'
  error=json.loads(result.read_text()).get('error','') if result.exists() else ''
  if any(s in error for s in ['Foreign GPU','not idle','Resource guard','Resource ceiling']):raise SystemExit('Prior resource guard stopped; no new GPU work')
 time.sleep(5)
check=json.loads((r/'results/alignment-v3/result.json').read_text())
assert check['passed'] and check['timings']['gpu_alignment']['mean_ms']<check['timings']['cpu_alignment']['mean_ms']
plan=json.loads((r/'v3-followup-plan.json').read_text());status=[]
for job in plan['jobs']:
 for p,h in plan['inputs_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 start=time.time();print('START',job['name'],flush=True)
 env=os.environ.copy();env.update(TMPDIR=str(r/'tmp'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 with (r/'logs'/(job['name']+'.txt')).open('w') as log:proc=subprocess.run(job['argv'],env=env,stdout=log,stderr=subprocess.STDOUT)
 status.append(dict(name=job['name'],returncode=proc.returncode,start_unix_s=start,end_unix_s=time.time()));(r/'results/v3-followup-status.json').write_text(json.dumps(status,indent=2));print('END',status[-1],flush=True)
 result=r/'results'/job['name']/'result.json'
 if result.exists():
  error=json.loads(result.read_text()).get('error','')
  if any(s in error for s in ['Foreign GPU','not idle','Resource guard','Resource ceiling']):break
