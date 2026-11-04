import json,time,subprocess,hashlib
from pathlib import Path
p=Path(__file__).resolve().parents[1];plan=json.loads((p/'first-plan.json').read_text());statuses=[]
while '[100%] Built target' not in (p/'logs/build-specialized-validation.txt').read_text():time.sleep(5)
for job in plan['jobs']:
 for name,sha in plan['inputs_sha256'].items():
  if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Frozen input changed: '+name)
 if job['name'] in ('r1-specialized','native-special-profile'):
  q=p/'results/native-query-validation/result.json'
  if not q.exists() or not json.loads(q.read_text()).get('native_validation',{}).get('passed'):
   statuses.append(dict(name=job['name'],skipped=True,reason='native equivalence was not passed'));(p/'results/first-status.json').write_text(json.dumps(statuses,indent=2));continue
 print('START',job['name'],flush=True);start=time.time()
 with (p/'logs'/str(job['name']+'.txt')).open('w') as stream:r=subprocess.run(job['argv'],stdout=stream,stderr=subprocess.STDOUT)
 row=dict(name=job['name'],returncode=r.returncode,start_unix_s=start,end_unix_s=time.time());statuses.append(row);(p/'results/first-status.json').write_text(json.dumps(statuses,indent=2));print('END',row,flush=True)
 q=p/'results'/job['name']/'result.json'
 if q.exists() and any(x in json.loads(q.read_text()).get('error','') for x in ['Foreign GPU','not idle','Resource guard','Resource ceiling']):break
