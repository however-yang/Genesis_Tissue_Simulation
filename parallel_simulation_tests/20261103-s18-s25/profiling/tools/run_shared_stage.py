import json,os,subprocess,time,sys
from pathlib import Path
r=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';out=r/'results/shared-memory-stage';out.mkdir(exist_ok=False)
result=dict(completed=False,passed=False,jobs=[])
args=[py,'-m','unittest','discover','-s',str(r/'candidate-v5-source/endoscopic_tissue/training'),'-t',str(r/'candidate-v5-source')]
with (r/'logs/cpu-regression-v5.txt').open('w') as log:proc=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
result['cpu_returncode']=proc.returncode
if proc.returncode:
 (out/'result.json').write_text(json.dumps(result,indent=2));raise SystemExit(2)
assert json.loads((r/'results/shared-images-regression.json').read_text())['passed']
for n in (1,2):
 name='pool-shared-'+str(n);args=[py,str(r/'tools/pool_benchmark_shared.py'),'--output',str(r/'results'/name),'--workers',str(n),'--vision','--optimized','--gpu-aligned-depth','--shared-images']
 env=os.environ.copy();env['PROFILE_POOL_SOURCE']=str(r/'candidate-v5-source');start=time.time();print('START',name,flush=True)
 with (r/'logs'/(name+'.txt')).open('w') as log:proc=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT)
 result['jobs'].append(dict(name=name,returncode=proc.returncode,start_unix_s=start,end_unix_s=time.time()));(out/'result.json').write_text(json.dumps(result,indent=2));print('END',result['jobs'][-1],flush=True)
 p=r/'results'/name/'result.json'
 if p.exists() and any(s in json.loads(p.read_text()).get('error','') for s in ['Foreign GPU','not idle','Resource guard','Resource ceiling']):break
result.update(completed=len(result['jobs'])==2,passed=len(result['jobs'])==2 and all(j['returncode']==0 for j in result['jobs']));(out/'result.json').write_text(json.dumps(result,indent=2))
if not result['passed']:raise SystemExit(2)
