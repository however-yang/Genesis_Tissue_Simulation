"""Compare one uniform-computation hoist; reject on any strict query mismatch."""
import json,time,subprocess,statistics
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';h=p/'tools/profile_r7.py';statuses=[]
while not (p/'results/sixth-r8-v2-selection.json').exists():
 for name in ['sixth-v2-queue.txt','fifth-r8-queue.txt']:
  log=(p/'logs'/name).read_text() if (p/'logs'/name).exists() else ''
  if any(x in log for x in ['Resource guard','GPU resource guard','Resource ceiling','Foreign work','regression failed','compatibility failed','ABBA run failed','V2 validation failed']):raise SystemExit('Predecessor stopped; no automatic retry')
 time.sleep(5)
assert all(x['returncode']==0 for x in json.loads((p/'results/private-scratch-v2-build-status.json').read_text()))
prior=json.loads((p/'results/sixth-r8-v2-selection.json').read_text());source=Path(prior['source']);flags=prior['flags'];baseline=Path(prior['dll']);native=p/'native-precomputed-v2/build/lib/libLiverCudaSim.so'
def run(name,args,required=True):
 print('START',name,flush=True);start=time.time()
 with (p/'seventh-v2-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 with (p/'logs'/str(name+'.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,returncode=r.returncode,start_unix_s=start,end_unix_s=time.time()));(p/'results/seventh-v2-status.json').write_text(json.dumps(statuses,indent=2));print('END',statuses[-1],flush=True)
 q=p/'results'/name/'result.json';d=json.loads(q.read_text()) if q.exists() else {}
 if any(x in d.get('error','') for x in ['Foreign GPU','not idle','Resource guard','Resource ceiling']):raise SystemExit('GPU resource guard')
 if required and (r.returncode or d.get('error') or not d.get('completed') or not d.get('quality_passed')):raise SystemExit('Full run failed: '+name)
 return d
old=p.parents[1]/'native-s25/build/lib/libLiverCudaSim.so'
control=run('v4-idle-control-a',[py,str(h),'--source',str(p.parent/'candidate-v4-source'),'--dll',str(old),'--output',str(p/'results/v4-idle-control-a'),'--cpus','7,19,20,25','--vision','--gpu-postprocess','--gpu-aligned-depth'])
v=run('query-precomputed-v2',[py,str(h),'--source',str(source),'--dll',str(native),'--output',str(p/'results/query-precomputed-v2'),'--cpus','7,19,20,25','--no-render','--validate-native','--start','480','--count','40','--max-frames','520'],False)
keep=False;runs=[]
if v.get('native_validation',{}).get('passed') and v.get('quality_passed'):
 for name,modified in [('r8-native-v2-control-a',False),('r8-native-v2-precomputed-a',True),('r8-native-v2-precomputed-b',True),('r8-native-v2-control-b',False)]:
  d=run(name,[py,str(h),'--source',str(source),'--dll',str(native if modified else baseline),'--output',str(p/'results'/name),'--cpus','7,19,20,25']+flags);runs.append(dict(name=name,modified=modified,performance=d['performance']))
 mean=lambda variant,key:statistics.mean(x['performance'][key] for x in runs if x['modified']==variant)
 keep=mean(True,'mean_ms')<=.95*mean(False,'mean_ms') and mean(True,'p95_ms')<=mean(False,'p95_ms')
result=dict(completed=True,source=str(source),dll=str(native if keep else baseline),flags=flags,native_precomputed_kept=keep,native_validation=v.get('native_validation'),runs=runs,old_v4_control=control.get('performance'),formal_physics_domain_approved=False)
(p/'results/seventh-v2-selection.json').write_text(json.dumps(result,indent=2));print('DONE',json.dumps(result),flush=True)
