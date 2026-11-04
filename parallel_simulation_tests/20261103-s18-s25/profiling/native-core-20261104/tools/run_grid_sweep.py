"""Full RGB-D sweep of grid-stride launch settings; no work or accuracy removed."""
import json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';h=p/'tools/profile_scheduled.py';manifest=json.loads((p/'manifest.json').read_text());source=manifest['baseline_source'];dll=p/'native-scheduled/build/lib/libLiverCudaSim.so';statuses=[]
assert json.loads((p/'results/original-device-binding.json').read_text())['all_original_functions_unchanged']
assert json.loads((p/'results/workload-original-grid/result.json').read_text())['completed']
assert json.loads((p/'results/batch-captured-validation/result.json').read_text())['native_validation']['passed']
flags=['--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh','--persistent-soup','--direct-rgb','--opencv-threads','1','--gpu-topology']
def run(name,q,a,m):
 args=[py,str(h),'--source',source,'--dll',str(dll),'--output',str(p/'results'/name),'--cpus','7,19,20,25','--query-mode',str(q),'--assembly-blocks',str(a),'--manifold-blocks',str(m)]+flags
 print('START',name,flush=True);start=time.time()
 with (p/'grid-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 with (p/'logs'/str(name+'.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,returncode=r.returncode,start_unix_s=start,end_unix_s=time.time()));(p/'results/grid-status.json').write_text(json.dumps(statuses,indent=2));print('END',statuses[-1],flush=True)
 file=p/'results'/name/'result.json';d=json.loads(file.read_text()) if file.exists() else {}
 if any(x in d.get('error','') for x in ['Foreign GPU','not idle','Resource ceiling','Resource guard']):raise SystemExit('Resource guard stopped sweep')
 return d
runs=[]
for name,q,a,m in [('control-same-binary',0,512,1024),('grid-assembly128',1,128,1024),('grid-manifold128',1,512,128),('grid-both128',1,128,128),('grid-both64',1,64,64),('grid-both256',1,256,256)]:
 d=run(name,q,a,m);runs.append(dict(name=name,query_mode=q,assembly_blocks=a,manifold_blocks=m,completed=d.get('completed'),quality_passed=d.get('quality_passed'),error=d.get('error'),performance=d.get('performance')))
 valid=[x for x in runs if x['completed'] and x['quality_passed']]
 (p/'results/grid-progress.json').write_text(json.dumps(runs,indent=2))
base=runs[0];valid=[x for x in runs[1:] if x['completed'] and x['quality_passed'] and x['performance']['mean_ms']<base['performance']['mean_ms'] and x['performance']['p95_ms']<base['performance']['p95_ms']]
best=min(valid,key=lambda x:(x['performance']['p95_ms'],x['performance']['mean_ms'])) if valid else base
result=dict(completed=True,source=source,dll=str(dll),flags=flags,runs=runs,selected=best,formal_physics_domain_approved=False,production_replaced=False)
(p/'results/grid-selection.json').write_text(json.dumps(result,indent=2));print('DONE',json.dumps(result),flush=True)
