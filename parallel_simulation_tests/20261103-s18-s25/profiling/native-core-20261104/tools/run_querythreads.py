"""Full RGB-D sweep of grid-stride launch settings; no work or accuracy removed."""
import json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';h=p/'tools/profile_launch.py';manifest=json.loads((p/'manifest.json').read_text());source=manifest['baseline_source'];dll=p/'native-launch/build/lib/libLiverCudaSim.so';statuses=[]
assert json.loads((p/'results/original-device-binding.json').read_text())['all_original_functions_unchanged']
assert json.loads((p/'results/workload-original-grid/result.json').read_text())['completed']
assert json.loads((p/'results/batch-captured-validation/result.json').read_text())['native_validation']['passed']
flags=['--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh','--persistent-soup','--direct-rgb','--opencv-threads','1','--gpu-topology']
def run(name,q,a,m,t):
 args=[py,str(h),'--source',source,'--dll',str(dll),'--output',str(p/'results'/name),'--cpus','7,19,20,25','--query-mode',str(q),'--assembly-blocks',str(a),'--manifold-blocks',str(m),'--query-threads',str(t)]+flags
 print('START',name,flush=True);start=time.time()
 with (p/'querythreads-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 with (p/'logs'/str(name+'.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,returncode=r.returncode,start_unix_s=start,end_unix_s=time.time()));(p/'results/querythreads-status.json').write_text(json.dumps(statuses,indent=2));print('END',statuses[-1],flush=True)
 file=p/'results'/name/'result.json';d=json.loads(file.read_text()) if file.exists() else {}
 if any(x in d.get('error','') for x in ['Foreign GPU','not idle','Resource ceiling','Resource guard']):raise SystemExit('Resource guard stopped sweep')
 return d
runs=[]
for t in [64,32,128]:
 name='querythreads-'+str(t)
 d=run(name,1,128,1024,t)
 runs.append(dict(name=name,threads=t,completed=d.get('completed'),quality_passed=d.get('quality_passed'),error=d.get('error'),performance=d.get('performance')))
 (p/'results/querythreads-progress.json').write_text(json.dumps(runs,indent=2))
(p/'results/querythreads-selection.json').write_text(json.dumps(dict(runs=runs),indent=2))
