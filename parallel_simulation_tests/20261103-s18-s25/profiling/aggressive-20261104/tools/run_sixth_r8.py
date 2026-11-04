"""Resolve the measured topology-stage gain vs end-to-end regression with ABBA.

Same source, binary, images, physics and checks; only checker location changes.
No GPU job starts before the preceding queue completes and resource guard passes.
"""
import json,time,subprocess,statistics
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';statuses=[]
while not (p/'results/fifth-r8-selection.json').exists():
 log=(p/'logs/fifth-r8-queue.txt').read_text() if (p/'logs/fifth-r8-queue.txt').exists() else ''
 if any(x in log for x in ['Resource guard','GPU resource guard','Resource ceiling','Foreign work','regression failed','compatibility failed']):raise SystemExit('Predecessor stopped; no automatic retry')
 time.sleep(5)
prior=json.loads((p/'results/fifth-r8-selection.json').read_text());source=Path(prior['source']);dll=p/'native-gpu-diagnostics/build/lib/libLiverCudaSim.so';flags=[x for x in prior['flags'] if x!='--gpu-topology'];h=p/'tools/profile_r7.py'
def run(name,args):
 print('START',name,flush=True);start=time.time()
 with (p/'sixth-r8-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 with (p/'logs'/str(name+'.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,returncode=r.returncode,start_unix_s=start,end_unix_s=time.time()));(p/'results/sixth-r8-status.json').write_text(json.dumps(statuses,indent=2));print('END',statuses[-1],flush=True)
 q=p/'results'/name/'result.json';d=json.loads(q.read_text()) if q.exists() else {}
 if r.returncode or d.get('error') or not d.get('completed',d.get('passed',False)) or not d.get('quality_passed',True):raise SystemExit('ABBA run failed or resource guard stopped: '+name)
 return d
runs=[]
for name,gpu in [('r8-topo-cpu-a',False),('r8-topo-gpu-a',True),('r8-topo-gpu-b',True),('r8-topo-cpu-b',False)]:
 d=run(name,[py,str(h),'--source',str(source),'--dll',str(dll),'--output',str(p/'results'/name),'--cpus','7,19,20,25']+flags+(['--gpu-topology'] if gpu else []));runs.append(dict(name=name,gpu=gpu,performance=d['performance']))
mean=lambda gpu,key:statistics.mean(x['performance'][key] for x in runs if x['gpu']==gpu)
keep=mean(True,'mean_ms')<=.95*mean(False,'mean_ms') and mean(True,'p95_ms')<=mean(False,'p95_ms')
reg={}
if keep:
 args=[py,str(p/'tools/check_current_frame_r7.py'),'--source',str(source),'--dll',str(dll),'--output',str(p/'results/current-frame-r8-gpu'),'--gpu-topology']
 for f in ['--parallel-frame','--render-nogil','--persistent-soup','--direct-rgb']:
  if f in flags:args.append(f)
 reg=run('current-frame-r8-gpu',args);keep=bool(reg.get('passed'))
result=dict(completed=True,source=str(source),dll=str(dll),flags=flags+(['--gpu-topology'] if keep else []),topology_gpu_kept=keep,selection_rule='At least 5% lower ABBA mean frame cost, no worse mean P95, all full checks and current-frame regression pass; 20Hz remains separately required',runs=runs,cpu_mean_ms=mean(False,'mean_ms'),gpu_mean_ms=mean(True,'mean_ms'),cpu_mean_p95_ms=mean(False,'p95_ms'),gpu_mean_p95_ms=mean(True,'p95_ms'),formal_physics_domain_approved=False)
(p/'results/sixth-r8-selection.json').write_text(json.dumps(result,indent=2));print('DONE',json.dumps(result),flush=True)
