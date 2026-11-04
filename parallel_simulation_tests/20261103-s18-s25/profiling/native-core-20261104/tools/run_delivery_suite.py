"""Quiet final comparison after all compilation has finished."""
import json,subprocess,time,hashlib,statistics
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';m=json.loads((p/'manifest.json').read_text());status=[]
def read(name):return json.loads((p/'results'/name/'result.json').read_text())
cache_file=p/'results/bound-cache-selection.json';cache=json.loads(cache_file.read_text()) if cache_file.exists() else {};use_cache=False
if cache.get('first_pair_benefit') and 'candidate_b' in cache:
 names=['bound-cache-control-a','bound-cache-control-b','bound-cache-candidate-a','bound-cache-candidate-b'];runs=[read(n) for n in names]
 if all(d['completed'] and d['quality_passed'] for d in runs):
  means=[statistics.mean(d['performance']['mean_ms'] for d in runs[a:b]) for a,b in [(0,2),(2,4)]];tails=[statistics.mean(d['performance']['p95_ms'] for d in runs[a:b]) for a,b in [(0,2),(2,4)]];use_cache=means[1]<means[0] and tails[1]<tails[0]
choice=json.loads((p/'results/conservative-candidate.json').read_text());choice['execution']['bound_cache']=use_cache
if use_cache:choice.update(native_source=str(p/'native-bound-cache/source'),dll=str(p/'native-bound-cache/build/lib/libLiverCudaSim.so'),reason='Dispatch batching plus exact adjacent-stage coarse-bound coefficient reuse')
choice['dll_sha256']=hashlib.sha256(Path(choice['dll']).read_bytes()).hexdigest();choice['harness']='profile_bound_cache.py' if use_cache else 'profile_fallback.py';choice['qualification_pending']=True;(p/'results/recommended-candidate.json').write_text(json.dumps(choice,indent=2))
flags=['--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh','--persistent-soup','--direct-rgb','--opencv-threads','1','--gpu-topology']
def execflags(values):
 out=[]
 for k,v in values.items():
  if isinstance(v,bool):
   if v:out+=['--'+k.replace('_','-')]
  else:out+=['--'+k.replace('_','-'),str(v)]
 return out
def run(name,baseline=False,extra=None,harness=None,prefix=None):
 values=dict(choice['execution'])
 if baseline:values['query_mode']=0;values['bound_cache']=False
 if harness and harness.startswith('check_'):values.pop('compact_blocks',None)
 args=[py,str(p/'tools'/(harness or choice['harness'])),'--source',m['baseline_source'],'--dll',choice['dll'],'--output',str(p/'results'/name)]+execflags(values)+(extra or [])
 if not harness or harness.startswith('profile_'):args+=['--cpus','7,19,20,25']
 if prefix:args=prefix+args
 with (p/'delivery-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 print('START',name,flush=True);begin=time.time()
 with (p/'logs'/str(name+'.txt')).open('w') as f:ret=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 f=p/'results'/name/'result.json';d=json.loads(f.read_text()) if f.exists() else {};row=dict(name=name,returncode=ret.returncode,start_unix_s=begin,end_unix_s=time.time(),completed=d.get('completed'),quality_passed=d.get('quality_passed'),passed=d.get('passed'),error=d.get('error'),performance=d.get('performance'));status.append(row);(p/'results/delivery-suite-progress.json').write_text(json.dumps(status,indent=2));print('END',json.dumps(row),flush=True)
 if any(x in d.get('error','') for x in ['Foreign GPU','not idle','Resource ceiling','Resource guard']):raise SystemExit('Resource guard stopped delivery suite')
 return d
print('SELECTED',json.dumps(choice),flush=True)
for name,baseline in [('delivery-baseline-a',True),('delivery-candidate-a',False),('delivery-candidate-b',False),('delivery-baseline-b',True)]:run(name,baseline,flags)
for name,baseline in [('delivery-baseline-physics',True),('delivery-candidate-physics',False)]:run(name,baseline,['--no-render','--gpu-topology'])
suffix='_cache' if use_cache else ''
run('delivery-current-frame',extra=['--gpu-topology','--direct-rgb','--persistent-soup'],harness='check_current_frame_final'+suffix+'.py')
run('delivery-snapshot-roundtrip',extra=['--gpu-topology'],harness='check_snapshot_roundtrip'+suffix+'.py')
nsys='/opt/nvidia/nsight-systems/2025.3.2/bin/nsys'
for name,baseline in [('delivery-baseline-profile',True),('delivery-candidate-profile',False)]:
 trace=p/'results'/str(name+'-trace');prefix=[nsys,'profile','--trace=cuda,nvtx,opengl','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','--output',str(trace)]
 run(name,baseline,flags+['--profile','--start','470','--count','12','--max-frames','482'],prefix=prefix)
 with (p/'logs'/str(name+'-export.txt')).open('w') as f:ret=subprocess.run([nsys,'export','--type=sqlite','--force-overwrite=true','--output',str(trace)+'.sqlite',str(trace)+'.nsys-rep'],stdout=f,stderr=subprocess.STDOUT)
 print('EXPORT',name,ret.returncode,flush=True)
(p/'results/delivery-suite-status.json').write_text(json.dumps(dict(completed=True,use_bound_cache=use_cache,runs=status),indent=2));print('DELIVERY SUITE FINISHED',flush=True)
