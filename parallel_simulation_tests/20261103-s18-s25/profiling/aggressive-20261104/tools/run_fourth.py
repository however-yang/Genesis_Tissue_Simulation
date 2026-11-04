import json,time,subprocess,hashlib
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';source=p/'candidate-r6-source';h=p/'tools/profile_r6.py';diag=p/'native-gpu-diagnostics/build/lib/libLiverCudaSim.so';combined=p/'native-all-optimized/build/lib/libLiverCudaSim.so';statuses=[]
while not (p/'results/third-selection.json').exists():time.sleep(5)
while True:
 q=p/'results/diagnostics-build-status.json'
 if q.exists():
  rows=json.loads(q.read_text())
  if any(x['returncode'] for x in rows):raise SystemExit('GPU diagnostics build failed; earlier candidates retained')
  if len(rows)==2:break
 time.sleep(5)
def run(name,args):
 print('START',name,flush=True);start=time.time()
 with (p/'fourth-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 with (p/'logs'/str(name+'.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,returncode=r.returncode,start_unix_s=start,end_unix_s=time.time()));(p/'results/fourth-status.json').write_text(json.dumps(statuses,indent=2));print('END',statuses[-1],flush=True)
 q=p/'results'/name/'result.json';d=json.loads(q.read_text()) if q.exists() else {}
 if any(x in d.get('error','') for x in ['Foreign GPU','not idle','Resource guard','Resource ceiling']):raise SystemExit('Resource guard stopped follow-up')
 return d,r.returncode
cpu,code=run('cpu-regression-r6',[py,'-m','unittest','discover','-s',str(source/'endoscopic_tissue/training'),'-t',str(source)])
if code:raise SystemExit('CPU compatibility regression failed')
(p/'fourth-inputs.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in list(source.rglob('*.py'))+[diag,combined,h]},indent=2))
test,code=run('gpu-topology-equivalence',[py,str(p/'tools/check_gpu_topology.py'),'--dll',str(diag),'--output',str(p/'results/gpu-topology-equivalence')])
if code or not test.get('passed'):raise SystemExit('GPU topology equivalence failed; no performance adoption')
prior=json.loads((p/'results/third-selection.json').read_text());flags=['--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh']
for flag in ['--persistent-soup','--parallel-frame','--render-nogil']:
 if flag in prior['flags']:flags.append(flag)
def frame(name,dll,extra):return run(name,[py,str(h),'--source',str(source),'--dll',str(dll),'--output',str(p/'results'/name),'--cpus','7,19,20,25']+flags+extra)[0]
validation=frame('r6-topology-validation',diag,['--gpu-topology','--topology-ab'])
if validation.get('error') or validation.get('topology_frames_compared',0)!=len(validation.get('frames',[])):raise SystemExit('Current-frame topology validation incomplete')
base=frame('r6-cpu-checks',diag,[]);gpu=frame('r6-gpu-checks',diag,['--gpu-topology']);best=base;best_dll=diag;best_name='r6-cpu-checks';best_flags=list(flags)
def better(d):return bool(d.get('completed') and d.get('quality_passed') and d.get('performance',{}).get('fps',0)>best.get('performance',{}).get('fps',0))
if better(gpu):best=gpu;best_flags=flags+['--gpu-topology'];best_name='r6-gpu-checks'
# The combined native implementation must pass the old-query test again after linking diagnostics.
q=json.loads((p/'results/query-cutting-optimized/result.json').read_text())
if q.get('native_validation',{}).get('passed'):
 v,_=run('query-all-optimized',[py,str(h),'--source',str(source),'--dll',str(combined),'--output',str(p/'results/query-all-optimized'),'--cpus','7,19,20,25','--no-render','--gpu-topology','--topology-ab','--validate-native','--start','480','--count','8','--max-frames','488'])
 if v.get('native_validation',{}).get('passed') and v.get('quality_passed'):
  for name in ['r6-all-optimized','r6-all-repeat']:
   d=frame(name,combined,['--gpu-topology'])
   if better(d):best=d;best_dll=combined;best_name=name;best_flags=flags+['--gpu-topology']
check=[py,str(p/'tools/check_current_frame_r6.py'),'--source',str(source),'--dll',str(best_dll),'--output',str(p/'results/current-frame-r6')]
for flag in ['--gpu-topology','--parallel-frame','--render-nogil','--persistent-soup']:
 if flag in best_flags:check.append(flag)
regression,_=run('current-frame-r6',check)
frame('r6-unchanged-control',diag,[])
name='r6-final-profile';args=[py,str(h),'--source',str(source),'--dll',str(best_dll),'--output',str(p/'results'/name),'--cpus','7,19,20,25']+best_flags+['--profile','--start','470','--count','12','--max-frames','482']
run(name,['/opt/nvidia/nsight-systems/2025.3.2/bin/nsys','profile','--trace=cuda,nvtx,opengl','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','--output',str(p/'results'/str(name+'-trace'))]+args)
result=dict(completed=True,source=str(source),dll=str(best_dll),flags=best_flags,selected_run=best_name,performance=best.get('performance'),current_frame_regression_passed=regression.get('passed',False),stable_20hz_passed=bool(best.get('completed') and best.get('quality_passed') and regression.get('passed') and best['performance']['fps']>=20 and best['performance']['p95_ms']<=50),formal_physics_domain_approved=False)
(p/'results/fourth-selection.json').write_text(json.dumps(result,indent=2));print('DONE',json.dumps(result),flush=True)
