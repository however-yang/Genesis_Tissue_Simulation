import json,time,subprocess,hashlib
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';old=p.parent.parent/'native-s25/build/lib/libLiverCudaSim.so';native=p/'native-cutting-optimized/build/lib/libLiverCudaSim.so';statuses=[]
while not (p/'results/second-completed.json').exists():time.sleep(5)
while '[100%] Built target' not in (p/'logs/build-cutting-optimized.txt').read_text():time.sleep(5)
(p/'third-inputs.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in list((p/'candidate-r5-source').rglob('*.py'))+[native,p/'tools/profile_r5.py']},indent=2))
def run(name,args):
 print('START',name,flush=True);start=time.time()
 with (p/'third-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 with (p/'logs'/str(name+'.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,returncode=r.returncode,start_unix_s=start,end_unix_s=time.time()));(p/'results/third-status.json').write_text(json.dumps(statuses,indent=2));print('END',statuses[-1],flush=True)
 q=p/'results'/name/'result.json';d=json.loads(q.read_text()) if q.exists() else {}
 if any(x in d.get('error','') for x in ['Foreign GPU','not idle','Resource guard','Resource ceiling']):raise SystemExit('Foreign work/resource guard')
 return d

def full(name,source,h,dll,flags,extra=()):return run(name,[py,str(h),'--source',str(source),'--dll',str(dll),'--output',str(p/'results'/name),'--cpus','7,19,20,25']+list(flags)+list(extra))
static=run('render-static-visuals',[py,str(p/'tools/check_static_visuals.py'),'--output',str(p/'results/render-static-visuals')])
soup=run('render-persistent-soup',[py,str(p/'tools/check_persistent_soup.py'),'--output',str(p/'results/render-persistent-soup')])
validation=full('query-cutting-optimized',p/'candidate-r3-source',p/'tools/profile_r3.py',native,[],['--no-render','--validate-native','--start','480','--count','40','--max-frames','520'])
flags=['--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh'];source=p/'candidate-r5-source';h=p/'tools/profile_r5.py'
if soup.get('passed'):flags+=['--persistent-soup']
if static.get('passed'):flags+=['--static-visuals']
base=full('r4-serial',source,h,old,flags);best=base;best_flags=list(flags);best_dll=old;best_name='r4-serial'
if json.loads((p/'results/render-nogil/result.json').read_text()).get('passed'):
 trial=flags+['--parallel-frame','--render-nogil'];d=full('r4-parallel',source,h,old,trial)
 if d.get('completed') and d.get('quality_passed') and d['performance']['fps']>best.get('performance',{}).get('fps',0):best=d;best_flags=trial;best_name='r4-parallel'
if validation.get('native_validation',{}).get('passed') and validation.get('quality_passed'):
 for name in ['r4-cutting-optimized','r4-cutting-repeat']:
  d=full(name,source,h,native,best_flags)
  if d.get('completed') and d.get('quality_passed') and d['performance']['fps']>best.get('performance',{}).get('fps',0):best=d;best_dll=native;best_name=name
 # Actual node timings, not just total wall time.
 name='profile-cutting-optimized';args=[py,str(h),'--source',str(source),'--dll',str(native),'--output',str(p/'results'/name),'--cpus','7,19,20,25','--no-render','--profile','--start','470','--count','12','--max-frames','482']
 run(name,['/opt/nvidia/nsight-systems/2025.3.2/bin/nsys','profile','--trace=cuda,nvtx,opengl','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','--output',str(p/'results'/str(name+'-trace'))]+args)
# Same-state output and retained-frame ownership tests must use the selected actual paths.
check=[py,str(p/'tools/check_current_frame.py'),'--source',str(source),'--dll',str(best_dll),'--output',str(p/'results/current-frame-regression')]
for flag in ['--parallel-frame','--render-nogil','--persistent-soup','--static-visuals']:
 if flag in best_flags:check.append(flag)
regression=run('current-frame-regression',check)
full('r4-unchanged-control',source,h,old,best_flags)
name='render-final-profile';args=[py,str(h),'--source',str(source),'--dll',str(best_dll),'--output',str(p/'results'/name),'--cpus','7,19,20,25']+best_flags+['--profile','--start','470','--count','12','--max-frames','482']
run(name,['/opt/nvidia/nsight-systems/2025.3.2/bin/nsys','profile','--trace=cuda,nvtx,opengl','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','--output',str(p/'results'/str(name+'-trace'))]+args)
result=dict(completed=True,selected_run=best_name,flags=best_flags,dll=str(best_dll),source=str(source),performance=best.get('performance'),current_frame_regression_passed=regression.get('passed',False),stable_20hz_passed=bool(best.get('completed') and best.get('quality_passed') and regression.get('passed') and best.get('performance',{}).get('fps',0)>=20 and best.get('performance',{}).get('p95_ms',1e9)<=50),formal_physics_domain_approved=False)
(p/'results/third-selection.json').write_text(json.dumps(result,indent=2));print('DONE',json.dumps(result),flush=True)
