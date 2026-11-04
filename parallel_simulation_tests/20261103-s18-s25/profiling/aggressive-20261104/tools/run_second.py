"""Sequential, adaptive tests; failed or unhelpful optimizations are not adopted."""
import json,time,subprocess,hashlib
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';h=p/'tools/profile_r3.py';old=p.parent.parent/'native-s25/build/lib/libLiverCudaSim.so';source=p/'candidate-r3-source';statuses=[]
while True:
 q=p/'results/first-status.json'
 if q.exists() and len(json.loads(q.read_text()))==9:break
 time.sleep(5)
for name in ['grasper-specialized','compacted']:
 while '[100%] Built target' not in (p/'logs'/str('build-'+name+'.txt')).read_text():time.sleep(5)
(p/'second-inputs.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in list(source.rglob('*.py'))+[p/'native-grasper-specialized/build/lib/libLiverCudaSim.so',p/'native-compacted/build/lib/libLiverCudaSim.so',h]},indent=2))
def run(name,args):
 start=time.time();print('START',name,flush=True)
 with (p/'second-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 with (p/'logs'/str(name+'.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,returncode=r.returncode,start_unix_s=start,end_unix_s=time.time()));(p/'results/second-status.json').write_text(json.dumps(statuses,indent=2));print('END',statuses[-1],flush=True)
 q=p/'results'/name/'result.json';data=json.loads(q.read_text()) if q.exists() else {}
 if any(s in data.get('error','') for s in ['Foreign GPU','not idle','Resource guard','Resource ceiling']):raise SystemExit('GPU resource guard stopped follow-up')
 return data

def frame(name,dll,flags,extra=()):return run(name,[py,str(h),'--source',str(source),'--dll',str(dll),'--output',str(p/'results'/name),'--cpus','7,19,20,25']+list(flags)+list(extra))
pbo=run('render-pbo',[py,str(p/'tools/check_pbo.py'),'--output',str(p/'results/render-pbo')])
nogil=run('render-nogil',[py,str(p/'tools/check_render_nogil.py'),'--output',str(p/'results/render-nogil')])
validations={}
for native in ['grasper-specialized','compacted']:
 dll=p/str('native-'+native)/'build/lib/libLiverCudaSim.so'
 data=frame('query-'+native,dll,[],['--no-render','--validate-native','--start','480','--count','40','--max-frames','520']);validations[native]=data.get('native_validation',{}).get('passed',False) and data.get('quality_passed',False)
flags=['--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh']
if nogil.get('passed'):flags+=['--render-nogil']
base=frame('r3-serial',old,flags);best=base;best_flags=list(flags);best_name='r3-serial'
def better(data,best):return bool(data.get('completed') and data.get('quality_passed') and data.get('performance',{}).get('fps',0)>best.get('performance',{}).get('fps',0))
if pbo.get('passed'):
 data=frame('r3-serial-pbo',old,flags+['--pbo'])
 if better(data,best):best=data;best_flags=flags+['--pbo'];best_name='r3-serial-pbo'
if nogil.get('passed'):
 trial_flags=list(best_flags)+['--parallel-frame'];data=frame('r3-parallel-nogil',old,trial_flags)
 if better(data,best):best=data;best_flags=trial_flags;best_name='r3-parallel-nogil'
# Always record why a mode is selected; mean alone does not certify a 20 Hz floor.
selected=dict(name=best_name,flags=best_flags,performance=best.get('performance'),stable_20hz_passed=bool(best.get('completed') and best.get('quality_passed') and best.get('performance',{}).get('p95_ms',1e9)<=50 and best.get('performance',{}).get('fps',0)>=20))
(p/'results/second-selection.json').write_text(json.dumps(selected,indent=2))
for native,passed in validations.items():
 if passed:frame('r3-'+native,p/str('native-'+native)/'build/lib/libLiverCudaSim.so',best_flags)
 else:print('SKIP native full benchmark, equivalence not passed:',native,flush=True)
# A final unchanged-DLL control brackets the native trials under shared CPU load.
frame('r3-control-repeat',old,best_flags)
# Actual CPU/GPU overlap, with no skipped pre-window renders.
if nogil.get('passed'):
 name='pipeline-profile';argv=[py,str(h),'--source',str(source),'--dll',str(old),'--output',str(p/'results'/name),'--cpus','7,19,20,25']+flags+['--parallel-frame','--profile','--start','470','--count','12','--max-frames','482']
 run(name,['/opt/nvidia/nsight-systems/2025.3.2/bin/nsys','profile','--trace=cuda,nvtx,opengl','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','--output',str(p/'results'/str(name+'-trace'))]+argv)
for native,passed in validations.items():
 if not passed:continue
 name='profile-'+native;argv=[py,str(h),'--source',str(source),'--dll',str(p/str('native-'+native)/'build/lib/libLiverCudaSim.so'),'--output',str(p/'results'/name),'--cpus','7,19,20,25','--no-render','--profile','--start','470','--count','12','--max-frames','482']
 run(name,['/opt/nvidia/nsight-systems/2025.3.2/bin/nsys','profile','--trace=cuda,nvtx,opengl','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','--output',str(p/'results'/str(name+'-trace'))]+argv)
(p/'results/second-completed.json').write_text(json.dumps(dict(completed=True,jobs=len(statuses)),indent=2))
