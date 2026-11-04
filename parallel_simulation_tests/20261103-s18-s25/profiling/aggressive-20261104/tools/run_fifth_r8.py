import json,time,subprocess,hashlib
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';source=p/'candidate-r8-source';h=p/'tools/profile_r7.py';statuses=[]
while True:
 q=p/'results/fourth-resume-selection.json'
 if q.exists():prior=json.loads(q.read_text());break
 log=(p/'logs/fourth-resume-queue.txt').read_text() if (p/'logs/fourth-resume-queue.txt').exists() else ''
 if any(x in log for x in ['GPU topology equivalence failed','GPU diagnostics build failed','CPU compatibility regression failed','Current-frame topology validation incomplete']):
  prior=json.loads((p/'results/third-selection.json').read_text());break
 if 'Resource guard' in log or 'Foreign work' in log:raise SystemExit('Resource guard stopped predecessor')
 time.sleep(5)
dll=Path(prior['dll']);flags=['--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh']
for flag in ['--persistent-soup','--gpu-topology']:
 if flag in prior['flags']:flags.append(flag)
(p/'fifth-r8-inputs.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in list(source.rglob('*.py'))+[dll,h]},indent=2))
def run(name,args):
 print('START',name,flush=True);start=time.time()
 with (p/'fifth-r8-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 with (p/'logs'/str(name+'.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,returncode=r.returncode,start_unix_s=start,end_unix_s=time.time()));(p/'results/fifth-r8-status.json').write_text(json.dumps(statuses,indent=2));print('END',statuses[-1],flush=True)
 q=p/'results'/name/'result.json';d=json.loads(q.read_text()) if q.exists() else {}
 if any(x in d.get('error','') for x in ['Foreign GPU','not idle','Resource guard','Resource ceiling']):raise SystemExit('GPU resource guard')
 return d,r.returncode
_,code=run('cpu-regression-r7',[py,'-m','unittest','discover','-s',str(source/'endoscopic_tissue/training'),'-t',str(source)])
if code:raise SystemExit('R7 CPU compatibility failed')
image,code=run('render-direct-rgb-r8',[py,str(p/'tools/check_direct_rgb_r8.py'),'--output',str(p/'results/render-direct-rgb-r8')])
if code or not image.get('passed'):raise SystemExit('Direct RGB image regression failed')
def frame(name,f):return run(name,[py,str(h),'--source',str(source),'--dll',str(dll),'--output',str(p/'results'/name),'--cpus','7,19,20,25']+f)[0]
base=frame('r8-control',flags);best=base;best_name='r8-control';best_flags=list(flags)
for name,f in [('r8-direct-rgb',flags+['--direct-rgb']),('r8-direct-rgb-cv1',flags+['--direct-rgb','--opencv-threads','1']),('r8-parallel-rgb',flags+['--direct-rgb','--opencv-threads','1','--parallel-frame','--render-nogil'])]:
 d=frame(name,f)
 if d.get('completed') and d.get('quality_passed') and d['performance']['fps']>best.get('performance',{}).get('fps',0):best=d;best_name=name;best_flags=f
repeat=frame('r8-best-repeat',best_flags)
check=[py,str(p/'tools/check_current_frame_r7.py'),'--source',str(source),'--dll',str(dll),'--output',str(p/'results/current-frame-r8')]
for flag in ['--gpu-topology','--parallel-frame','--render-nogil','--persistent-soup','--direct-rgb']:
 if flag in best_flags:check.append(flag)
reg,_=run('current-frame-r8',check)
name='r8-final-profile';args=[py,str(h),'--source',str(source),'--dll',str(dll),'--output',str(p/'results'/name),'--cpus','7,19,20,25']+best_flags+['--profile','--start','470','--count','12','--max-frames','482']
run(name,['/opt/nvidia/nsight-systems/2025.3.2/bin/nsys','profile','--trace=cuda,nvtx,opengl','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','--output',str(p/'results'/str(name+'-trace'))]+args)
result=dict(completed=True,selected_run=best_name,source=str(source),dll=str(dll),flags=best_flags,performance=best.get('performance'),repeat_performance=repeat.get('performance'),current_frame_regression_passed=reg.get('passed',False),stable_20hz_passed=bool(best.get('completed') and best.get('quality_passed') and repeat.get('completed') and repeat.get('quality_passed') and reg.get('passed') and best['performance']['fps']>=20 and repeat['performance']['fps']>=20 and max(best['performance']['p95_ms'],repeat['performance']['p95_ms'])<=50),formal_physics_domain_approved=False)
(p/'results/fifth-r8-selection.json').write_text(json.dumps(result,indent=2));print('DONE',json.dumps(result),flush=True)
