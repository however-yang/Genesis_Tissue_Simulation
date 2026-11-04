"""Final minimal candidate: independently test it, including same-time V4 ABBA.

Never replace production or treat failed physics-domain acceptance as approved.
"""
import json,time,subprocess,hashlib
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';h=p/'tools/profile_r7.py';statuses=[]
while not (p/'results/seventh-v2-selection.json').exists():
 for name in ['seventh-v2-queue.txt','sixth-v2-queue.txt','fifth-r8-queue.txt']:
  log=(p/'logs'/name).read_text() if (p/'logs'/name).exists() else ''
  if any(x in log for x in ['Resource guard','GPU resource guard','Resource ceiling','Foreign work','regression failed','compatibility failed','ABBA run failed','Full run failed','V2 validation failed']):raise SystemExit('Predecessor stopped; no automatic retry')
 time.sleep(5)
prior=json.loads((p/'results/seventh-v2-selection.json').read_text());flags=prior['flags'];source=p/'final-source';dll=Path(prior['dll']);old=p.parents[1]/'native-s25/build/lib/libLiverCudaSim.so'
assert '--parallel-frame' not in flags,'Final minimal serial source must not silently drop a selected pipeline'
if '--gpu-topology' not in flags and not prior['native_precomputed_kept']:dll=old

def run(name,args,kind='full'):
 print('START',name,flush=True);start=time.time()
 with (p/'final-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 with (p/'logs'/str(name+'.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,returncode=r.returncode,start_unix_s=start,end_unix_s=time.time()));(p/'results/final-status.json').write_text(json.dumps(statuses,indent=2));print('END',statuses[-1],flush=True)
 q=p/'results'/name/'result.json';d=json.loads(q.read_text()) if q.exists() else {}
 if r.returncode:raise SystemExit('Final validation failed: '+name)
 if kind=='full' and (not d.get('completed') or not d.get('quality_passed')):raise SystemExit('Final full scene or enabled checks failed: '+name)
 if kind=='regression' and not d.get('passed'):raise SystemExit('Final current-frame regression failed')
 return d
assembly=[py,str(p/'tools/assemble_final.py'),'--output',str(source)]
if '--direct-rgb' in flags:assembly.append('--rgb')
if '--gpu-topology' in flags:assembly.append('--gpu-topology')
run('assemble-final',assembly,'command')
run('cpu-regression-final',[py,'-m','unittest','discover','-s',str(source/'endoscopic_tissue/training'),'-t',str(source)],'command')
check=(p/'tools/check_current_frame_r7.py').read_text().replace('gpu_topology_checks=a.gpu_topology','**(dict(gpu_topology_checks=True) if a.gpu_topology else {})');(p/'tools/check_current_frame_final.py').write_text(check)
# Same OpenCV setting for both sides of final ABBA; source-only speedup is kept separate from host-load changes.
cv_flags=['--opencv-threads',flags[flags.index('--opencv-threads')+1]] if '--opencv-threads' in flags else []
base_flags=['--vision','--gpu-postprocess','--gpu-aligned-depth']+cv_flags
runs=[]
for name,modified in [('final-v4-control-a',False),('final-rgbd-a',True),('final-rgbd-b',True),('final-v4-control-b',False)]:
 d=run(name,[py,str(h),'--source',str(source if modified else p.parent/'candidate-v4-source'),'--dll',str(dll if modified else old),'--output',str(p/'results'/name),'--cpus','7,19,20,25']+(flags if modified else base_flags));runs.append(dict(name=name,modified=modified,performance=d['performance']))
physical_flags=[x for x in flags if x in ['--gpu-topology']]
checked=run('final-checked-physics',[py,str(h),'--source',str(source),'--dll',str(dll),'--output',str(p/'results/final-checked-physics'),'--cpus','7,19,20,25','--no-render']+physical_flags)
args=[py,str(p/'tools/check_current_frame_final.py'),'--source',str(source),'--dll',str(dll),'--output',str(p/'results/current-frame-final')]
for f in ['--gpu-topology','--persistent-soup','--direct-rgb']:
 if f in flags:args.append(f)
reg=run('current-frame-final',args,'regression')
# Verify physical parameters, action space and acceptance thresholds against the actual control context.
a=json.loads((p/'results/final-v4-control-a/context.json').read_text());b=json.loads((p/'results/final-rgbd-a/context.json').read_text());keys=['settings','canonical_settings','native_physics_parameters','scene_config','action_bounds','quality_limits','cutting_enabled','full_surface_checks','instrument_collision_checks','outcome_semantics'];parameters={k:a.get(k)==b.get(k) for k in keys};assert all(parameters.values()),parameters
(p/'results/final-parameter-equivalence.json').write_text(json.dumps(parameters,indent=2))
name='final-profile';args=[py,str(h),'--source',str(source),'--dll',str(dll),'--output',str(p/'results'/name),'--cpus','7,19,20,25']+flags+['--profile','--start','470','--count','12','--max-frames','482']
run(name,['/opt/nvidia/nsight-systems/2025.3.2/bin/nsys','profile','--trace=cuda,nvtx,opengl','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','--output',str(p/'results/final-profile-trace')]+args,'profile')
new=[x['performance'] for x in runs if x['modified']];control=[x['performance'] for x in runs if not x['modified']];mean=lambda values,key:sum(x[key] for x in values)/len(values)
result=dict(completed=True,source=str(source),dll=str(dll),dll_sha256=hashlib.sha256(dll.read_bytes()).hexdigest(),flags=flags,production_replaced=False,runs=runs,candidate_mean_frame_ms=mean(new,'mean_ms'),control_mean_frame_ms=mean(control,'mean_ms'),aggregate_wall_speedup=mean(control,'mean_ms')/mean(new,'mean_ms'),checked_no_image_performance=checked['performance'],current_frame_regression_passed=reg['passed'],parameters_and_thresholds_unchanged=all(parameters.values()),mean_and_p95_20hz=all(x['fps']>=20 and x['p95_ms']<=50 for x in new),rolling20_floor_20hz=all(x['rolling20_min_hz']>=20 for x in new),every_frame_50ms=all(x['over_50ms']==0 for x in new),formal_physics_domain_approved=False)
result['stable_20hz_passed']=result['mean_and_p95_20hz'] and result['rolling20_floor_20hz'] and result['every_frame_50ms']
(p/'results/final-selection.json').write_text(json.dumps(result,indent=2));print('DONE',json.dumps(result),flush=True)
