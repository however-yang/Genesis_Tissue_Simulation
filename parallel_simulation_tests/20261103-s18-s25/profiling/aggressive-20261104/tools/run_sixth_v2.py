"""Repair private-workspace snapshot accounting without changing old state fields."""
import json,time,subprocess
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';h=p/'tools/profile_r7.py';source=p/'candidate-r8-source';dll=p/'native-gpu-diagnostics-v2/build/lib/libLiverCudaSim.so';flags=json.loads((p/'results/fifth-r8-selection.json').read_text())['flags'];statuses=[]
assert all(x['returncode']==0 for x in json.loads((p/'results/private-scratch-v2-build-status.json').read_text()))
root_cause=json.loads((p/'results/archive-v1-root-cause.json').read_text());assert root_cause['all_changed_sites_are_new_topology_workspace'] and root_cause['all_existing_host_fields_unchanged']
def run(name,args,kind='full'):
 print('START',name,flush=True);start=time.time()
 with (p/'sixth-v2-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 with (p/'logs'/str(name+'.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,returncode=r.returncode,start_unix_s=start,end_unix_s=time.time()));(p/'results/sixth-v2-status.json').write_text(json.dumps(statuses,indent=2));print('END',statuses[-1],flush=True)
 q=p/'results'/name/'result.json';d=json.loads(q.read_text()) if q.exists() else {}
 if r.returncode or d.get('error') or not (d.get('passed') if kind=='regression' else d.get('completed') and d.get('quality_passed')):raise SystemExit('V2 validation failed or resource guard stopped: '+name)
 return d
original=(p/'tools/check_gpu_topology.py').read_text().replace('native-gpu-diagnostics/build','native-gpu-diagnostics-v2/build');(p/'tools/check_gpu_topology_v2.py').write_text(original)
run('gpu-topology-equivalence-v2',[py,str(p/'tools/check_gpu_topology_v2.py'),'--dll',str(dll),'--output',str(p/'results/gpu-topology-equivalence-v2')],'regression')
regargs=[py,str(p/'tools/check_current_frame_archive.py'),'--source',str(source),'--dll',str(dll),'--output',str(p/'results/current-frame-r8-gpu-v2'),'--gpu-topology']
for f in ['--persistent-soup','--direct-rgb']:
 if f in flags:regargs.append(f)
reg=run('current-frame-r8-gpu-v2',regargs,'regression')
def frame(name,extra):return run(name,[py,str(h),'--source',str(source),'--dll',str(dll),'--output',str(p/'results'/name),'--cpus','7,19,20,25']+flags+extra)
validation=frame('r8-topology-validation-v2',['--gpu-topology','--topology-ab']);assert validation.get('topology_frames_compared')==1410
cpu=frame('r8-topo-cpu-v2',[]);gpu=frame('r8-topo-gpu-v2',['--gpu-topology'])
keep=gpu['performance']['mean_ms']<=.95*cpu['performance']['mean_ms'] and gpu['performance']['p95_ms']<=cpu['performance']['p95_ms']
result=dict(completed=True,source=str(source),dll=str(dll),flags=flags+(['--gpu-topology'] if keep else []),topology_gpu_kept=keep,cpu_mean_ms=cpu['performance']['mean_ms'],gpu_mean_ms=gpu['performance']['mean_ms'],cpu_mean_p95_ms=cpu['performance']['p95_ms'],gpu_mean_p95_ms=gpu['performance']['p95_ms'],full_topology_frames_compared=1410,strict_archive_and_current_frame_passed=reg['passed'],snapshot_standard_unchanged=True,existing_physical_allocations_and_fields_unchanged=True,formal_physics_domain_approved=False)
(p/'results/sixth-r8-v2-selection.json').write_text(json.dumps(result,indent=2));print('DONE',json.dumps(result),flush=True)
