"""Optional grasper batching: validate first, compare original / blade-only / all kinds."""
import json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';m=json.loads((p/'manifest.json').read_text());dll=p/'native-all-queries/build/lib/libLiverCudaSim.so';status=[]
assert json.loads((p/'results/all-queries-device-binding.json').read_text())['all_original_functions_unchanged']
base=[py,str(p/'tools/profile_all_queries.py'),'--source',m['baseline_source'],'--dll',str(dll),'--cpus','7,19,20,25','--query-threads','64','--assembly-blocks','512','--manifold-blocks','1024','--gpu-topology']
def run(name,mode,cache,flags):
 args=base+['--output',str(p/'results'/name),'--query-mode',str(mode)]+(['--bound-cache'] if cache else [])+flags
 with (p/'all-queries-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 print('START',name,flush=True);start=time.time()
 with (p/'logs'/str(name+'.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 file=p/'results'/name/'result.json';d=json.loads(file.read_text()) if file.exists() else {};row=dict(name=name,returncode=r.returncode,start_unix_s=start,end_unix_s=time.time(),completed=d.get('completed'),quality_passed=d.get('quality_passed'),performance=d.get('performance'),error=d.get('error'));status.append(row);(p/'results/all-queries-progress.json').write_text(json.dumps(status,indent=2));print('END',json.dumps(row),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 return d
validation=run('all-queries-full-validation',3,True,['--no-render','--validate-native','--start','0','--count','1410'])
assert validation['completed'] and validation['quality_passed'] and validation['native_validation']['passed'] and validation['coarse_bound_validation']['passed']
flags=['--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh','--persistent-soup','--direct-rgb','--opencv-threads','1']
for name,mode,cache in [('all-queries-original-a',0,False),('all-queries-blade-a',1,True),('all-queries-candidate-a',3,True),('all-queries-candidate-b',3,True),('all-queries-blade-b',1,True),('all-queries-original-b',0,False)]:run(name,mode,cache,flags)
(p/'results/all-queries-status.json').write_text(json.dumps(dict(completed=True,runs=status),indent=2));print('ALL QUERIES SUITE FINISHED',flush=True)
