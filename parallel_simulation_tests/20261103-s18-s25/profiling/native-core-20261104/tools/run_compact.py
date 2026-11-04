import json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';m=json.loads((p/'manifest.json').read_text());h=p/'tools/profile_compact.py';dll=p/'native-compact/build/lib/libLiverCudaSim.so'
assert json.loads((p/'results/querythreads-selection.json').read_text())['runs'][-1]['completed']
base=[py,str(h),'--source',m['baseline_source'],'--dll',str(dll),'--cpus','7,19,20,25','--query-mode','1','--query-threads','128','--assembly-blocks','128','--manifold-blocks','1024','--gpu-topology']
def run(name,flags):
 args=base+['--output',str(p/'results'/name)]+flags
 with (p/'compact-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 print('START',name,flush=True)
 with (p/'logs'/str(name+'.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 d=json.loads((p/'results'/name/'result.json').read_text());print('END',name,r.returncode,{k:d.get(k) for k in ['completed','quality_passed','performance','native_validation','coarse_validation','error']},flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 return d

validation=run('compact-window-validation',['--no-render','--compact-blocks','32','--validate-native','--start','470','--count','4','--max-frames','474'])
assert validation['native_validation']['passed']
validation=run('compact-full-validation',['--no-render','--compact-blocks','32','--validate-native','--start','0','--count','1410'])
assert validation['completed'] and validation['quality_passed'] and validation['native_validation']['passed']
flags=['--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh','--persistent-soup','--direct-rgb','--opencv-threads','1']
control=run('compact-control-rgbd',flags)
candidate=run('compact-32-rgbd',flags+['--compact-blocks','32'])
(p/'results/compact-selection.json').write_text(json.dumps(dict(validation_passed=True,control={k:control.get(k) for k in ['completed','quality_passed','performance']},candidate={k:candidate.get(k) for k in ['completed','quality_passed','performance']}),indent=2))
