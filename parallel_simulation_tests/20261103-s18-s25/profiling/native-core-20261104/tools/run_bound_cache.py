import json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';m=json.loads((p/'manifest.json').read_text());h=p/'tools/profile_bound_cache.py';dll=p/'native-bound-cache/build/lib/libLiverCudaSim.so'
assert all(x['returncode']==0 for x in json.loads((p/'results/conservative-suite-status.json').read_text())['runs'])
assert json.loads((p/'results/bound-cache-device-binding.json').read_text())['all_original_functions_unchanged']
base=[py,str(h),'--source',m['baseline_source'],'--dll',str(dll),'--cpus','7,19,20,25','--query-mode','1','--query-threads','64','--assembly-blocks','512','--manifold-blocks','1024','--gpu-topology']
def run(name,flags):
 args=base+['--output',str(p/'results'/name)]+flags
 with (p/'bound-cache-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 print('START',name,flush=True)
 with (p/'logs'/str(name+'.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 d=json.loads((p/'results'/name/'result.json').read_text());print('END',name,r.returncode,{k:d.get(k) for k in ['completed','quality_passed','performance','native_validation','coarse_validation','coarse_bound_validation','error']},flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 return d

validation=run('bound-cache-window-validation',['--no-render','--bound-cache','--validate-native','--start','470','--count','4','--max-frames','474'])
assert validation['coarse_bound_validation']['passed'] and validation['native_validation']['passed']
validation=run('bound-cache-full-validation',['--no-render','--bound-cache','--validate-native','--start','0','--count','1410'])
assert validation['completed'] and validation['quality_passed'] and validation['coarse_bound_validation']['passed'] and validation['native_validation']['passed']
flags=['--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh','--persistent-soup','--direct-rgb','--opencv-threads','1']
control=run('bound-cache-control-a',flags)
candidate=run('bound-cache-candidate-a',flags+['--bound-cache'])
benefit=candidate['completed'] and candidate['quality_passed'] and candidate['performance']['mean_ms']<control['performance']['mean_ms'] and candidate['performance']['p95_ms']<control['performance']['p95_ms']
result=dict(validation_passed=True,first_pair_benefit=benefit,control={k:control.get(k) for k in ['completed','quality_passed','performance']},candidate={k:candidate.get(k) for k in ['completed','quality_passed','performance']})
(p/'results/bound-cache-selection.json').write_text(json.dumps(result,indent=2))
if benefit:
 candidate_b=run('bound-cache-candidate-b',flags+['--bound-cache'])
 control_b=run('bound-cache-control-b',flags)
 result['candidate_b']={k:candidate_b.get(k) for k in ['completed','quality_passed','performance']};result['control_b']={k:control_b.get(k) for k in ['completed','quality_passed','performance']};(p/'results/bound-cache-selection.json').write_text(json.dumps(result,indent=2))
