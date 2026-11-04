"""Final matched tests; every failure remains in the record."""
import json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';m=json.loads((p/'manifest.json').read_text());choice=json.loads((p/'results/conservative-candidate.json').read_text());source=m['baseline_source'];status=[]
flags=['--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh','--persistent-soup','--direct-rgb','--opencv-threads','1','--gpu-topology']
execflags=lambda values:[x for k,v in values.items() for x in ['--'+k.replace('_','-'),str(v)]]
base=dict(query_mode=0,query_threads=64,assembly_blocks=512,manifold_blocks=1024,compact_blocks=0)
def run(name,harness='profile_fallback.py',extra=None,baseline=False,dll=None,settings=None,prefix=None):
 settings=settings or (base if baseline else choice['execution']);args=[py,str(p/'tools'/harness),'--source',source,'--dll',dll or (m['baseline_dll'] if baseline else choice['dll']),'--output',str(p/'results'/name)]+execflags(settings)+(extra or [])
 if harness.startswith('profile_'):args+=['--cpus','7,19,20,25']
 if prefix:args=prefix+args
 with (p/'conservative-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 print('START',name,flush=True);start=time.time()
 with (p/'logs'/str(name+'.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 file=p/'results'/name/'result.json';d=json.loads(file.read_text()) if file.exists() else {}
 item=dict(name=name,returncode=r.returncode,start_unix_s=start,end_unix_s=time.time(),completed=d.get('completed'),quality_passed=d.get('quality_passed'),passed=d.get('passed'),error=d.get('error'),performance=d.get('performance'));status.append(item);(p/'results/conservative-suite-progress.json').write_text(json.dumps(status,indent=2));print('END',json.dumps(item),flush=True)
 if any(x in d.get('error','') for x in ['Foreign GPU','not idle','Resource ceiling','Resource guard']):raise SystemExit('Resource guard stopped final suite')
 return d
run('conservative-captured-validation',extra=['--no-render','--gpu-topology','--validate-native','--start','0','--count','1410'])
run('conservative-captured-validation-repeat',extra=['--no-render','--gpu-topology','--validate-native','--start','0','--count','1410'])
run('conservative-current-frame','check_current_frame_final.py',['--gpu-topology','--direct-rgb','--persistent-soup'])
run('conservative-snapshot-roundtrip','check_snapshot_roundtrip.py',['--gpu-topology'])
for name,baseline in [('conservative-baseline-a',True),('conservative-candidate-a',False),('conservative-candidate-b',False),('conservative-baseline-b',True)]:run(name,extra=flags,baseline=baseline)
run('conservative-checked-physics',extra=['--no-render','--gpu-topology'])
nsys='/opt/nvidia/nsight-systems/2025.3.2/bin/nsys'
for name,baseline in [('conservative-core-profile',False)]:
 trace=p/'results'/str(name+'-trace');prefix=[nsys,'profile','--trace=cuda,nvtx,opengl','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','--output',str(trace)]
 run(name,extra=flags+['--profile','--start','470','--count','12','--max-frames','482'],baseline=baseline,prefix=prefix)
 args=[nsys,'export','--type=sqlite','--force-overwrite=true','--output',str(trace)+'.sqlite',str(trace)+'.nsys-rep']
 with (p/'logs'/str(name+'-export.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 print('EXPORT',name,r.returncode,flush=True)
(p/'results/conservative-suite-status.json').write_text(json.dumps(dict(completed=True,runs=status),indent=2));print('CONSERVATIVE SUITE FINISHED',flush=True)
