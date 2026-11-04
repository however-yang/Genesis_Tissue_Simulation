"""Qualify the final optional dispatch extension without replacing the verified fallback."""
import json,subprocess,statistics,time,hashlib
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';m=json.loads((p/'manifest.json').read_text());status=[]
read=lambda n:json.loads((p/'results'/n/'result.json').read_text())
file=p/'results/all-queries-v2-status.json'
if not file.exists():raise SystemExit('Complete 6-run comparison required first')
runs=json.loads(file.read_text())['runs'];assert all(r['returncode']==0 for r in runs)
blade=[read(n) for n in ['all-queries-v2-blade-a','all-queries-v2-blade-b']];both=[read(n) for n in ['all-queries-v2-candidate-a','all-queries-v2-candidate-b']]
mean=lambda ds,key:statistics.mean(x['performance'][key] for x in ds)
benefit=all(d['completed'] and d['quality_passed'] for d in blade+both) and mean(both,'mean_ms')<mean(blade,'mean_ms')*.995 and mean(both,'p95_ms')<mean(blade,'p95_ms')
selection=dict(performance_benefit=benefit,blade_mean_ms=mean(blade,'mean_ms'),both_mean_ms=mean(both,'mean_ms'),blade_mean_p95_ms=mean(blade,'p95_ms'),both_mean_p95_ms=mean(both,'p95_ms'),qualified=False,production_replaced=False)
(p/'results/all-queries-v2-selection.json').write_text(json.dumps(selection,indent=2));print('SELECTION',json.dumps(selection),flush=True)
if not benefit:raise SystemExit(0)
choice=json.loads((p/'results/recommended-candidate.json').read_text());choice.update(native_source=str(p/'native-all-queries-v2/source'),dll=str(p/'native-all-queries-v2/build/lib/libLiverCudaSim.so'),harness='profile_all_queries.py');choice['execution']['query_mode']=3;choice['dll_sha256']=hashlib.sha256(Path(choice['dll']).read_bytes()).hexdigest()
def run(name,harness=None,extra=None,mode=3,prefix=None):
 values=dict(choice['execution'],query_mode=mode);args=[py,str(p/'tools'/(harness or choice['harness'])),'--source',m['baseline_source'],'--dll',choice['dll'],'--output',str(p/'results'/name)]
 for k,v in values.items():
  if isinstance(v,bool):
   if v:args+=['--'+k.replace('_','-')]
  else:args+=['--'+k.replace('_','-'),str(v)]
 args+=(extra or [])
 if not harness:args+=['--cpus','7,19,20,25']
 if prefix:args=prefix+args
 with (p/'all-queries-v2-final-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=args))+'\n')
 print('START',name,flush=True)
 with (p/'logs'/str(name+'.txt')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 d=read(name);status.append(dict(name=name,returncode=r.returncode,passed=d.get('passed'),completed=d.get('completed'),quality_passed=d.get('quality_passed'),error=d.get('error'),performance=d.get('performance')));print('END',json.dumps(status[-1]),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 return d
run('all-queries-v2-checked-physics',extra=['--no-render','--gpu-topology'])
run('all-queries-v2-current-frame',harness='check_current_frame_final_cache.py',extra=['--gpu-topology','--persistent-soup','--direct-rgb'])
run('all-queries-v2-snapshot-roundtrip',harness='check_snapshot_roundtrip_cache.py',extra=['--gpu-topology'])
nsys='/opt/nvidia/nsight-systems/2025.3.2/bin/nsys';flags=['--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh','--persistent-soup','--direct-rgb','--opencv-threads','1','--gpu-topology','--profile','--start','470','--count','12','--max-frames','482']
for name,mode in [('all-queries-v2-blade-profile',1),('all-queries-v2-candidate-profile',3)]:
 trace=p/'results'/str(name+'-trace');prefix=[nsys,'profile','--trace=cuda,nvtx,opengl','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','--output',str(trace)]
 run(name,extra=flags,mode=mode,prefix=prefix)
 with (p/'logs'/str(name+'-export.txt')).open('w') as f:r=subprocess.run([nsys,'export','--type=sqlite','--force-overwrite=true','--output',str(trace)+'.sqlite',str(trace)+'.nsys-rep'],stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0
selection.update(qualified=True,final_checks=status,candidate=choice);(p/'results/all-queries-v2-selection.json').write_text(json.dumps(selection,indent=2));choice['qualification_pending']=False;choice['local_validation_passed']=True;(p/'results/recommended-candidate.json').write_text(json.dumps(choice,indent=2));print('ALL QUERY EXTENSION QUALIFIED LOCALLY; FORMAL DOMAIN REMAINS UNAPPROVED',flush=True)
