import os,sys,subprocess,json,time
from pathlib import Path
p=Path(__file__).resolve().parents[1];os.chdir(p);py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';dll=p/'native-readonly-v8/build/lib/libLiverCudaSim.so';source=p/'python-gpu-check';results=[]
common=['--source',str(source),'--dll',str(dll),'--query-mode','3','--bound-cache','--core-optimizations','0','--gpu-topology','--gpu-body-check','--query-threads','64','--assembly-blocks','512','--manifold-blocks','1024']
jobs=[('readonly-v8-synthetic',[py,str(p/'tools/test_gpu_body_check.py'),'--dll',str(dll),'--output',str(p/'results/gpu-body-v8-synthetic')]),('readonly-v8-current-rgbd',[py,str(p/'tools/check_current_frame_final_cache.py'),*common,'--output',str(p/'results/readonly-v8-current-rgbd'),'--persistent-soup','--direct-rgb']),('readonly-v8-snapshot',[py,str(p/'tools/check_snapshot_roundtrip_cache.py'),*common,'--output',str(p/'results/readonly-v8-snapshot')]),('readonly-v8-checked-no-render',[py,str(p/'tools/profile_all_queries.py'),*common,'--output',str(p/'results/readonly-v8-checked-no-render'),'--cpus','7,19,20,25','--no-render'])]
for name,cmd in jobs:
 start=time.time();print(json.dumps(dict(start=name,unix_s=start)),flush=True)
 with (p/'logs'/(name+'.txt')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 record=dict(name=name,returncode=r.returncode,elapsed_s=time.time()-start);results.append(record);(p/'results/post-checks.json').write_text(json.dumps(results,indent=2));print(json.dumps(record),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
