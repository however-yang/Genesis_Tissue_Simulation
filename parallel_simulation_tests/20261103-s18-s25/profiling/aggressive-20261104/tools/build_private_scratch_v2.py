import subprocess,json,time
from pathlib import Path
p=Path(__file__).resolve().parents[1];cm='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/cmake';statuses=[]
for name in ['native-gpu-diagnostics-v2','native-precomputed-v2']:
 start=time.time();flags=[cm,'-S',str(p/name/'source'),'-B',str(p/name/'build'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_COMPILER=/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/nvcc','-DCMAKE_CXX_COMPILER=/usr/bin/g++-12']
 with (p/'logs'/str('build-'+name+'.txt')).open('w') as f:
  r=subprocess.run(flags,stdout=f,stderr=subprocess.STDOUT)
  if r.returncode==0:r=subprocess.run(['taskset','-c','30,31',cm,'--build',str(p/name/'build'),'-j2'],stdout=f,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,returncode=r.returncode,start_unix_s=start,end_unix_s=time.time()));(p/'results/private-scratch-v2-build-status.json').write_text(json.dumps(statuses,indent=2));print(statuses[-1],flush=True)
 if r.returncode:raise SystemExit(r.returncode)
