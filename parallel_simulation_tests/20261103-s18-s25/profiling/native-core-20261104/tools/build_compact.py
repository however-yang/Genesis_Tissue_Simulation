import json,subprocess,time,hashlib,os
from pathlib import Path
p=Path(__file__).resolve().parents[1];os.sched_setaffinity(0,[30,31]);cm='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/cmake';base=p/'native-compact';start=time.time()
with (p/'logs/build-compact.txt').open('w') as f:
 r=subprocess.run([cm,'-S',str(base/'source'),'-B',str(base/'build'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_COMPILER=/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/nvcc','-DCMAKE_CXX_COMPILER=/usr/bin/g++-12'],stdout=f,stderr=subprocess.STDOUT)
 if r.returncode==0:r=subprocess.run([cm,'--build',str(base/'build'),'-j2'],stdout=f,stderr=subprocess.STDOUT)
dll=base/'build/lib/libLiverCudaSim.so';result=dict(returncode=r.returncode,started_unix_s=start,finished_unix_s=time.time(),dll=str(dll),sha256=hashlib.sha256(dll.read_bytes()).hexdigest() if dll.exists() else None);(p/'results/build-compact.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(r.returncode)
