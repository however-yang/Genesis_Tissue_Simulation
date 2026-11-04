import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parents[1]
project=p.parents[3];prefix=project/'envs/genesis-py311-cu125-v1'
os.sched_setaffinity(0,[30,31])
parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
assert all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.label)
start=time.time();log=p/'logs'/('build-'+args.label+'.txt');assert not log.exists()
commands=[[str(prefix/'bin/cmake'),'-S',str(p/'native/source'),'-B',str(p/'native/build'),'-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_CUDA_COMPILER='+str(prefix/'bin/nvcc'),'-DCMAKE_CXX_COMPILER=/usr/bin/g++-12'],
    [str(prefix/'bin/cmake'),'--build',str(p/'native/build'),'-j2']]
with log.open('w') as stream:
    for command in commands:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:break
dll=p/'native/build/lib/libLiverCudaSim.so'
data={'returncode':result.returncode,'started_unix':start,'finished_unix':time.time(),'commands':commands,
      'dll':str(dll),'sha256':hashlib.sha256(dll.read_bytes()).hexdigest() if result.returncode==0 and dll.exists() else None}
data['source_sha256']={str(f.relative_to(p/'native/source')):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((p/'native/source').rglob('*')) if f.is_file()}
(p/'results'/('build-'+args.label+'.json')).write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({k:v for k,v in data.items() if k!='source_sha256'}),flush=True)
raise SystemExit(result.returncode)
