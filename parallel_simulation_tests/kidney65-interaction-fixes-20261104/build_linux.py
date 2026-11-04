"""Build and archive a uniquely named native candidate with existing tools."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parent
PREFIX=ROOT.parents[1]/'envs/genesis-py311-cu125-v1'
parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--trace',action='store_true');parser.add_argument('--precise-coarse',action='store_true');parser.add_argument('--warp-stack',action='store_true');parser.add_argument('--fragment-probe',action='store_true');parser.add_argument('--precise-interaction',action='store_true');parser.add_argument('--warp-scratch',action='store_true');parser.add_argument('--precise-tool',action='store_true');args=parser.parse_args()
assert args.label.replace('-','').replace('_','').isalnum()
for folder in ('logs','results','binaries','tmp'):(ROOT/folder).mkdir(exist_ok=True)
library=ROOT/'binaries'/(args.label+'.so');record=ROOT/'results'/('build-'+args.label+'.json')
assert not library.exists() and not record.exists()
available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[v for v in (30,31) if v in available] or available[-2:])
commands=[[str(PREFIX/'bin/cmake'),'-S',str(ROOT/'native'),'-B',str(ROOT/'build'),'-DCMAKE_BUILD_TYPE=Release',
           '-DCMAKE_CUDA_COMPILER='+str(PREFIX/'bin/nvcc'),'-DCMAKE_CXX_COMPILER=/usr/bin/g++-12',
           '-DENDO_KIDNEY_CONTACT_TRACE='+('ON' if args.trace else 'OFF'),
           '-DENDO_KIDNEY_PRECISE_COARSE='+('ON' if args.precise_coarse else 'OFF'),
           '-DENDO_KIDNEY_WARP_STACK='+('ON' if args.warp_stack else 'OFF'),
           '-DENDO_KIDNEY_WARP_SCRATCH='+('ON' if args.warp_scratch else 'OFF'),
           '-DENDO_FRAGMENT_PROBE='+('ON' if args.fragment_probe else 'OFF'),
           '-DENDO_KIDNEY_PRECISE_INTERACTION='+('ON' if args.precise_interaction else 'OFF'),
           '-DENDO_KIDNEY_PRECISE_TOOL='+('ON' if args.precise_tool else 'OFF')],
          [str(PREFIX/'bin/cmake'),'--build',str(ROOT/'build'),'-j2']]
start=time.time()
with (ROOT/'logs'/('build-'+args.label+'.log')).open('x') as stream:
    for cmd in commands:
        result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')))
        if result.returncode:break
data=dict(label=args.label,returncode=result.returncode,commands=commands,started_unix=start,finished_unix=time.time(),trace=args.trace,precise_coarse=args.precise_coarse,warp_stack=args.warp_stack,warp_scratch=args.warp_scratch,fragment_probe=args.fragment_probe,precise_interaction=args.precise_interaction,precise_tool=args.precise_tool,
          source_sha256={str(p.relative_to(ROOT/'native')):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'native').rglob('*')) if p.is_file()})
if result.returncode==0:
    shutil.copy2(ROOT/'build/lib/libLiverCudaSim.so',library)
    data['sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
record.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({k:v for k,v in data.items() if k!='source_sha256'}),flush=True)
raise SystemExit(result.returncode)
