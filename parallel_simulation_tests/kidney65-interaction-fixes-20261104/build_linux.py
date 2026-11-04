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
parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--trace',action='store_true');parser.add_argument('--precise-coarse',action='store_true');parser.add_argument('--warp-stack',action='store_true');parser.add_argument('--fragment-probe',action='store_true');parser.add_argument('--precise-interaction',action='store_true');parser.add_argument('--warp-scratch',action='store_true');parser.add_argument('--precise-tool',action='store_true');parser.add_argument('--angle-budget',choices=('0.02','0.04'),default='0.02');parser.add_argument('--dipole',action='store_true');parser.add_argument('--stage-trace',action='store_true');parser.add_argument('--precise-line',action='store_true');parser.add_argument('--precise-coarse-warp',action='store_true');parser.add_argument('--precise-mg',action='store_true');parser.add_argument('--mg-warp-reduce',action='store_true');parser.add_argument('--cache-tree-bounds',action='store_true');parser.add_argument('--split-fragment',action='store_true');parser.add_argument('--closed-ray-probe',action='store_true');parser.add_argument('--closed-ray',action='store_true');parser.add_argument('--closed-stats',action='store_true');parser.add_argument('--split-pressure',action='store_true');parser.add_argument('--specialize-pressure',action='store_true');parser.add_argument('--stencil-cache',action='store_true');args=parser.parse_args()
assert args.label.replace('-','').replace('_','').isalnum()
for folder in ('logs','results','binaries','tmp'):(ROOT/folder).mkdir(exist_ok=True)
library=ROOT/'binaries'/(args.label+'.so');record=ROOT/'results'/('build-'+args.label+'.json')
assert not library.exists() and not record.exists()
source=ROOT/('source-'+args.label);build=ROOT/('build-'+args.label)
assert not source.exists() and not build.exists()
# Freeze all build inputs before starting the compiler. A later experiment can
# edit native/ without changing this label's sources or build directory.
shutil.copytree(ROOT/'native',source)
available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[v for v in (30,31) if v in available] or available[-2:])
commands=[[str(PREFIX/'bin/cmake'),'-S',str(source),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
           '-DCMAKE_CUDA_COMPILER='+str(PREFIX/'bin/nvcc'),'-DCMAKE_CXX_COMPILER=/usr/bin/g++-12',
           '-DENDO_FRAGMENT_ANGLE_BUDGET='+args.angle_budget,
           '-DENDO_KIDNEY_STENCIL_CACHE='+('ON' if args.stencil_cache else 'OFF'),
           '-DENDO_KIDNEY_SPECIALIZE_PRESSURE='+('ON' if args.specialize_pressure else 'OFF'),
           '-DENDO_KIDNEY_SPLIT_PRESSURE='+('ON' if args.split_pressure else 'OFF'),
           '-DENDO_KIDNEY_CLOSED_STATS='+('ON' if args.closed_stats else 'OFF'),
           '-DENDO_KIDNEY_CLOSED_RAY='+('ON' if args.closed_ray else 'OFF'),
           '-DENDO_KIDNEY_CLOSED_RAY_PROBE='+('ON' if args.closed_ray_probe else 'OFF'),
           '-DENDO_KIDNEY_SPLIT_FRAGMENT='+('ON' if args.split_fragment else 'OFF'),
           '-DENDO_KIDNEY_CACHE_TREE_BOUNDS='+('ON' if args.cache_tree_bounds else 'OFF'),
           '-DENDO_KIDNEY_MG_WARP_REDUCE='+('ON' if args.mg_warp_reduce else 'OFF'),
           '-DENDO_KIDNEY_PRECISE_MG='+('ON' if args.precise_mg else 'OFF'),
           '-DENDO_KIDNEY_PRECISE_COARSE_WARP='+('ON' if args.precise_coarse_warp else 'OFF'),
           '-DENDO_KIDNEY_PRECISE_LINE='+('ON' if args.precise_line else 'OFF'),
           '-DENDO_KIDNEY_STAGE_TRACE='+('ON' if args.stage_trace else 'OFF'),
           '-DENDO_FRAGMENT_DIPOLE='+('ON' if args.dipole else 'OFF'),
           '-DENDO_KIDNEY_CONTACT_TRACE='+('ON' if args.trace else 'OFF'),
           '-DENDO_KIDNEY_PRECISE_COARSE='+('ON' if args.precise_coarse else 'OFF'),
           '-DENDO_KIDNEY_WARP_STACK='+('ON' if args.warp_stack else 'OFF'),
           '-DENDO_KIDNEY_WARP_SCRATCH='+('ON' if args.warp_scratch else 'OFF'),
           '-DENDO_FRAGMENT_PROBE='+('ON' if args.fragment_probe else 'OFF'),
           '-DENDO_KIDNEY_PRECISE_INTERACTION='+('ON' if args.precise_interaction else 'OFF'),
           '-DENDO_KIDNEY_PRECISE_TOOL='+('ON' if args.precise_tool else 'OFF')],
          [str(PREFIX/'bin/cmake'),'--build',str(build),'-j2']]
start=time.time()
with (ROOT/'logs'/('build-'+args.label+'.log')).open('x') as stream:
    for cmd in commands:
        result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')))
        if result.returncode:break
data=dict(label=args.label,returncode=result.returncode,commands=commands,started_unix=start,finished_unix=time.time(),trace=args.trace,precise_coarse=args.precise_coarse,warp_stack=args.warp_stack,warp_scratch=args.warp_scratch,fragment_probe=args.fragment_probe,precise_interaction=args.precise_interaction,precise_tool=args.precise_tool,angle_budget=args.angle_budget,dipole=args.dipole,stage_trace=args.stage_trace,precise_line=args.precise_line,precise_coarse_warp=args.precise_coarse_warp,precise_mg=args.precise_mg,mg_warp_reduce=args.mg_warp_reduce,cache_tree_bounds=args.cache_tree_bounds,split_fragment=args.split_fragment,closed_ray_probe=args.closed_ray_probe,closed_ray=args.closed_ray,closed_stats=args.closed_stats,split_pressure=args.split_pressure,specialize_pressure=args.specialize_pressure,stencil_cache=args.stencil_cache,
          source_directory=str(source),build_directory=str(build),
          source_sha256={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()})
if result.returncode==0:
    shutil.copy2(build/'lib/libLiverCudaSim.so',library)
    data['sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
record.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({k:v for k,v in data.items() if k!='source_sha256'}),flush=True)
raise SystemExit(result.returncode)
