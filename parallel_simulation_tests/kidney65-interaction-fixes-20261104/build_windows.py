"""Isolated RTX 3090 build from the reviewed candidate; never replaces a default DLL."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--precise-coarse',action='store_true');parser.add_argument('--warp-stack',action='store_true');parser.add_argument('--precise-interaction',action='store_true');parser.add_argument('--angle-budget',choices=('0.02','0.04'),default='0.02');
    for option in ('precise-line','precise-coarse-warp','precise-mg','precise-tool'):
        parser.add_argument('--'+option,action='store_true')
    args=parser.parse_args()
    if os.name!='nt' or not args.label.replace('-','').isalnum():raise ValueError('Windows and a plain build label required')
    source=ROOT/('source-'+args.label);build=ROOT/('build-'+args.label)
    source.mkdir(exist_ok=False)
    hashes={}
    for p in sorted((ROOT/'native').rglob('*')):
        if not p.is_file():continue
        name=p.relative_to(ROOT/'native');data=p.read_bytes().replace(b'\r\n',b'\n')
        if str(name)=='CMakeLists.txt':
            assert data.count(b'CUDA_ARCHITECTURES "80"')==1
            data=data.replace(b'CUDA_ARCHITECTURES "80"',b'CUDA_ARCHITECTURES "86"')
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        hashes[str(name)]=hashlib.sha256(data).hexdigest()
    for folder in ('logs','results','binaries','tmp'):(ROOT/folder).mkdir(exist_ok=True)
    dll=ROOT/'binaries'/(args.label+'.dll');record=ROOT/'results'/('build-'+args.label+'.json')
    assert not dll.exists() and not record.exists()
    command=ROOT/'tmp'/('build-'+args.label+'.cmd')
    precise='ON' if args.precise_coarse else 'OFF'
    warp='ON' if args.warp_stack else 'OFF'
    interaction='ON' if args.precise_interaction else 'OFF'
    extra=' '.join('-DENDO_KIDNEY_'+name.upper()+'='+('ON' if getattr(args,name) else 'OFF') for name in ('precise_line','precise_coarse_warp','precise_mg','precise_tool'))
    command.write_text(f'''@echo off
call "D:\\develop\\Mocrosoft VS\\Microsoft Visual Studio\\2022 Comumnity\\VC\\Auxiliary\\Build\\vcvars64.bat" -vcvars_ver=14.29
if errorlevel 1 exit /b 1
"D:\\develop\\CMake\\bin\\cmake.exe" -S "{source}" -B "{build}" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="D:/develop/Mocrosoft VS/Microsoft Visual Studio/2022 Comumnity/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe" -DCMAKE_CUDA_COMPILER="D:/develop/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin/nvcc.exe" -DCMAKE_OBJECT_PATH_MAX=220 -DENDO_KIDNEY_CONTACT_TRACE=OFF -DENDO_KIDNEY_PRECISE_COARSE={precise} -DENDO_KIDNEY_WARP_STACK={warp} -DENDO_KIDNEY_PRECISE_INTERACTION={interaction} -DENDO_FRAGMENT_PROBE=OFF -DENDO_FRAGMENT_ANGLE_BUDGET={args.angle_budget} -DENDO_KIDNEY_STAGE_TRACE=OFF -DENDO_FRAGMENT_DIPOLE=OFF -DENDO_KIDNEY_WARP_SCRATCH=OFF {extra}
if errorlevel 1 exit /b 1
"D:\\develop\\CMake\\bin\\cmake.exe" --build "{build}" --target LiverCudaSim --parallel 2
exit /b %errorlevel%
''',encoding='utf-8')
    start=time.time()
    with (ROOT/'logs'/('build-'+args.label+'.log')).open('x') as log:
        status=subprocess.run(['cmd.exe','/d','/c',str(command)],stdout=log,stderr=subprocess.STDOUT).returncode
    result=dict(label=args.label,platform='Windows',architecture='sm_86',returncode=status,
                started_unix=start,finished_unix=time.time(),source_sha256=hashes,source_directory=str(source),build_directory=str(build),
                precise_line=args.precise_line,precise_coarse_warp=args.precise_coarse_warp,precise_mg=args.precise_mg,precise_tool=args.precise_tool,
                precise_coarse=args.precise_coarse,warp_stack=args.warp_stack,precise_interaction=args.precise_interaction,angle_budget=args.angle_budget,trace=False,default_replaced=False)
    if status==0:
        shutil.copy2(build/'lib/LiverCudaSim.dll',dll)
        result['sha256']=hashlib.sha256(dll.read_bytes()).hexdigest()
    record.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))
    raise SystemExit(status)

if __name__=='__main__':main()
