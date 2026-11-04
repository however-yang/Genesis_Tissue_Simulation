"""CPU compilation of isolated baseline/training DLLs, never default output."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
p=argparse.ArgumentParser()
p.add_argument('--variant',choices=['baseline','training-state1','training-snapshot1'],required=True)
args=p.parse_args()
source=HERE/'native-baseline' if args.variant=='baseline' else HERE/'artifacts'/args.variant/'source'
if args.variant=='training-state1':
    for header in (HERE/'training-native').glob('*'):
        shutil.copyfile(header,source/'vendor/cuda/training'/header.name)
if args.variant=='training-snapshot1':
    for header in (HERE/'snapshot-native').glob('*'):
        shutil.copyfile(header,source/'vendor/cuda/training'/header.name)
artifact=HERE/'artifacts'/args.variant
build=artifact/'build'
binary=ROOT/'endoscopic_tissue/native'/('bin-s15-'+args.variant)
assert binary.resolve().parent== (ROOT/'endoscopic_tissue/native').resolve()
assert binary.name.startswith('bin-s15-')
artifact.mkdir(parents=True,exist_ok=True)
vcvars=Path(r'D:\develop\Mocrosoft VS\Microsoft Visual Studio\2022 Comumnity\VC\Auxiliary\Build\vcvars64.bat')
nvcc=Path(r'D:\develop\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe')
script=artifact/'build.cmd'
script.write_text('\n'.join(['@echo off','setlocal',f'call "{vcvars}"','if errorlevel 1 exit /b 1',
    f'cmake -S "{source.as_posix()}" -B "{build.as_posix()}" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="{nvcc.as_posix()}" -DENDO_BINARY_DIR="{binary.as_posix()}" -DCMAKE_CUDA_FLAGS=""',
    'if errorlevel 1 exit /b 1',f'cmake --build "{build.as_posix()}" --target LiverCudaSim','exit /b %errorlevel%'])+'\n',encoding='utf8')
files={f.relative_to(source).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*') if f.is_file()}
(artifact/'build-inputs.json').write_text(json.dumps({'variant':args.variant,'source':str(source),'files':files},indent=2),encoding='utf8')
with (artifact/'build.log').open('w',encoding='utf8') as log:
    result=subprocess.run([os.environ.get('COMSPEC','cmd.exe'),'/d','/c',str(script)],stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    print((artifact/'build.log').read_text(encoding='utf8',errors='replace')[-7000:])
    raise SystemExit(result.returncode)
dll=binary/'LiverCudaSim.dll'
record={'variant':args.variant,'dll':str(dll),'sha256':hashlib.sha256(dll.read_bytes()).hexdigest(),
        'source_files':files,'default_replaced':False,'physics_approved':False}
(artifact/'build-result.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in record.items() if k!='source_files'},indent=2))
