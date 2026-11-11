"""Compile and execute archive ownership tests without CUDA or GPU access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
p=argparse.ArgumentParser()
p.add_argument('--name',required=True)
args=p.parse_args()
out=ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/host-state-tests'/args.name
out.mkdir(parents=True,exist_ok=False)
stub=HERE/'host-state-tests'
source=HERE/'snapshot-native'
vcvars=Path(r'D:\develop\Mocrosoft VS\Microsoft Visual Studio\2022 Comumnity\VC\Auxiliary\Build\vcvars64.bat')
cleanup='discard_session' in (source/'state_archive.h').read_text(encoding='utf8')
flag='/DARCHIVE_HAS_DISCARD' if cleanup else ''
script=out/'test.cmd'
script.write_text('\n'.join(['@echo off','setlocal',f'call "{vcvars}"','if errorlevel 1 exit /b 1',
    f'cl /nologo /EHsc /std:c++17 /Od /Zi {flag} /I"{stub}" /I"{source}" '
    f'"{source / "state_archive.cpp"}" "{stub / "archive_test.cpp"}" /Fe:archive_test.exe',
    'if errorlevel 1 exit /b 1','archive_test.exe','exit /b %errorlevel%'])+'\n',encoding='utf8')
with (out/'run.log').open('w',encoding='utf8') as log:
    run=subprocess.run([os.environ.get('COMSPEC','cmd.exe'),'/d','/c',str(script)],cwd=out,
                       stdout=log,stderr=subprocess.STDOUT)
record={'passed':run.returncode==0,'returncode':run.returncode,'gpu_used':False,
    'discard_cleanup_enabled':cleanup,
    'archive_source_sha256':hashlib.sha256((source/'state_archive.cpp').read_bytes()).hexdigest(),
    'scope':'CPU fault-injection ownership checks only; no real CUDA physics or performance acceptance'}
(out/'result.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print((out/'run.log').read_text(encoding='utf8',errors='replace')[-4000:])
print(json.dumps(record,indent=2))
raise SystemExit(run.returncode)
