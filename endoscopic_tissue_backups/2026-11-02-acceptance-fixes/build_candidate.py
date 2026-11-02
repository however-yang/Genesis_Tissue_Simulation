"""Build a separately named S15/S16 native variant from frozen experiment source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--base', choices=['training-state1', 'training-snapshot1'], default='training-state1')
parser.add_argument('--section-trace', action='store_true')
args = parser.parse_args()
assert re.fullmatch(r'[a-z0-9-]+', args.name)
artifact = HERE / 'artifacts' / args.name
source = artifact / 'source'
baseline = ROOT / 'endoscopic_tissue_backups/2026-11-01-s15-s17/artifacts' / args.base / 'source'
if not source.exists():
    shutil.copytree(baseline, source)
    override = HERE / 'overrides' / args.name
    if override.exists():
        manifest = json.loads((override / 'manifest.json').read_text(encoding='utf8'))
        for relative, expected in manifest['base_sha256'].items():
            current = hashlib.sha256((source / relative).read_bytes()).hexdigest()
            if current != expected:
                raise RuntimeError('Frozen base mismatch: ' + relative)
        for relative, expected in manifest['candidate_sha256'].items():
            file = override / relative
            if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Candidate source mismatch: ' + relative)
            shutil.copyfile(file, source / relative)
    if args.section_trace:
        file = source / 'vendor/cuda/xpbd/xpbd_solver.cu'
        text = file.read_text(encoding='utf8')
        old = 'extern "C" void xpbd_section_mark(const char* name){\n    if(!g_sectionProfile)return;'
        new = old + '\n    fprintf(stderr,"TRACE section waiting: %s\\n",name);fflush(stderr);\n' \
                    '    cudaError_t traceError=cudaDeviceSynchronize();\n' \
                    '    fprintf(stderr,"TRACE section done: %s status=%d\\n",name,(int)traceError);fflush(stderr);'
        assert text.count(old) == 1
        file.write_text(text.replace(old, new), encoding='utf8')
build = artifact / 'build'
binary = ROOT / 'endoscopic_tissue/native' / ('bin-acceptance-' + args.name)
assert binary.resolve().parent == (ROOT / 'endoscopic_tissue/native').resolve()
files = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in source.rglob('*') if p.is_file()}
(artifact / 'build-inputs.json').write_text(json.dumps(dict(base=args.base, files=files), indent=2), encoding='utf8')
vcvars = Path(r'D:\develop\Mocrosoft VS\Microsoft Visual Studio\2022 Comumnity\VC\Auxiliary\Build\vcvars64.bat')
nvcc = Path(r'D:\develop\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe')
script = artifact / 'build.cmd'
script.write_text('\n'.join(['@echo off', 'setlocal', f'call "{vcvars}"', 'if errorlevel 1 exit /b 1',
    f'cmake -S "{source.as_posix()}" -B "{build.as_posix()}" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="{nvcc.as_posix()}" -DENDO_BINARY_DIR="{binary.as_posix()}" -DCMAKE_CUDA_FLAGS=""',
    'if errorlevel 1 exit /b 1', f'cmake --build "{build.as_posix()}" --target LiverCudaSim',
    'exit /b %errorlevel%']) + '\n', encoding='utf8')
with (artifact / 'build.log').open('w', encoding='utf8') as log:
    result = subprocess.run([os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/c', str(script)],
                            stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    print((artifact / 'build.log').read_text(encoding='utf8', errors='replace')[-8000:])
    raise SystemExit(result.returncode)
dll = binary / 'LiverCudaSim.dll'
record = dict(dll=str(dll), sha256=hashlib.sha256(dll.read_bytes()).hexdigest(),
              source_files=files, diagnostic=args.section_trace, default_replaced=False, accepted=False)
(artifact / 'build-result.json').write_text(json.dumps(record, indent=2), encoding='utf8')
print(json.dumps({k: v for k, v in record.items() if k != 'source_files'}, indent=2))
