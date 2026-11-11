"""Rebuild an isolated experimental DLL from the pinned Git baseline and audited overrides."""
import argparse, hashlib, json, os, shutil, subprocess
from pathlib import Path
P = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['release-r16', 'release-r17', 'vertex-r18', 'baseline-rim-trace2', 'interaction-r13-trace', 'release-r19-volume-gate'], required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=P, text=True).strip())
    manifest = json.loads((P/(args.variant+'.json')).read_text(encoding='utf8'))
    base, prefix = manifest['base_commit'], manifest['base_source_prefix']
    directory = P/'artifacts'/args.variant
    source, build, binary = directory/'source', directory/'build', directory/'bin'
    source.mkdir(parents=True, exist_ok=True)
    assert source.resolve().is_relative_to((P/'artifacts').resolve())
    paths = subprocess.check_output(['git', 'ls-tree', '-rz', '--name-only', base, '--', prefix+'vendor'], cwd=root).split(b'\0')
    written = {}
    for raw in paths:
        if not raw:
            continue
        name = raw.decode('utf8')
        relative = name[len(prefix):]
        assert name.startswith(prefix) and '..' not in Path(relative).parts
        data = subprocess.check_output(['git', 'show', base+':'+name], cwd=root)
        target = source/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written[relative] = hashlib.sha256(data).hexdigest()
    for item in manifest['files']:
        relative = item['path']
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        source_variant = item.get('source_variant', args.variant)
        if source_variant not in ('release-r16', 'release-r17', 'vertex-r18', 'baseline-rim-trace2', 'interaction-r13-trace', 'release-r19-volume-gate'):
            raise RuntimeError('Unknown override source variant')
        original = P/'overrides'/source_variant/relative
        if digest(original) != item['sha256']:
            raise RuntimeError('Override hash mismatch: '+relative)
        target = source/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        written[relative] = item['sha256']
    shutil.copyfile(P/'CMakeLists.txt', source/'CMakeLists.txt')
    vcvars = Path(r'D:\develop\Mocrosoft VS\Microsoft Visual Studio\2022 Comumnity\VC\Auxiliary\Build\vcvars64.bat')
    nvcc = Path(r'D:\develop\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe')
    if not vcvars.is_file() or not nvcc.is_file():
        raise RuntimeError('This recipe requires the recorded MSVC 2022 and CUDA 12.5 installations; adjust only these tool paths for another machine.')
    lines = ['@echo off', 'setlocal', f'call "{vcvars}"', 'if errorlevel 1 exit /b 1',
        f'cmake -S "{source.as_posix()}" -B "{build.as_posix()}" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="{nvcc.as_posix()}" -DENDO_BINARY_DIR="{binary.as_posix()}" -DCMAKE_CUDA_FLAGS=""',
        'if errorlevel 1 exit /b 1', f'cmake --build "{build.as_posix()}" --target LiverCudaSim', 'exit /b %errorlevel%']
    script = directory/'build.cmd'
    script.write_text('\n'.join(lines)+'\n', encoding='utf8')
    (directory/'build-inputs.json').write_text(json.dumps(dict(base_commit=base, files=written), indent=2), encoding='utf8')
    print('Prepared', source, flush=True)
    if args.prepare_only:
        return
    with (directory/'build.log').open('w', encoding='utf8') as log:
        subprocess.run([os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/c', str(script)], stdout=log, stderr=subprocess.STDOUT, check=True)
    dll = binary/'LiverCudaSim.dll'
    result = dict(dll=str(dll), sha256=digest(dll), base_commit=base, experimental=True, default_replaced=False)
    (directory/'build-result.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
