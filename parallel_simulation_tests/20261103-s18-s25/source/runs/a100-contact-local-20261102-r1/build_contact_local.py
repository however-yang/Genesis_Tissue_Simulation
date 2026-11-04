"""Build only the immutable R7 source package in its own server directory."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def main():
    root = Path('/srv/disk2/hryang/genesis-exec')
    here = Path(__file__).resolve().parent
    if here != root/'runs/a100-contact-local-20261102-r1':
        raise RuntimeError('Unexpected isolated build location')
    build = root/'builds/bf21a47-contact-local-r7-sm80'
    source = here/'source'
    if build.exists() or source.exists() or (here/'build-result.json').exists():
        raise FileExistsError('Do not replace prior builds or sources')
    manifest = json.loads((here/'manifest.json').read_text())
    archive = here/'source.tar'
    digest = lambda b: hashlib.sha256(b).hexdigest()
    assert digest(archive.read_bytes()) == manifest['archive_sha256']
    expected = {'source/'+x['path']:x for x in manifest['files']}
    with tarfile.open(archive) as tar:
        assert set(tar.getnames()) == set(expected)
        for member in tar.getmembers():
            assert member.isfile() and member.name in expected
            dest = here/member.name
            assert dest.resolve().is_relative_to(source)
            data = tar.extractfile(member).read()
            assert len(data) == expected[member.name]['bytes']
            assert digest(data) == expected[member.name]['sha256']
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as stream:
                stream.write(data)
    prefix = root/'envs/genesis-py311-cu125-v1'
    env = dict(os.environ, PATH=str(prefix/'bin')+':'+os.environ['PATH'],
               CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa')
    commands = [[str(prefix/'bin/cmake'), '-S', str(source), '-B', str(build), '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=/usr/bin/g++-12',
        '-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12',
        '-DCMAKE_CUDA_COMPILER='+str(prefix/'bin/nvcc'),
        '-DCUDAToolkit_ROOT='+str(prefix), '-DCMAKE_CUDA_ARCHITECTURES=80'],
        [str(prefix/'bin/cmake'), '--build', str(build), '--parallel', '2']]
    record = dict(started=time.time(), source_archive_sha256=manifest['archive_sha256'],
        commands=commands, default_replaced=False, training_started=False, completed=False)
    try:
        with (here/'build.log').open('xb') as log:
            for command in commands:
                subprocess.run(command, cwd=here, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        library = build/'lib/libLiverCudaSim.so'
        record.update(completed=True, library=str(library), library_sha256=digest(library.read_bytes()))
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        record['wall_s'] = time.time()-record['started']
        with (here/'build-result.json').open('x') as stream:
            json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
