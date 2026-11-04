"""R8 algebra, same-library snapshots and full regrasp, all on physical GPU1."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root = Path('/srv/disk2/hryang/genesis-exec')
    here = Path(__file__).resolve().parent
    assert here == root/'runs/a100-cone-repair-20261102-r1'
    release = root/'releases/0643e10991cf'
    python = root/'envs/genesis-py311-cu125-v1/bin/python'
    fixture = release/'endoscopic_tissue_backups/2026-11-02-acceptance-fixes'
    old = release/'endoscopic_tissue_backups/2026-11-01-s15-s17'
    outputs = release/'endoscopic_tissue/outputs/acceptance-fixes-20261102'
    library = root/'builds/1a22a9e-cone-repair-r8-sm80/lib'
    build = json.loads((here/'build-result.json').read_text())
    assert build['completed']
    assert hashlib.sha256((library/'libLiverCudaSim.so').read_bytes()).hexdigest() == build['library_sha256']
    name = 'a100-cone8-regrasp-video-r1'
    raw = 'a100-cone8-snapshot-raw-r1'
    assert not (outputs/name).exists() and not (outputs/raw).exists()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',
        CUDA_DEVICE_ORDER='PCI_BUS_ID', EGL_DEVICE_ID='1', PYOPENGL_PLATFORM='egl',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    record = dict(started=time.time(), library_sha256=build['library_sha256'],
        candidate_adopted=False, training_started=False, stages=[], completed=False)
    def run(label, arguments, timeout, allow_physical_failure=False):
        command = [str(python), '-B', '-u', *map(str, arguments)]
        stage = dict(name=label, command=command, started=time.time())
        record['stages'].append(stage)
        (here/'suite-progress.json').write_text(json.dumps(record, indent=2))
        with (here/(label+'.log')).open('xb') as log:
            result = subprocess.run(command, cwd=release, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        stage.update(exit_code=result.returncode, wall_s=time.time()-stage['started'])
        if result.returncode not in ((0, 2) if allow_physical_failure else (0,)):
            raise RuntimeError(f'{label} failed: {result.returncode}')
        return result.returncode
    try:
        run('algebra', [here/'verify_cone_gpu.py'], 240)
        run('snapshot-raw', [old/'validate_native_snapshot.py', '--name', raw,
            '--stage', 'blade_approach', '--dll', library, '--output-root', outputs], 600)
        code = run('regrasp', [fixture/'regrasp_trial.py', '--name', name, '--repeat', '0',
            '--capture', '--dll', library], 600, True)
        for branch in ['original', 'restored']:
            run('video-'+branch, [fixture/'render_capture.py', outputs/name/branch], 600)
        if code == 0:
            code = run('regrasp-matrix', [here/'snapshot_matrix.py', '--release', release,
                '--library-dir', library, '--fork', 'regrasp', '--output',
                outputs/'a100-cone8-T4-regrasp-r1'], 3600, True)
        record.update(completed=True, physical_passed=code == 0)
        return code
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        record['wall_s'] = time.time()-record['started']
        with (here/'suite-result.json').open('x') as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
