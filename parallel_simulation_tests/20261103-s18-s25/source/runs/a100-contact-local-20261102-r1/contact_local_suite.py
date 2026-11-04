"""Fresh R7 regrasp and complete R6/R7 movies, at one owned idle boundary.

Exit 2 is a recorded physical rejection, not a reason to omit its movie.
No training, default replacement, foreign process control, or cross-library
snapshot restore is performed here.
"""
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root = Path('/srv/disk2/hryang/genesis-exec')
    here = Path(__file__).resolve().parent
    assert here == root/'runs/a100-contact-local-20261102-r1'
    release = root/'releases/0643e10991cf'
    python = root/'envs/genesis-py311-cu125-v1/bin/python'
    fixture = release/'endoscopic_tissue_backups/2026-11-02-acceptance-fixes'
    outputs = release/'endoscopic_tissue/outputs/acceptance-fixes-20261102'
    library = root/'builds/bf21a47-contact-local-r7-sm80/lib'
    build = json.loads((here/'build-result.json').read_text())
    assert build['completed']
    name = 'a100-local7-regrasp-video-r1'
    assert not (outputs/name).exists(), 'Keep all previous trial evidence'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',
               CUDA_DEVICE_ORDER='PCI_BUS_ID', EGL_DEVICE_ID='1', PYOPENGL_PLATFORM='egl',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    record = dict(started=time.time(), library_sha256=build['library_sha256'],
                  candidate_adopted=False, training_started=False, stages=[])
    jobs = [('regrasp', [str(python), '-B', '-u', str(fixture/'regrasp_trial.py'),
                        '--name', name, '--repeat', '0', '--capture', '--dll', str(library)])]
    for case in ['a100-safe6-regrasp-video-r1', name]:
        for branch in ['original', 'restored']:
            target = outputs/case/branch
            assert not (target/'review').exists(), 'Do not overwrite review artifacts'
            jobs.append((case+'-'+branch, [str(python), '-B', '-u',
                         str(fixture/'render_capture.py'), str(target)]))
    for label, command in jobs:
        stage = dict(name=label, command=command, started=time.time())
        record['stages'].append(stage)
        (here/'suite-progress.json').write_text(json.dumps(record, indent=2))
        with (here/(label+'.log')).open('xb') as log:
            result = subprocess.run(command, cwd=release, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=600)
        stage.update(exit_code=result.returncode, wall_s=time.time()-stage['started'])
        if result.returncode not in ((0, 2) if label == 'regrasp' else (0,)):
            raise RuntimeError(f'{label} failed: {result.returncode}')
    record.update(completed=True, physical_passed=record['stages'][0]['exit_code'] == 0,
                  wall_s=time.time()-record['started'])
    with (here/'suite-result.json').open('x') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)
    return 0 if record['physical_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
