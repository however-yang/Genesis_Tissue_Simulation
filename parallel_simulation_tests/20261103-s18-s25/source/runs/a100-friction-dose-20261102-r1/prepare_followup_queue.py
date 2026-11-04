"""Prepare an immutable GPU-1 test list; run on Ubuntu after upload.

This queues tests, not training, and starts no worker itself.
"""
import hashlib
import json
from pathlib import Path


ROOT = Path('/srv/disk2/hryang/genesis-exec')
RELEASE = ROOT/'releases/0643e10991cf'
OUT = RELEASE/'endoscopic_tissue/outputs/acceptance-fixes-20261102'
HERE = Path(__file__).resolve().parent
PYTHON = ROOT/'envs/genesis-py311-cu125-v1/bin/python'
BASE = ROOT/'builds/0643e10991cf-contact-groups31-sm80/lib'
R3 = ROOT/'builds/57f184c-release-training-r3-sm80/lib'
R5 = ROOT/'builds/c805c77-boundary-fans-r5-sm80/lib'
PREVIOUS = ROOT/'runs/a100-functional-remaining-20261102-r3'


def main():
    if HERE != ROOT/'runs/a100-friction-dose-20261102-r1':
        raise ValueError('Unexpected execution directory')
    if (HERE/'jobs.json').exists() or (HERE/'status.json').exists():
        raise FileExistsError('Never replace an existing queue')
    jobs = []
    def add(name, args, wall=2400, idle=600):
        if (OUT/name).exists():
            raise FileExistsError('Existing result: '+name)
        jobs.append(dict(name=name, args=[str(PYTHON), '-B', '-u', *map(str, args)],
                         max_wall=wall, max_idle=idle))
    strict = RELEASE/'endoscopic_tissue_backups/2026-10-31-interaction-fixes/validate_strict.py'
    render = RELEASE/'endoscopic_tissue_backups/2026-11-02-acceptance-fixes/render_capture.py'
    videos = []
    # Alternate the order for every repeat. Both variants get the unchanged
    # pure friction model, trajectory, material, grid and iteration budgets.
    for case in ['grasp-wall', 'grasp-flap', 'grasp-deeper']:
        for repeat in range(3):
            variants = [('base', BASE), ('fans5', R5)]
            if repeat % 2:
                variants.reverse()
            for variant, library in variants:
                name = f'a100-friction-{case}-{variant}-r{repeat}'
                args = [strict, '--case', case, '--dll', library/'libLiverCudaSim.so', '--output', OUT/name]
                if repeat == 0:
                    args += ['--capture', '--archive-capture']
                    videos.append(name)
                add(name, args)
    for name in videos:
        add('render-'+name, [render, OUT/name], 3600, 900)
    # First complete the two candidates that survived the initial nominal
    # screen. Rejected lower tear thresholds are not silently adopted, and
    # this list is not evidence that the other force thresholds were scanned.
    for law, value, label in [('tear', 2.0, 'tear200'), ('force', 0.2794878101, 'force150')]:
        for direction in ['+x', '-x', '+y', '+z', '-z']:
            suffix = direction.replace('+', 'plus').replace('-', 'minus')
            name = f'a100-s16-fresh-{label}-{suffix}-r1'
            add(name, [HERE/'dose_matrix.py', '--release', RELEASE, '--library-dir', R3,
                       '--name', name, '--direction='+direction, '--'+law, value], 10800, 600)
    code = (PREVIOUS/'run_batch.py').read_bytes()
    compile(code, str(HERE/'run_batch.py'), 'exec')
    (HERE/'run_batch.py').write_bytes(code)
    (HERE/'jobs.json').write_text(json.dumps(jobs, indent=2))
    manifest = dict(predecessor=str(PREVIOUS), jobs=len(jobs), gpu_uuid='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',
                    friction_pairs=9, full_friction_videos=6, dose_trials=900,
                    dose_uses_fresh_preparation=True, training_started=False,
                    candidates_adopted=False, default_replaced=False,
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in HERE.glob('*.py')},
                    scope='R5 friction comparison; R3 S16 tear=2 and force=1.5 nominal peak only. '
                          'All other candidates and S16d remain separately reported.')
    (HERE/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
