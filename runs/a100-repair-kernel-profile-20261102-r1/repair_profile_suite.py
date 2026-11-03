"""Profile three same-library states serially on the authorized A100."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path('/srv/disk2/hryang/genesis-exec')
    here = Path(__file__).resolve().parent
    assert here == root/'runs/a100-repair-kernel-profile-20261102-r1'
    release = root/'releases/0643e10991cf'
    outputs = release/'endoscopic_tissue/outputs/acceptance-fixes-20261102'
    cases = [
        ('r3', '57f184c-release-training-r3-sm80',
         outputs/'a100-combined-T4-regrasp-r1/regrasp-r00/first-failures/01-tick-975/state.lcssnap'),
        ('r6', '9651c46-safe-repair-r6-sm80',
         outputs/'a100-safe6-regrasp-video-r1/first-failures/01-tick-957/state.lcssnap'),
    ]
    r7 = outputs/'a100-local7-regrasp-video-r1'
    assert (r7/'result.json').is_file()
    failures = sorted((r7/'first-failures').glob('*/state.lcssnap'))
    cases.append(('r7', 'bf21a47-contact-local-r7-sm80', failures[0] if failures else r7/'fork.lcssnap'))
    records = []
    for label, build, snapshot in cases:
        args = [sys.executable, '-B', '-u', str(here/'profile_repair_kernels.py'),
                '--release', str(release), '--library-dir', str(root/'builds'/build/'lib'),
                '--snapshot', str(snapshot), '--output', str(here/label)]
        with (here/(label+'.log')).open('xb') as log:
            result = subprocess.run(args, cwd=release, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=240)
        record = dict(label=label, args=args, exit_code=result.returncode)
        records.append(record)
        (here/'progress.json').write_text(json.dumps(records, indent=2))
        if result.returncode:
            raise RuntimeError('Diagnostic failed; keep its report: '+label)
    with (here/'result.json').open('x') as stream:
        json.dump(dict(completed=True, diagnostics_only=True, cases=records), stream, indent=2)


if __name__ == '__main__':
    main()
