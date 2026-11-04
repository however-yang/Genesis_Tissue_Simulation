"""Serial independent timing. Start only when capture/render/build jobs have finished."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
P = Path(__file__).resolve().parent
OUT = ROOT/'endoscopic_tissue/outputs/release-next-20261101'
BASE = ROOT/'endoscopic_tissue/native/bin-contact-groups31/LiverCudaSim.dll'
R21 = P/'package/artifacts/release-r21-fragment-submission/bin/LiverCudaSim.dll'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['controls', 'abba'], required=True)
    args = parser.parse_args()
    if args.suite == 'controls':
        jobs = [(f'dispatch-{i+1}-{mode}', R21, 450, mode) for i, mode in enumerate((0, 1, 1, 0))]
    else:
        jobs = [('time-base-a1', BASE, 1215, None), ('time-r21-b1', R21, 0, 1),
                ('time-r21-b2', R21, 0, 1), ('time-base-a2', BASE, 1215, None)]
    results = []
    for name, dll, end, mode in jobs:
        destination = OUT/name
        if destination.exists():
            raise FileExistsError(destination)
        command = [sys.executable, '-X', 'utf8', str(P/'time_dispatch.py'), '--case', 'kidney-reference',
                   '--dll', str(dll), '--output', str(destination)]
        if end:
            command += ['--end-step', str(end)]
        if mode is not None:
            command += ['--submission-graph', str(mode)]
        print('START', name, flush=True)
        started = time.time()
        with (P/(name+'.log')).open('w', encoding='utf8') as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        results.append(dict(name=name, command=command, returncode=completed.returncode,
                            elapsed_s=time.time()-started))
        (P/(args.suite+'-jobs.json')).write_text(json.dumps(results, indent=2)+'\n')
        print('END', name, completed.returncode, flush=True)
        if completed.returncode:
            raise RuntimeError('Timing job failed; inspect '+name+'.log')


if __name__ == '__main__':
    main()
