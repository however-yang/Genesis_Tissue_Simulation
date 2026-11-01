"""Serial matched controls plus baseline brackets, with process provenance."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[2]
P = Path(__file__).resolve().parent
OUT = ROOT/'endoscopic_tissue/outputs/release-next-20261101'
BASE = ROOT/'endoscopic_tissue/native/bin-contact-groups31/LiverCudaSim.dll'


def inventory():
    return [dict(pid=p.pid, scripts=[Path(x).name for x in (p.info['cmdline'] or [])
                                    if x.lower().endswith('.py')],
                 created_utc=datetime.fromtimestamp(p.create_time(), timezone.utc).isoformat())
            for p in psutil.process_iter(['name', 'cmdline'])
            if 'python' in (p.info['name'] or '').lower()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', choices=['r22', 'r23', 'r24', 'r25'], default='r22')
    parser.add_argument('--include-r24-screen', action='store_true',
                        help='One R24 screen after the first baseline, before the corrected R25 ABBA.')
    args = parser.parse_args()
    revision = args.revision
    variant = {'r22': 'release-r22-repair-empty-manifold', 'r23': 'release-r23-repair-prelude-proof',
               'r24': 'release-r24-material-boundary',
               'r25': 'release-r25-boundary-contact-identity'}[revision]
    script = 'time_boundary.py' if revision in ('r24', 'r25') else 'time_repair.py'
    control_flag = '--boundary-graph' if revision in ('r24', 'r25') else '--repair-empty'
    candidate = P/'package/artifacts'/variant/'bin/LiverCudaSim.dll'
    jobs = [(revision+'-time-base-a1', BASE, 1215, None),
            (revision+'-control-off1', candidate, 0, 0), (revision+'-control-on1', candidate, 0, 1),
            (revision+'-control-on2', candidate, 0, 1), (revision+'-control-off2', candidate, 0, 0),
            (revision+'-time-base-a2', BASE, 1215, None)]
    if args.include_r24_screen:
        assert revision == 'r25'
        r24 = P/'package/artifacts/release-r24-material-boundary/bin/LiverCudaSim.dll'
        jobs.insert(1, ('r24-control-on-screen', r24, 0, 1))
    results = []
    for name, dll, end, mode in jobs:
        before = inventory()
        foreign = [p for p in before if p['pid'] != os.getpid()]
        if foreign:
            raise RuntimeError('Refusing timing with another Python process: '+json.dumps(foreign))
        destination = OUT/name
        if destination.exists():
            raise FileExistsError(destination)
        command = [sys.executable, '-X', 'utf8', str(P/script), '--case', 'kidney-reference',
                   '--dll', str(dll), '--output', str(destination)]
        if end:
            command += ['--end-step', str(end)]
        if mode is not None:
            command += [control_flag, str(mode)]
        started = datetime.now(timezone.utc).isoformat()
        telemetry = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu',
                                    '--format=csv,noheader'], capture_output=True, text=True)
        print('START', name, started, flush=True)
        begin = time.perf_counter()
        with (P/(name+'.log')).open('w', encoding='utf8') as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        results.append(dict(name=name, command=command, returncode=completed.returncode,
                            started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                            elapsed_s=time.perf_counter()-begin, processes_before=before,
                            processes_after=inventory(), gpu_before=telemetry.stdout.strip()))
        (P/(revision+'-timing-jobs.json')).write_text(json.dumps(results, indent=2)+'\n')
        print('END', name, completed.returncode, flush=True)
        if completed.returncode:
            raise RuntimeError('Timing job failed; inspect '+name+'.log')


if __name__ == '__main__':
    main()
