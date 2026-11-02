"""Wait for free GPU, then supervise exactly one owned diagnostic subprocess."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'endoscopic_tissue/outputs/acceptance-fixes-20261102'
HIDDEN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def atomic(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf8')
    temp.replace(path)


def foreign_python():
    own = psutil.Process()
    ids = {own.pid, *(p.pid for p in own.children(recursive=True))}
    found = []
    for p in psutil.process_iter(['name', 'cmdline']):
        if p.pid not in ids and 'python' in (p.info['name'] or '').lower():
            found.append({'pid': p.pid, 'scripts': [Path(a).name for a in
                          (p.info['cmdline'] or []) if a.endswith('.py')]})
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--stall-seconds', type=float, default=120)
    parser.add_argument('--worker-script', type=Path, default=HERE / 'trace_damage.py')
    args, forwarded = parser.parse_known_args()
    OUT.mkdir(parents=True, exist_ok=True)
    record = OUT / (args.name + '-supervisor.json')
    if record.exists():
        raise FileExistsError(record)
    idle, last_report = 0, 0.
    while idle < 3:
        foreign = foreign_python()
        gpu = float(subprocess.check_output([
            'nvidia-smi', '--query-gpu=utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, creationflags=HIDDEN).strip())
        # Diagnostic tracing already changes timings. Permit ordinary desktop
        # rendering load; this runner must never supply performance acceptance.
        idle = idle + 1 if not foreign and gpu <= 35 else 0
        if time.time() - last_report > 30:
            atomic(record, {'phase': 'waiting', 'foreign': foreign, 'gpu': gpu})
            print('Waiting for idle GPU:', foreign, gpu, flush=True)
            last_report = time.time()
        time.sleep(5)
    command = [sys.executable, '-X', 'utf8', str(args.worker_script.resolve()),
               '--name', args.name, *forwarded]
    started = time.time()
    interference = []
    with (OUT / (args.name + '.log')).open('w', encoding='utf8') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, creationflags=HIDDEN)
        identity = psutil.Process(child.pid)
        data = {'phase': 'running', 'pid': child.pid, 'created': identity.create_time(),
                'command': command, 'started': started, 'user_processes_touched': False}
        data.update(performance_evidence=False, starting_gpu_percent=gpu)
        atomic(record, data)
        trace = OUT / (args.name + '-trace/events.jsonl')
        while child.poll() is None:
            foreign = foreign_python()
            if foreign:
                interference.append({'unix': time.time(), 'foreign': foreign})
            modified = trace.stat().st_mtime if trace.exists() else (OUT / (args.name + '.log')).stat().st_mtime
            if time.time() - modified > args.stall_seconds:
                # Popen is the owned child handle; never terminate a discovered PID.
                data['stalled_seconds'] = time.time() - modified
                data['trace_tail'] = trace.read_text(encoding='utf8')[-12000:] if trace.exists() else None
                child.terminate()
                child.wait(timeout=15)
                break
            time.sleep(2)
    data.update(phase='finished', returncode=child.returncode,
                wall_s=time.time() - started, interference=interference)
    atomic(record, data)
    print(json.dumps(data, indent=2), flush=True)
    return child.returncode


if __name__ == '__main__':
    raise SystemExit(main())
