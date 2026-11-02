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


def cpu_inventory():
    result = {}
    for process in psutil.process_iter(['name', 'cpu_times']):
        try:
            times = process.info['cpu_times']
            if times is not None:
                result[process.pid] = (process.info['name'], times.user + times.system)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def load_sample(previous, previous_time):
    now = time.time()
    current = cpu_inventory()
    busy = []
    for pid, (name, seconds) in current.items():
        if pid in previous and name == previous[pid][0]:
            percent = 100 * max(0., seconds-previous[pid][1]) / (now-previous_time)
            if percent > 1 and name not in ('System Idle Process', 'Idle'):
                busy.append({'pid': pid, 'name': name, 'percent_of_one_cpu': percent})
    sample = {'unix': now, 'cpu_percent': psutil.cpu_percent(),
              'available_memory_bytes': psutil.virtual_memory().available,
              'top_cpu_processes': sorted(busy, key=lambda x: -x['percent_of_one_cpu'])[:8]}
    try:
        sample['gpu_csv'] = subprocess.check_output(['nvidia-smi',
            '--query-gpu=utilization.gpu,utilization.memory,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu',
            '--format=csv,noheader,nounits'], text=True, creationflags=HIDDEN, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        sample['gpu_query_error'] = repr(error)
    return sample, current, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--stall-seconds', type=float, default=120)
    parser.add_argument('--worker-script', type=Path, default=HERE / 'trace_damage.py')
    parser.add_argument('--performance', action='store_true')
    parser.add_argument('--cpu-affinity', help='Comma-separated logical CPUs, only for this owned child')
    args, forwarded = parser.parse_known_args()
    affinity = None
    if args.cpu_affinity is not None:
        try:
            affinity = sorted({int(x) for x in args.cpu_affinity.split(',')})
        except ValueError:
            parser.error('Invalid CPU affinity list')
        if not affinity or not set(affinity) <= set(psutil.Process().cpu_affinity()):
            parser.error('CPU affinity must be a nonempty subset of available CPUs')
    if args.performance and (args.worker_script.resolve() != (HERE / 'damage_trial.py').resolve()
                             or '--capture' in forwarded):
        parser.error('Timing requires the uninstrumented, no-recording full worker')
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
        idle = idle + 1 if not foreign and gpu <= (15 if args.performance else 35) else 0
        if time.time() - last_report > 30:
            atomic(record, {'phase': 'waiting', 'foreign': foreign, 'gpu': gpu})
            print('Waiting for idle GPU:', foreign, gpu, flush=True)
            last_report = time.time()
        time.sleep(5)
    command = [sys.executable, '-X', 'utf8', str(args.worker_script.resolve()),
               '--name', args.name, *forwarded]
    started = time.time()
    interference = []
    load_samples = []
    cpu_previous, cpu_time = cpu_inventory(), time.time()
    psutil.cpu_percent()
    with (OUT / (args.name + '.log')).open('w', encoding='utf8') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, creationflags=HIDDEN)
        identity = psutil.Process(child.pid)
        if affinity is not None:
            try:
                identity.cpu_affinity(affinity)
            except Exception:
                child.terminate()
                child.wait(timeout=15)
                raise
        data = {'phase': 'running', 'pid': child.pid, 'created': identity.create_time(),
                'command': command, 'started': started, 'user_processes_touched': False}
        data.update(performance_evidence=args.performance, starting_gpu_percent=gpu)
        data['cpu_affinity'] = identity.cpu_affinity()
        data['explicit_cpu_affinity'] = affinity is not None
        atomic(record, data)
        trace = OUT / (args.name + '-trace/events.jsonl')
        while child.poll() is None:
            foreign = foreign_python()
            if foreign:
                interference.append({'unix': time.time(), 'foreign': foreign})
            if args.performance and time.time()-cpu_time >= 10:
                sample, cpu_previous, cpu_time = load_sample(cpu_previous, cpu_time)
                load_samples.append(sample)
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
                wall_s=time.time() - started, interference=interference, system_load_samples=load_samples)
    atomic(record, data)
    print(json.dumps({k: v for k, v in data.items() if k != 'system_load_samples'}, indent=2), flush=True)
    return child.returncode


if __name__ == '__main__':
    raise SystemExit(main())
