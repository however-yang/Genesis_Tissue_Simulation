"""Run one bounded diagnostic or acceptance job between owned jobs.

Only the exact, same-UID queue controller can be briefly suspended, while it
has no child and is waiting for a GPU. A finally block resumes it. If it has
already launched a child, resume immediately without running the probe.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

GPU = 'GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--controller-pid', type=int, required=True)
    p.add_argument('--command-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=240)
    args = p.parse_args()
    if not 1 <= args.timeout <= 7200:
        p.error('An owned boundary job must have a bounded timeout of 1..7200 seconds')
    queue = args.queue.resolve(strict=True)
    controller = str(queue/'run_batch.py')
    proc = Path('/proc')/str(args.controller_pid)
    assert proc.stat().st_uid == os.getuid(), 'Controller ownership mismatch'
    command = (proc/'cmdline').read_bytes().split(b'\0')
    assert controller.encode() in command and b'codex' not in b' '.join(command).lower()
    birth = (proc/'stat').read_text().split()[21]
    args.output.mkdir(parents=True, exist_ok=False)
    record = {'controller_pid': args.controller_pid, 'controller': controller,
              'gpu_uuid': GPU, 'phase': 'waiting_owned_queue_idle', 'existing_workers_stopped': False}
    def save():
        (args.output/'status.json').write_text(json.dumps(record, indent=2))
    def read_status():
        for _ in range(50):
            try:
                return json.loads((queue/'status.json').read_text())
            except json.JSONDecodeError:
                time.sleep(.02)
        raise RuntimeError('Could not read a complete controller status')
    def same_process():
        return (proc/'stat').read_text().split()[21] == birth and controller.encode() in (proc/'cmdline').read_bytes().split(b'\0')
    def idle_gpu():
        status = subprocess.check_output(['nvidia-smi', '-i', GPU,
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        memory, usage = [float(x.strip()) for x in status.strip().split(',')]
        tasks = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'], text=True)
        return memory < 256 and usage < 5 and GPU not in tasks
    suspended = False
    def interrupted(sig, frame):
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    started = time.time()
    try:
        save()
        while time.time()-started < 3600:
            assert same_process(), 'Controller exited or PID was reused'
            status = read_status()
            if status['phase'] == 'waiting_gpu':
                os.kill(args.controller_pid, signal.SIGSTOP)
                suspended = True
                # Waiting for the stopped state closes the race with its next
                # launch. Any child means this boundary is not available.
                for _ in range(100):
                    if (proc/'stat').read_text().split()[2] in ('T', 't'):
                        break
                    time.sleep(.01)
                state = read_status()
                children = (proc/'task'/str(args.controller_pid)/'children').read_text().strip()
                if state['phase'] == 'waiting_gpu' and not children and idle_gpu():
                    record.update(phase='running_probe_at_idle_boundary', boundary=state.get('current'))
                    save()
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
                    cmd = json.loads(args.command_file.read_text())
                    with (args.output/'probe.log').open('xb') as log:
                        result = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL,
                                                stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                    record.update(phase='probe_finished', exit_code=result.returncode)
                    return result.returncode
                os.kill(args.controller_pid, signal.SIGCONT)
                suspended = False
            if status['phase'] in ('complete', 'failed'):
                record.update(phase='not_run', reason='Queue ended; use a fresh serial job')
                return 2
            time.sleep(1)
        record.update(phase='not_run', reason='No idle boundary in one hour')
        return 2
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        if suspended and same_process():
            os.kill(args.controller_pid, signal.SIGCONT)
            record['controller_resumed'] = True
        record['wall_s'] = time.time()-started
        save()


if __name__ == '__main__':
    raise SystemExit(main())
