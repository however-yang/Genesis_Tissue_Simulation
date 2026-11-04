"""Transfer only this turn's idle controller to a corrected continuation.

Completed workers and their output directories are retained. No simulation
worker is stopped. The replacement changes timeout field spelling only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--next', dest='next_queue', type=Path, required=True)
    p.add_argument('--controller-pid', type=int, required=True)
    p.add_argument('--after-status', type=Path, required=True)
    p.add_argument('--insert-jobs', type=Path)
    args = p.parse_args()
    previous = args.previous.resolve(strict=True)
    assert not args.next_queue.exists(), 'Never overwrite another queue'
    proc = Path('/proc')/str(args.controller_pid)
    assert proc.stat().st_uid == os.getuid()
    expected = str(previous/'run_batch.py').encode()
    assert expected in (proc/'cmdline').read_bytes().split(b'\0')
    birth = (proc/'stat').read_text().split()[21]
    jobs = json.loads((previous/'jobs.json').read_text())
    suspended, replaced = False, False
    def resume():
        if proc.exists() and (proc/'stat').read_text().split()[21] == birth:
            os.kill(args.controller_pid, signal.SIGCONT)
    def interrupted(sig, frame):
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    started = time.time()
    try:
        while time.time()-started < 3600:
            try:
                dependency = json.loads(args.after_status.read_text())
                state = json.loads((previous/'status.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(1)
                continue
            if dependency.get('phase') != 'probe_finished' or not dependency.get('controller_resumed'):
                time.sleep(1)
                continue
            assert (proc/'stat').read_text().split()[21] == birth
            if state['phase'] in ('complete', 'failed'):
                raise RuntimeError('Queue ended; inspect its final state before any continuation')
            if state['phase'] != 'waiting_gpu':
                time.sleep(1)
                continue
            os.kill(args.controller_pid, signal.SIGSTOP)
            suspended = True
            for _ in range(100):
                if (proc/'stat').read_text().split()[2] in ('T', 't'):
                    break
                time.sleep(.01)
            state = json.loads((previous/'status.json').read_text())
            children = (proc/'task'/str(args.controller_pid)/'children').read_text().strip()
            if state['phase'] != 'waiting_gpu' or children:
                resume(); suspended = False
                time.sleep(1)
                continue
            completed = state['jobs']
            assert all(j.get('exit_code') in (0, 2) and 'finished' in j for j in completed)
            assert [j['name'] for j in completed] == [j['name'] for j in jobs[:len(completed)]]
            remaining = jobs[len(completed):]
            assert remaining and remaining[0]['name'] == state['current']
            corrections = []
            for job in remaining:
                for old, new in [('max_wall_s', 'max_wall'), ('max_idle_s', 'max_idle')]:
                    if old in job:
                        assert new not in job
                        job[new] = job.pop(old)
                        corrections.append(dict(job=job['name'], field=new, value=job[new]))
            inserted = json.loads(args.insert_jobs.read_text()) if args.insert_jobs else []
            names = [j['name'] for j in inserted+remaining]
            assert len(names) == len(set(names))
            args.next_queue.mkdir(exist_ok=False)
            (args.next_queue/'jobs.json').write_text(json.dumps(inserted+remaining, indent=2))
            code = (previous/'run_batch.py').read_bytes()
            compile(code, str(args.next_queue/'run_batch.py'), 'exec')
            (args.next_queue/'run_batch.py').write_bytes(code)
            record = dict(previous=str(previous), old_controller=args.controller_pid,
                completed_jobs=[j['name'] for j in completed], corrections=corrections,
                inserted_jobs=[j['name'] for j in inserted], remaining_jobs=len(remaining),
                controller_source_sha256=hashlib.sha256(code).hexdigest(),
                no_worker_stopped=True, old_outputs_preserved=True, time=time.time())
            (args.next_queue/'continuation.json').write_text(json.dumps(record, indent=2))
            # The controller is stopped, has no children, and has not begun
            # the next worker. End only that owned controller, then resume the
            # exact pending jobs with the corrected limits in a fresh queue.
            os.kill(args.controller_pid, signal.SIGTERM)
            resume(); suspended = False; replaced = True
            (previous/'continued-in.json').write_text(json.dumps({'queue': str(args.next_queue), 'time': time.time()}, indent=2))
            with (args.next_queue/'supervisor.log').open('xb') as log:
                child = subprocess.Popen([sys.executable, '-B', '-u', str(args.next_queue/'run_batch.py')],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (args.next_queue/'supervisor.json').write_text(json.dumps({'pid': child.pid}, indent=2))
            print(json.dumps({'next_queue': str(args.next_queue), 'pid': child.pid, **record}), flush=True)
            return 0
        raise TimeoutError('No safe idle boundary; original controller remains active')
    finally:
        if suspended and not replaced:
            resume()


if __name__ == '__main__':
    raise SystemExit(main())
