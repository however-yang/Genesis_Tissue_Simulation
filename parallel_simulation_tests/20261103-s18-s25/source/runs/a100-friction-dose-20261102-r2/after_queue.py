"""Start this isolated test queue only after its predecessor has completed.

No process is interrupted. A failed predecessor requires diagnosis and a new
explicit continuation, rather than silently bypassing the failure.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    args = parser.parse_args()
    previous = args.previous.resolve(strict=True)
    queue = args.queue.resolve(strict=True)
    if (queue/'status.json').exists() or (queue/'supervisor.json').exists():
        raise FileExistsError('This queue has already been started')
    state_path = queue/'dependency-status.json'
    def save(**extra):
        temp = queue/'dependency-status.tmp'
        temp.write_text(json.dumps(dict(previous=str(previous), queue=str(queue),
            updated=time.time(), controller_pid=os.getpid(), **extra), indent=2))
        temp.replace(state_path)
    save(phase='waiting_for_predecessor')
    while True:
        try:
            state = json.loads((previous/'status.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(5)
            continue
        if state['phase'] == 'failed':
            save(phase='blocked_by_failed_predecessor', error=state.get('error'))
            return 2
        if (previous/'continued-in.json').exists():
            save(phase='blocked_by_replaced_predecessor',
                 continuation=json.loads((previous/'continued-in.json').read_text()))
            return 2
        if state['phase'] == 'complete':
            break
        time.sleep(5)
    with (queue/'supervisor.log').open('xb') as log:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(queue/'run_batch.py')],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (queue/'supervisor.json').write_text(json.dumps(dict(pid=child.pid), indent=2))
    save(phase='queue_started', queue_controller_pid=child.pid)
    print(json.dumps(dict(queue=str(queue), pid=child.pid)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
