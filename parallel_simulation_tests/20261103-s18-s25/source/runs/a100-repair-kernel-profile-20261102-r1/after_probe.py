"""Wait for the previous owned probe to resume its controller before chaining.

Only then start the existing idle-boundary scheduler. Two schedulers never
suspend the same controller concurrently; no simulation worker is stopped.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--here', type=Path, required=True)
    args = p.parse_args()
    root = Path('/srv/disk2/hryang/genesis-exec')
    previous = args.previous.resolve(strict=True)
    here = args.here.resolve(strict=True)
    assert previous.is_relative_to(root/'runs') and here.is_relative_to(root/'runs')
    assert previous.stat().st_uid == here.stat().st_uid == os.getuid()
    status_path = here/'dependency-status.json'
    if status_path.exists():
        raise FileExistsError('Do not replace previous scheduling evidence')
    record = dict(phase='waiting_for_previous_probe', previous=str(previous), started=time.time())
    def save():
        status_path.write_text(json.dumps(record, indent=2))
    save()
    for _ in range(3600):
        try:
            prior = json.loads((previous/'status.json').read_text())
        except json.JSONDecodeError:
            time.sleep(1)
            continue
        if 'error' in prior or prior['phase'] == 'not_run':
            record.update(phase='blocked_by_failed_probe', prior=prior)
            save()
            return 2
        if prior['phase'] == 'probe_finished' and prior.get('controller_resumed'):
            assert prior['exit_code'] in (0, 2)
            command = json.loads((here/'scheduler-command.json').read_text())
            record.update(phase='starting_idle_scheduler', command=command)
            save()
            result = subprocess.run(command, stdin=subprocess.DEVNULL, timeout=4600)
            record.update(phase='finished', exit_code=result.returncode)
            save()
            return result.returncode
        time.sleep(1)
    record.update(phase='not_run', reason='Preceding probe did not finish within one hour')
    save()
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
