"""Serial task-owned GPU runs; keep failures and stop dependent work honestly."""
import argparse
import json
from pathlib import Path
import sys
import time

import run_diagnostic as supervisor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('plan', type=Path)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf8'))
    output = args.plan.with_suffix('.results.json')
    if output.exists():
        raise FileExistsError(output)
    results = []
    for job in plan['jobs']:
        blocked = []
        for path in job.get('requires_pass', []):
            record = json.loads(Path(path).read_text(encoding='utf8')) if Path(path).exists() else {}
            if record.get('passed') is not True:
                blocked.append(path)
        if blocked:
            results.append(dict(name=job['name'], executed=False, blocked_by=blocked))
        else:
            started = time.time()
            sys.argv = ['run_diagnostic.py', '--name', job['name'], '--worker-script',
                        str(supervisor.HERE / job['worker']), *job.get('args', [])]
            if job.get('performance'):
                sys.argv.append('--performance')
            code = supervisor.main()
            results.append(dict(name=job['name'], executed=True, returncode=code,
                                elapsed_s=time.time()-started))
        supervisor.atomic(output, dict(jobs=results, completed=len(results)==len(plan['jobs'])))
        print('Batch progress', len(results), '/', len(plan['jobs']), job['name'], flush=True)


if __name__ == '__main__':
    main()
