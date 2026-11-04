"""Run an expensive GPU job only after its explicit prerequisite evidence passes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--require', action='append', default=[], help='JSON path; its passed field must be true')
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    for name in args.require:
        path = Path(name)
        if not path.is_file():
            raise SystemExit('Prerequisite has not run: '+str(path))
        result = json.loads(path.read_text(encoding='utf8'))
        if result.get('passed') is not True:
            raise SystemExit('Prerequisite failed or is incomplete: '+str(path))
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        p.error('Missing worker script')
    raise SystemExit(subprocess.run([sys.executable, '-X', 'utf8', *command]).returncode)


if __name__ == '__main__':
    main()
