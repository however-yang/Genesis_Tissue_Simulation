"""Queue wrapper: render existing captures, never replay physics for video."""
import argparse
from pathlib import Path
import sys

import render_capture


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--case', required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    folder = root / 'endoscopic_tissue/outputs/acceptance-fixes-20261102' / args.case
    print('Render recorded case:', args.case, flush=True)
    sys.argv = ['render_capture.py', str(folder)]
    render_capture.main()


if __name__ == '__main__':
    main()
