"""Run the unchanged S17a native endpoint worker with an explicit Linux library."""
import argparse
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--library-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.release), str(args.release/'endoscopic_tissue_backups/2026-11-01-s15-s17')]
    import validate_parameters as worker
    from endoscopic_tissue.training.provenance import sha256
    if not 0 <= args.index < len(worker.CASES):
        p.error('Unknown endpoint case')
    worker.OUT = args.output
    worker.DLL = args.library_dir/'LiverCudaSim.dll'
    worker.SHA = sha256(worker.DLL)
    worker.worker(args.index)


if __name__ == '__main__':
    main()
