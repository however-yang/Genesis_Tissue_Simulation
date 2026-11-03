"""Isolated Linux acceptance entry points; preserve all existing criteria."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys


GPU = 'GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'


def selected_gpu_memory():
    # CUDA ordinal 0 here is physical A100 index 1. Never sample physical GPU 0.
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise RuntimeError('Acceptance must run on the authorized second A100')
    result = subprocess.check_output([
        'nvidia-smi', '-i', GPU, '--query-gpu=uuid,memory.used',
        '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(result) != 1:
        raise RuntimeError('Ambiguous GPU memory measurement')
    identity, memory = [part.strip() for part in result[0].split(',')]
    if identity != GPU:
        raise RuntimeError('Unexpected GPU identity')
    return int(memory) * 1024**2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--library-dir', required=True, type=Path)
    parser.add_argument('--mode', choices=['resources', 'damage'], required=True)
    parser.add_argument('--name', required=True)
    args, forwarded = parser.parse_known_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('name must be a new output directory name')
    root = args.release.resolve(strict=True)
    library = args.library_dir.resolve(strict=True)
    (library / 'LiverCudaSim.dll').resolve(strict=True)
    selected_gpu_memory()
    sys.path[:0] = [str(root), str(root / 'endoscopic_tissue_backups/2026-11-01-s15-s17')]
    from threadpoolctl import threadpool_limits
    from endoscopic_tissue.training.data import atomic_json
    from endoscopic_tissue.training.provenance import sha256
    out = root / 'endoscopic_tissue/outputs/acceptance-fixes-20261102' / args.name
    provenance = dict(worker_sha256=sha256(__file__), library_sha256=sha256(library / 'LiverCudaSim.dll'),
                      gpu_uuid=GPU, criteria_changed=False, default_replaced=False)
    with threadpool_limits(limits=1, user_api='blas'):
        if args.mode == 'resources':
            if forwarded:
                parser.error('The resource acceptance always runs the required 1000 cycles')
            import validate_environment_snapshot as worker
            worker.environment = partial(worker.environment, dll=str(library))
            worker.gpu_memory = selected_gpu_memory
            out.mkdir(parents=True, exist_ok=False)
            atomic_json(out / 'launch.json', provenance)
            result = worker.resources(out, 'wall-traction', 1000)
            print(json.dumps({k: v for k, v in result.items() if k not in ('samples', 'metadata')}), flush=True)
            return 0 if result['passed'] else 2
        import validate_damage as worker
        worker.OUT = out.parent
        sys.argv = ['validate_damage', '--name', args.name, '--dll', str(library), *forwarded]
        try:
            worker.main()
        except RuntimeError as error:
            # A recorded physical rejection may continue the serial experiment
            # matrix. Unexpected crashes and timeouts still stop the queue.
            result = out / 'result.json'
            terminal = out / 'terminal-failure.json'
            if (not str(error).startswith('Validation stopped at existing local-domain/quality failure:')
                    or not result.is_file() or not terminal.is_file()):
                raise
            record = json.loads(result.read_text())
            if record.get('passed') or not record.get('error'):
                raise
            print('REJECTED physical case; evidence retained:', str(error), flush=True)
            return 2
        finally:
            if out.is_dir():
                atomic_json(out / 'launch.json', provenance)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
