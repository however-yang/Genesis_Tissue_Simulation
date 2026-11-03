"""Run the original full S16 curve with fresh preparations in an isolated output.

No cached reset is used before the identical library has passed all of S15.
The original worker retains censored trajectories and its original pass rules.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--library-dir', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--direction', choices=['+x', '-x', '+y', '+z', '-z'], required=True)
    law = parser.add_mutually_exclusive_group(required=True)
    law.add_argument('--tear', type=float)
    law.add_argument('--force', type=float)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('name must identify a new directory')
    root = args.release.resolve(strict=True)
    library = args.library_dir.resolve(strict=True)
    sys.path[:0] = [str(root), str(root/'endoscopic_tissue_backups/2026-11-01-s15-s17')]
    import validate_failure_dose as worker
    from endoscopic_tissue.training.data import atomic_json
    from endoscopic_tissue.training.provenance import sha256
    from functional_worker import selected_gpu_memory, GPU
    selected_gpu_memory()
    worker.OUT = root/'endoscopic_tissue/outputs/acceptance-fixes-20261102'
    out = worker.OUT/args.name
    if out.exists():
        raise FileExistsError('Previous dose evidence must not be overwritten')
    option, value = ('--tear', args.tear) if args.tear is not None else ('--force', args.force)
    sys.argv = ['validate_failure_dose', '--name', args.name,
                '--direction='+args.direction, option, str(value), '--fresh',
                '--repeats', '10', '--max-dose', '4', '--dll', str(library)]
    try:
        worker.main()
        return 0
    finally:
        if out.exists():
            atomic_json(out/'launch.json', dict(gpu_uuid=GPU, fresh_preparations=True,
                required_trials=90, required_all_directions_trials=450,
                worker_sha256=sha256(worker.__file__), wrapper_sha256=sha256(__file__),
                library_sha256=sha256(library/'LiverCudaSim.dll'),
                criteria_changed=False, default_replaced=False))
            p = out/'result.json'
            if p.exists():
                result = json.loads(p.read_text())
                print(json.dumps(dict(name=args.name, passed=result.get('passed'),
                    completed_trials=len(result.get('trials', [])), required_trials=90)), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
