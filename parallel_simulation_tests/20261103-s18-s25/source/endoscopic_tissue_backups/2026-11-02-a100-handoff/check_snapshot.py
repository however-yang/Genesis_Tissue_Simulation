"""Run existing S15 criteria with an explicit Linux build, without Windows probes."""
import argparse
from functools import partial
import json
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'endoscopic_tissue_backups/2026-11-01-s15-s17'), str(ROOT)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--library-dir', type=Path, required=True)
    p.add_argument('--scene', choices=['wall-traction', 'flap'], default='wall-traction')
    p.add_argument('--resources', type=int, default=0)
    p.add_argument('--vision', action='store_true')
    args = p.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in ('.', '..'):
        p.error('--name must be one new directory name')
    if args.resources < 0:
        p.error('--resources cannot be negative')
    library_dir = args.library_dir.resolve(strict=True)
    library = (library_dir / 'LiverCudaSim.dll').resolve(strict=True)
    from threadpoolctl import threadpool_limits
    import validate_environment_snapshot as worker
    from endoscopic_tissue.training.provenance import sha256
    worker.environment = partial(worker.environment, dll=str(library_dir))
    out = ROOT / 'endoscopic_tissue/outputs/acceptance-fixes-20261102' / args.name
    out.mkdir(parents=True, exist_ok=False)
    (out / 'launch.json').write_text(json.dumps(dict(
        platform=platform.platform(), python=sys.version, library=str(library),
        library_sha256=sha256(library), worker=str(Path(worker.__file__).resolve()),
        worker_sha256=sha256(worker.__file__), criteria_changed=False,
        default_replaced=False), indent=2) + '\n', encoding='utf8')
    with threadpool_limits(limits=1, user_api='blas'):
        result = (worker.resources(out, args.scene, args.resources) if args.resources
                  else worker.smoke(out, args.scene, args.vision))
    print(json.dumps({k: v for k, v in result.items() if k not in ('samples', 'metadata', 'T6')}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
