"""Apply unchanged S15 smoke/resource criteria to an explicitly selected DLL."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'endoscopic_tissue_backups/2026-11-01-s15-s17'), str(ROOT)]
import validate_environment_snapshot as worker


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--dll', default='bin-acceptance-boundary-snapshot2')
    p.add_argument('--scene', choices=['wall-traction', 'flap'], default='wall-traction')
    p.add_argument('--resources', type=int, default=0)
    p.add_argument('--vision', action='store_true')
    args = p.parse_args()
    worker.environment = partial(worker.environment, dll=args.dll)
    out = ROOT / 'endoscopic_tissue/outputs/acceptance-fixes-20261102' / args.name
    out.mkdir(parents=True, exist_ok=False)
    from cpu_topology import probe
    (out/'cpu-topology.json').write_text(json.dumps(probe(), indent=2), encoding='utf8')
    result = (worker.resources(out, args.scene, args.resources) if args.resources
              else worker.smoke(out, args.scene, args.vision))
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples', 'metadata', 'T6')}), flush=True)
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
