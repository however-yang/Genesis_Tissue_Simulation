"""Isolated full regrasp and paired continuation using the existing S15 checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'endoscopic_tissue_backups/2026-11-01-s15-s17'
sys.path[:0] = [str(ROOT), str(OLD)]
import validate_snapshot_distribution as worker


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--repeat', type=int, default=0)
    p.add_argument('--capture', action='store_true')
    p.add_argument('--dll', default='bin-s15-training-snapshot1')
    p.add_argument('--fork', choices=['regrasp', 'before_contact', 'first_cut', 'squeezing'], default='regrasp')
    args = p.parse_args()
    args.cache = None
    out = ROOT / 'endoscopic_tissue/outputs/acceptance-fixes-20261102' / args.name
    out.mkdir(parents=True, exist_ok=False)
    original_step = worker.environment
    def environment(**kwargs):
        env = original_step(dll=args.dll, **kwargs)
        advance = env.replay_command
        def step(command):
            result = advance(command)
            if env.native.tick % 30 == 0:
                print('tick', env.native.tick, 'grips', env.observed.grip_points,
                      'jaw_load_n', env.observed.jaw_load_n, flush=True)
            return result
        env.replay_command = step
        return env
    worker.environment = environment
    source = Path(worker.__file__)
    (out / 'fixture-source.json').write_text(json.dumps(dict(source=str(source),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(), change='3 mm toward attached end'), indent=2), encoding='utf8')
    result = worker.paired_run(out, args)
    print(json.dumps(result), flush=True)
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
