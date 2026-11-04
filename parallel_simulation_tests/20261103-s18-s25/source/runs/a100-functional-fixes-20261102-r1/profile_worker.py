"""Run original validation workers with diagnostics, never a speed acceptance."""
import argparse
import cProfile
import ctypes as ct
import json
from pathlib import Path
import pstats
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--mode', choices=['training', 'sections'], required=True)
    args = p.parse_args()
    root = args.release.resolve()
    out = root / 'endoscopic_tissue/outputs/acceptance-fixes-20261102' / args.name
    diagnostic = Path(__file__).resolve().parent / args.name
    diagnostic.mkdir(exist_ok=False)
    sys.path[:0] = [str(root), str(root / 'endoscopic_tissue_backups/2026-11-01-s15-s17'),
                    str(root / 'endoscopic_tissue_backups/2026-11-01-release-next')]
    record = {'diagnostic_only': True, 'performance_accepted': False, 'completed': False}
    try:
        if args.mode == 'training':
            import validate_damage as worker
            worker.OUT = out.parent
            sys.argv = ['validate_damage', '--name', args.name, '--dll', str(args.library.parent), '--stop-after', 'hold']
            profiler = cProfile.Profile()
            try:
                profiler.runcall(worker.main)
            finally:
                profiler.dump_stats(str(diagnostic / 'training.prof'))
                with (diagnostic / 'profile.txt').open('w') as stream:
                    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(70)
        else:
            import profile_release as worker
            original = worker.NativeTissue
            class Guarded(original):
                def step(self, command):
                    observed = super().step(command)
                    material = self.grid.inside != 0
                    maximum = float(np.linalg.norm(observed.nodes[material] - self.grid.positions[material], axis=1).max() * 10)
                    if not np.isfinite(observed.nodes).all() or maximum > 60:
                        (diagnostic / 'failure.json').write_text(json.dumps({'tick': self.tick, 'max_displacement_mm': maximum}))
                        raise RuntimeError('Existing finite-state/60 mm material guard failed')
                    if self.tick % 30 == 0:
                        print('tick', self.tick, flush=True)
                    return observed
            worker.NativeTissue = Guarded
            sys.argv = ['profile_release', '--case', 'kidney-reference', '--dll', str(args.library),
                        '--output', str(out), '--end-step', '1246', '--profile-steps', '300', '600', '900', '1230', '1244', '1245']
            worker.main()
        record['completed'] = True
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        (diagnostic / 'status.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
