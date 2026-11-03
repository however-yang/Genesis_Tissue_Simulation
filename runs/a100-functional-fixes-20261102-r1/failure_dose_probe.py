"""Fresh high-dose diagnostics before the required full S16 dose matrix."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--library-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    law = parser.add_mutually_exclusive_group(required=True)
    law.add_argument('--force', type=float)
    law.add_argument('--tear', type=float)
    args = parser.parse_args()
    root = args.release.resolve(strict=True)
    sys.path[:0] = [str(root), str(root/'endoscopic_tissue_backups/2026-11-01-s15-s17')]
    from threadpoolctl import threadpool_limits
    from validation_support import environment
    from validate_failure_dose import one_dose, DIRECTIONS
    from endoscopic_tissue.training.physics import PhysicsOverrides
    from endoscopic_tissue.training.data import atomic_json
    from endoscopic_tissue.training.provenance import sha256
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(diagnostic_only=True, passed=False, full_required_matrix=False,
                  required_matrix='5 directions x 9 doses x 10 independent repeats',
                  actual_matrix='5 directions x 4 mm x 1 fresh preparation',
                  library_sha256=sha256(args.library_dir/'LiverCudaSim.dll'),
                  source_sha256=sha256(__file__), trials=[], default_replaced=False)
    physics = PhysicsOverrides(tear_stretch_ratio=args.tear, grasp_core_break_force_n=args.force)
    try:
        with threadpool_limits(limits=1, user_api='blas'):
            for direction, vector in DIRECTIONS.items():
                folder = args.output / direction.replace('+', 'plus').replace('-', 'minus')
                folder.mkdir(exist_ok=False)
                with environment(dll=str(args.library_dir), preparation='cut_ready',
                                 physics=physics, telemetry=True) as env:
                    env.reset(0)
                    atomic_json(folder/'context.json', env.context)
                    trial = one_dose(env, 4., vector)
                    trial['direction_name'] = direction
                    trial['preparation_failure_steps'] = sum(
                        not row['quality']['checked_metrics_passed']
                        or not row['quality']['within_declared_local_scope'] for row in env.preparation)
                    atomic_json(folder/'steps.json', trial.pop('rows'))
                    atomic_json(folder/'result.json', trial)
                    if trial['quality']['quality_failure_steps'] or trial['terminated_at_existing_limit']:
                        env.snapshot().save(folder/'rejected-state.lcssnap')
                        np.savez_compressed(folder/'rejected-state.npz', nodes=env.observed.nodes,
                                            neighbors=env.observed.neighbors, rest=env.grid.positions,
                                            material=env.grid.inside, active=env.grid.active)
                    result['trials'].append(trial)
                    atomic_json(args.output/'result.json', result)
                    print(direction, json.dumps({k: trial[k] for k in
                        ('quality', 'completed_requested_steps', 'initial_counts', 'final_failure_physics')}), flush=True)
        result['completed_diagnostic'] = True
        return 0
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        atomic_json(args.output/'result.json', result)


if __name__ == '__main__':
    raise SystemExit(main())
