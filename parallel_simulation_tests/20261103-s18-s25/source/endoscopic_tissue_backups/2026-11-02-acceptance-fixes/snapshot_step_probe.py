"""Localize a T3 failure; retain every continuation and never relax its gate."""
import argparse
from dataclasses import asdict
import ctypes
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'endoscopic_tissue_backups/2026-11-01-s15-s17'), str(ROOT)]
from validation_support import environment, command_sequence, exact_values
from endoscopic_tissue.training.data import atomic_json
from endoscopic_tissue.training.snapshot import SnapshotBackend, EnvSnapshot, binding
from endoscopic_tissue.training.provenance import json_hash


def difference(reference, current, grid):
    delta = np.linalg.norm(reference-current, axis=1)*1000
    index = int(np.argmax(delta))
    material = np.asarray(grid.inside, dtype=bool)
    return {'max_node_mm': float(delta[index]), 'node': index,
            'material_node': bool(grid.inside[index]), 'active_node': bool(grid.active[index]),
            'max_material_mm': float(delta[material].max()),
            'nodes_over_001mm': int(np.count_nonzero(delta >= .01)),
            'reference_position_m': reference[index].tolist(), 'current_position_m': current[index].tolist()}


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--dll', default='bin-acceptance-boundary-snapshot2')
    p.add_argument('--scene', default='flap')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--direct', action='store_true')
    p.add_argument('--snapshot', type=Path)
    args = p.parse_args()
    if args.repeats < 2:
        p.error('At least two repeats are required to measure restored-to-restored variation')
    out = ROOT/'endoscopic_tissue/outputs/acceptance-fixes-20261102'/args.name
    out.mkdir(parents=True, exist_ok=False)
    record = {'diagnostic_only': True, 'passed': False, 'candidate_adopted': False,
              'direct_section_profile': args.direct, 'repeats': [],
              'reference': 'fresh preparation with warm graphs' if not args.snapshot else 'loaded snapshot with cold graphs'}
    try:
        with environment(dll=args.dll, scene=args.scene, preparation='cut_ready') as env:
            _, context = env.reset(0)
            atomic_json(out/'context.json', context)
            atomic_json(out/'preparation.json', env.preparation)
            if args.snapshot:
                initial = EnvSnapshot.load(args.snapshot, dll_sha256=context['dll_sha256'],
                                           physics_config_sha256=json_hash(binding(env)))
                env.restore(initial)
            if args.direct:
                call = env.native.dll.LCS_ProfileSections
                call.argtypes = [ctypes.c_int]
                call.restype = None
            def advance(command):
                # Profiling events are intentionally outside the state ABI.
                # Toggle only around execution, leaving every save/load at its
                # ordinary completed boundary with profiling disabled.
                if args.direct:
                    call(1)
                try:
                    return env.replay_command(command)
                finally:
                    if args.direct:
                        call(0)
            stage = next(s for s in env.scene.stages if s.name == 'incise_1')
            prior = next(s.command for s in env.scene.stages if s.name == 'blade_approach')
            command = command_sequence([stage], prior)[0][2]
            atomic_json(out/'command.json', asdict(command))
            before = asdict(env.observed)
            snapshot = env.snapshot()
            snapshot.save(out/'parent.lcssnap')
            source = advance(command)
            assert source.after is not None
            reference = source.after.arrays['positions_m'].copy()
            atomic_json(out/'original-step.json', source.after.metadata)
            np.savez_compressed(out/'original-step.npz', positions_m=reference,
                                rest=env.grid.positions, material=env.grid.inside, active=env.grid.active)
            previous = []
            for repeat in range(args.repeats):
                env.restore(snapshot)
                assert SnapshotBackend(env.native).save() == snapshot.native_bytes()
                exact_values(before, asdict(env.native.observe()))
                result = advance(command)
                assert result.after is not None
                current = result.after.arrays['positions_m'].copy()
                item = {'repeat': repeat, 'T1_exact': True, 'T2_exact': True,
                        'vs_original': difference(reference, current, env.grid),
                        'vs_restored': [difference(x, current, env.grid) for x in previous],
                        'quality': result.after.metadata['quality']}
                record['repeats'].append(item)
                atomic_json(out/f'restored-{repeat:02}.json', result.after.metadata)
                np.savez_compressed(out/f'restored-{repeat:02}.npz', positions_m=current)
                previous.append(current)
                atomic_json(out/'result.json', record)
                print('repeat', repeat, item['vs_original'], flush=True)
            record['passed'] = all(x['vs_original']['max_node_mm'] < .01
                and x['quality']['checked_metrics_passed'] and x['quality']['within_declared_local_scope']
                for x in record['repeats'])
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        atomic_json(out/'result.json', record)


if __name__ == '__main__':
    main()
