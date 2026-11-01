"""Serial native endpoint checks; each failure is retained, never retried away."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from endoscopic_tissue.training.collect import local_bounds
from endoscopic_tissue.training.data import atomic_json, save_arrays
from endoscopic_tissue.training.env import EnvironmentConfig, NativeEnvironment, SceneConfig
from endoscopic_tissue.training.physics import PhysicsOverrides
from endoscopic_tissue.training.provenance import sha256

BASE = Path(__file__).parent
OUT = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/parameters'
DLL = ROOT/'endoscopic_tissue/native/bin-contact-groups31/LiverCudaSim.dll'
SHA = '4515f14aa15309162c2c54b6370792e128e63a7e8d3054c506da19e3aeb37d2f'
CASES = [('default', {}), ('empty', {})]
for field, values in {'young_pa': (3000., 24000.), 'start_force_n': (.2, .6),
                      'blade_friction': (.05, .4), 'grasp_friction': (.3, 1.2),
                      'tear_stretch_ratio': (0., 2.), 'tear_exempt_rings': (0, 2),
                      'grasp_core_break_force_n': (.1, 150.)}.items():
    CASES.extend((f'{field}-{i}', {field: value}) for i, value in enumerate(values))


@threadpool_limits.wrap(limits=1, user_api='blas')
def worker(index):
    name, overrides = CASES[index]
    directory = OUT/name
    directory.mkdir(parents=True, exist_ok=False)
    # Wall is the existing thin-wall pure-friction scene; wall-traction uses
    # assisted anchors by default. Do not invent a successful grip on failure.
    friction = 'grasp_friction' in overrides
    scene = SceneConfig(scene='wall' if friction else 'wall-traction', preparation='held',
                        contact_model=1 if friction else None,
                        physics=None if name == 'default' else PhysicsOverrides(**overrides))
    cfg = EnvironmentConfig(str(DLL), SHA, local_bounds(scene), scene, max_steps=8)
    row = {'name': name, 'config': asdict(cfg), 'native_preparation_passed': False,
           'quality_passed': False, 'exception': None}
    start = time.perf_counter()
    with NativeEnvironment(cfg) as env:
        try:
            state, context = env.reset(seed=0)
            atomic_json(directory/'context.json', context)
            atomic_json(directory/'preparation.json', env.preparation)
            save_arrays(directory/'initial.npz.zst', {**state.arrays, 'neighbors': state.topology['neighbors']})
            steps = []
            for _ in range(8):
                r = env.step(env.previous_action.copy())
                steps.append({'timings_ms': r.timings_ms, 'reason': r.reason,
                              'metadata': r.after.metadata if r.after is not None else None})
                if env.done:
                    break
            atomic_json(directory/'steps.json', steps)
            row.update(native_preparation_passed=True,
                       quality_passed=all(s['metadata'] is not None and
                         s['metadata']['quality']['checked_metrics_passed'] and
                         s['metadata']['quality']['within_declared_local_scope'] for s in steps),
                       reset_wall_time_ms=context['reset_wall_time_ms'],
                       material_nodes=context['material_node_count'],
                       physical_defaults_changed=context['physical_defaults_changed'],
                       overridden_fields=context['overridden_fields'],
                       native_step_mean_ms=float(np.mean([s['timings_ms']['native_step'] for s in steps])),
                       native_step_p95_ms=float(np.percentile([s['timings_ms']['native_step'] for s in steps],95)))
        except Exception as error:
            row['exception'] = repr(error)
            diagnostic = env.last_reset_diagnostic or {'error': repr(error)}
            if env.last_reset_failure_arrays is not None:
                diagnostic['arrays_sha256'] = save_arrays(directory/'reset-failure.npz.zst', env.last_reset_failure_arrays)
            atomic_json(directory/'failure.json', diagnostic)
    row['wall_seconds'] = time.perf_counter()-start
    atomic_json(directory/'result.json', row)
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    if args.worker is not None:
        return worker(args.worker)
    OUT.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((BASE/'baseline.json').read_text(encoding='utf8'))
    for p, expected in baseline['protected'].items():
        if sha256(ROOT/p) != expected:
            raise RuntimeError('Protected baseline changed: '+p)
    rows = []
    for i, (name, _) in enumerate(CASES):
        print(f'Running {i+1}/{len(CASES)}: {name}', flush=True)
        with (OUT/(name+'.log')).open('w', encoding='utf8') as log:
            run = subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--worker', str(i)],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        result_path = OUT/name/'result.json'
        row = json.loads(result_path.read_text(encoding='utf8')) if result_path.exists() else {
            'name': name, 'native_preparation_passed': False, 'quality_passed': False,
            'worker_exit': run.returncode, 'missing_result': True}
        rows.append(row)
        atomic_json(OUT/'summary.json', {'cases': rows, 'all_passed': all(
            r['native_preparation_passed'] and r['quality_passed'] for r in rows),
            'complete': len(rows) == len(CASES), 'physical_defaults_promoted': False})
        print(f"{name}: preparation={row['native_preparation_passed']}, quality={row['quality_passed']}", flush=True)


if __name__ == '__main__':
    main()
