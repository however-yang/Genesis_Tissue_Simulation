"""Same-state native repair comparison; diagnostic only, never S15 acceptance.

The optional candidate differs only in the repair execution graph. This tool
uses the raw native ABI to transplant its fixed parent's state, then checks
every saved byte and observed value before making exactly one repair call.
The production EnvSnapshot DLL/configuration guards remain untouched.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'endoscopic_tissue_backups/2026-11-01-s15-s17'), str(ROOT)]
from validation_support import environment, exact_values
from endoscopic_tissue.training.data import atomic_json
from endoscopic_tissue.training.snapshot import EnvSnapshot, SnapshotBackend, binding
from endoscopic_tissue.training.provenance import json_hash
from endoscopic_tissue.instrument_geometry import load_scissors, scissors_body_depths

EXPECTED_PARENT = '14c0b76eefc893126fe6d858af197949e142a8e6964dde28476c7e0992ad5957'
EXPECTED_CANDIDATE = 'f26280bc11284fac6feb2b6c21b69aebb74a40c45aa3ec78ee27349d6660fd48'
FAILURE = ROOT/'endoscopic_tissue/outputs/acceptance-fixes-20261102/regrasp-boundary-forensics-r1/first-failures/01-tick-954/state.lcssnap'


def inspect(native, folder):
    observed, surface = native.observe(), native.surface()
    s = observed.scissors
    depths = scissors_body_depths(surface.vertices, surface.triangles, load_scissors(),
                                  s.hinge, s.axis, s.normal, s.gap)
    np.savez_compressed(folder.with_suffix('.npz'), nodes=observed.nodes,
                        vertices=surface.vertices, triangles=surface.triangles,
                        anchors=surface.anchors, neighbors=observed.neighbors,
                        grip_flags=observed.grip_flags)
    result = {'body_depth_mm': float(depths[0]*10), 'wrist_depth_mm': float(depths[1]*10),
              'native_all_contact_depth_mm': float(native.contact_depth_m()*1000),
              'material_quality': native.material_quality().tolist(),
              'scissors_gap_mm': float(s.gap*10), 'scissors_hinge_cm': s.hinge.tolist(),
              'grasper_hinge_cm': observed.hinge.tolist(), 'grip_points': observed.grip_points}
    atomic_json(folder.with_suffix('.json'), result)
    return observed, result


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--dll', choices=['bin-acceptance-boundary-snapshot2',
                                    'bin-acceptance-repair-feasible-snapshot3'], required=True)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    out = ROOT/'endoscopic_tissue/outputs/acceptance-fixes-20261102'/args.name
    out.mkdir(parents=True, exist_ok=False)
    record = {'diagnostic_only': True, 'passed': False, 'candidate_adopted': False,
              'same_dll_S15_acceptance': False, 'snapshot': str(FAILURE), 'repeats': []}
    try:
        with environment(dll=args.dll, preparation='settled') as env:
            _, context = env.reset(0)
            target = context['dll_sha256']
            expected = EXPECTED_PARENT if args.dll.endswith('boundary-snapshot2') else EXPECTED_CANDIDATE
            if target != expected:
                raise ValueError('Unreviewed diagnostic DLL')
            snap = EnvSnapshot.load(FAILURE, dll_sha256=EXPECTED_PARENT,
                                    physics_config_sha256=json_hash(binding(env)))
            record.update(source_dll_sha256=EXPECTED_PARENT, target_dll_sha256=target,
                          cross_version_diagnostic=target != EXPECTED_PARENT, snapshot_sha256=snap.sha256)
            data, raw = snap.python_state(), snap.native_bytes()
            backend = SnapshotBackend(env.native)
            for repeat in range(args.repeats):
                backend.load(raw)
                if backend.save() != raw:
                    raise ValueError('Raw native state changed at diagnostic import')
                env.native.tick = data['native']['tick']
                env.native.neighbors = data['native']['neighbors']
                env.native.topology_revision = data['native']['topology_revision']
                exact_values(asdict(data['observed']), asdict(env.native.observe()))
                before, bm = inspect(env.native, out/f'{repeat:02}-before')
                started = time.perf_counter()
                env.native.check('LCS_RefineContactSurface', env.settings.contact_refine_passes)
                milliseconds = (time.perf_counter()-started)*1000
                after, am = inspect(env.native, out/f'{repeat:02}-after')
                item = {'repeat': repeat, 'import_bytes_and_observation_exact': True,
                        'before': bm, 'after': am, 'diagnostic_repair_ms': milliseconds,
                        'max_node_motion_mm': float(np.linalg.norm(after.nodes-before.nodes, axis=1).max()*10)}
                record['repeats'].append(item)
                atomic_json(out/'result.json', record)
                print(json.dumps(item), flush=True)
            record['completed'] = True
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        atomic_json(out/'result.json', record)


if __name__ == '__main__':
    main()
