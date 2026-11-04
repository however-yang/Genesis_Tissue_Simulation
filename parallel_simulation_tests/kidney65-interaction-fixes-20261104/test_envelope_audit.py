"""CPU regressions using a saved real failure, including deliberately bad evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/python-gpu-check'))
from endoscopic_tissue import run, scenario
from scene_config import configure
from envelope_audit import certify


def main():
    configure(run, 'frontal65', 1.25)
    grid = run.SCISSORS_SCENES['kidney'].build_grid(run.scissors_settings().density_kg_m3)
    path = ROOT / 'fixtures/air-release-r1a.npz'
    provenance=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==provenance['fixture_sha256']
    fixture=dict(np.load(path))
    neighbors=fixture['neighbors']
    evidence={k.removeprefix('evidence_'):v for k,v in fixture.items() if k.startswith('evidence_')}
    original=neighbors.copy()
    sequence = object.__new__(scenario.Sequence)
    sequence.initial_neighbors, sequence.rest_positions, sequence.inside = grid.neighbors, grid.positions, grid.inside != 0
    count = sequence.stray_skin_cuts
    raw = count(neighbors)
    cert, diagnostic = certify(grid, neighbors, evidence)
    assert raw == 14 and count(diagnostic) == 0 and not cert['invalid_release_edges']
    assert np.array_equal(neighbors, original)
    tests = ['real_air_releases_verified_without_mutation']
    missing = deepcopy(evidence); missing['flags'][:] = 0
    _, diagnostic = certify(grid, neighbors, missing)
    assert count(diagnostic) == raw
    tests.append('unmarked_far_cuts_still_fail')
    fake = deepcopy(evidence)
    first = cert['certified_edge_ids'][0]
    fake['points'].view('<i4').reshape(-1, 8)[0, 4] = first
    invalid, _ = certify(grid, neighbors, fake)
    assert first in invalid['invalid_release_edges']
    tests.append('actual_wound_samples_cannot_be_exempted')
    bad_grid = deepcopy(grid)
    axis, x, y, z = grid.edges[first]
    dims = evidence['counts'][:3]+1
    a = x + dims[0]*(y+dims[1]*z)
    bad_grid.inside[a] = 1
    invalid, _ = certify(bad_grid, neighbors, evidence)
    assert first in invalid['invalid_release_edges']
    tests.append('material_endpoint_release_cannot_be_exempted')
    bad_grid = deepcopy(grid); bad_grid.edges = bad_grid.edges[::-1].copy()
    try:
        certify(bad_grid, neighbors, evidence)
    except ValueError:
        tests.append('mismatched_grid_rejected')
    else:
        raise AssertionError('mismatched grid accepted')
    result = dict(passed=True, checks=tests, raw_stray_count=raw,
                  certified_releases=cert['certified_air_releases'])
    (ROOT/'results/envelope-negative-tests.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
