"""Supplement T4 with each fresh cutting gesture, retaining the original verdict."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads((args.matrix/'result.json').read_text())
    assert matrix['completed_pairs'] == matrix['required_pairs'] == 20
    paths = [Path(path) for path in matrix['results']]
    rows = [json.loads(path.read_text()) for path in paths]
    assert len(rows) == 20 and {row['repeat'] for row in rows} == set(range(20))
    assert {row['dll_sha256'] for row in rows} == {matrix['dll_sha256']}
    phases = sorted({phase for row in rows for arm in ('original', 'restored')
                     for phase in row[arm]['snips']})
    report = dict(dll_sha256=matrix['dll_sha256'], original_matrix_passed=matrix['passed'],
                  original_criteria_unchanged=True, supplemental_only=True, phases={},
                  source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    for phase in phases:
        metrics = {}
        for key, limit in [('delta_length_mm', .8), ('first_new_material_cut_step', 5),
                           ('max_deviation_rest_mm', .16)]:
            groups = {arm:[row[arm]['snips'].get(phase, {}).get(key) for row in rows]
                      for arm in ('original', 'restored')}
            missing = {arm:values.count(None) for arm, values in groups.items()}
            a, b = ([v for v in groups[arm] if v is not None] for arm in ('original', 'restored'))
            difference = abs(float(np.median(a))-float(np.median(b))) if a and b else None
            ks = ks_2samp(a, b) if a and b else None
            metrics[key] = dict(**groups, missing=missing, median_difference=difference, limit=limit,
                within_limit=not any(missing.values()) and difference is not None and difference <= limit,
                ks_statistic=None if ks is None else float(ks.statistic),
                ks_pvalue=None if ks is None else float(ks.pvalue))
        report['phases'][phase] = metrics
    report['all_gestures_within_original_noise_limits'] = all(
        metric['within_limit'] for values in report['phases'].values() for metric in values.values())
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({phase:{key:metric['median_difference'] for key,metric in values.items()}
                      for phase,values in report['phases'].items()}))


if __name__ == '__main__':
    main()
