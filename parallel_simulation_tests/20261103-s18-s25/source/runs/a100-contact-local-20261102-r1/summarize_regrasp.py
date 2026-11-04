"""Read-only, phase-aligned diagnostics; these are not controlled FPS trials."""
import argparse
import json
from pathlib import Path

import numpy as np


def distribution(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('Missing/nonfinite timing data must not be silently dropped')
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), p99=float(np.percentile(values, 99)),
                maximum=float(values.max()))


def summarize(case):
    result = json.loads((case/'result.json').read_text())
    record = dict(case=str(case), dll_sha256=result['dll_sha256'], passed=result['passed'],
                  restore_ms=result['restore_ms'], T1_native_exact=result['T1_native_exact'],
                  T2_observation_exact=result['T2_observation_exact'], branches={})
    for branch in ('original', 'restored'):
        rows = json.loads((case/(branch+'-steps.json')).read_text())
        expected = result[branch]['quality']['steps']
        assert len(rows) == expected
        assert all(b['native_tick'] == a['native_tick']+1 for a, b in zip(rows, rows[1:]))
        phases = {}
        for phase in dict.fromkeys(row['stage'] for row in rows):
            subset = [row for row in rows if row['stage'] == phase]
            phases[phase] = {
                'steps': len(subset),
                'timings_ms': {key: distribution([row['timings_ms'][key] for row in subset])
                               for key in ('native_step', 'quality_and_export', 'snapshot', 'total')},
                'max_blade_overlap_mm': max(row['quality']['metrics']['blade_body_overlap_m'] for row in subset)*1000,
                'min_volume_ratio': min(row['quality']['metrics']['min_volume_ratio'] for row in subset),
                'quality_failure_steps': sum(not row['quality']['checked_metrics_passed'] for row in subset),
                'scope_failure_steps': sum(not row['quality']['within_declared_local_scope'] for row in subset),
                'grip_points_min_max': [min(row['grasp']['anchor_points'] for row in subset),
                                        max(row['grasp']['anchor_points'] for row in subset)],
                'new_material_cut_edges': sum(row['new_material_cut_edges'] for row in subset),
            }
        record['branches'][branch] = dict(outcome=result[branch]['outcome'], phases=phases)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(cases=[summarize(path) for path in args.cases], controlled_performance_acceptance=False,
                  note='Separate trajectories with GPU atomic variability and shared CPU load. '
                       'Capture, snapshots and graph reconstruction can add overhead. '
                       'surface_export is already inside quality_and_export; no double count.',
                  candidate_adopted=False)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({str(row['case']): row['passed'] for row in report['cases']}))


if __name__ == '__main__':
    main()
