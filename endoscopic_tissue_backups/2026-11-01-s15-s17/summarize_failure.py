"""S16 calibration tables from actual native records; never adopt a parameter."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from validation_support import ROOT
from endoscopic_tissue.training.data import atomic_json
from endoscopic_tissue.training.provenance import json_hash, sha256

DIRECTIONS = ('+x', '-x', '+y', '+z', '-z')
EVENTS = ('material_tear', 'anchor_break', 'no_anchor_points', 'geometric_hold_lost')


def read_nominal(path):
    path = Path(path)
    record = json.loads(path.read_text(encoding='utf8'))
    prep = json.loads((path.parent/'preparation.json').read_text(encoding='utf8'))
    steps = json.loads((path.parent/'steps.json').read_text(encoding='utf8'))
    samples = [{'stage': r['stage'], 'metrics': r.get('failure_physics')} for r in prep]
    samples.extend({'stage': r['stage'], 'metrics': (r['metadata'] or {}).get('failure_physics')} for r in steps)
    if not samples or any(s['metrics'] is None for s in samples):
        raise ValueError('Missing per-step preparation/native failure telemetry: '+str(path))
    phases = {}
    for sample in samples:
        phase = phases.setdefault(sample['stage'], {'max_stretch': 0., 'peak_core_ema_n': 0.,
            'tear_edges': 0, 'broken_support_corners': 0, 'broken_material_points': 0})
        m = sample['metrics']
        phase['max_stretch'] = max(phase['max_stretch'], m['stretch']['material_max'])
        phase['peak_core_ema_n'] = max(phase['peak_core_ema_n'], m['anchors']['core_force_ema_max_n'])
        phase['tear_edges'] += m['new_tear_edges']
        phase['broken_support_corners'] += m['anchors']['new_broken_support_corners']
        phase['broken_material_points'] += m['anchors']['new_broken_material_points']
    no_events = not any(v[k] for v in phases.values() for k in
        ('tear_edges', 'broken_support_corners', 'broken_material_points'))
    return {'source': str(path), 'result_sha256': sha256(path), 'config': record['config'],
        'passed': record.get('passed') is True and record.get('full_script') is True,
        'snips': record['snips'], 'phases': phases, 'no_unintended_failure_events': no_events}


def nominal(paths, baseline_paths):
    rows = [read_nominal(p) for p in paths]
    baseline = [read_nominal(p) for p in baseline_paths]
    if not rows or not baseline:
        raise ValueError('Candidate and failure-disabled nominal runs are both required')
    configurations = {json_hash(r['config']) for r in rows}
    if len(configurations) != 1:
        raise ValueError('Nominal candidate records mix configurations')
    def common_configuration(record):
        config = json.loads(json.dumps(record['config']))
        physics = config['scene']['physics']
        for key in ('tear_stretch_ratio', 'grasp_core_break_force_n'):
            physics.pop(key, None)
        return json_hash(config)
    if len({common_configuration(r) for r in rows+baseline}) != 1:
        raise ValueError('Nominal comparisons may differ only in the selected failure law')
    if any(r['config']['scene']['physics'].get('tear_stretch_ratio') not in (None, 0) or
           r['config']['scene']['physics'].get('grasp_core_break_force_n') not in (None, 150) for r in baseline):
        raise ValueError('The nominal reference must have both failure laws disabled')
    peak = max(p['peak_core_ema_n'] for r in baseline for p in r['phases'].values())
    gravity_max = max(r['phases']['settle']['max_stretch'] for r in baseline)
    force_candidates = [{'multiplier': f, 'force_n': peak*f, 'inside_supported_range': .1 <= peak*f < 150.}
                        for f in (1.5, 2., 3.)]
    comparisons = {}
    for snip in ('incise_1', 'incise_2', 'incise_3'):
        comparisons[snip] = {}
        for metric in ('first_material_cut_stage_step', 'material_edges'):
            a, b = [r['snips'][snip][metric] for r in baseline], [r['snips'][snip][metric] for r in rows]
            av, bv = [v for v in a if v is not None], [v for v in b if v is not None]
            ks = ks_2samp(av, bv) if av and bv else None
            comparisons[snip][metric] = {'disabled': a, 'candidate': b,
                'median_difference': float(np.median(bv)-np.median(av)) if av and bv else None,
                'ks_pvalue': None if ks is None else float(ks.pvalue),
                'missing_disabled': a.count(None), 'missing_candidate': b.count(None)}
    tear = rows[0]['config']['scene']['physics']['tear_stretch_ratio']
    return {'schema': 's16_nominal_comparison_v1', 'candidate_runs': rows, 'disabled_runs': baseline,
        'quality_and_no_failure_events': all(r['passed'] and r['no_unintended_failure_events'] for r in rows+baseline),
        'comparison': comparisons, 'gravity_max_stretch': gravity_max,
        'tear_threshold_above_gravity': None if not tear else tear > gravity_max,
        'nominal_peak_core_ema_n': peak, 'force_candidates': force_candidates,
        'candidate_adopted': False,
        'note': 'Small-sample KS values do not prove equivalence. Review paired first-cut/count distributions and complete videos; do not clamp unsupported force candidates into range.'}


def dose(paths):
    curves = [json.loads(Path(p).read_text(encoding='utf8')) for p in paths]
    if len(curves) != 5 or {c['direction'] for c in curves} != set(DIRECTIONS):
        raise ValueError('A candidate requires all five directions exactly once')
    if len({json_hash(c['physics']) for c in curves}) != 1 or len({c['dll_sha256'] for c in curves}) != 1:
        raise ValueError('Dose curves mix native versions or failure laws')
    expected = {(d*.5, r) for d in range(9) for r in range(10)}
    tables = []
    for curve in curves:
        trials = curve['trials']
        if len(trials) != 90 or {(t['dose_mm'], t['repeat']) for t in trials} != expected:
            raise ValueError('Keep every failed/missing trial; 9 doses x 10 repeats are required')
        table = {'direction': curve['direction'], 'recorded_complete_curve': curve['complete_required_curve'],
            'quality_failure_steps': sum(t['quality']['quality_failure_steps'] for t in trials),
            'scope_failure_steps': sum(t['quality']['scope_failure_steps'] for t in trials),
            'preparation_failure_steps': sum(t['preparation_failure_steps'] for t in trials),
            'baseline_geometric_hold_not_established': sum(not t['initial_geometric_loaded_hold'] for t in trials),
            'events': {}}
        for event in EVENTS:
            first_doses, examples, nonmonotonic = [], [], []
            for repeat in range(10):
                rs = sorted((r for r in trials if r['repeat'] == repeat), key=lambda r: r['dose_mm'])
                hits = [r for r in rs if r['first_events'][event] is not None]
                first = hits[0] if hits else None
                first_doses.append(None if first is None else first['dose_mm'])
                examples.append(None if first is None else first['first_events'][event])
                if first and any(r['dose_mm'] > first['dose_mm'] and r['first_events'][event] is None for r in rs):
                    nonmonotonic.append(repeat)
            table['events'][event] = {'first_tested_dose_mm': first_doses,
                'not_observed_up_to_4mm_count': first_doses.count(None), 'first_event_records': examples,
                'nonmonotonic_repeats': nonmonotonic}
        tables.append(table)
    return {'schema': 's16_dose_table_v1', 'physics': curves[0]['physics'], 'dll_sha256': curves[0]['dll_sha256'],
        'directions': tables, 'passed': all(c.get('passed') is True for c in curves),
        'all_within_existing_local_scope': all(t['scope_failure_steps'] == 0 for t in tables),
        'candidate_adopted': False,
        'sources': [{'path': str(p), 'sha256': sha256(Path(p))} for p in paths],
        'semantics': 'First tested dose is a 0.5 mm discrete dose response, not an exact rupture displacement. Null means not observed within the tested range. Anchor exhaustion and geometric hold loss remain separate. Missing baseline geometric hold cannot establish a loss threshold.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['nominal', 'dose'])
    parser.add_argument('results', type=Path, nargs='+')
    parser.add_argument('--baseline', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.kind == 'nominal' and not args.baseline:
        parser.error('Nominal comparison requires failure-disabled baseline results')
    result = nominal(args.results, args.baseline) if args.kind == 'nominal' else dose(args.results)
    atomic_json(args.output, result)
    print(json.dumps({'output': str(args.output), 'candidate_adopted': False}))


if __name__ == '__main__':
    main()
