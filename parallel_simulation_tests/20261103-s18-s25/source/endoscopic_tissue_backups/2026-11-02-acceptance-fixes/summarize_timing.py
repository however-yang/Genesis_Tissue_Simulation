"""Summarize whole-run ABBA timings without treating frames as independent trials."""
import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'endoscopic_tissue/outputs/acceptance-fixes-20261102'


def summary(rows):
    result = {}
    for key in ('total', 'native_step', 'quality_and_export', 'surface_export', 'snapshot'):
        values = sorted(r['timings_ms'][key] for r in rows)
        # Explicit nearest-rank percentile; every accepted transition included.
        result[key] = {'mean': statistics.mean(values), 'median': statistics.median(values),
                       'p95': values[max(0, (95*len(values)+99)//100-1)],
                       'p99': values[max(0, (99*len(values)+99)//100-1)], 'max': max(values)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    runs = []
    for arm in ('a1', 'b1', 'b2', 'a2'):
        name = args.prefix+'-'+arm
        case = OUT/name
        supervisor = json.loads((OUT/(name+'-supervisor.json')).read_text(encoding='utf8'))
        result = json.loads((case/'result.json').read_text(encoding='utf8'))
        config = result['config']
        rows = json.loads((case/'steps.json').read_text(encoding='utf8'))
        valid = (supervisor['phase'] == 'finished' and supervisor['returncode'] == 0
                 and supervisor.get('performance_evidence') is True and not supervisor['interference']
                 and result['passed'] and result['full_script'] and config['full_surface_checks']
                 and '--capture' not in supervisor['command'] and len(rows) == result['transitions'])
        runs.append({'name': name, 'arm': arm[0], 'valid_timing': valid,
                     'dll_sha256': config['dll_sha256'], 'frames': len(rows),
                     'summary_ms': summary(rows),
                     'stages_ms': {s: summary([r for r in rows if r['stage'] == s])
                                   for s in dict.fromkeys(r['stage'] for r in rows)}})
    # Runs, rather than thousands of correlated frames, are the independent units.
    compare = {}
    for key in runs[0]['summary_ms']:
        compare[key] = {}
        for stat in ('mean', 'p95', 'p99', 'max'):
            a, b = [statistics.mean(r['summary_ms'][key][stat] for r in runs if r['arm'] == arm)
                    for arm in ('a', 'b')]
            compare[key][stat] = {'baseline_ms': a, 'candidate_ms': b,
                                  'change_percent': (b/a-1)*100}
    report = {'runs': runs, 'valid_complete_abba': all(r['valid_timing'] for r in runs),
              'comparison': compare, 'candidate_adopted': False,
              'limits': ['Only two independent trials per version; not a statistical equivalence certificate.',
                         'Headless wall task, 76050 grid nodes; not >100k nodes or rendered GUI FPS.',
                         'surface_export is included in quality_and_export; do not add it twice.',
                         'native_step includes observation readback; snapshot is per-step Python packing.',
                         'Physical realism, T4 distributions and all remaining acceptance gates remain separate.']}
    path = OUT/(args.prefix+'-summary.json')
    with path.open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({'valid': report['valid_complete_abba'], 'comparison': compare}, indent=2))


if __name__ == '__main__':
    main()
