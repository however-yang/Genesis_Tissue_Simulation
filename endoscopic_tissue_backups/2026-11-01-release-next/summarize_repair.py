"""Summarize a completed repair candidate without promoting it."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P = Path(__file__).resolve().parent
OUT = ROOT/'endoscopic_tissue/outputs/release-next-20261101'


def stats(values):
    return dict(n=len(values), mean=float(np.mean(values)), p95=float(np.percentile(values, 95)),
                maximum=float(np.max(values))) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', choices=['r22', 'r23', 'r24', 'r25'], required=True)
    parser.add_argument('--single-screen', action='store_true')
    args = parser.parse_args()
    revision = args.revision
    names = [revision+s for s in ['-time-base-a1', '-control-off1', '-control-on1',
                                  '-control-on2', '-control-off2', '-time-base-a2']]
    if args.single_screen:
        assert revision == 'r24'
        names = ['r25-time-base-a1', 'r24-control-on-screen', 'r25-time-base-a2']
    parts = dict(prefix=(31, 1215), before_cut=(31, 450), cutting=(451, 840),
                 held_after=(841, 1215), free_release=(1244, 1320))
    runs = {}
    for name in names:
        rows = json.loads((OUT/name/'steps.json').read_text())
        report = json.loads((OUT/name/'report.json').read_text())
        runs[name] = dict(final_state=report['final_state'],
            phases={phase: {key: stats([r[key] for r in rows if low <= r['step'] <= high])
                            for key in ('physics_ms', 'pipeline_ms', 'repair_ms')}
                    for phase, (low, high) in parts.items()})
    comparisons = {}
    groups = [('baseline_vs_on', (0, 5), (2, 3)), ('off_vs_on', (1, 4), (2, 3))]
    if args.single_screen:
        groups = [('baseline_vs_on', (0, 2), (1,))]
    for label, left, right in groups:
        comparisons[label] = {}
        for phase in parts:
            comparisons[label][phase] = {}
            for key in ('physics_ms', 'pipeline_ms', 'repair_ms'):
                a = [runs[names[i]]['phases'][phase][key]['mean'] for i in left
                     if runs[names[i]]['phases'][phase][key]]
                b = [runs[names[i]]['phases'][phase][key]['mean'] for i in right
                     if runs[names[i]]['phases'][phase][key]]
                comparisons[label][phase][key] = dict(left_means=a, right_means=b,
                    left_mean=float(np.mean(a)) if a else None, right_mean=float(np.mean(b)) if b else None,
                    delta_ms=float(np.mean(b)-np.mean(a)) if a and b else None)
    case = OUT/(revision+'-kidney')
    report = json.loads((case/'report.json').read_text())
    overlap = json.loads((case/'release-overlap-audit.json').read_text())
    movie = json.loads((case/'review/video-verification.json').read_text())
    dispatch_path = case/'repair-dispatch-audit.json'
    dispatch = json.loads(dispatch_path.read_text()) if dispatch_path.exists() else dict(
        applicable=False, note='Boundary grouping revisions do not use the R22/R23 repair-dispatch optimization.')
    protection = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == value
                  for n, value in json.loads((P/'machine.json').read_text())['protected'].items()}
    quality = dict(steps=report['steps'], completed=report['completed'], invalid_tets=report['max_invalid_tets'],
        healed_connections=report['healed_connections'], cuts_while_disabled=report['cuts_while_disabled'],
        free_release_frames=len(overlap), closed_body_frames=sum(r.get('body_closed', False) for r in overlap),
        closed_piece_frames=sum(r.get('piece_closed', False) for r in overlap),
        observed_contains_max_mm=max(r.get('max_penetration_mm', 0) for r in overlap),
        contains_valid_for_acceptance=bool(overlap) and all(r.get('body_closed') and r.get('piece_closed') for r in overlap),
        stage_nonmanifold_edges={s['stage']: s['audit']['positional_nonmanifold_edges'] for s in report['stages']},
        video_frames=movie['frames'], decoded_frames=movie['decoded'],
        original_quality_failed=bool(report['diagnostic_errors']),
        dispatch_audit={k:v for k,v in dispatch.items() if k!='rows'},
        visual_review='See the explicit review notes in the experiment report; no automated realism pass.')
    jobs = json.loads((P/(('r25' if args.single_screen else revision)+'-timing-jobs.json')).read_text())
    if args.single_screen:
        jobs = [job for job in jobs if job['name'] in names]
        (P/'r24-timing-jobs.json').write_text(json.dumps(jobs, indent=2)+'\n')
    result = dict(revision=revision, independent_timing=True, display_fps_measured=False,
        candidate_repeated=not args.single_screen,
        timing_scope='Single screen only; R24 rejected for cache-identity alias before stability testing.' if args.single_screen else 'Repeated matched controls and baseline brackets.',
        timing_jobs=jobs,
        runs=runs, comparisons=comparisons, quality=quality, protected_files=protection,
        accepted=False, default_replaced=False,
        acceptance_note='Requires stable no-regression timing, complete surgical video review, valid closed meshes, and interaction regressions. No pass inferred from completing a run or from containment on nonclosed meshes.')
    (P/(revision+'-result.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(comparisons=comparisons, quality=quality, protected=all(protection.values())), indent=2))


if __name__ == '__main__':
    main()
