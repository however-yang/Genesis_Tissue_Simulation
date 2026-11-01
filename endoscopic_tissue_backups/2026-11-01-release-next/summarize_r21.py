"""Evidence summary; never promotes a DLL or changes acceptance thresholds."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P = Path(__file__).resolve().parent
OUT = ROOT/'endoscopic_tissue/outputs/release-next-20261101'


def stats(values):
    return dict(n=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(np.max(values))) if values else None


def summarize(directory):
    rows = json.loads((OUT/directory/'steps.json').read_text())
    parts = dict(prefix=(31, 1215), before_cut=(31, 450), cutting=(451, 840),
                 held_after=(841, 1215), free_release=(1244, 1320))
    return {phase: {key: stats([r[key] for r in rows if low <= r['step'] <= high])
                    for key in ('physics_ms', 'pipeline_ms')} for phase, (low, high) in parts.items()}


def main():
    names = ['time-base-a1', 'time-r21-b1', 'time-r21-b2', 'time-base-a2']
    runs = {name: summarize(name) for name in names}
    paired = {}
    for phase in runs[names[0]]:
        paired[phase] = {}
        for key in ('physics_ms', 'pipeline_ms'):
            a = [runs[n][phase][key]['mean'] for n in (names[0], names[3]) if runs[n][phase][key]]
            b = [runs[n][phase][key]['mean'] for n in (names[1], names[2]) if runs[n][phase][key]]
            paired[phase][key] = dict(baseline_run_means=a, candidate_run_means=b,
                baseline_mean=float(np.mean(a)) if a else None, candidate_mean=float(np.mean(b)) if b else None,
                delta_ms=float(np.mean(b)-np.mean(a)) if a and b else None)
    report = json.loads((OUT/'r21-kidney/report.json').read_text())
    overlap = json.loads((OUT/'r21-kidney/release-overlap-audit.json').read_text())
    movie = json.loads((OUT/'r21-kidney/review/video-verification.json').read_text())
    hashes = json.loads((P/'machine.json').read_text())['protected']
    protection = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == value for n, value in hashes.items()}
    quality = dict(steps=report['steps'], completed=report['completed'], invalid_tets=report['max_invalid_tets'],
        healed_connections=report['healed_connections'], cuts_while_disabled=report['cuts_while_disabled'],
        free_release_frames=len(overlap), closed_body_frames=sum(r.get('body_closed', False) for r in overlap),
        closed_piece_frames=sum(r.get('piece_closed', False) for r in overlap),
        observed_contains_max_mm=max(r.get('max_penetration_mm', 0) for r in overlap),
        contains_valid_for_acceptance=all(r.get('body_closed') and r.get('piece_closed') for r in overlap),
        video_frames=movie['frames'], decoded_frames=movie['decoded'],
        original_quality_failed=bool(report['diagnostic_errors']),
        visual_review='All 8 full-trajectory sheets plus every one of 77 free-release frames inspected. Early grip/lift/cuts remain qualitatively similar to baseline. Release avoids the old immediate spiked collapse, but shows repeated bouncing then departure out of the view; not accepted as realistic equivalence.')
    result = dict(candidate='release-r21-fragment-submission', base='current bin-contact-groups31',
        independent_timing=True, display_fps_measured=False, runs=runs, paired_means=paired,
        quality=quality, protected_files=protection, accepted=False, default_replaced=False,
        reasons=['Candidate prefix physics and pipeline means exceed baseline in this ABBA.',
                 'All 77 release body meshes fail the closed-manifold prerequisite; a small contains depth cannot certify nonpenetration.',
                 'Existing cutting quality errors remain; full interaction realism is not established.'])
    (P/'r21-result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(paired_means=paired, quality=quality, protected=all(protection.values())), indent=2))


if __name__ == '__main__':
    main()
