"""Render and verify every recorded state in this isolated acceptance batch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'endoscopic_tissue_backups/2026-11-01-s15-s17'
sys.path.insert(0, str(OLD))
from render_validation import preflight
from endoscopic_tissue.training.data import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case', type=Path)
    args = p.parse_args()
    folder = args.case.resolve()
    if not folder.is_relative_to(ROOT / 'endoscopic_tissue/outputs/acceptance-fixes-20261102'):
        p.error('Only this isolated batch is allowed')
    evidence = preflight(folder)
    atomic_json(folder / 'capture-preflight.json', evidence)
    renderer = ROOT / 'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation/render_review.py'
    subprocess.run([sys.executable, '-X', 'utf8', str(renderer), str(folder)], check=True)
    video_file = folder / 'review/video-verification.json'
    video = json.loads(video_file.read_text(encoding='utf8'))
    if video['decoded'] != evidence['frames'] or video['frames'] != evidence['frames']:
        raise ValueError('Video omitted recorded states')
    video.update(capture_preflight=evidence, physical_or_performance_approved=False)
    atomic_json(video_file, video)
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
