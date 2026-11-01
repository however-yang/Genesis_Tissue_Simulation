"""Check every captured frame/annotation before using the existing full renderer."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OLD))
from replay_archive import open_replay
from endoscopic_tissue.training.data import atomic_json


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def preflight(folder):
    report = json.loads((folder/'report.json').read_text(encoding='utf8'))
    rows = [json.loads(line) for line in (folder/'trajectory.jsonl').read_text(encoding='utf8').splitlines()]
    archive = json.loads((folder/'replay-archive.json').read_text(encoding='utf8'))
    if archive.get('exact_encoded_byte_roundtrip') is not True:
        raise ValueError('Archive has no complete byte-roundtrip evidence')
    if len(rows) != report['steps'] or [r['step'] for r in rows] != list(range(1, report['steps']+1)):
        raise ValueError('Annotation ticks must include every preparation and continuation step')
    count = 0
    with open_replay(folder/'replay.bin') as stream:
        settings, scene, provenance = pickle.load(stream)
        while prefix := stream.read(4):
            if len(prefix) != 4:
                raise ValueError('Truncated captured frame prefix')
            encoded = stream.read(struct.unpack('<I', prefix)[0])
            step, stage, _, observation = pickle.loads(zlib.decompress(encoded))
            if step != count:
                raise ValueError('Recorded native frames are missing, repeated or reordered')
            if step:
                row = rows[step-1]
                if (row['stage'] != stage or row['grip_points'] != observation.grip_points or
                        list(row['jaw_load_n']) != list(observation.jaw_load_n)):
                    raise ValueError('Frame and numeric annotation describe different states')
            count += 1
    if count != len(rows)+1 or archive['frames'] != count:
        raise ValueError('Recorded frame count does not match the annotations/archive')
    source = folder/'replay.rdx'
    return {'passed': True, 'frames': count, 'source_sha256': file_sha(source),
        'scene': scene, 'provenance': provenance, 'simulation_time_fps': 1/settings.dt,
        'full_original_scenario': report['completed'],
        'scope': report.get('scope'), 'expected_steps': report.get('expected_steps'),
        'completed_requested_continuation': report.get('completed_requested_continuation'),
        'note': 'Every captured state and label checked; this is not a physical-quality or speed acceptance.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    folder = args.case.resolve()
    if not folder.is_relative_to(ROOT/'endoscopic_tissue/outputs/s15-s17-20261101'):
        parser.error('Only this task\'s generated replay directories are accepted')
    try:
        evidence = preflight(folder)
    except Exception as error:
        atomic_json(folder/'capture-preflight.json', {'passed': False, 'error': repr(error)})
        raise
    atomic_json(folder/'capture-preflight.json', evidence)
    if not args.check_only:
        subprocess.run([sys.executable, '-X', 'utf8', str(OLD/'render_review.py'), str(folder)], check=True)
        video_file = folder/'review/video-verification.json'
        video = json.loads(video_file.read_text(encoding='utf8'))
        if video['decoded'] != evidence['frames'] or video['frames'] != evidence['frames']:
            raise ValueError('Video omitted recorded states')
        video.update(capture_preflight=evidence, physical_or_performance_approved=False)
        atomic_json(video_file, video)
    print(json.dumps(evidence))


if __name__ == '__main__':
    main()
