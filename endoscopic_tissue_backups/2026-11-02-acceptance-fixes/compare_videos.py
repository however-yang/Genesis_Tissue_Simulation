"""Whole-sequence side-by-side videos: baseline left, candidate right."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'endoscopic_tissue/outputs/acceptance-fixes-20261102'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    args = p.parse_args()
    sources, records = [], []
    for label in (args.baseline, args.candidate):
        case = (OUT/label).resolve()
        if not case.is_relative_to(OUT):
            p.error('Only recorded cases from this batch are allowed')
        video = case/'review/interactions-h264.mp4'
        record = json.loads((case/'review/video-verification.json').read_text(encoding='utf8'))
        if record['frames'] != record['decoded'] or digest(video) != record['movie_sha256']:
            raise ValueError('Input video no longer matches its full-frame verification')
        sources.append(video)
        records.append(record)
    if records[0]['frames'] != records[1]['frames'] or records[0]['fps'] != records[1]['fps']:
        raise ValueError('No truncation or temporal warping is allowed to make a comparison fit')
    out = OUT/'comparisons'/args.name
    video = out/'baseline-left-candidate-right.mp4'
    if out.exists():
        # A CPU-only review can finish while the serial GPU queue is waiting.
        # Reuse only the exact already-decoded artifact, never a partial encode.
        previous = json.loads((out/'result.json').read_text(encoding='utf8'))
        if not (previous.get('passed') is True
                and previous.get('physical_or_performance_approved') is False
                and previous.get('left') == str(sources[0])
                and previous.get('right') == str(sources[1])
                and previous.get('input_sha256') == [r['movie_sha256'] for r in records]
                and previous.get('decoded_frames') == records[0]['frames']
                and previous.get('expected_frames') == records[0]['frames']
                and previous.get('metadata', {}).get('fps') == records[0]['fps']
                and previous.get('movie_sha256') == digest(video)):
            raise ValueError('Existing comparison is incomplete or has different inputs; retained without overwrite')
        print('Reused byte-identical verified comparison:', args.name, flush=True)
        return
    out.mkdir(parents=True, exist_ok=False)
    record = {'passed': False, 'left': str(sources[0]), 'right': str(sources[1]),
              'input_sha256': [r['movie_sha256'] for r in records],
              'physical_or_performance_approved': False,
              'note': 'Every recorded frame; common simulation time, no temporal alignment or frame removal. Playback rate is not runtime FPS.'}
    try:
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-n', '-loglevel', 'error',
                   '-i', str(sources[0]), '-i', str(sources[1]),
                   '-filter_complex', '[0:v]scale=640:480:flags=lanczos[a];[1:v]scale=640:480:flags=lanczos[b];[a][b]hstack=inputs=2[v]',
                   '-map', '[v]', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
                   '-pix_fmt', 'yuv420p', '-threads', '2', str(video)]
        print('Encoding whole-sequence comparison:', args.name, flush=True)
        with (out/'encode.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
        metadata = next(reader)
        frames = sum(1 for _ in reader)
        record.update(decoded_frames=frames, expected_frames=records[0]['frames'], metadata=metadata,
                      movie_sha256=digest(video))
        if frames != records[0]['frames'] or abs(metadata['fps']-records[0]['fps']) > .001:
            raise ValueError('Comparison did not preserve every frame and the original time scale')
        record['passed'] = True
        print('Verified comparison:', frames, 'frames', flush=True)
    finally:
        (out/'result.json').write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf8')


if __name__ == '__main__':
    main()
