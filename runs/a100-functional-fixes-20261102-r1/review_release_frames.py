"""Show every decoded release frame, without rendering or modifying physics."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('review', type=Path)
    p.add_argument('--first', type=int, required=True)
    p.add_argument('--last', type=int, required=True)
    args = p.parse_args()
    source = args.review/'interactions-h264.mp4'
    verification = json.loads((args.review/'video-verification.json').read_text())
    with source.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == verification['movie_sha256']
    out = args.review/'all-release-frames'
    out.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    video = cv2.VideoCapture(str(source))
    selected, count = [], 0
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if args.first <= count <= args.last:
                thumb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((426, 320))
                ImageDraw.Draw(thumb).rectangle((0, 0, 426, 20), fill='black')
                ImageDraw.Draw(thumb).text((5, 4), f'Frame {count}; every release frame', fill='white')
                selected.append(thumb)
            count += 1
    finally:
        video.release()
    assert count == verification['decoded']
    assert len(selected) == args.last-args.first+1
    for start in range(0, len(selected), 12):
        sheet = Image.new('RGB', (1278, 1280), '#101010')
        for i, frame in enumerate(selected[start:start+12]):
            sheet.paste(frame, ((i % 3)*426, (i//3)*320))
        sheet.save(out/f'frames-{args.first+start:04d}.jpg', quality=94)
    (out/'index.json').write_text(json.dumps(dict(first=args.first, last=args.last,
        frames=len(selected), source_sha256=verification['movie_sha256'], all_frames_in_interval=True), indent=2)+'\n')
    print('Reviewed interval prepared:', len(selected), 'frames', flush=True)


if __name__ == '__main__':
    main()
