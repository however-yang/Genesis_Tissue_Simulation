"""CPU extraction of existing, verified comparison frames; no simulation."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw

from compare_videos import OUT, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('name')
    args = p.parse_args()
    folder = (OUT/'comparisons'/args.name).resolve()
    if not folder.is_relative_to(OUT/'comparisons'):
        p.error('Only this batch comparison directory is allowed')
    result = json.loads((folder/'result.json').read_text(encoding='utf8'))
    movie = folder/'baseline-left-candidate-right.mp4'
    assert result['passed'] and digest(movie) == result['movie_sha256']
    report = json.loads((Path(result['left']).parents[1]/'report.json').read_text(encoding='utf8'))
    names = ({'lift', 'blade_approach', 'incise_1', 'incise_2', 'incise_3', 'withdraw', 'release', 'hold'}
             if 'three-snips' in args.name else
             {'incise_1', 'regrasp_release_hold', 'regrasp_shift', 'regrasp_lower',
              'regrasp_close_hold', 'regrasp_lift', 'reposition_2', 'incise_2'})
    selected = {s['end']['step']: s['name'] for s in report['stages'] if s['name'] in names}
    reader = imageio_ffmpeg.read_frames(str(movie), pix_fmt='rgb24')
    metadata = next(reader)
    width, height = metadata['size']
    frames = []
    for index, raw in enumerate(reader):
        if index in selected:
            panel = Image.new('RGB', (width, height+28), 'white')
            panel.paste(Image.frombytes('RGB', (width, height), raw), (0, 28))
            ImageDraw.Draw(panel).text((8, 7), f'Step {index}: {selected[index]} | baseline LEFT / candidate RIGHT', fill='black')
            frames.append((index, panel))
    assert {i for i, _ in frames} == set(selected)
    paths = []
    for page in range(0, len(frames), 3):
        group = frames[page:page+3]
        sheet = Image.new('RGB', (width, (height+28)*len(group)), 'white')
        for row, (_, panel) in enumerate(group):
            sheet.paste(panel, (0, row*(height+28)))
        target = folder/f'milestones-{page//3+1:02}.png'
        sheet.save(target)
        paths.append(str(target))
    (folder/'milestones.json').write_text(json.dumps({'source_sha256': result['movie_sha256'],
        'frames': selected, 'images': paths, 'scope': 'Selected real frames, not every-frame visual certification',
        'physical_or_performance_approved': False}, indent=2), encoding='utf8')
    print(json.dumps({'comparison': args.name, 'selected_frames': len(frames), 'images': paths}))


if __name__ == '__main__':
    main()
