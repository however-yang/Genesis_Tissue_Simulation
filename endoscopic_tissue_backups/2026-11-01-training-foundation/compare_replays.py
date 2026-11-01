"""Compare every captured node state and transition of paired full replays."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import zstandard

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue.scenario import SCISSORS_SCENES
from endoscopic_tissue.training.data import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports, logs = [], []
    for folder in (args.reference, args.candidate):
        reports.append(json.loads((folder/'report.json').read_text()))
        logs.append([json.loads(s) for s in (folder/'trajectory.jsonl').read_text().splitlines()])
    assert reports[0]['completed'] and reports[1]['completed']
    assert reports[0]['steps'] == reports[1]['steps']
    scene = SCISSORS_SCENES[reports[0]['metadata']['scene']]
    grid = scene.build_grid(reports[0]['metadata']['settings']['density_kg_m3'])
    active = grid.active != 0
    material = grid.inside != 0
    count, nbytes = reports[0]['steps']+1, len(grid.positions)*3*4
    differences = []
    with (args.reference/'nodes.f32.zst').open('rb') as first, (args.candidate/'nodes.f32.zst').open('rb') as second, \
            zstandard.ZstdDecompressor().stream_reader(first) as a, \
            zstandard.ZstdDecompressor().stream_reader(second) as b:
        for frame in range(count):
            raw_a, raw_b = a.read(nbytes), b.read(nbytes)
            assert len(raw_a) == len(raw_b) == nbytes
            x, y = (np.frombuffer(raw, '<f4').reshape(-1, 3) for raw in (raw_a, raw_b))
            distance = np.linalg.norm(x-y, axis=1)*10
            differences.append(dict(frame=frame, max_active_mm=float(distance[active].max()),
                rms_material_mm=float(np.sqrt(np.mean(distance[material]**2))),
                max_material_mm=float(distance[material].max())))
        assert not a.read(1) and not b.read(1)
    summary = dict(reference=str(args.reference), candidate=str(args.candidate), frames=count,
        max_active_mm=max(d['max_active_mm'] for d in differences),
        max_material_mm=max(d['max_material_mm'] for d in differences),
        max_frame_rms_material_mm=max(d['rms_material_mm'] for d in differences),
        frames_with_changed_topology=sum(x['topology_sha256'] != y['topology_sha256'] for x,y in zip(*logs)),
        frames_with_changed_target=sum(x['executed'] != y['executed'] for x,y in zip(*logs)),
        final_cut_edges=[rows[-1]['cut_edges'] for rows in logs],
        failed_quality_steps=[r['failed_quality_steps'] for r in reports],
        timing=[r['metrics'] for r in reports],
        stage_ends=[dict(stage=x['stage'], cuts=[x['cut_edges'],y['cut_edges']],
                        grips=[x['grip_points'],y['grip_points']])
                    for x,y in zip(*logs) if x['step'] in {s['end']['step'] for s in reports[0]['stages']}],
        frame_differences=differences,
        acceptance='diagnostic comparison only; known physical acceptance failures are not waived')
    atomic_json(args.output, summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('frame_differences','stage_ends','timing')}, indent=2))


if __name__ == '__main__':
    main()
