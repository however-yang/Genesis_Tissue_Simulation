"""Full interaction replay after the identical 30-step gravity preparation.

Legacy and adapter both use the pinned production DLL. Capture, quality checks
and compression are timed separately from native step. Diagnostic replay never
grants training admission or bypasses the production native numerical guards.
"""
import argparse
from collections import Counter
from contextlib import closing
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation'))
sys.path.insert(0, str(ROOT/'endoscopic_tissue_backups/2026-10-31-interaction-fixes'))
import numpy as np
from threadpoolctl import threadpool_limits
import zstandard
from replay_archive import ReplayArchiveWriter
from validate_strict import pack_frame, summary
from endoscopic_tissue.native_backend import NativeTissue
from endoscopic_tissue.scenario import SCISSORS_SCENES, drive_command, interpolate, scissors_settings, severed_edges, stage_start
from endoscopic_tissue.training.actions import ActionBounds, command_to_action
from endoscopic_tissue.training.env import EnvironmentConfig, NativeEnvironment, SceneConfig
from endoscopic_tissue.training.quality import QualityMonitor
from endoscopic_tissue.training.provenance import pinned_dll, sha256
from endoscopic_tissue.training.data import atomic_json

PIN = '4515f14aa15309162c2c54b6370792e128e63a7e8d3054c506da19e3aeb37d2f'


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('legacy', 'adapter'), required=True)
    parser.add_argument('--scene', choices=('wall', 'wall-traction', 'flap'), default='wall')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dll = pinned_dll(ROOT/'endoscopic_tissue/native/bin-contact-groups31/LiverCudaSim.dll', PIN)
    scene = SCISSORS_SCENES[args.scene]
    settings = replace(scissors_settings(), young_pa=scene.young_pa, contact_model=scene.contact_model)
    grid = scene.build_grid(settings.density_kg_m3)
    requests = np.stack([command_to_action(s.command) for s in scene.stages])
    lo, hi = requests.min(0)-.01, requests.max(0)+.01
    lo[[3, 10]], hi[[3, 10]] = 0, 1
    bounds = ActionBounds(tuple(lo), tuple(hi), (10.,)*11)
    env = None
    native = None
    rows, stages = [], []
    timings = Counter()
    overall = time.perf_counter()
    metadata = dict(case=args.mode+'-'+args.scene, scene=args.scene, dll_sha256=PIN,
                    settings=asdict(settings), material_nodes=int(grid.inside.sum()),
                    active_nodes=int(grid.active.sum()), total_grid_nodes=len(grid.positions),
                    start='after identical scripted settle (30 steps); all interaction stages included',
                    domain_approved=False)
    atomic_json(args.output/'manifest.json', metadata)
    try:
        if args.mode == 'adapter':
            env = NativeEnvironment(EnvironmentConfig(str(dll), PIN, bounds, SceneConfig(args.scene, 'settled'),
                                    max_steps=10000, diagnostic_continue=True))
            _, context = env.reset(0)
            atomic_json(args.output/'context.json', context)
            native, observed, surface = env.native, env.observed, env.last_surface
        else:
            native = NativeTissue(grid, settings, dll, scene.stages[0].command)
            observed = native.observe()
            stage = scene.stages[0]
            start = stage.command
            for i in range(stage.steps):
                requested = interpolate(start, stage.command, (i+1)/(stage.steps*stage.ramp))
                observed = native.step(drive_command(requested, observed, settings))
            surface = native.surface()
        monitor = QualityMonitor(grid)
        monitor.previous_neighbors = observed.neighbors
        original_check = native.check
        def timed_check(name, *values):
            begin = time.perf_counter()
            try:
                return original_check(name, *values)
            finally:
                timings[name] += (time.perf_counter()-begin)*1000
        native.check = timed_check
        with closing(ReplayArchiveWriter(args.output/'replay.rdx')) as replay, \
                (args.output/'trajectory.jsonl').open('w', encoding='utf8') as trajectory, \
                (args.output/'nodes.f32.zst').open('wb') as node_file, \
                zstandard.ZstdCompressor(level=1).stream_writer(node_file, closefd=False) as node_stream:
            pickle.dump((settings, args.scene, args.mode+': unchanged native DLL; full interaction after settling'), replay, protocol=5)
            pack_frame(replay, 0, 'prepared', surface, observed)
            node_stream.write(observed.nodes.tobytes())
            previous = scene.stages[0].command
            for stage in scene.stages[1:]:
                print(args.mode, len(rows), stage.name, flush=True)
                start = stage_start(stage.name, previous, observed)
                first = len(rows)
                for i in range(stage.steps):
                    requested = interpolate(start, stage.command, (i+1)/(stage.steps*stage.ramp))
                    old = observed
                    timings.clear()
                    if env:
                        result = env.replay_command(requested)
                        if result.after is None:
                            raise RuntimeError(result.reason)
                        observed, surface = env.observed, env.last_surface
                        quality = result.after.metadata['quality']
                        native_ms = result.timings_ms['native_step']
                        pipeline_ms = native_ms+result.timings_ms['surface_export']
                        total_ms = result.timings_ms['total']
                        executed = result.action['executed_command_native']
                    else:
                        begin = time.perf_counter()
                        command = drive_command(requested, observed, settings)
                        observed = native.step(command)
                        native_ms = (time.perf_counter()-begin)*1000
                        surface = native.surface()
                        pipeline_ms = (time.perf_counter()-begin)*1000
                        quality = monitor.assess(observed, old, command, settings, native.material_quality(),
                                                 surface, native.scissors_geometry)
                        total_ms = (time.perf_counter()-begin)*1000
                        executed = asdict(command)
                    metrics = quality['metrics']
                    if metrics['max_displacement_m'] > .06:
                        raise RuntimeError('Unchanged 60 mm displacement guard exceeded')
                    row = dict(step=len(rows)+1, stage=stage.name, physics_ms=timings['LCS_Step'],
                        native_step_ms=native_ms, pipeline_ms=pipeline_ms, checked_step_ms=total_ms,
                        grip_points=observed.grip_points, jaw_load_n=list(observed.jaw_load_n),
                        pad_slip_mm=None, held_bite_nodes=0, bite_nodes=0,
                        cut_edges=int(severed_edges(grid.neighbors, observed.neighbors)),
                        fractures=int(observed.scissors.loads.fractures),
                        scissors_edges_cm=observed.scissors.edges.tolist(),
                        topology_sha256=hashlib.sha256(observed.neighbors.tobytes()).hexdigest(),
                        requested=asdict(requested), executed=executed, quality=quality)
                    rows.append(row)
                    trajectory.write(json.dumps(row, allow_nan=False)+'\n')
                    node_stream.write(observed.nodes.tobytes())
                    pack_frame(replay, len(rows), stage.name, surface, observed)
                previous = stage.command
                stages.append(dict(stage=stage.name, first=first+1, end=rows[-1]))
        expected = sum(s.steps for s in scene.stages[1:])
        report = dict(metadata=metadata, steps=len(rows), completed=len(rows)==expected, stages=stages,
                      metrics={k: summary([r[k] for r in rows]) for k in ('physics_ms','native_step_ms','pipeline_ms','checked_step_ms')},
                      failed_quality_steps=sum(not r['quality']['checked_metrics_passed'] for r in rows),
                      out_of_scope_steps=sum(not r['quality']['within_declared_local_scope'] for r in rows),
                      wall_time_s=time.perf_counter()-overall,
                      nodes_file_sha256=sha256(args.output/'nodes.f32.zst'))
        atomic_json(args.output/'report.json', report)
        print('complete', len(rows), report['metrics'], flush=True)
    finally:
        if env:
            env.close()
        elif native:
            native.close()


if __name__ == '__main__':
    main()
