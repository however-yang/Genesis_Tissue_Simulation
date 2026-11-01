"""Isolated S16 evidence worker; run only through the serial GPU queue.

This does not approve a candidate. It retains all quality failures and checks
the compact native reductions against independent exported-array calculations.
"""
import argparse
from dataclasses import asdict
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
from replay_archive import ReplayArchiveWriter
from validate_strict import pack_frame
from endoscopic_tissue.native_backend import NativeTissue
from endoscopic_tissue.scenario import SCISSORS_SCENES, stage_start, interpolate
from endoscopic_tissue.training import env as entry
from endoscopic_tissue.training.actions import ActionBounds, command_to_action
from endoscopic_tissue.training.physics import PhysicsOverrides
from endoscopic_tissue.training.geometry import WallGeometry, scene_for_config
from endoscopic_tissue.training.data import atomic_json, EpisodeWriter
from endoscopic_tissue.training.grasp import GraspMonitor
from endoscopic_tissue.training.provenance import sha256

OUT = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/damage'


def independently_check(native):
    compact = native.material_quality()
    original = NativeTissue.material_quality(native)
    np.testing.assert_array_equal(compact, original)
    m = native.last_training_metrics
    obs = native.observe()
    force = native.core_forces_n()
    supports = obs.grip_flags != 0
    np.testing.assert_allclose(m['coreForceMax']*.01, force.max(), rtol=2e-6, atol=1e-8)
    # Native exports the union of valid witness supports. Check its exact median.
    assert m['activeCoreCorners'] == int(supports.sum()), (m['activeCoreCorners'], int(supports.sum()))
    median = float(np.median(force[supports])) if supports.any() else 0.
    np.testing.assert_allclose(m['coreForceMedian']*.01, median, rtol=2e-6, atol=1e-8)
    nbr = obs.neighbors[:, 1::2]
    inside = np.asarray(native.grid.inside, bool)
    owner, axis = np.where((nbr >= 0) & inside[:, None] & inside[np.maximum(nbr, 0)])
    other = nbr[owner, axis]
    rest = native.grid.positions
    stretch = np.linalg.norm(obs.nodes[other]-obs.nodes[owner], axis=1)/np.linalg.norm(rest[other]-rest[owner], axis=1)
    np.testing.assert_allclose(m['maxStretch'], stretch.max(initial=0), rtol=3e-6, atol=1e-7)
    initial = native.grid.neighbors
    rim = np.any((initial >= 0) & inside[np.maximum(initial, 0)] & (obs.neighbors < 0), axis=1) & inside
    rim_edges = rim[owner] | rim[other]
    np.testing.assert_allclose(m['rimStretch'], stretch[rim_edges].max(initial=0), rtol=3e-6, atol=1e-7)
    causes = native.tear_causes()
    assert int(np.count_nonzero(causes & 1)) == m['tearEdges']
    assert int(np.count_nonzero(causes & 2)) == m['materialTearEdges']
    return {'tick': native.tick, 'quality_exact': True, 'force_and_stretch_reductions_match': True,
            'cause_array_matches': True, 'metrics': m}


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--dll', default='bin-s15-training-state1')
    p.add_argument('--scene', default='wall-traction', choices=['wall-traction', 'wall', 'flap'])
    p.add_argument('--tear', type=float)
    p.add_argument('--force', type=float)
    p.add_argument('--contact',type=int,choices=[1,2])
    p.add_argument('--geometry',type=Path)
    p.add_argument('--telemetry', action='store_true')
    p.add_argument('--capture', action='store_true')
    p.add_argument('--stop-after', default='hold')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--probe-dir', type=Path,
                   help='Optional actual two-state episode for geometry split validation; full-script evidence remains separate')
    args = p.parse_args()
    out = OUT/args.name
    out.mkdir(parents=True, exist_ok=False)
    dll = ROOT/'endoscopic_tissue/native'/args.dll/'LiverCudaSim.dll'
    geometry=WallGeometry(**json.loads(args.geometry.read_text(encoding='utf8'))) if args.geometry else None
    scene_config=entry.SceneConfig(args.scene,'settled',contact_model=args.contact,geometry=geometry)
    scene = scene_for_config(scene_config)
    if args.stop_after not in {s.name for s in scene.stages}:
        raise ValueError('Requested stop stage is absent from the scene')
    center = command_to_action(scene.stages[0].command)
    lower, upper = center-.2, center+.2
    lower[[3,10]], upper[[3,10]] = 0., 1.
    bounds = ActionBounds(tuple(lower), tuple(upper), (10.,)*11)
    physics = PhysicsOverrides(tear_stretch_ratio=args.tear, grasp_core_break_force_n=args.force)
    config = entry.EnvironmentConfig(str(dll), sha256(dll), bounds,
        entry.SceneConfig(args.scene, 'settled',contact_model=args.contact,physics=physics,geometry=geometry), cutting_enabled=True,
        max_steps=sum(s.steps for s in scene.stages), failure_telemetry=args.telemetry,
        diagnostic_continue=True)
    stream = ReplayArchiveWriter(out/'replay.rdx') if args.capture else None
    active = {'stage': 'settle'}
    trajectory, stages, checks, rows, preparation = [], [], [], [], []
    probe = None
    snips={}
    original_class = entry.TrainingNativeTissue

    class Recorder(original_class):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if stream:
                self.capture_grasp = GraspMonitor(self.settings, self.grid.inside)
                pickle.dump((self.settings, args.scene, args.name), stream, protocol=5)
                pack_frame(stream, 0, 'initial', self.surface(), self.observe())

        def step(self, command):
            obs = super().step(command)
            if stream:
                grasp = self.capture_grasp.assess_native(obs)
                trajectory.append({'step': self.tick, 'stage': active['stage'],
                    'pad_slip_mm': grasp['witness_tangential_slip_m']*1000,
                    'grip_points': obs.grip_points, 'bite_nodes': len(grasp['witness_ids']),
                    'held_bite_nodes': round(len(grasp['witness_ids'])*grasp['witness_retained_fraction']),
                    'cut_edges': int(np.count_nonzero((self.grid.neighbors[:,1::2]>=0)&(obs.neighbors[:,1::2]<0))),
                    'jaw_load_n': list(obs.jaw_load_n), 'grasp': grasp})
                pack_frame(stream, self.tick, active['stage'], self.surface(), obs)
            return obs

    if stream:
        entry.TrainingNativeTissue = Recorder
    started = time.perf_counter()
    error = None
    try:
        with entry.NativeEnvironment(config) as env:
            initial, context = env.reset(args.seed)
            atomic_json(out/'context.json', context)
            preparation = env.preparation
            if args.probe_dir:
                probe = EpisodeWriter(args.probe_dir, context, initial, env.static_arrays())
            stages.append({'name': 'settle', 'end': {'step': env.native.tick}})
            if env.native.training_flags:
                checks.append(independently_check(env.native))
            requested = scene.stages[0].command
            for stage in (() if args.stop_after == 'settle' else scene.stages[1:]):
                active['stage'] = stage.name
                start = stage_start(stage.name, requested, env.observed)
                for i in range(stage.steps):
                    requested = interpolate(start, stage.command, (i+1)/(stage.steps*stage.ramp))
                    r = env.replay_command(requested)
                    meta = r.after.metadata if r.after else None
                    rows.append({'tick': env.native.tick, 'stage': stage.name, 'metadata': meta,
                                 'timings_ms': r.timings_ms, 'reason': r.reason})
                    if r.after is None:
                        raise RuntimeError(f'No state returned: {r.reason}')
                    if probe is not None:
                        probe.append(r)
                        probe.close('one_transition_geometry_split_probe; full_script_checked_in_separate_result')
                        probe = None
                    if stage.name.startswith('incise_'):
                        cut=meta.get('cut_origin_bracket')
                        snip=snips.setdefault(stage.name,{'first_material_cut_stage_step':None,'material_edges':0,'steps':0})
                        snip['steps']+=1
                        if cut:
                            if snip['first_material_cut_stage_step'] is None:snip['first_material_cut_stage_step']=i+1
                            snip['material_edges']+=cut['new_material_cut_edges']
                    g = meta['grasp']
                    if not stream:
                        trajectory.append({'step': env.native.tick, 'stage': stage.name,
                            'pad_slip_mm': g['witness_tangential_slip_m']*1000,
                            'grip_points': env.observed.grip_points, 'bite_nodes': len(g['witness_ids']),
                            'held_bite_nodes': round(len(g['witness_ids'])*g['witness_retained_fraction']),
                            'cut_edges': int(np.count_nonzero((env.grid.neighbors[:,1::2]>=0)&(env.observed.neighbors[:,1::2]<0))),
                            'jaw_load_n': list(env.observed.jaw_load_n), 'grasp': g})
                stages.append({'name': stage.name, 'end': {'step': env.native.tick}})
                if env.native.training_flags:
                    checks.append(independently_check(env.native))
                print(args.name, stage.name, env.native.tick, flush=True)
                if stage.name == args.stop_after:
                    break
            atomic_json(out/'preparation.json', env.preparation)
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        entry.TrainingNativeTissue = original_class
        if probe is not None:
            probe.close('initial_state_only; no_transition_requested')
        if stream:
            stream.close()
        atomic_json(out/'steps.json', rows)
        atomic_json(out/'independent-checks.json', checks)
        recorded_steps = trajectory[-1]['step'] if trajectory else (stages[-1]['end']['step'] if stages else 0)
        if stream:
            assert [r['step'] for r in trajectory] == list(range(1, recorded_steps+1)), 'Missing or repeated captured step'
        expected_steps = sum(s.steps for s in scene.stages)
        atomic_json(out/'report.json', {'metadata': {'case': args.name}, 'stages': stages,
            'steps': recorded_steps, 'completed': error is None and recorded_steps == expected_steps,
            'expected_steps': expected_steps, 'scope': 'original scenario up to '+args.stop_after})
        with (out/'trajectory.jsonl').open('w', encoding='utf8') as f:
            for row in trajectory:
                f.write(json.dumps(row)+'\n')
        quality_failures = sum(not r['metadata']['quality']['checked_metrics_passed'] for r in rows if r['metadata'])
        scope_failures = sum(not r['metadata']['quality']['within_declared_local_scope'] for r in rows if r['metadata'])
        prep_failures = sum(not r['quality']['checked_metrics_passed'] or not r['quality']['within_declared_local_scope'] for r in preparation)
        full_script = args.stop_after == scene.stages[-1].name
        completed = bool(stages) and stages[-1]['name'] == args.stop_after
        snips_passed = (len(snips) == 3 and all(s['material_edges'] > 0 for s in snips.values())) if full_script else None
        passed = error is None and completed and not (quality_failures or scope_failures or prep_failures) and (snips_passed is not False)
        atomic_json(out/'result.json', {'case': args.name, 'config': asdict(config), 'error': error,
            'wall_s': time.perf_counter()-started, 'transitions': len(rows),
            'snips':snips, 'completed_requested_stages': completed, 'full_script': full_script,
            'three_snips_created_material_cuts': snips_passed,
            'quality_failures': quality_failures, 'scope_failures': scope_failures,
            'preparation_failures': prep_failures, 'passed': passed,
            'split_probe_note': 'Two adjacent native states only; complete script metrics and video are separate' if args.probe_dir else None,
            'independent_checks': len(checks), 'candidate_adopted': False})
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
