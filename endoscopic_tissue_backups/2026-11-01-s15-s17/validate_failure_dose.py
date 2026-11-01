"""S16 one-direction dose curve, preserving every failure and quality check.

Fresh runs are available for pilots. Cached runs require completed S15 evidence
for the identical DLL, plus a local round-trip check with this failure law on.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from validation_support import ROOT, environment, command_sequence, row, quality_summary, exact_values
from endoscopic_tissue.scenario import Stage
from endoscopic_tissue.training.physics import PhysicsOverrides
from endoscopic_tissue.training.data import atomic_json
from endoscopic_tissue.training.provenance import sha256
from endoscopic_tissue.training.snapshot import SnapshotBackend, source_binding

DIRECTIONS = {'+x': (1, 0, 0), '-x': (-1, 0, 0), '+y': (0, 1, 0), '+z': (0, 0, 1), '-z': (0, 0, -1)}


def native_cache_probe(env):
    observed = asdict(env.observed)
    snapshot = env.snapshot()
    command = next(s.command for s in env.scene.stages if s.name == 'blade_approach')
    first = env.replay_command(command)
    assert first.after is not None
    expected = first.after.arrays['positions_m'].copy()
    env.restore(snapshot)
    assert SnapshotBackend(env.native).save() == snapshot.native_bytes()
    exact_values(observed, asdict(env.native.observe()))
    second = env.replay_command(command)
    assert second.after is not None
    mm = float(np.linalg.norm(expected-second.after.arrays['positions_m'], axis=1).max()*1000)
    assert mm < .01, 'Failure-law state failed native single-step snapshot check'
    env.restore(snapshot)
    return snapshot, {'T1_exact': True, 'T2_exact': True, 'T3_max_node_difference_mm': mm,
                      'snapshot_bytes': snapshot.size_bytes, 'metadata': snapshot.metadata}


def one_dose(env, dose, direction):
    start = next(s.command for s in env.scene.stages if s.name == 'blade_approach')
    target = replace(start, hinge=tuple(np.asarray(start.hinge)+np.asarray(direction)*dose*.1),
                     scissors_cutting=False)
    # 90 smooth-ramp steps (3 s) and 30 hold steps (1 s) for EVERY dose.
    # At 4 mm the peak requested speed is 2 mm/s. Equal elapsed time prevents
    # longer high-dose trials alone from changing the 1 s EMA observation.
    sequence = command_sequence([Stage('dose_ramp', 90, target), Stage('dose_hold', 30, target)], start)
    hinge0 = env.observed.hinge.copy()
    points0 = env.observed.grip_points
    held0 = bool(env.state.metadata['grasp']['geometric_loaded_hold'])
    rows, first = [], {'tear': None, 'material_tear': None, 'anchor_break': None,
                      'no_anchor_points': None, 'geometric_hold_lost': None}
    previous = env.observed.neighbors
    initial_counts = env.failure_metrics['native_cumulative_counts'].copy()
    for label, index, command in sequence:
        result = env.replay_command(command)
        item = row(env, result, label, index)
        f = item['failure_physics']
        item['requested_displacement_mm'] = (np.asarray(command.hinge)-start.hinge).tolist()
        item['requested_displacement_mm'] = (np.asarray(item['requested_displacement_mm'])*10).tolist()
        item['actual_hinge_displacement_mm'] = ((env.observed.hinge-hinge0)*10).tolist()
        event = {'native_tick': env.native.tick, 'phase': label, 'phase_step': index,
                 'requested_displacement_mm': item['requested_displacement_mm'],
                 'actual_hinge_displacement_mm': item['actual_hinge_displacement_mm'],
                 'failure_physics': f, 'grasp': item['grasp']}
        if first['tear'] is None and f['new_tear_edges']:
            broken = (previous[:, 1::2] >= 0) & (env.observed.neighbors[:, 1::2] < 0)
            ids, axes = np.where(broken)
            targets = previous[ids, 2*axes+1]
            material = np.asarray(env.grid.inside, bool)
            keep = material[ids] & material[targets]
            ids, targets = ids[keep], targets[keep]
            first['tear'] = {**event, 'new_material_edge_ids': np.stack((ids, targets), axis=1).tolist(),
                'new_material_edge_midpoints_mm': ((env.observed.nodes[ids]+env.observed.nodes[targets])*5).tolist()}
        if first['material_tear'] is None and f['new_material_tear_edges']:
            broken = (previous[:, 1::2] >= 0) & (env.observed.neighbors[:, 1::2] < 0)
            ids, axes = np.where(broken)
            targets = previous[ids, 2*axes+1]
            material = np.asarray(env.grid.inside, bool)
            keep = material[ids] & material[targets]
            ids, targets = ids[keep], targets[keep]
            first['material_tear'] = {**event,
                'new_material_edge_ids': np.stack((ids, targets), axis=1).tolist(),
                'new_material_edge_midpoints_mm': ((env.observed.nodes[ids]+env.observed.nodes[targets])*5).tolist()}
        if first['anchor_break'] is None and f['anchors']['new_broken_support_corners']:
            first['anchor_break'] = event
        if first['no_anchor_points'] is None and points0 > 0 and env.observed.grip_points == 0:
            first['no_anchor_points'] = event
        if held0 and first['geometric_hold_lost'] is None and not item['grasp']['geometric_loaded_hold']:
            first['geometric_hold_lost'] = event
        rows.append(item)
        previous = env.observed.neighbors
    return {'dose_mm': dose, 'direction': list(direction), 'initial_grip_points': points0,
            'initial_geometric_loaded_hold': held0,
            'initial_counts': initial_counts, 'first_events': first,
            'final_failure_physics': rows[-1]['failure_physics'], 'final_grasp': rows[-1]['grasp'],
            'quality': quality_summary(rows), 'rows': rows,
            'hold_semantics': 'no_anchor_points and loss of measured bilateral geometric hold are distinct; no invented friction force'}


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--direction', choices=DIRECTIONS, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--tear', type=float)
    group.add_argument('--force', type=float)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--max-dose', type=float, default=4.)
    p.add_argument('--fresh', action='store_true')
    p.add_argument('--snapshot-acceptance', type=Path)
    args = p.parse_args()
    if args.repeats < 1 or not 0 <= args.max_dose <= 4 or args.max_dose*2 != int(args.max_dose*2):
        p.error('Dose range must be 0..4 mm in 0.5 mm increments; repeats must be positive')
    physics = PhysicsOverrides(tear_stretch_ratio=args.tear, grasp_core_break_force_n=args.force)
    dll = ROOT/'endoscopic_tissue/native/bin-s15-training-snapshot1/LiverCudaSim.dll'
    if not args.fresh:
        if args.snapshot_acceptance is None:
            p.error('Cached dose trials require completed S15 acceptance; use --fresh for an uncached pilot')
        acceptance = json.loads(args.snapshot_acceptance.read_text(encoding='utf8'))
        if (acceptance.get('schema') != 's15_native_acceptance_v1' or acceptance.get('passed') is not True
                or acceptance.get('dll_sha256') != sha256(dll)
                or acceptance.get('python_source_sha256') != source_binding()):
            raise ValueError('Unvalidated cached dose trials are prohibited')
    out = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/dose'/args.name
    out.mkdir(parents=True, exist_ok=False)
    record = {'name': args.name, 'physics': asdict(physics), 'direction': args.direction,
        'dll_sha256': sha256(dll), 'snapshot_cached': not args.fresh, 'passed': False,
        'complete_required_curve': args.repeats == 10 and args.max_dose == 4.,
        'full_surface_checks': True, 'trials': [], 'candidate_adopted': False}
    env, snapshot = None, None
    started = time.perf_counter()
    try:
        if not args.fresh:
            env = environment(preparation='cut_ready', physics=physics, telemetry=True)
            env.reset(0)
            snapshot, record['failure_law_snapshot_probe'] = native_cache_probe(env)
            atomic_json(out/'cached-parent-context.json', env.context)
        for dose in np.arange(0, args.max_dose+.01, .5):
            for repeat in range(args.repeats):
                case = f'dose-{dose:.1f}-r{repeat:02}'
                folder = out/case
                folder.mkdir()
                if args.fresh:
                    env = environment(preparation='cut_ready', physics=physics, telemetry=True)
                    env.reset(repeat)
                else:
                    env.restore(snapshot, branch_label=case)
                atomic_json(folder/'context.json', env.context)
                preparation_bad = sum(not r['quality']['checked_metrics_passed'] or
                    not r['quality']['within_declared_local_scope'] for r in env.preparation)
                trial = one_dose(env, float(dose), DIRECTIONS[args.direction])
                trial.update(repeat=repeat, case=case, preparation_failure_steps=preparation_bad)
                trial['failure_events_during_preparation'] = any(trial['initial_counts'][k] > 0 for k in
                    ('tearEdges', 'brokenCoreCorners', 'brokenGripPoints'))
                atomic_json(folder/'steps.json', trial.pop('rows'))
                atomic_json(folder/'result.json', trial)
                record['trials'].append(trial)
                atomic_json(out/'result.json', record)
                if args.fresh:
                    env.close()
                    env = None
                print(args.name, case, trial['quality'], flush=True)
        record['passed'] = record['complete_required_curve'] and all(
            r['quality']['quality_failure_steps'] == 0 and r['preparation_failure_steps'] == 0
            and r['initial_grip_points'] > 0 and not r['failure_events_during_preparation'] for r in record['trials'])
        record['scope_note'] = 'Detached fragments remain explicitly outside the existing local training scope; no scope threshold is relaxed.'
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        if env is not None:
            env.close()
        record['wall_s'] = time.perf_counter()-started
        atomic_json(out/'result.json', record)
    if not record['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
