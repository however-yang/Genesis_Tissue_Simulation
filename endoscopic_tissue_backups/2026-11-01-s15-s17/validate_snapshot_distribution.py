"""One independent original/restored pair for S15 T4 or T7.

Run 20 separately named pairs per fork through the GPU queue, then summarize.
Original arms always continue their actual freshly prepared native session.
"""
import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import ks_2samp
from threadpoolctl import threadpool_limits

from validation_support import (ROOT, environment, command_sequence, exact_values,
                                replay, row, quality_summary, WallCutMeasure)
from validation_capture import PairedCapture
from endoscopic_tissue.scenario import Stage
from endoscopic_tissue.training.data import atomic_json
from endoscopic_tissue.training.snapshot import SnapshotBackend, EnvSnapshot


def prepare_fork(env, fork, capture):
    scene = env.scene
    requested = next(s.command for s in scene.stages if s.name == 'lift')
    start = next(i for i, s in enumerate(scene.stages) if s.name == 'blade_approach')
    end = next(i for i, s in enumerate(scene.stages) if s.name == 'incise_1')+1
    sequence = command_sequence(scene.stages[start:end], requested)
    prefix = []
    if fork == 'before_contact':
        assert not env.observed.scissors.loads.is_squeezing
        assert env.observed.scissors.contacts == 0, 'Before-contact fixture already touches scissors'
        return sequence, prefix
    measure = WallCutMeasure(env)
    original_edges = measure.sample(env)['material_broken_edges']
    for i, (label, index, command) in enumerate(sequence):
        if capture:
            capture.label = label
        result = env.replay_command(command)
        prefix.append(row(env, result, label, index))
        cut = measure.sample(env)['material_broken_edges'] > original_edges
        squeezing = env.observed.scissors.loads.is_squeezing
        if fork == 'first_cut' and cut:
            # Include the second snip, so the continuation measures a fresh
            # cutting gesture as well as the tail of the first one.
            end2 = next(j for j, s in enumerate(scene.stages) if s.name == 'incise_2')+1
            tail = command_sequence(scene.stages[end:end2], sequence[-1][2])
            return sequence[i+1:]+tail, prefix
        if fork == 'squeezing' and squeezing and not cut:
            return sequence[i+1:], prefix
    if fork != 'regrasp':
        raise RuntimeError('Required fork event did not occur: '+fork)
    # Explicit release and fresh friction/assisted capture through the same
    # physical jaw mechanism; success requires actual new grip witnesses.
    # Move toward the attached end, inside the remaining upper wall. Moving
    # +3 mm put the jaw tips beyond the free edge: at that x the entire finite
    # pad was above the tissue, so closing produced zero load on both jaws.
    # This changes only the test's requested trajectory, never capture physics.
    held = sequence[-1][2]
    opened_scissors = replace(held, scissors_closure=0, scissors_cutting=False)
    open_jaw = replace(opened_scissors, closure=0)
    shifted = replace(open_jaw, hinge=tuple(np.asarray(open_jaw.hinge)+(-.3, 0, 0)))
    lowered = replace(shifted, hinge=tuple(np.asarray(shifted.hinge)+(0, -.12, 0)))
    closed = replace(lowered, closure=1)
    raised = replace(closed, hinge=tuple(np.asarray(closed.hinge)+(0, .12, 0)))
    stages = [Stage('regrasp_scissors_open', 30, opened_scissors),
        Stage('regrasp_release', 30, open_jaw), Stage('regrasp_release_hold', 15, open_jaw),
        Stage('regrasp_shift', 45, shifted), Stage('regrasp_lower', 20, lowered),
        Stage('regrasp_close', 30, closed), Stage('regrasp_close_hold', 10, closed),
        Stage('regrasp_lift', 20, raised)]
    saw_release = False
    for label, index, command in command_sequence(stages, held):
        if capture:
            capture.label = label
        result = env.replay_command(command)
        prefix.append(row(env, result, label, index))
        if label == 'regrasp_release_hold' and env.observed.grip_points == 0:
            saw_release = True
    if not saw_release or env.observed.grip_points <= 0:
        raise RuntimeError('Regrasp fixture failed actual release/reacquisition; no artificial attachment is permitted')
    second = [replace(s, command=replace(s.command, hinge=raised.hinge)) for s in scene.stages
              if s.name in ('reposition_2', 'incise_2')]
    return command_sequence(second, raised), prefix


def paired_run(out, args):
    record = {'fork': args.fork, 'repeat': args.repeat, 'passed': False,
              'full_surface_checks': True, 'original_is_fresh_continuation': True,
              'cached_snapshot': str(args.cache) if args.cache else None}
    capture = PairedCapture(out, 'wall-traction', args.name) if args.capture else None
    try:
        with environment(preparation='cut_ready' if args.fork == 'cache' else 'held') as env:
            _, context = env.reset(args.repeat)
            atomic_json(out/'parent-context.json', context)
            record['dll_sha256'] = context['dll_sha256']
            preparation = env.preparation
            if args.fork == 'cache':
                stage = next(s for s in env.scene.stages if s.name == 'incise_1')
                prior = next(s.command for s in env.scene.stages if s.name == 'blade_approach')
                sequence, prefix = command_sequence([stage], prior), []
            else:
                sequence, prefix = prepare_fork(env, args.fork, capture)
            atomic_json(out/'prefix-steps.json', prefix)
            if capture:
                capture.expected_end_step = env.native.tick+len(sequence)
            record['prefix_quality'] = quality_summary(prefix)
            record['preparation_quality_failures'] = sum(not r['quality']['checked_metrics_passed'] for r in preparation)
            record['preparation_scope_failures'] = sum(not r['quality']['within_declared_local_scope'] for r in preparation)
            record['fork_native_tick'] = env.native.tick
            record['fork_loads'] = asdict(env.observed.scissors.loads)
            record['fork_grip_points'] = env.observed.grip_points
            native_before = asdict(env.observed)
            snapshot = env.snapshot()
            if args.capture:
                snapshot.save(out/'fork.lcssnap')
            record['snapshot_metadata'] = snapshot.metadata
            record['snapshot_bytes'] = snapshot.size_bytes
            if args.fork == 'cache' and args.cache:
                snapshot = EnvSnapshot.load(args.cache, dll_sha256=context['dll_sha256'],
                    physics_config_sha256=snapshot.metadata['physics_config_sha256'])
                record['cache_sha256'] = snapshot.sha256
                record['cached_snapshot_metadata'] = snapshot.metadata
            if capture:
                monitor_before = copy.deepcopy(capture.monitor.__dict__)
                capture.arm = 'original'
            # replay() sets the capture label before, not after, the native step.
            def run(which):
                if capture:
                    capture.arm = which
                return replay(env, sequence, before_step=(lambda label: setattr(capture, 'label', label)) if capture else None)
            original = run('original')
            atomic_json(out/'original-steps.json', original.pop('rows'))
            _, branch = env.restore(snapshot, branch_label=args.name)
            atomic_json(out/'branch-context.json', branch)
            record['restore_ms'] = branch['restore_wall_time_ms']
            assert SnapshotBackend(env.native).save() == snapshot.native_bytes(), 'Native T1 mismatch at fork'
            record['T1_native_exact'] = True
            if args.fork != 'cache':
                exact_values(native_before, asdict(env.native.observe()))
                record['T2_observation_exact'] = True
            if capture:
                capture.monitor.__dict__.clear()
                capture.monitor.__dict__.update(monitor_before)
            restored = run('restored')
            atomic_json(out/'restored-steps.json', restored.pop('rows'))
            record.update(original=original, restored=restored)
            record['mechanism_passed'] = True
            record['both_continuations_cut_material'] = all(v['outcome']['new_material_edges'] > 0 for v in (original, restored))
            record['passed'] = (record['preparation_quality_failures'] == 0 and record['preparation_scope_failures'] == 0
                and record['both_continuations_cut_material'] and record['restore_ms'] < 11500
                and all(v['quality_failure_steps'] == v['scope_failure_steps'] == 0 for v in
                        (record['prefix_quality'], original['quality'], restored['quality'])))
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        if capture:
            capture.close()
        atomic_json(out/'result.json', record)
    return record


def summarize(paths):
    rows = [json.loads(Path(p).read_text(encoding='utf8')) for p in paths]
    assert len(rows) == 20, 'T4/T7 acceptance requires exactly 20 independent pairs'
    assert len({r['repeat'] for r in rows}) == 20 and len({r['fork'] for r in rows}) == 1
    assert all(r.get('mechanism_passed') for r in rows), 'Retain failed pairs; do not omit them from the sample'
    evidence = {}
    passed = all(r['passed'] for r in rows)
    for metric, limit in [('delta_length_mm', .8), ('first_new_material_cut_step', 5), ('max_deviation_rest_mm', .16)]:
        a = [r['original']['outcome'][metric] for r in rows]
        b = [r['restored']['outcome'][metric] for r in rows]
        missing_a, missing_b = a.count(None), b.count(None)
        av, bv = [x for x in a if x is not None], [x for x in b if x is not None]
        delta = abs(float(np.median(av))-float(np.median(bv))) if av and bv else None
        ks = ks_2samp(av, bv) if av and bv else None
        ok = missing_a == missing_b and ((delta is not None and delta <= limit) or not av and not bv)
        evidence[metric] = {'original': a, 'restored': b, 'median_difference': delta, 'limit': limit,
            'original_missing_events': missing_a, 'restored_missing_events': missing_b,
            'ks_statistic': None if ks is None else float(ks.statistic),
            'ks_pvalue': None if ks is None else float(ks.pvalue), 'within_limit': bool(ok)}
        passed &= bool(ok)
    return {'fork': rows[0]['fork'], 'pairs': 20, 'metrics': evidence, 'passed': passed,
            'all_quality_checks_passed': all(r['passed'] for r in rows),
            'note': 'Missing initiation events are reported, never encoded as zero or dropped.'}


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--fork', choices=['before_contact', 'first_cut', 'squeezing', 'regrasp', 'cache'], required=True)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    if (args.fork == 'cache') != (args.cache is not None):
        parser.error('T7 requires one explicitly pinned common cached cut_ready snapshot')
    if args.fork == 'cache' and args.capture:
        parser.error('Cached and fresh prefixes differ; use the separate T4 paired complete videos')
    out = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/snapshot-distribution'/args.name
    out.mkdir(parents=True, exist_ok=False)
    result = paired_run(out, args)
    print(json.dumps({k: v for k, v in result.items() if k not in ('snapshot_metadata', 'original', 'restored')}), flush=True)
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
