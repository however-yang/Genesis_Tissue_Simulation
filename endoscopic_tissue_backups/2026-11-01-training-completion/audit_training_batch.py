"""Audit completed batch metadata without starting physics or altering recordings.

The original episode split is reported, not silently reinterpreted as a held-out
physical scene. Same-template repeats are useful repeatability evidence only.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue.training.provenance import json_hash, sha256
from endoscopic_tissue.training.data import load_arrays


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def audit(directory):
    results = read(directory / 'results.json')
    spec = read(directory / 'spec.json')
    planned = {c['id']: c for c in spec['episodes']}
    records = []
    action_fingerprints = defaultdict(list)
    for row in results['episodes']:
        case = planned[row['id']]
        entry = {k: row.get(k) for k in ('id', 'status', 'reason', 'split', 'worker_peak_rss_bytes')}
        folder = directory / row['id'] / 'episode'
        if row['status'] != 'recorded':
            records.append(entry)
            continue
        manifest = read(folder / 'manifest.json')
        context = read(folder / 'context.json')
        transitions = [json.loads(line) for line in (folder / 'transitions.jsonl').read_text().splitlines()]
        metadata = [read(folder / f'state-{i:08}.json') for i in range(manifest['states'])]
        errors = []
        if manifest['states'] != manifest['normal_transitions'] + 1:
            errors.append('N_plus_one_mismatch')
        if len(transitions) != manifest['normal_transitions'] + manifest['failed_attempts']:
            errors.append('transition_count_mismatch')
        if any(sha256(folder / p) != digest for p, digest in manifest['metadata_sha256'].items()):
            errors.append('metadata_hash_mismatch')
        if json_hash({k: v for k, v in context.items() if k != 'context_sha256'}) != context['context_sha256']:
            errors.append('context_hash_mismatch')
        topology_checks, asymmetric_links, healed_links, invalid_neighbor_ids = 0, 0, 0, 0
        previous_neighbors = None
        for name in sorted(n for n in manifest['arrays_sha256'] if n.startswith('topology-')):
            if sha256(folder/name) != manifest['arrays_sha256'][name]:
                errors.append('topology_array_hash_mismatch:'+name)
            neighbors = load_arrays(folder/name)['neighbors']
            topology_checks += 1
            invalid_neighbor_ids += int(np.count_nonzero((neighbors < -1) | (neighbors >= len(neighbors))))
            for slot in range(neighbors.shape[1]):
                owners = np.flatnonzero((neighbors[:, slot] >= 0) & (neighbors[:, slot] < len(neighbors)))
                targets = neighbors[owners, slot]
                asymmetric_links += int(np.count_nonzero(neighbors[targets, slot ^ 1] != owners))
            if previous_neighbors is not None:
                healed_links += int(np.count_nonzero((previous_neighbors < 0) & (neighbors >= 0)))
                if np.any((previous_neighbors >= 0) & (neighbors >= 0) & (previous_neighbors != neighbors)):
                    errors.append('existing_neighbor_changed')
            previous_neighbors = neighbors
        if asymmetric_links or healed_links or invalid_neighbor_ids:
            errors.append('raw_topology_reciprocity_or_irreversibility_failure')
        failures, scope = Counter(), Counter()
        new_cuts, material_cuts = 0, 0
        max_overlap_m = max_speed_m_s = 0.
        min_volume_ratio = 1.
        for i, state in enumerate(metadata):
            if state['step_id'] != i or state['episode_id'] != context['episode_id']:
                errors.append(f'state_identity_{i}')
            if abs(state['t_sim_s'] - i * state['dt_s']) > 1e-10:
                errors.append(f'state_time_{i}')
            q = state['quality']
            failures.update(q['numerical_failure_reasons'])
            scope.update(q['scope_failure_reasons'])
            m = q['metrics']
            max_overlap_m = max(max_overlap_m, m.get('blade_body_overlap_m', 0), m.get('wrist_overlap_m', 0))
            max_speed_m_s = max(max_speed_m_s, m.get('max_estimated_speed_m_s') or 0)
            min_volume_ratio = min(min_volume_ratio, m.get('min_volume_ratio') or 0)
            if i:
                new_cuts += m['new_cut_edges']
                material_cuts += (state.get('cut_origin_bracket') or {}).get('new_material_cut_edges', 0)
        if not case.get('cutting_enabled', False) and new_cuts:
            errors.append('disabled_cutting_changed_topology')
        digest = hashlib.sha256()
        for i, transition in enumerate(transitions):
            if transition['step_id'] != i or transition['before'] != i:
                errors.append(f'transition_order_{i}')
            if transition['terminated'] and transition['bootstrap_allowed']:
                errors.append(f'terminal_bootstrap_{i}')
            if transition['task'] is not None:
                task = transition['task']
                if task['success'] and (task['failure'] or not task['physical_valid_for_task']):
                    errors.append(f'invalid_success_{i}')
            digest.update(json.dumps(transition['action']['policy'], separators=(',', ':')).encode())
        template = json_hash(context['scene_config'])
        fingerprint = template + ':' + digest.hexdigest()
        action_fingerprints[fingerprint].append({'id': row['id'], 'split': row['split']})
        entry.update(states=manifest['states'], transitions=manifest['normal_transitions'],
                     failed_attempts=manifest['failed_attempts'], errors=errors,
                     numerical_failures=dict(failures), scope_failures=dict(scope),
                     total_new_edges=new_cuts, new_material_edges=material_cuts,
                     max_body_overlap_m=max_overlap_m, max_estimated_speed_m_s=max_speed_m_s,
                     min_volume_ratio=min_volume_ratio,
                     topology_arrays_checked=topology_checks, asymmetric_directed_links=asymmetric_links,
                     healed_directed_links=healed_links, invalid_neighbor_ids=invalid_neighbor_ids,
                     initial_cut_edges=metadata[0]['quality']['metrics']['new_cut_edges'],
                     final_grasp=metadata[-1]['grasp'],
                     final_task=transitions[-1]['task'] if transitions else None,
                     writer_enqueue_ms=manifest['record_enqueue_ms'], writer_work_ms=manifest['record_worker_ms'])
        records.append(entry)
    duplicate_actions = [v for v in action_fingerprints.values() if len(v) > 1]
    cross_split_duplicates = [v for v in duplicate_actions if len({r['split'] for r in v}) > 1]
    return {'schema': 'batch_metadata_audit_v1', 'source': str(directory),
            'expected_episodes': len(planned), 'completed_attempts': len(records),
            'all_expected_attempts_present': set(planned) == {r['id'] for r in records},
            'statuses': dict(Counter(r['status'] for r in records)),
            'reasons': dict(Counter(r.get('reason') for r in records)),
            'total_transitions': sum(r.get('transitions', 0) for r in records),
            'metadata_integrity_errors': sum(len(r.get('errors', [])) for r in records),
            'topology_arrays_checked': sum(r.get('topology_arrays_checked', 0) for r in records),
            'asymmetric_directed_links': sum(r.get('asymmetric_directed_links', 0) for r in records),
            'healed_directed_links': sum(r.get('healed_directed_links', 0) for r in records),
            'numerical_failure_episodes': sum(bool(r.get('numerical_failures')) for r in records),
            'scope_failure_episodes': sum(bool(r.get('scope_failures')) for r in records),
            'same_template_exact_policy_action_sequences': duplicate_actions,
            'cross_seed_split_duplicate_action_sequences': cross_split_duplicates,
            'split_limitation': 'Seed-based episode split is not a held-out physical template. '
                                'Repeated hold/feedback paths may cross this provisional split; '
                                'use the whole physical template as a group before model evaluation.',
            'training_domain_approved': False, 'records': records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'same_template_exact_policy_action_sequences',
                                                                'cross_seed_split_duplicate_action_sequences')}, indent=2))


if __name__ == '__main__':
    main()
