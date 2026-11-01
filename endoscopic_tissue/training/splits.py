"""Build a conservative split index over complete recorded episodes.

Seed-based collector labels remain historical metadata. This index groups by
the recorded, source-verified tissue geometry recipe, so a changed policy seed,
preparation, contact mode or mesh spacing cannot create held-out anatomy.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from endoscopic_tissue.training.data import atomic_json, load_arrays
from endoscopic_tissue.training.provenance import json_hash, sha256
from endoscopic_tissue.training.tasks import material_reference
from endoscopic_tissue.scenario import SCISSORS_SCENES

ROOT = Path(__file__).resolve().parents[2]


def geometry_recipe(context, source_hashes):
    for path, digest in source_hashes.items():
        if context['current_adapter_and_asset_sha256'].get(path) != digest:
            raise ValueError('Recorded geometry sources differ; use their archived version to build the index')
    scene = SCISSORS_SCENES[context['scene_config']['scene']]
    return ({'kind': 'shaped_tissue', 'shape': asdict(scene.shape)} if scene.shape is not None
            else {'kind': 'block_with_tab', 'tab': asdict(scene.tab)})


def assign_groups(groups, validation_fraction=.2):
    """Assign whole physical groups; one group cannot supply held-out validation."""
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between zero and one')
    ordered = sorted(set(groups))
    if len(ordered) < 2:
        return {g: 'train' for g in ordered}
    count = min(len(ordered)-1, max(1, round(len(ordered)*validation_fraction)))
    validation = set(ordered[:count])
    return {g: 'validation' if g in validation else 'train' for g in ordered}


def build_index(result_paths, *, evaluation_scope='held_out_geometry', validation_fraction=.2):
    if evaluation_scope not in ('held_out_geometry', 'repeatability'):
        raise ValueError('Explicit evaluation scope must be held_out_geometry or repeatability')
    episodes, failures, inputs = [], [], []
    seen_episodes = set()
    source_hashes = {p: sha256(ROOT/p) for p in ('endoscopic_tissue/scenario.py',
                                               'endoscopic_tissue/grid.py', 'endoscopic_tissue/kidney.py')}
    for result_path in map(Path, result_paths):
        result_path = result_path.resolve(strict=True)
        result = json.loads(result_path.read_text(encoding='utf8'))
        spec = json.loads((result_path.parent/'spec.json').read_text(encoding='utf8'))
        if json_hash(spec) != result['spec_sha256']:
            raise ValueError('Result/spec hash mismatch')
        if {r['id'] for r in result['episodes']} != {c['id'] for c in spec['episodes']}:
            raise ValueError('Batch is incomplete; finish or explicitly retain its failure results first')
        inputs.append({'results': str(result_path), 'sha256': sha256(result_path)})
        for row in result['episodes']:
            directory = result_path.parent/row['id']/'episode'
            if row['status'] != 'recorded':
                failures.append({'results': str(result_path), 'case': row['id'], 'status': row['status'],
                                 'error': row.get('error')})
                continue
            context = json.loads((directory/'context.json').read_text(encoding='utf8'))
            manifest = json.loads((directory/'manifest.json').read_text(encoding='utf8'))
            if manifest['status'] != 'closed' or manifest['states'] != manifest['normal_transitions']+1:
                raise ValueError('Cannot index incomplete episode')
            if manifest['context_sha256'] != context['context_sha256'] or json_hash(
                    {k: v for k, v in context.items() if k != 'context_sha256'}) != context['context_sha256']:
                raise ValueError('Episode context/hash mismatch')
            if sha256(directory/'static.npz.zst') != manifest['arrays_sha256']['static.npz.zst']:
                raise ValueError('Static geometry/hash mismatch')
            reference = material_reference(load_arrays(directory/'static.npz.zst'))
            if context.get('goal') is not None and context['goal'].get('material_reference_sha256') != reference:
                raise ValueError('Goal reference differs from recorded physical geometry')
            episode_id = context['episode_id']
            if episode_id in seen_episodes:
                raise ValueError('Same episode supplied more than once')
            seen_episodes.add(episode_id)
            recipe = geometry_recipe(context, source_hashes)
            episodes.append({'episode_id': episode_id, 'path': str(directory),
                             'context_sha256': context['context_sha256'],
                             'material_reference_sha256': reference,
                             'geometry_group': json_hash(recipe), 'geometry_recipe': recipe,
                             'purpose': spec['purpose'], 'historical_seed_split': row.get('split'),
                             'status': row['status'], 'reason': row.get('reason'),
                             'transitions': manifest['normal_transitions'],
                             'failed_attempts': manifest['failed_attempts'],
                             'training_domain_approved': False})
    collection_groups = {e['geometry_group'] for e in episodes if e['purpose'] == 'collection'}
    evaluation_groups = {e['geometry_group'] for e in episodes if e['purpose'] == 'evaluation'}
    overlap = sorted(collection_groups & evaluation_groups)
    if overlap and evaluation_scope != 'repeatability':
        raise ValueError('Evaluation shares physical geometry with collection; use explicit repeatability scope '
                         'or collect a different held-out geometry')
    assignments = assign_groups(collection_groups, validation_fraction)
    for episode in episodes:
        episode['split'] = (assignments[episode['geometry_group']] if episode['purpose'] == 'collection'
                            else 'repeatability' if evaluation_scope == 'repeatability' else 'evaluation')
    counts = defaultdict(int)
    for episode in episodes:
        counts[episode['split']] += 1
    result = {'schema': 'physical_geometry_split_v1', 'inputs': inputs,
              'grouping': 'source-verified geometry recipe; conservative across policies, contact modes, preparation and spacing',
              'geometry_source_sha256': source_hashes,
              'validation_fraction_requested': validation_fraction,
              'collection_groups': assignments, 'evaluation_scope': evaluation_scope,
              'evaluation_collection_geometry_overlap': overlap, 'episodes_by_split': dict(counts),
              'has_held_out_validation': bool(counts['validation']),
              'collection_failures_preserved': failures, 'episodes': episodes,
              'training_domain_approved': False,
              'reader_requirement': 'Use EpisodeReader hash/quality checks for windows. This index does not approve '
                                    'any physical domain or make corrupt/numerically invalid transitions trainable.'}
    result['index_sha256'] = json_hash(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--evaluation-scope', choices=('held_out_geometry', 'repeatability'), default='held_out_geometry')
    parser.add_argument('--validation-fraction', type=float, default=.2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output index must not already exist')
    result = build_index(args.results, evaluation_scope=args.evaluation_scope, validation_fraction=args.validation_fraction)
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('episodes_by_split', 'has_held_out_validation',
                                           'evaluation_scope', 'training_domain_approved')}, indent=2))


if __name__ == '__main__':
    main()
