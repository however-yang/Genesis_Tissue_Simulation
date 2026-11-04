"""Freeze and summarize the original 3 x 5 S17b geometry protocol on Linux."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    args = p.parse_args()
    sys.path.insert(0, str(args.release))
    from endoscopic_tissue.training.data import atomic_json, EpisodeReader
    from endoscopic_tissue.training.provenance import json_hash, sha256
    from endoscopic_tissue.training.geometry import WallGeometry
    from endoscopic_tissue.training.splits import build_index
    if args.prepare:
        args.output.mkdir(parents=True, exist_ok=False)
        for purpose, variants in [('collection', ('wall-24mm', 'wall-25mm')),
                                   ('evaluation', ('wall-26mm-heldout',))]:
            batch = args.output/purpose
            batch.mkdir()
            cases = []
            for variant in variants:
                path = args.release/'endoscopic_tissue_backups/2026-11-01-s15-s17/geometries'/(variant+'.json')
                parameters = json.loads(path.read_text())
                for repeat in range(5):
                    cases.append(dict(id=f'a100-combined-geometry-{variant}-r{repeat}', seed=repeat,
                        geometry=parameters, geometry_file_sha256=sha256(path),
                        geometry_recipe_sha256=json_hash(WallGeometry(**parameters).recipe())))
            atomic_json(batch/'spec.json', dict(schema='geometry_acceptance_probe_v1', purpose=purpose,
                episodes=cases, evaluation_frozen_before_results=True,
                data_scope='One actual adjacent transition for split integration; full-script quality stored separately',
                full_script_results=str(args.results)))
        print('Frozen 15 cases, with wall-26mm reserved for evaluation', flush=True)
        return 0
    paths, full = [], []
    for purpose in ('collection', 'evaluation'):
        batch = args.output/purpose
        spec = json.loads((batch/'spec.json').read_text())
        rows = []
        for case in spec['episodes']:
            path = args.results/case['id']/'result.json'
            if not path.is_file():
                raise RuntimeError('Missing full-script result: '+case['id'])
            result = json.loads(path.read_text())
            full.append(dict(id=case['id'], purpose=purpose, geometry_group=case['geometry_recipe_sha256'],
                passed=result.get('passed', False), snips=result.get('snips'),
                quality_failures=result.get('quality_failures'), scope_failures=result.get('scope_failures'),
                preparation_failures=result.get('preparation_failures'), result_sha256=sha256(path)))
            directory = batch/case['id']/'episode'
            status, error = 'reset_failure', result.get('error')
            if directory.is_dir():
                try:
                    reader = EpisodeReader(directory)
                    assert reader.manifest['normal_transitions'] == 1
                    status = 'recorded'
                except Exception as exc:
                    status, error = 'probe_integrity_failure', repr(exc)
            rows.append(dict(id=case['id'], status=status, error=error,
                             full_script_passed=result.get('passed', False)))
        path = batch/'results.json'
        atomic_json(path, dict(spec_sha256=json_hash(spec), episodes=rows))
        paths.append(path)
    index = build_index(paths)
    atomic_json(args.output/'splits.json', index)
    assert len(full) == 15
    passed = (all(r['passed'] for r in full) and index['has_held_out_validation']
              and not index['evaluation_collection_geometry_overlap'])
    result = dict(schema='s17b_geometry_acceptance_v1', full_scripts=full,
        has_held_out_validation=index['has_held_out_validation'],
        evaluation_groups_disjoint=not index['evaluation_collection_geometry_overlap'],
        passed=passed, candidate_adopted=False,
        storage_note='15 full scripts; 3 full video captures; indexed probe episodes contain one actual transition each')
    atomic_json(args.output/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'full_scripts'}), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
