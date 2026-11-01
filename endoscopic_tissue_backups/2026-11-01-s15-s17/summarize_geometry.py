"""Index real probe episodes and report full three-snip acceptance separately."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue.training.data import atomic_json, EpisodeReader
from endoscopic_tissue.training.provenance import json_hash, sha256
from endoscopic_tissue.training.splits import build_index


def main():
    out = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101'
    paths, full = [], []
    for purpose in ('collection', 'evaluation'):
        batch = out/'geometry-batches'/purpose
        spec = json.loads((batch/'spec.json').read_text(encoding='utf8'))
        rows = []
        for case in spec['episodes']:
            path = out/'damage'/case['id']/'result.json'
            if not path.exists():
                raise RuntimeError('Full script has not completed: '+case['id'])
            result = json.loads(path.read_text(encoding='utf8'))
            full.append({'id': case['id'], 'purpose': purpose, 'geometry_group': case['geometry_recipe_sha256'],
                'passed': result.get('passed', False), 'snips': result.get('snips'),
                'quality_failures': result.get('quality_failures'), 'scope_failures': result.get('scope_failures'),
                'preparation_failures': result.get('preparation_failures'), 'result_sha256': sha256(path)})
            directory = batch/case['id']/'episode'
            status = 'reset_failure'
            error = result.get('error')
            if directory.is_dir():
                try:
                    reader = EpisodeReader(directory)
                    assert reader.manifest['normal_transitions'] == 1
                    status = 'recorded'
                except Exception as exc:
                    status, error = 'probe_integrity_failure', repr(exc)
            rows.append({'id': case['id'], 'status': status, 'error': error,
                'full_script_passed': result.get('passed', False),
                'note': 'This row describes the small split probe, not a full raw-state recording'})
        path = batch/'results.json'
        atomic_json(path, {'spec_sha256': json_hash(spec), 'episodes': rows})
        paths.append(path)
    index = build_index(paths)
    atomic_json(out/'geometry-batches/splits.json', index)
    assert len(full) == 15
    record = {'schema': 's17b_geometry_acceptance_v1', 'full_scripts': full,
        'has_held_out_validation': index['has_held_out_validation'],
        'evaluation_groups_disjoint': not index['evaluation_collection_geometry_overlap'],
        'passed': all(r['passed'] for r in full) and index['has_held_out_validation'] and not index['evaluation_collection_geometry_overlap'],
        'candidate_adopted': False,
        'storage_note': '15 full quality/interaction logs and 3 complete video inputs; only actual adjacent probes are indexed as small episodes'}
    atomic_json(out/'geometry-batches/result.json', record)
    print(json.dumps({k: v for k, v in record.items() if k != 'full_scripts'}, indent=2))
    if not record['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
