"""Exercise conservative grouping against the real final-v1 datasets, read-only."""
import importlib.util
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('splits_candidate', P/'splits_candidate.py')
splits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splits)
paths = sorted((ROOT/'endoscopic_tissue/outputs/training-completion-20261101/final-v1').glob('*/results.json'))

try:
    splits.build_index(paths)
except ValueError as error:
    if 'shares physical geometry' not in str(error):
        raise
    rejected = str(error)
else:
    raise AssertionError('Same-geometry evaluation incorrectly accepted as held-out')

index = splits.build_index(paths, evaluation_scope='repeatability')
assert index['episodes_by_split'] == {'train': 10, 'repeatability': 4}, index['episodes_by_split']
assert len(index['collection_failures_preserved']) == 1
assert not index['has_held_out_validation']
(P/'final-v1/split-index.json').write_text(json.dumps(index, indent=2)+'\n')
result = {'real_datasets': len(paths), 'same_geometry_held_out_evaluation_rejected': rejected,
          'episodes_by_split': index['episodes_by_split'], 'failed_attempts_preserved': 1,
          'has_held_out_validation': index['has_held_out_validation'], 'passed': True}
(P/'split-integration-test.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
