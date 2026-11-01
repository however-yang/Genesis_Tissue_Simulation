"""Regression checks for physical-template leakage and incomplete data indexing."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue.training.data import atomic_json, save_arrays
from endoscopic_tissue.training.provenance import json_hash, sha256
from endoscopic_tissue.training.tasks import material_reference

_spec = importlib.util.spec_from_file_location('splits_candidate', Path(__file__).with_name('splits_candidate.py'))
splits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(splits)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='split-test-', dir=Path(__file__).parent)
        self.root = Path(self.temporary.name)
        self.sources = {p: sha256(ROOT/p) for p in ('endoscopic_tissue/scenario.py',
                                                  'endoscopic_tissue/grid.py', 'endoscopic_tissue/kidney.py')}
        self.static = {'rest_positions_m': np.array([[0., 0., 0.], [.001, 0., 0.]], np.float32),
                       'initial_neighbors': np.array([[-1, 1], [0, -1]], np.int32),
                       'material_mask': np.ones(2, np.int32), 'active_mask': np.ones(2, np.int32)}

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, name, purpose='collection', scene='wall-traction', ids=('a',), failure=False, bad_goal=False):
        directory = self.root/name
        directory.mkdir()
        cases, rows = [], []
        for number, episode_id in enumerate(ids):
            case_id = f'case-{number}'
            cases.append({'id': case_id})
            folder = directory/case_id/'episode'
            folder.mkdir(parents=True)
            reference = material_reference(self.static)
            context = {'episode_id': episode_id, 'scene_config': {'scene': scene, 'spacing_scale': 1+number*.25},
                       'current_adapter_and_asset_sha256': self.sources,
                       'goal': {'material_reference_sha256': 'bad' if bad_goal else reference}}
            context['context_sha256'] = json_hash(context)
            atomic_json(folder/'context.json', context)
            digest = save_arrays(folder/'static.npz.zst', self.static)
            atomic_json(folder/'manifest.json', {'status': 'closed', 'states': 2, 'normal_transitions': 1,
                        'failed_attempts': 0, 'context_sha256': context['context_sha256'],
                        'arrays_sha256': {'static.npz.zst': digest}})
            rows.append({'id': case_id, 'status': 'recorded', 'split': 'train' if number%2 == 0 else 'validation'})
        if failure:
            cases.append({'id': 'failed'})
            rows.append({'id': 'failed', 'status': 'reset_failure', 'error': 'controlled fixture'})
        spec = {'purpose': purpose, 'episodes': cases}
        atomic_json(directory/'spec.json', spec)
        atomic_json(directory/'results.json', {'episodes': rows, 'spec_sha256': json_hash(spec)})
        return directory/'results.json'

    def test_one_group_cannot_supply_validation(self):
        self.assertEqual(splits.assign_groups(['same', 'same']), {'same': 'train'})

    def test_groups_are_indivisible_and_order_independent(self):
        result = splits.assign_groups(['b', 'a', 'a'])
        self.assertEqual(result, splits.assign_groups(['a', 'b']))
        self.assertEqual(set(result.values()), {'train', 'validation'})

    def test_seed_labels_and_spacing_do_not_split_same_geometry(self):
        result = splits.build_index([self.fixture('collection', ids=('a', 'b'), failure=True)])
        self.assertEqual(result['episodes_by_split'], {'train': 2})
        self.assertFalse(result['has_held_out_validation'])
        self.assertEqual(len(result['collection_failures_preserved']), 1)
        self.assertEqual({e['historical_seed_split'] for e in result['episodes']}, {'train', 'validation'})

    def test_alias_preparation_scenes_share_geometry(self):
        a = {'scene_config': {'scene': 'wall'}, 'current_adapter_and_asset_sha256': self.sources}
        b = {'scene_config': {'scene': 'wall-traction'}, 'current_adapter_and_asset_sha256': self.sources}
        self.assertEqual(splits.geometry_recipe(a, self.sources), splits.geometry_recipe(b, self.sources))

    def test_same_geometry_evaluation_requires_explicit_repeatability(self):
        paths = [self.fixture('collection'), self.fixture('evaluation', purpose='evaluation', ids=('b',))]
        with self.assertRaisesRegex(ValueError, 'shares physical geometry'):
            splits.build_index(paths)
        result = splits.build_index(paths, evaluation_scope='repeatability')
        self.assertEqual(result['episodes_by_split'], {'train': 1, 'repeatability': 1})
        self.assertTrue(result['evaluation_collection_geometry_overlap'])

    def test_duplicate_episode_is_rejected(self):
        paths = [self.fixture('one'), self.fixture('two')]
        with self.assertRaisesRegex(ValueError, 'more than once'):
            splits.build_index(paths)

    def test_changed_geometry_source_is_rejected(self):
        context = {'scene_config': {'scene': 'wall'}, 'current_adapter_and_asset_sha256': {}}
        with self.assertRaisesRegex(ValueError, 'geometry sources differ'):
            splits.geometry_recipe(context, self.sources)

    def test_wrong_material_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Goal reference differs'):
            splits.build_index([self.fixture('bad-reference', bad_goal=True)])

    def test_incomplete_batch_cannot_be_silently_indexed(self):
        path = self.fixture('partial', ids=('a', 'b'))
        result = json.loads(path.read_text())
        result['episodes'].pop()
        atomic_json(path, result)
        with self.assertRaisesRegex(ValueError, 'Batch is incomplete'):
            splits.build_index([path])


if __name__ == '__main__':
    unittest.main()
