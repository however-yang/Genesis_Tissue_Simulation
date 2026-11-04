from types import SimpleNamespace
import unittest
import numpy as np

from snapshot_step_probe import difference


class DiagnosticMasks(unittest.TestCase):
    def test_native_integer_material_mask_is_not_an_index_list(self):
        grid = SimpleNamespace(inside=np.array([0, 0, 1, 1], np.int32), active=np.ones(4, np.int32))
        before = np.zeros((4, 3))
        after = before.copy()
        after[2, 0] = .00002
        after[0, 0] = .00003
        result = difference(before, after, grid)
        self.assertAlmostEqual(result['max_node_mm'], .03)
        self.assertAlmostEqual(result['max_material_mm'], .02)
        self.assertFalse(result['material_node'])
        self.assertEqual(result['nodes_over_001mm'], 2)

    def test_bool_material_mask_matches_native_integer_mask(self):
        before = np.zeros((3, 3))
        after = before.copy()
        after[2, 1] = .00004
        results = [difference(before, after, SimpleNamespace(
            inside=np.array([0, 0, 1], dtype=dtype), active=np.ones(3, np.int32)))
            for dtype in (bool, np.int32)]
        self.assertEqual(*results)


if __name__ == '__main__':
    unittest.main()
