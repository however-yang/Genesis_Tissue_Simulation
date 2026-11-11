"""A goal with fractional material IDs must be rejected rather than truncated."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue.training.tasks import TaskGoal, TaskMonitor
from endoscopic_tissue.training.test_tasks import fixture, sample


class TargetIdentityTests(unittest.TestCase):
    def test_fractional_and_boolean_edges_are_rejected(self):
        for edge in ((.9, 1.9), (False, True)):
            with self.subTest(edge=edge), self.assertRaises(ValueError):
                TaskMonitor(TaskGoal(kind='incision', target_edges=(edge,)), fixture(), sample())


if __name__ == '__main__':
    unittest.main()
