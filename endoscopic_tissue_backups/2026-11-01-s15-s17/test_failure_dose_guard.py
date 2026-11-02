"""No valid-looking partial dose curve may be accepted as a full trajectory."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import validate_failure_dose as worker
from endoscopic_tissue.native_backend import Command
from endoscopic_tissue.scenario import Stage


def fake_environment():
    env = SimpleNamespace(
        scene=SimpleNamespace(stages=[Stage('blade_approach', 1, Command(hinge=(0, 0, 0)))]),
        observed=SimpleNamespace(hinge=np.zeros(3), grip_points=1,
                                 neighbors=np.full((2, 6), -1, dtype=np.int32), nodes=np.zeros((2, 3))),
        grid=SimpleNamespace(positions=np.zeros((2, 3)), inside=np.ones(2, dtype=bool)),
        native=SimpleNamespace(tick=0),
        state=SimpleNamespace(metadata={'grasp': {'geometric_loaded_hold': True}}),
        failure_metrics={'native_cumulative_counts': {
            'tearEdges': 0, 'brokenCoreCorners': 0, 'brokenGripPoints': 0}},
        context={}, preparation=[])

    def advance(command):
        env.native.tick += 1
        env.observed.hinge = np.asarray(command.hinge)

    env.replay_command = advance
    env.reset = lambda _: None
    env.close = lambda: None
    return env


def recorded_row(env, result, label, index, terminal_tick=None):
    stopped = env.native.tick == terminal_tick
    return {
        'quality': {'checked_metrics_passed': True,
                    'within_declared_local_scope': not stopped,
                    'numerical_failure_reasons': [],
                    'scope_failure_reasons': ['detached_material_outside_local_task_scope'] if stopped else [],
                    'metrics': {'healed_edges': 0}},
        'grasp': {'geometric_loaded_hold': True},
        'failure_physics': {'new_tear_edges': 0, 'new_material_tear_edges': 0,
                            'anchors': {'new_broken_support_corners': int(stopped)}}}


class DoseGuardTests(unittest.TestCase):
    def test_existing_limit_preserves_terminal_event_and_prevents_another_step(self):
        env = fake_environment()
        with patch.object(worker, 'row', side_effect=lambda *a: recorded_row(*a, terminal_tick=2)):
            trial = worker.one_dose(env, 4., (1, 0, 0))
        self.assertEqual(env.native.tick, 2)
        self.assertEqual(len(trial['rows']), 2)
        self.assertEqual(trial['first_events']['anchor_break']['native_tick'], 2)
        self.assertEqual(trial['quality']['scope_failure_steps'], 1)
        self.assertFalse(trial['completed_requested_steps'])
        self.assertTrue(trial['terminated_at_existing_limit'])

    def test_valid_trajectory_still_runs_full_ramp_and_hold(self):
        env = fake_environment()
        with patch.object(worker, 'row', side_effect=recorded_row):
            trial = worker.one_dose(env, 4., (1, 0, 0))
        self.assertEqual(env.native.tick, 120)
        np.testing.assert_allclose(env.observed.hinge, [.4, 0, 0])
        self.assertTrue(trial['completed_requested_steps'])
        self.assertEqual(trial['terminated_at_existing_limit'], [])

    def test_complete_dose_grid_cannot_hide_truncated_trajectories(self):
        env = fake_environment()
        with patch.object(worker, 'row', side_effect=lambda *a: recorded_row(*a, terminal_tick=2)):
            partial = worker.one_dose(env, 0., (1, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'dose'
            with patch.object(worker, 'OUT', out), patch.object(worker, 'environment', return_value=env), \
                    patch.object(worker, 'sha256', return_value='fixture'), \
                    patch.object(worker, 'one_dose', side_effect=lambda *a: copy.deepcopy(partial)), \
                    patch('sys.argv', ['dose', '--name', 'truncated', '--direction', '+x', '--tear', '2', '--fresh']), \
                    patch('builtins.print'):
                with self.assertRaises(SystemExit) as raised:
                    worker.main()
            self.assertEqual(raised.exception.code, 2)
            folder = out / 'truncated'
            record = json.loads((folder / 'result.json').read_text(encoding='utf8'))
            self.assertTrue(record['complete_required_curve'])
            self.assertEqual(len(record['trials']), 90)
            self.assertFalse(record['passed'])
            self.assertEqual(len(json.loads((folder / 'dose-0.0-r00/steps.json').read_text(encoding='utf8'))), 2)
            with np.load(folder / 'dose-0.0-r00/terminal-state.npz') as state:
                self.assertEqual(int(state['native_tick']), 2)


if __name__ == '__main__':
    unittest.main()
