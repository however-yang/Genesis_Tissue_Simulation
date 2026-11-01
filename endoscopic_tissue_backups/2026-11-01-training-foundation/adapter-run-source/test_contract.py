"""CPU fault-injection and independent data contracts; no GPU is required."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ..native_backend import Command, Settings
from .actions import ActionBounds, action_to_command, command_to_action
from .data import EpisodeReader, EpisodeWriter, load_arrays
from .env import EnvironmentConfig, NativeEnvironment, State, StepResult
from .provenance import json_hash, pinned_dll
from .quality import QualityMonitor


def context():
    value = {"episode_id": "cpu-fixture", "domain_validation": "unvalidated", "diagnostic_continue": False}
    value["context_sha256"] = json_hash(value)
    return value


def state(index, *, valid=True, revision=0):
    neighbors = np.full((3, 6), -1, np.int32)
    if revision == 0:
        neighbors[0, 1], neighbors[1, 0] = 1, 0
    return State({"episode_id": "cpu-fixture", "step_id": index, "t_sim_s": index/30, "dt_s": 1/30,
                  "native_tick": 30+index, "topology_revision": revision,
                  "quality": {"checked_metrics_passed": valid, "within_declared_local_scope": True}},
                 {"positions_m": np.full((3, 3), index, np.float32), "grip_flags": np.zeros(3, np.int32)},
                 {"revision": revision, "neighbors": neighbors, "cut_fractions": np.full((3, 6), -1, np.float32)})


def result(before, after, **overrides):
    values = dict(before=before, after=after, action={"policy": [float(before.metadata["step_id"])]*11},
                  terminated=False, truncated=False, reason=None, timings_ms={"total": 1.})
    return StepResult(**{**values, **overrides})


class ActionTests(unittest.TestCase):
    def test_si_translation_and_euler_round_trip_preserve_native_float_payload(self):
        command = Command((1.13, 3.19, 1.8), .875, (2.05, 1.0468, 2.6), True, .4, True, 19.9, 90, -55)
        vector = command_to_action(command)
        np.testing.assert_allclose(vector[:3], [.0113, .0319, .018], rtol=0, atol=1e-16)
        np.testing.assert_allclose(vector[7:10], np.array([19.9, 90, -55])*np.pi/180, rtol=0, atol=1e-16)
        restored = action_to_command(vector, cutting_enabled=True)
        for name in ("hinge", "blade_center", "closure", "scissors_closure", "scissors_pitch_deg", "scissors_roll_deg", "scissors_yaw_deg"):
            np.testing.assert_array_equal(np.asarray(getattr(command, name), np.float32), np.asarray(getattr(restored, name), np.float32))

    def test_separate_bounding_slew_and_environment_owned_cut_flag(self):
        bounds = ActionBounds((0,)*11, (1,)*11, (.3,)*11)
        processed, record = bounds.process(np.full(11, 2.), np.full(11, .5), .1)
        np.testing.assert_allclose(processed, .53)
        self.assertEqual(record["policy"], [2.]*11)
        self.assertEqual(record["bounded"], [1.]*11)
        self.assertFalse(action_to_command(processed, cutting_enabled=False).scissors_cutting)
        for bad in ([0.]*12, [float("nan")]*11, [float("inf")]*11):
            with self.assertRaises(ValueError):
                bounds.process(bad, np.full(11, .5), .1)

    def test_dll_pin_rejects_wrong_binary_before_native_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/"candidate.dll"
            file.write_bytes(b"not the pinned DLL")
            with self.assertRaises(ValueError):
                pinned_dll(file, "0"*64)

    def test_raw_replay_is_unavailable_in_standard_environment(self):
        bounds = ActionBounds((0,)*11, (1,)*11, (1,)*11)
        env = NativeEnvironment(EnvironmentConfig("unused.dll", "0"*64, bounds))
        with self.assertRaises(RuntimeError):
            env.replay_command(Command())

    def test_headless_import_does_not_import_genesis(self):
        self.assertNotIn("genesis", sys.modules)
        self.assertNotIn("endoscopic_tissue.rendering", sys.modules)

    def test_observation_failure_keeps_last_state_and_quarantines_raw_scene(self):
        bounds = ActionBounds((0,)*11, (1,)*11, (1,)*11)
        env = NativeEnvironment(EnvironmentConfig("unused.dll", "0"*64, bounds))
        env.state, env.observed = state(0), SimpleNamespace(blade=np.zeros(4))
        failed = SimpleNamespace(nodes=np.full((3, 3), 4., np.float32))
        env.native = SimpleNamespace(step=lambda command: failed)
        env.settings, env.step_id, env.done = Settings(), 0, False
        with patch.object(env, "_assess", side_effect=RuntimeError("surface export failed")):
            response = env._advance(Command(), {})
        self.assertTrue(response.terminated)
        self.assertIsNone(response.after)
        self.assertEqual(env.state.metadata["step_id"], 0)
        np.testing.assert_array_equal(response.failure_nodes_native_cm, failed.nodes)
        self.assertIn("observation_failure", response.reason)

    def test_cut_ready_requires_grasp_before_first_scissors_approach(self):
        from .env import SceneConfig
        with self.assertRaises(ValueError):
            SceneConfig("wall", "cut_ready")


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)/"episode"

    def tearDown(self):
        self.temp.cleanup()

    def record(self, states):
        with EpisodeWriter(self.directory, context(), states[0], {"node_ids": np.arange(3)}, queue_size=1) as writer:
            for before, after in zip(states[:-1], states[1:]):
                writer.append(result(before, after))

    def test_n_transitions_n_plus_one_states_and_topology_alignment(self):
        self.record([state(0), state(1), state(2, revision=1), state(3, revision=1)])
        reader = EpisodeReader(self.directory)
        self.assertEqual(reader.manifest["normal_transitions"], 3)
        self.assertEqual(reader.manifest["states"], 4)
        self.assertEqual(len(list(self.directory.glob("topology-*.zst"))), 2)
        windows = list(reader.windows(2, allow_unvalidated_scope=True))
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["states"][0][1]["positions_m"][0, 0], 0)
        self.assertEqual(windows[0]["states"][2][1]["positions_m"][0, 0], 2)
        self.assertEqual(windows[0]["states"][1][2]["neighbors"][0, 1], 1)
        self.assertEqual(windows[0]["states"][2][2]["neighbors"][0, 1], -1)
        with self.assertRaises(ValueError):
            list(reader.windows(1))

    def test_no_crossing_invalid_state_or_reset(self):
        self.record([state(0), state(1), state(2, valid=False), state(3), state(4)])
        windows = list(EpisodeReader(self.directory).windows(1, allow_unvalidated_scope=True))
        self.assertEqual([w["transitions"][0]["step_id"] for w in windows], [0, 3])
        other = Path(self.temp.name)/"other"
        with EpisodeWriter(other, context(), state(0), {}) as writer:
            s = state(1)
            s.metadata["episode_id"] = "after-reset"
            with self.assertRaises(ValueError):
                writer.append(result(state(0), s))

    def test_failure_attempt_does_not_invent_next_state(self):
        with EpisodeWriter(self.directory, context(), state(0), {}) as writer:
            writer.append(result(state(0), None, terminated=True, reason="native_failure",
                                 failure_nodes_native_cm=np.array([[np.nan, 0, 0]], np.float32)))
        reader = EpisodeReader(self.directory)
        self.assertEqual(reader.manifest["states"], 1)
        self.assertEqual(reader.manifest["normal_transitions"], 0)
        self.assertEqual(reader.manifest["failed_attempts"], 1)
        self.assertEqual(list(reader.windows(1, allow_unvalidated_scope=True)), [])
        self.assertTrue(np.isnan(load_arrays(self.directory/"failure-00000000.npz.zst")["nodes_native_cm"][0, 0]))

    def test_array_snapshot_owned_before_async_write(self):
        started, release = threading.Event(), threading.Event()
        real = EpisodeWriter._write_state
        def blocked(writer, snapshot):
            started.set()
            self.assertTrue(release.wait(5))
            return real(writer, snapshot)
        initial = state(0)
        with patch.object(EpisodeWriter, "_write_state", blocked):
            writer = EpisodeWriter(self.directory, context(), initial, {})
            self.assertTrue(started.wait(5))
            initial.arrays["positions_m"][:] = 99
            initial.topology["neighbors"][:] = 42
            transition = result(state(0), state(1))
            writer.append(transition)
            transition.action["policy"][0] = 99
            release.set()
            writer.close()
        reader = EpisodeReader(self.directory)
        stored = reader.state(0)
        self.assertTrue(np.all(stored[1]["positions_m"] == 0))
        self.assertEqual(stored[2]["neighbors"][0, 1], 1)
        self.assertEqual(reader.transitions[0]["action"]["policy"][0], 0)

    def test_worker_outer_failure_is_reported_without_waiting_for_second_sentinel(self):
        writer = EpisodeWriter(self.directory, context(), state(0), {})
        writer.close()
        # Equivalent to a file-close exception after the shutdown token was consumed.
        writer.closed = False
        with patch.object(writer, "_write_loop", side_effect=OSError("injected close failure")):
            writer._work()
        with self.assertRaises(RuntimeError):
            writer.close()
        self.assertEqual(json.loads((self.directory/"manifest.json").read_text())["status"], "write_failure")

    def test_write_failure_is_explicit_and_recovery_uses_new_directory(self):
        with patch.object(EpisodeWriter, "_write_state", side_effect=OSError("injected disk failure")):
            writer = EpisodeWriter(self.directory, context(), state(0), {})
            with self.assertRaises(RuntimeError):
                writer.close()
        with self.assertRaises(ValueError):
            EpisodeReader(self.directory)
        fresh = Path(self.temp.name)/"fresh"
        with EpisodeWriter(fresh, context(), state(0), {}):
            pass
        self.assertEqual(EpisodeReader(fresh).manifest["states"], 1)

    def test_corruption_reordering_and_time_misalignment_are_rejected(self):
        self.record([state(0), state(1), state(2)])
        path = self.directory/"transitions.jsonl"
        original = path.read_text()
        lines = original.splitlines()
        path.write_text("\n".join(reversed(lines))+"\n")
        with self.assertRaises(ValueError):
            EpisodeReader(self.directory, verify_hashes=False)
        path.write_text(original)
        meta = self.directory/"state-00000001.json"
        data = json.loads(meta.read_text())
        data["t_sim_s"] += .1
        meta.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            EpisodeReader(self.directory, verify_hashes=False)

    def test_terminal_transition_rejects_following_append_and_existing_directory(self):
        with EpisodeWriter(self.directory, context(), state(0), {}) as writer:
            writer.append(result(state(0), state(1), truncated=True, reason="time_limit"))
            with self.assertRaises(ValueError):
                writer.append(result(state(1), state(2)))
        with self.assertRaises(FileExistsError):
            EpisodeWriter(self.directory, context(), state(0), {})


class QualityTests(unittest.TestCase):
    def setUp(self):
        neighbors = np.full((2, 6), -1, np.int32)
        neighbors[0, 1], neighbors[1, 0] = 1, 0
        self.grid = SimpleNamespace(positions=np.zeros((2, 3), np.float32), active=np.ones(2, np.int32),
                                    inside=np.ones(2, np.int32), neighbors=neighbors)
        self.initial = SimpleNamespace(nodes=self.grid.positions.copy(), neighbors=neighbors)

    def check(self, monitor, observation, cutting=False, material=(1, 0)):
        return monitor.assess(observation, self.initial, Command(scissors_cutting=cutting), Settings(),
                              np.asarray(material, np.float32), None, None)

    def test_no_cut_is_valid_dynamics_not_automatic_task_success(self):
        q = self.check(QualityMonitor(self.grid), self.initial)
        self.assertTrue(q["checked_metrics_passed"])
        self.assertFalse(q["training_domain_approved"])
        self.assertIn("tool_body_overlap", q["unchecked"])

    def test_disabled_cut_and_healing_fail_and_free_piece_is_out_of_scope(self):
        monitor = QualityMonitor(self.grid)
        severed = SimpleNamespace(nodes=self.grid.positions.copy(), neighbors=np.full((2, 6), -1, np.int32))
        first = self.check(monitor, severed)
        self.assertIn("new_cut_while_disabled", first["numerical_failure_reasons"])
        self.assertIn("detached_material_outside_local_task_scope", first["scope_failure_reasons"])
        healed = self.check(monitor, self.initial)
        self.assertIn("topology_irreversibility_violation", healed["numerical_failure_reasons"])

    def test_finite_but_large_spike_and_nan_quality_do_not_pass(self):
        spike = SimpleNamespace(nodes=np.full((2, 3), 7., np.float32), neighbors=self.grid.neighbors)
        q = self.check(QualityMonitor(self.grid), spike)
        self.assertIn("workspace_displacement_exceeded", q["numerical_failure_reasons"])
        q = self.check(QualityMonitor(self.grid), self.initial, material=(np.nan, np.nan))
        self.assertIn("non_finite_state", q["numerical_failure_reasons"])
        json.dumps(q, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
