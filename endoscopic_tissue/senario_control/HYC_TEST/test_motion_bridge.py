"""Check external-input acceptance, independent motion and geometric correctness without CUDA."""
import json
import subprocess
import sys
import unittest
from math import ceil
from pathlib import Path

import numpy as np

from .. import config
from ..instrument_control import Grasper, JointTargets, Scissors, joint_transition, load_config
from ..instrument_driver import MotionBridge


class MotionBridgeTests(unittest.TestCase):
    def setUp(self):
        configured = load_config(Path(__file__).parents[1] / "instrument_control" / "configs" / "instruments.toml")
        self.grasper = Grasper(configured.grasper)
        self.scissors = Scissors(configured.scissors)
        self.bridge = MotionBridge(self.grasper, self.scissors)

    def test_import_does_not_load_tissue_or_cuda(self):
        command = (
            "import sys; from endoscopic_tissue.senario_control.instrument_driver import MotionBridge; "
            "from endoscopic_tissue.senario_control.instrument_control import Grasper, Scissors; "
            "MotionBridge(Grasper(), Scissors()); "
            "assert not any(name in sys.modules for name in "
            "('endoscopic_tissue.native_backend', 'endoscopic_tissue.grid', "
            "'endoscopic_tissue.senario_control.simulator', 'endoscopic_tissue.senario_control.cuda_adapter'))"
        )
        completed = subprocess.run([sys.executable, "-B", "-c", command], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_default_motion_config_loads(self):
        state = self.bridge.snapshot()
        self.assertAlmostEqual(state["grasper"]["joints"]["opening"]["angle_deg"], 20)
        self.assertAlmostEqual(state["scissors"]["joints"]["opening"]["angle_deg"], 79.9998)
        json.dumps(state, allow_nan=False)

    def test_scenario_config_initial_angles_also_fit_motion_limits(self):
        configured = config.load_config(Path(__file__).parents[1] / "configs" / "kidney.toml")
        self.assertLessEqual(configured.scissors.opening.initial_deg, configured.scissors.opening.upper_deg)
        Scissors(configured.scissors)

    def test_submit_records_targets_without_advancing_or_affecting_other_instrument(self):
        before = self.bridge.snapshot()
        self.bridge.submit({"grasper": {"pitch_deg": 25, "opening_deg": 10}})
        after = self.bridge.snapshot()
        self.assertEqual(after, before)
        after = self.bridge.step(dt_s=.01)
        self.assertEqual(after["grasper"]["joints"]["pitch"]["target_deg"], 25)

    def test_external_frame_updates_both_instruments_with_speed_limits(self):
        state = self.bridge.step({
            "grasper": {"roll_deg": 45, "pitch_deg": 20},
            "scissors": {"yaw_deg": -30, "opening_deg": 40}, "is_cutting": True,
        }, dt_s=.1)
        self.assertAlmostEqual(state["grasper"]["joints"]["roll"]["angle_deg"], 9)
        self.assertAlmostEqual(state["grasper"]["joints"]["pitch"]["angle_deg"], 6)
        self.assertAlmostEqual(state["scissors"]["joints"]["yaw"]["angle_deg"], -6)
        self.assertAlmostEqual(state["scissors"]["joints"]["opening"]["angle_deg"], 75.4998)
        self.assertTrue(state["is_cutting"])

    def test_base_translation_moves_every_link_by_the_same_displacement(self):
        before = self.bridge.snapshot()
        old_position = self.bridge.grasper.base[:3, 3].copy()
        delta = np.array((1.2, -.7, .4))
        after = self.bridge.step({"grasper": {"base_position_cm": (old_position + delta).tolist()}}, dt_s=.01)
        self.assertEqual(after["scissors"], before["scissors"])
        for name, transform in after["grasper"]["transforms"].items():
            expected = np.array(before["grasper"]["transforms"][name])
            expected[:3, 3] += delta
            np.testing.assert_allclose(transform, expected, atol=1e-12, rtol=0)

    def test_orientation_only_update_preserves_base_position(self):
        old_position = self.bridge.grasper.base[:3, 3].copy()
        self.bridge.step({"grasper": {"base_quaternion_xyzw": [0, 0, np.sqrt(2), np.sqrt(2)]}}, dt_s=.01)
        np.testing.assert_allclose(self.bridge.grasper.base[:3, 3], old_position, atol=1e-12, rtol=0)
        np.testing.assert_allclose(self.bridge.grasper.base[:3, 0], (0, 1, 0), atol=1e-12, rtol=0)

    def test_pitch_rotates_the_distal_link_about_its_actual_pivot(self):
        instrument = Grasper()
        instrument.set_joint_targets(pitch_deg=90)
        instrument.advance(1)
        pose = instrument.pose()
        np.testing.assert_allclose(pose.pitch[:3, 3], (.91, 0, 0), atol=1e-12, rtol=0)
        np.testing.assert_allclose(pose.yaw[:3, 3], (.91, 0, .91), atol=1e-12, rtol=0)
        np.testing.assert_allclose(pose.yaw[:3, 0], (0, 0, 1), atol=1e-12, rtol=0)

    def test_both_jaw_tip_distances_match_the_reported_opening_gap(self):
        for instrument, opening in ((Grasper(), 20), (Scissors(), 60)):
            instrument.set_joint_targets(roll_deg=35, pitch_deg=25, yaw_deg=-20, opening_deg=opening)
            instrument.advance(2)
            pose = instrument.pose()
            local_tip = np.array((instrument.config.jaw_length_cm, 0, 0, 1))
            upper = (pose.upper_jaw @ local_tip)[:3]
            lower = (pose.lower_jaw @ local_tip)[:3]
            self.assertAlmostEqual(np.linalg.norm(upper - lower), pose.gap_cm)
            for transform in (pose.roll, pose.pitch, pose.yaw, pose.upper_jaw, pose.lower_jaw):
                np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12, rtol=0)
                self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1)

    def test_invalid_inputs_leave_both_instruments_unchanged(self):
        invalid = (
            {"grasper": {"pitch_deg": 10}, "scissors": {"opening_deg": 100}},
            {"grasper": {"base_position_cm": [1, 2, 3]}, "scissors": {"yaw_deg": float("nan")}},
            {"grasper": {"roll_deg": 20}, "is_cutting": "false"},
            {"grasper": {"base_quaternion_xyzw": [0, 0, 0, 0]}},
            {"grasper": {"base_position_cm": [1, 2]}},
            {"scissors": {"roll_deg": True}},
            {"scissors": {"roll_deg": "10"}},
            {"scissors": {"unexpected": 1}},
            {"grasper": {"base_position_cm": [1, 2, 3], "linear_velocity_cm_s": [1, 0, 0]}},
            {"scissors": {"base_quaternion_xyzw": [0, 0, 0, 1], "angular_velocity_deg_s": [0, 0, 90]}},
            {"unknown_instrument": {}},
            {1: {}, "another_field": {}},
            {"scissors": {1: 10, "another_field": 20}},
            {"scissors": None},
            [],
        )
        for payload in invalid:
            with self.subTest(payload=payload):
                before = self.bridge.snapshot()
                with self.assertRaises(ValueError):
                    self.bridge.step(payload, dt_s=.1)
                self.assertEqual(self.bridge.snapshot(), before)

    def test_invalid_timestep_does_not_apply_valid_input(self):
        for dt_s in (0, -.1, float("inf"), float("nan"), True, "0.1"):
            with self.subTest(dt_s=dt_s):
                before = self.bridge.snapshot()
                with self.assertRaises(ValueError):
                    self.bridge.step({"grasper": {"roll_deg": 20}}, dt_s=dt_s)
                self.assertEqual(self.bridge.snapshot(), before)

    def test_zero_dimensional_vector_is_rejected_as_invalid_input(self):
        for value in (np.array(1.0), [[1], [2], [3]], np.zeros((3, 1))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.bridge.submit({"grasper": {"base_position_cm": value}})

    def test_snapshot_cannot_modify_live_state(self):
        before = self.bridge.snapshot()
        changed = self.bridge.snapshot()
        changed["grasper"]["base_transform"][0][3] = 999
        changed["scissors"]["transforms"]["yaw"][0][0] = 999
        changed["grasper"]["joints"]["pitch"]["target_deg"] = 999
        self.assertEqual(self.bridge.snapshot(), before)

    def test_smooth_transition_reaches_targets_and_respects_peak_speed(self):
        dt_s, duration_s = .01, 1.503
        samples = joint_transition((
            (self.grasper, JointTargets(pitch_deg=30, opening_deg=10)),
            (self.scissors, JointTargets(yaw_deg=-30, opening_deg=40)),
        ), duration_s=duration_s, dt_s=dt_s)
        frames = [self.bridge.step(dt_s=dt_s) for _ in samples]
        self.assertEqual(len(frames), ceil(duration_s / dt_s))
        self.assertAlmostEqual(frames[-1]["grasper"]["joints"]["pitch"]["angle_deg"], 30)
        self.assertAlmostEqual(frames[-1]["scissors"]["joints"]["opening"]["angle_deg"], 40)
        pitch = np.array([0] + [frame["grasper"]["joints"]["pitch"]["angle_deg"] for frame in frames])
        self.assertTrue(np.all(np.diff(pitch) >= 0))
        self.assertLess(np.diff(pitch)[0], np.max(np.diff(pitch)) / 10)
        self.assertLess(np.diff(pitch)[-1], np.max(np.diff(pitch)) / 10)
        for frame in frames:
            for name, instrument in (("grasper", self.bridge.grasper), ("scissors", self.bridge.scissors)):
                for field, joint in zip(("roll", "pitch", "yaw", "opening"), instrument.joints):
                    self.assertLessEqual(abs(frame[name]["joints"][field]["velocity_deg_s"]),
                                         joint.config.max_speed_deg_s + 1e-9)

    def test_external_smooth_sampler_rejects_invalid_targets_without_moving_instruments(self):
        for targets, duration_s in ((JointTargets(roll_deg=90), .01), (JointTargets(opening_deg=100), 1)):
            with self.subTest(targets=targets):
                before = self.bridge.snapshot()
                with self.assertRaises(ValueError):
                    list(joint_transition(((self.grasper, targets),), duration_s=duration_s, dt_s=.01))
                self.assertEqual(self.bridge.snapshot(), before)

    def test_external_scheduler_can_pause_smooth_target_sampling(self):
        iterator = joint_transition(((self.grasper, JointTargets(yaw_deg=25)),), duration_s=1, dt_s=.02)
        before = self.bridge.snapshot()
        self.assertEqual(self.bridge.snapshot(), before)
        next(iterator)
        first = self.bridge.step(dt_s=.02)
        self.assertGreater(first["grasper"]["joints"]["yaw"]["angle_deg"], 0)
        self.assertEqual(self.bridge.snapshot(), first)
        remaining = [self.bridge.step(dt_s=.02) for _ in iterator]
        self.assertAlmostEqual(remaining[-1]["grasper"]["joints"]["yaw"]["angle_deg"], 25)

    def test_feedback_round_trip_does_not_overwrite_commanded_angles(self):
        for instrument in (Grasper(), Scissors()):
            instrument.set_joint_targets(opening_deg=10)
            instrument.advance(1)
            angle_before = instrument.opening.state
            instrument.observe((1, 2, 3), instrument.pose().gap_cm)
            self.assertAlmostEqual(instrument.feedback.opening_deg, 10)
            self.assertEqual(instrument.opening.state, angle_before)

    def test_invalid_external_feedback_is_rejected_without_replacing_valid_feedback(self):
        instrument = self.bridge.grasper
        instrument.observe((1, 2, 3), .5)
        original = instrument.feedback
        for hinge, gap in (((1, 2), .5), ((1, 2, float("nan")), .5), ((1, 2, 3), -.1),
                           ((1, 2, 3), float("nan")), ((1, 2, 3), 99), ((1, 2, 3), [.5])):
            with self.subTest(hinge=hinge, gap=gap):
                with self.assertRaises(ValueError):
                    instrument.observe(hinge, gap)
                self.assertEqual(instrument.feedback, original)

    def test_feedback_uses_the_measured_gap_even_outside_the_control_range(self):
        instrument = self.bridge.grasper
        gap = 1.95 * instrument.config.jaw_length_cm
        instrument.observe((1, 2, 3), gap)
        recovered = 2 * instrument.config.jaw_length_cm * np.sin(np.radians(instrument.feedback.opening_deg) / 2)
        self.assertAlmostEqual(recovered, gap)

    def test_sample_json_messages_drive_the_real_instrument_instances(self):
        source = Path(__file__).with_name("sample_inputs.json")
        sequence = json.loads(source.read_text(encoding="utf-8"))
        for stage in sequence["stages"]:
            self.bridge.submit(stage["input"])
            for _ in range(stage["steps"]):
                state = self.bridge.step(dt_s=sequence["dt_s"])
                json.dumps(state, allow_nan=False)
        self.assertAlmostEqual(state["grasper"]["joints"]["pitch"]["angle_deg"], -15)
        self.assertAlmostEqual(state["scissors"]["joints"]["opening"]["angle_deg"], 30)
        self.assertFalse(state["is_cutting"])

    def test_driver_binds_existing_instances_without_resetting_them(self):
        self.grasper.set_joint_targets(pitch_deg=15)
        self.grasper.advance(.1)
        before = self.bridge.snapshot()
        another_driver = MotionBridge(self.grasper, self.scissors)
        self.assertIs(another_driver.grasper, self.grasper)
        self.assertIs(another_driver.scissors, self.scissors)
        self.assertEqual(another_driver.snapshot(), before)
        self.assertNotIn("time_s", before)
        self.assertNotIn("step", before)

    def test_world_linear_velocity_uses_the_external_timestep(self):
        start = self.grasper.base[:3, 3].copy()
        self.bridge.step({"grasper": {"linear_velocity_cm_s": [2, -1, .5]}}, dt_s=.1)
        np.testing.assert_allclose(self.grasper.base[:3, 3], start + (.2, -.1, .05), atol=1e-12, rtol=0)
        self.bridge.step(dt_s=.2)
        np.testing.assert_allclose(self.grasper.base[:3, 3], start + (.6, -.3, .15), atol=1e-12, rtol=0)

    def test_world_angular_velocity_rotates_an_existing_orientation(self):
        self.grasper.set_base_pose((1, 2, 3), (0, 0, 0, 1))
        self.bridge.step({"grasper": {"angular_velocity_deg_s": [0, 0, 90]}}, dt_s=1)
        np.testing.assert_allclose(self.grasper.base[:3, 0], (0, 1, 0), atol=1e-12, rtol=0)
        self.bridge.step({"grasper": {"angular_velocity_deg_s": [90, 0, 0]}}, dt_s=1)
        np.testing.assert_allclose(self.grasper.base[:3, 0], (0, 0, 1), atol=1e-12, rtol=0)
        np.testing.assert_allclose(self.grasper.base[:3, 3], (1, 2, 3), atol=1e-12, rtol=0)

    def test_absolute_pose_stops_velocity_and_is_consumed_once(self):
        self.bridge.step({"grasper": {"linear_velocity_cm_s": [10, 0, 0]}}, dt_s=.1)
        self.bridge.step({"grasper": {"base_position_cm": [1, 2, 3]}}, dt_s=.1)
        self.grasper.set_base_pose((4, 5, 6))
        self.bridge.step(dt_s=.1)
        np.testing.assert_allclose(self.grasper.base[:3, 3], (4, 5, 6), atol=1e-12, rtol=0)

    def test_invalid_frame_keeps_previously_submitted_motion(self):
        self.bridge.submit({"grasper": {"linear_velocity_cm_s": [1, 0, 0]}})
        start = self.grasper.base[:3, 3].copy()
        with self.assertRaises(ValueError):
            self.bridge.step({"grasper": {"linear_velocity_cm_s": [99, 0, 0]},
                              "scissors": {"opening_deg": 100}}, dt_s=.1)
        self.bridge.step(dt_s=.1)
        np.testing.assert_allclose(self.grasper.base[:3, 3], start + (.1, 0, 0), atol=1e-12, rtol=0)

    def test_nonfinite_motion_displacement_is_rejected_before_either_instance_moves(self):
        before = self.bridge.snapshot()
        with self.assertRaises(ValueError):
            self.bridge.step({"grasper": {"base_position_cm": [1, 2, 3]},
                              "scissors": {"linear_velocity_cm_s": [1e308, 0, 0]}}, dt_s=1e308)
        self.assertEqual(self.bridge.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
