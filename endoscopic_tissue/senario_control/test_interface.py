import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from ..native_backend import Observation, Surface
from .cutting import set_cutting
from .instrument_control.commands import Command
from .instrument_control.joints import scissors_frame, set_grasper, set_scissors
from .instrument_control.smoothing import Stage, interpolate
from .simulator import SimulationInterface


class SimulationInterfaceTests(unittest.TestCase):
    def test_independent_actuators_and_smooth_transition(self):
        initial = Command()
        target = set_grasper(initial, position_cm=(1.0, 2.0, 3.0), closure=1.0)
        self.assertEqual(target.blade_center, initial.blade_center)
        target = set_scissors(target, position_cm=(4.0, 5.0, 6.0), pitch_deg=0.0, roll_deg=90.0,
                              yaw_deg=-75.0, closure=1.0)
        target = set_cutting(target, is_cutting=True)
        self.assertEqual(target.hinge, (1.0, 2.0, 3.0))
        self.assertEqual(initial, Command())
        quarter = interpolate(initial, target, fraction=.25)
        self.assertEqual(quarter.closure, .15625)
        self.assertEqual(quarter.scissors_closure, .15625)
        np.testing.assert_allclose(quarter.hinge, .84375 * np.array(initial.hinge) + .15625 * np.array(target.hinge))
        self.assertTrue(interpolate(initial, target, fraction=0.0).scissors_cutting)
        self.assertEqual(interpolate(initial, target, fraction=1.0), target)
        self.assertEqual(interpolate(initial, target, fraction=2.0), target)
        axis, normal = scissors_frame(0.0, 90.0, -75.0)
        np.testing.assert_allclose(axis, (-np.cos(np.radians(75)), 0, -np.sin(np.radians(75))))
        np.testing.assert_allclose(normal, (-np.sin(np.radians(75)), 0, np.cos(np.radians(75))), atol=1e-15)

    def test_invalid_actuator_targets_and_stage_duration(self):
        for field in ("scissors_closure", "scissors_pitch_deg", "scissors_roll_deg", "scissors_yaw_deg"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                set_grasper(replace(Command(), **{field: np.nan}), closure=0.0)
        for closure in (-.1, 1.1):
            with self.subTest(closure=closure), self.assertRaises(ValueError):
                set_scissors(Command(), closure=closure)
        with self.assertRaises(ValueError):
            set_grasper(Command(), position_cm=(1, 2))
        with self.assertRaises(ValueError):
            set_cutting(Command(), is_cutting="false")
        for steps, ramp in ((0, 1.0), (1.5, 1.0), (True, 1.0), (1, 0.0), (1, 1.1)):
            with self.subTest(steps=steps, ramp=ramp), self.assertRaises(ValueError):
                Stage("move", steps, Command(), ramp=ramp)
        with self.assertRaises(ValueError):
            interpolate(Command(), Command(), fraction=np.nan)

    def test_step_transition_and_resource_lifecycle(self):
        observation = Observation(
            nodes=np.zeros((1, 3)), neighbors=np.full((1, 6), -1), grip_flags=np.zeros(1), hinge=np.zeros(3),
            jaw_gap=1.0, grip_points=0, grip_error_m=0.0, blade=np.zeros(10), reaction_n=np.zeros(3))
        surface = Surface(np.zeros((3, 3)), np.zeros((3, 3)), np.array(((0, 1, 2),)), np.zeros((3, 2)))
        with patch("endoscopic_tissue.senario_control.simulator.CudaJointAdapter",
                   autospec=True) as factory:
            backend = factory.return_value
            backend.closed = False
            backend.observe.return_value = observation
            backend.step.return_value = observation
            backend.surface.return_value = surface
            with SimulationInterface(scene="kidney") as simulation:
                target = set_scissors(simulation.command, position_cm=(10.0, 0.0, 0.0), closure=1.0)
                first = simulation.step(target, include_surface=True)
                self.assertEqual(first.step, 1)
                self.assertEqual(first.time_s, simulation.settings.dt)
                self.assertIs(first.surface, surface)
                self.assertIs(first.observation, observation)
                self.assertEqual(first.requested, target)
                offset_cm = np.linalg.norm(first.command.blade_center)
                self.assertAlmostEqual(offset_cm * .01 * simulation.settings.blade_drive_stiffness,
                                       simulation.settings.blade_preload_n)
                transition = simulation.transition(target, duration_s=2.5 * simulation.settings.dt)
                self.assertEqual(simulation.step_count, 1)
                states = list(transition)
                self.assertEqual(len(states), 3)
                self.assertEqual(states[-1].requested, target)
                self.assertTrue(all(state.surface is None for state in states))
                backend.surface.assert_called_once()
                stages = (Stage("incise", 4, set_cutting(target, is_cutting=True), ramp=.5),
                          Stage("withdraw", 2, set_cutting(target, is_cutting=False)))
                states = list(simulation.run_stages(stages))
                self.assertEqual([state.stage for state in states], ["incise"] * 4 + ["withdraw"] * 2)
                self.assertEqual(states[1].requested, stages[0].command)
                self.assertFalse(states[4].requested.scissors_cutting)
                self.assertEqual(simulation.step_count, 10)
                with self.assertRaises(ValueError):
                    list(simulation.transition(target, duration_s=0.0))
                self.assertIs(simulation.surface(), surface)
            backend.close.assert_called_once()
            backend.closed = True
            with self.assertRaises(RuntimeError):
                simulation.step(target)
            with self.assertRaises(RuntimeError):
                simulation.surface()

    def test_observation_failure_releases_native_session(self):
        with patch("endoscopic_tissue.senario_control.simulator.CudaJointAdapter",
                   autospec=True) as factory:
            factory.return_value.observe.side_effect = RuntimeError("Native observation failed")
            with self.assertRaisesRegex(RuntimeError, "Native observation failed"):
                SimulationInterface(scene="wall")
            factory.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
