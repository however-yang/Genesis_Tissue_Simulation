"""保留真实适配器及父类初始化，只用受调用顺序约束的接口替代 GPU。"""
import ctypes as ct
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

from ...instrument_geometry import SCISSORS_WRIST_FRONT
from ...native_backend import Blade, Grasper as NativeGrasper, NativeTissue, PhysicsDesc, ScissorsControl, ScissorsResult
from ...scenario import SCISSORS_SCENES
from ..config import ScenarioConfig, resolve_settings
from ..cuda_adapter import CudaJointAdapter
from ..instrument_control.config import grasper_config, scissors_config
from ..instrument_control.instruments import Grasper, JointCommand, Scissors


class NativeInitializationContract:
    """模拟原生接口的真实规则：剪刀启用后禁止重设腕部几何。"""

    def __init__(self, *, step_error=0):
        self.functions = {}
        self.events = []
        self.enabled = False
        self.wrists = []
        self.step_error = step_error
        self.abi = {
            "LCS_PhysicsAbiVersion": ct.sizeof(PhysicsDesc),
            "LCS_GraspToolAbiVersion": ct.sizeof(NativeGrasper),
            "LCS_PressureBladeAbiVersion": ct.sizeof(Blade),
            "LCS_ScissorsAbiVersion": ct.sizeof(ScissorsControl),
            "LCS_ScissorsResultAbiVersion": ct.sizeof(ScissorsResult),
            "LCS_ContactModelVersion": 28,
        }

    def __getitem__(self, name):
        if name not in self.functions:
            self.functions[name] = Mock(side_effect=lambda *args: self.call(name, *args))
        return self.functions[name]

    def call(self, name, *args):
        if name in self.abi:
            return self.abi[name]
        if name == "LCS_SetScissorsWrist":
            self.events.append("wrist")
            if self.enabled:
                return -1004
            self.wrists.append(args)
        elif name == "LCS_SetScissors":
            self.events.append("enable")
            self.enabled = bool(args[0]._obj.enabled)
        elif name == "LCS_Step":
            self.events.append("step")
            return self.step_error
        elif name == "LCS_Shutdown":
            self.enabled = False
        return 0


class CudaAdapterInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ScenarioConfig("wall", replace(grasper_config(), shaft_length_cm=5.5),
                                    replace(scissors_config(), shaft_length_cm=7.2))
        cls.settings = resolve_settings(cls.config)
        cls.grid = SCISSORS_SCENES["wall"].build_grid(cls.settings.density_kg_m3)

    def test_scene_geometry_is_set_once_before_enabling_in_both_command_modes(self):
        commands = (
            SCISSORS_SCENES["wall"].stages[0].command,
            JointCommand(Grasper(self.config.grasper).pose(), Scissors(self.config.scissors).pose(), False),
        )
        for command in commands:
            with self.subTest(command_type=type(command).__name__):
                dll = NativeInitializationContract()
                with patch.object(NativeTissue, "active_session", False), \
                     patch("endoscopic_tissue.native_backend.ct.CDLL", return_value=dll):
                    adapter = CudaJointAdapter(self.grid, self.settings, None, config=self.config, initial_command=command)
                    try:
                        self.assertEqual(dll.events, ["wrist", "enable", "step"])
                        self.assertEqual(len(dll.wrists), 1)
                        self.assertEqual(dll.wrists[0][0], SCISSORS_WRIST_FRONT)
                        self.assertEqual(dll.wrists[0][1], self.config.scissors.shaft_length_cm)
                        self.assertEqual(adapter.grasper_geometry.shaft_length, self.config.grasper.shaft_length_cm)
                        self.assertEqual(adapter.scissors_geometry.jaws.shaft_length, self.config.scissors.shaft_length_cm)
                    finally:
                        adapter.close()
                    self.assertFalse(NativeTissue.active_session)

    def test_initial_physics_failure_releases_the_session(self):
        dll = NativeInitializationContract(step_error=-2)
        with patch.object(NativeTissue, "active_session", False), \
             patch("endoscopic_tissue.native_backend.ct.CDLL", return_value=dll):
            with self.assertRaisesRegex(RuntimeError, "LCS_Step failed"):
                CudaJointAdapter(self.grid, self.settings, None, config=self.config,
                                 initial_command=SCISSORS_SCENES["wall"].stages[0].command)
            dll.functions["LCS_Shutdown"].assert_called_once()
            self.assertFalse(NativeTissue.active_session)


if __name__ == "__main__":
    unittest.main()
