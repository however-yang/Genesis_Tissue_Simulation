import ctypes as ct
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from ..config import load_config, resolve_settings
from ..hyc_movecontrol_config import SceneInput, build_scene_input


class CudaSceneInputTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).parents[1] / "configs/kidney.toml")
        self.settings = resolve_settings(self.config)

    def test_scene_defaults_are_passed_without_cuda_fallbacks(self):
        scene = build_scene_input(self.config, self.settings, grid_spacing_cm=.1)
        self.assertEqual(ct.sizeof(scene), 948)
        self.assertEqual(list(scene.instruments[0].base_position_cm),
                         list((ct.c_float * 3)(*self.config.grasper.base_position_cm)))
        self.assertEqual(scene.instruments[1].jaw_stiffness, self.settings.scissors_jaw_stiffness)
        self.assertEqual(scene.instruments[1].jaw_stiffness, 2500)
        self.assertEqual(scene.instruments[0].joints[1].max_speed_deg_s, self.config.grasper.pitch.max_speed_deg_s)
        self.assertAlmostEqual(scene.instruments[0].max_travel_cm, .1 * self.settings.tool_travel_fraction / self.settings.substeps)
        self.assertAlmostEqual(scene.instruments[0].shaft_radius_cm, .418, places=6)

    def test_scene_changes_update_the_next_runtime_input(self):
        original = build_scene_input(self.config, self.settings, grid_spacing_cm=.1)
        grasper = replace(self.config.grasper, base_position_cm=(1, 2, 3),
                          pitch=replace(self.config.grasper.pitch, max_speed_deg_s=15))
        changed = replace(self.config, grasper=grasper)
        settings = replace(self.settings, substeps=6, dt=.025, tool_mass_kg=.03,
                           blade_mass_kg=.04, blade_drive_stiffness=2400, scissors_jaw_stiffness=1700)
        scene = build_scene_input(changed, settings, grid_spacing_cm=.2)
        self.assertEqual(list(scene.instruments[0].base_position_cm), [1, 2, 3])
        self.assertEqual(scene.instruments[0].joints[1].max_speed_deg_s, 15)
        self.assertEqual(scene.instruments[1].jaw_stiffness, 1700)
        self.assertEqual(scene.instruments[1].stiffness, 2400)
        self.assertAlmostEqual(scene.instruments[1].mass_kg, .04)
        self.assertAlmostEqual(scene.instruments[0].max_travel_cm, .2 * settings.tool_travel_fraction / 6)
        self.assertEqual(original.instruments[1].jaw_stiffness, 2500)

    def test_physics_overrides_are_loaded_from_the_scene_toml(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scene.toml"
            path.write_text('scene="wall"\n[physics]\nsubsteps=5\ntool_mass_kg=0.07\n'
                            'jaw_speed_cm_s=0.8\nscissors_jaw_stiffness=1900\n', encoding="utf-8")
            config = load_config(path)
            scene = build_scene_input(config, resolve_settings(config), grid_spacing_cm=.1)
        self.assertEqual(scene.substeps, 5)
        self.assertAlmostEqual(scene.instruments[0].mass_kg, .07)
        self.assertAlmostEqual(scene.instruments[1].jaw_speed_cm_s, .8)
        self.assertEqual(scene.instruments[1].jaw_stiffness, 1900)

    def test_transport_round_trip(self):
        scene = build_scene_input(self.config, self.settings, grid_spacing_cm=.1)
        restored = SceneInput.from_buffer_copy(bytes(scene))
        self.assertEqual(bytes(restored), bytes(scene))
        self.assertEqual(restored.abi_version, 1)

    def test_invalid_physical_inputs_are_rejected(self):
        for changes in ({"substeps": 0}, {"substeps": True}, {"dt": 0}, {"dt": float("nan")},
                        {"tool_mass_kg": 0}, {"tool_drive_damping": -1}, {"scissors_guided": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_scene_input(self.config, replace(self.settings, **changes), grid_spacing_cm=.1)
        with self.assertRaises(ValueError):
            build_scene_input(self.config, self.settings, grid_spacing_cm=0)


if __name__ == "__main__":
    unittest.main()
