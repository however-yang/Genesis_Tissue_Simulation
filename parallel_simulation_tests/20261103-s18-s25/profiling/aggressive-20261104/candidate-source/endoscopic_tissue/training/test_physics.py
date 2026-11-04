"""Parameter ABI, truthful provenance, and original-default regression contracts."""
import ctypes as ct
from dataclasses import asdict, FrozenInstanceError, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..native_backend import CutDesc, GraspParams, NativeTissue, Settings
from ..scenario import SCISSORS_SCENES, scissors_settings
from .audit_physics import audit_context, audit_tree
from .env import SceneConfig
from .experiment import scene_configuration
from .native import TrainingNativeTissue
from .physics import (PhysicsOverrides, canonical_settings, native_defaults,
                      physical_differences, resolve_physics)
from .provenance import json_hash


class PhysicsTests(unittest.TestCase):
    def test_all_original_scenes_compare_equal_to_frozen_defaults(self):
        for name, scene in SCISSORS_SCENES.items():
            settings = replace(scissors_settings(), young_pa=scene.young_pa, contact_model=scene.contact_model)
            self.assertEqual(physical_differences(settings, asdict(SceneConfig(name, 'held'))), {})

    def test_frozen_overrides_and_strict_types_ranges(self):
        with self.assertRaises(FrozenInstanceError):
            PhysicsOverrides().young_pa = 9000
        for key in PhysicsOverrides.__dataclass_fields__:
            for bad in (True, False, '1', float('nan'), float('inf'), [], {}):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    PhysicsOverrides(**{key: bad})
        for key, bad in [('young_pa', 2999), ('young_pa', 24001), ('start_force_n', .199),
                         ('start_force_n', .601), ('blade_friction', .049), ('blade_friction', .401),
                         ('grasp_friction', .299), ('grasp_friction', 1.201), ('tear_stretch_ratio', 1.29),
                         ('tear_stretch_ratio', 2.01), ('tear_exempt_rings', 1.), ('tear_exempt_rings', -1),
                         ('tear_exempt_rings', 3), ('grasp_core_break_force_n', 0),
                         ('grasp_core_break_force_n', .099), ('grasp_core_break_force_n', 151)]:
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                PhysicsOverrides(**{key: bad})
        with self.assertRaises(TypeError):
            SceneConfig(physics={'young_pa': 8000})
        with self.assertRaises(TypeError):
            PhysicsOverrides(scissors_squeeze_cm=.1)

    def test_parameters_resolve_and_each_change_is_in_provenance(self):
        cases = {'young_pa': 24000, 'start_force_n': .6, 'blade_friction': .4,
                 'grasp_friction': 1.2, 'tear_stretch_ratio': 1.5,
                 'tear_exempt_rings': 2, 'grasp_core_break_force_n': 1.5}
        for key, value in cases.items():
            cfg = SceneConfig(contact_model=1, physics=PhysicsOverrides(**{key: value}))
            settings, native = resolve_physics(replace(scissors_settings(), contact_model=1), cfg.physics)
            changes = physical_differences(settings, asdict(cfg), native)
            prefix = 'settings' if hasattr(settings, key) else 'native'
            self.assertEqual(changes[f'{prefix}.{key}']['actual'], value)
            self.assertNotEqual(json_hash(asdict(cfg)), json_hash(asdict(SceneConfig(contact_model=1))))
        self.assertEqual(native_defaults()['grasp_core_break_force_n'], 150.)

    def test_friction_requires_pure_mode_and_changes_actual_static_dynamic_pair(self):
        with self.assertRaises(ValueError):
            resolve_physics(scissors_settings(), PhysicsOverrides(grasp_friction=.3))
        _, native = resolve_physics(replace(scissors_settings(), contact_model=1),
                                    PhysicsOverrides(grasp_friction=.3))
        self.assertEqual((native['grasp_friction'], native['grasp_dynamic_friction']), (.3, .2))

    def test_monkeypatched_factory_cannot_redefine_canonical_defaults(self):
        original = scissors_settings()
        with patch('endoscopic_tissue.training.env.scissors_settings',
                   return_value=replace(original, start_force_n=.4, substeps=4)) as patched:
            changed = physical_differences(patched(), asdict(SceneConfig()))
        self.assertEqual(changed['settings.start_force_n'], {'canonical': .2, 'actual': .4})
        self.assertEqual(changed['settings.substeps'], {'canonical': 3, 'actual': 4})
        copy = canonical_settings('wall-traction')
        copy['start_force_n'] = 99
        self.assertEqual(canonical_settings('wall-traction')['start_force_n'], .2)

    def test_unexposed_setting_changes_are_not_missed(self):
        changed = physical_differences(replace(scissors_settings(), scissors_squeeze_cm=.1,
                                                density_kg_m3=1100), asdict(SceneConfig()))
        self.assertIn('settings.scissors_squeeze_cm', changed)
        self.assertIn('settings.density_kg_m3', changed)

    def test_native_payload_unit_conversion_and_contact_path(self):
        obj = TrainingNativeTissue.__new__(TrainingNativeTissue)
        obj.training_parameters = {**native_defaults(), 'tear_stretch_ratio': 1.5,
                                   'tear_exempt_rings': 2, 'grasp_core_break_force_n': 3.,
                                   'grasp_friction': .9, 'grasp_dynamic_friction': .6}
        observed = {}
        def cut(ptr, *args):
            observed['tear'] = ct.cast(ptr, ct.POINTER(CutDesc)).contents.tear_ratio
            return 0
        def grasp(ptr):
            gp = ct.cast(ptr, ct.POINTER(GraspParams)).contents
            observed['grasp'] = (gp.break_force, gp.exempt_rings, gp.friction)
        def contact(*args):
            observed['contact'] = args
            return 0
        obj.functions = {'LCS_InitCut': cut, 'LCS_SetGraspParams': grasp, 'LCS_SetContactModel': contact}
        original = obj.functions.copy()
        def initialize(_):
            desc, gp = CutDesc(), GraspParams()
            obj.functions['LCS_InitCut'](ct.byref(desc), None, None, None, None)
            obj.functions['LCS_SetGraspParams'](ct.byref(gp))
            obj.functions['LCS_SetContactModel'](1, .6, .4, .1, .01)
        with patch.object(NativeTissue, 'initialize', initialize):
            obj.initialize()
        self.assertEqual(observed['tear'], 1.5)
        self.assertEqual(observed['grasp'][:2], (300., 2))
        self.assertAlmostEqual(observed['grasp'][2], .9, places=6)
        self.assertEqual(observed['contact'], (1, .9, .6, .1, .01))
        self.assertEqual(obj.functions, original)
        with patch.object(NativeTissue, 'initialize', side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):
                obj.initialize()
        self.assertEqual(obj.functions, original)

    def test_empty_override_preserves_exact_original_callables(self):
        obj = TrainingNativeTissue.__new__(TrainingNativeTissue)
        obj.training_parameters = native_defaults()
        obj.functions = {'marker': object()}
        original = obj.functions.copy()
        with patch.object(NativeTissue, 'initialize', lambda _: self.assertEqual(obj.functions, original)):
            obj.initialize()

    def test_experiment_json_preserves_parameters_without_coercion(self):
        cfg = scene_configuration({'physics': {'young_pa': 8000, 'tear_stretch_ratio': 1.5}})
        self.assertEqual(cfg.physics.young_pa, 8000)
        with self.assertRaises(ValueError):
            scene_configuration({'physics': {'young_pa': '8000'}})

    def test_historical_false_default_flag_is_reported_without_editing_input(self):
        context = {'episode_id': 'historical', 'scene_config': asdict(SceneConfig()),
                   'settings': asdict(replace(scissors_settings(), start_force_n=.4)),
                   'physical_defaults_changed': False}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'context.json'
            path.write_text(json.dumps(context), encoding='utf8')
            original = path.read_bytes()
            audited = audit_tree(Path(temp))
            self.assertEqual(audited['flag_mismatches'], 1)
            self.assertEqual(path.read_bytes(), original)
        self.assertEqual(audit_context(context)['overridden_fields']['settings.start_force_n']['actual'], .4)


if __name__ == '__main__':
    unittest.main()
