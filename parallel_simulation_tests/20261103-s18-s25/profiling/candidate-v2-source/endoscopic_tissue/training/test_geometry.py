"""Procedural wall recipe, unchanged reference and conservative split grouping."""
from dataclasses import asdict, replace
import unittest

import numpy as np

from ..scenario import WALL_TRACTION_SCENE
from .env import SceneConfig
from .geometry import WallGeometry, scene_for_config
from .provenance import json_hash


class GeometryTests(unittest.TestCase):
    def test_reference_geometry_and_complete_native_commands_preserved(self):
        geometry = WallGeometry()
        self.assertEqual(geometry.recipe(),{'kind':'block_with_tab','tab':asdict(WALL_TRACTION_SCENE.tab)})
        derived = geometry.scene()
        self.assertEqual(len(derived.stages),len(WALL_TRACTION_SCENE.stages))
        for a,b in zip(derived.stages,WALL_TRACTION_SCENE.stages):
            self.assertEqual((a.name,a.steps,a.ramp),(b.name,b.steps,b.ramp))
            for key,value in asdict(a.command).items():
                np.testing.assert_array_equal(np.asarray(value,dtype=np.float32),
                                              np.asarray(getattr(b.command,key),dtype=np.float32))
        self.assertIs(scene_for_config(SceneConfig()),WALL_TRACTION_SCENE)

    def test_grid_reference_is_exact_and_all_new_spacing_is_at_least_one_mm(self):
        old=WALL_TRACTION_SCENE.build_grid(1050)
        new=WallGeometry().build_grid(1050)
        for name in old.__dict__:
            np.testing.assert_array_equal(getattr(old,name),getattr(new,name))
        self.assertTrue((new.cell_size >= np.float32(.1)).all())
        with self.assertRaises(ValueError):
            SceneConfig(geometry=WallGeometry(),spacing_scale=.5)

    def test_targets_follow_geometry_without_instrument_or_budget_scaling(self):
        reference=WallGeometry().scene()
        variant=WallGeometry(length_m=.026,height_m=.008,thickness_m=.005,cut_height_m=.0045).scene()
        a={s.name:s for s in reference.stages}
        b={s.name:s for s in variant.stages}
        np.testing.assert_allclose(np.array(b['bite'].command.hinge)-a['bite'].command.hinge,[.1,.1,-.05],atol=1e-12)
        self.assertAlmostEqual(b['incise_1'].command.blade_center[1]-a['incise_1'].command.blade_center[1],.1)
        self.assertAlmostEqual(b['incise_1'].command.blade_center[0]-b['incise_2'].command.blade_center[0],.52)
        self.assertEqual([s.steps for s in reference.stages],[s.steps for s in variant.stages])
        self.assertEqual(reference.contact_model,variant.contact_model)

    def test_geometry_groups_change_for_anatomy_but_not_clamp_or_cut_target(self):
        hashes={json_hash(WallGeometry(length_m=x).recipe()) for x in (.024,.025,.026)}
        self.assertEqual(len(hashes),3)
        a=WallGeometry()
        b=replace(a,fixed_edge='x_min',cut_height_m=.004)
        self.assertEqual(json_hash(a.recipe()),json_hash(b.recipe()))
        ga,gb=a.build_grid(1050),b.build_grid(1050)
        np.testing.assert_array_equal(ga.positions,gb.positions)
        self.assertFalse(np.array_equal(ga.pinned,gb.pinned))
        self.assertTrue(np.all(gb.positions[gb.pinned!=0,0] <= -3.))

    def test_bad_parameters_are_rejected(self):
        for values in ({'height_m':True},{'length_m':float('nan')},{'thickness_m':.0005},
                       {'fixed_edge':'y_max'},{'height_m':.006,'cut_height_m':.005}):
            with self.assertRaises(ValueError):
                WallGeometry(**values)
        with self.assertRaises(ValueError):
            SceneConfig(scene='flap',geometry=WallGeometry())


if __name__=='__main__':
    unittest.main()
