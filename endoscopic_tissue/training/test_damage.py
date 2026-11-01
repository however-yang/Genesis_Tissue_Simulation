"""Fault tests for blade/tear attribution and honest overload telemetry."""
from types import SimpleNamespace
import unittest

import numpy as np

from .damage import DamageMonitor
from .native import TrainingMetrics
from .quality import QualityMonitor
from ..native_backend import Command, Settings


def fixture():
    nbr=np.full((4,6),-1,np.int32)
    for a,b in ((0,1),(1,2),(2,3)):
        nbr[a,1],nbr[b,0]=b,a
    return SimpleNamespace(neighbors=nbr,inside=np.array([1,1,1,0],np.int32),
                           active=np.ones(4,np.int32),positions=np.zeros((4,3),np.float32))


def metrics(**updates):
    result={name:0 for name,_ in TrainingMetrics._fields_}
    result.update(maxStretch=1.5,rimStretch=1.2,graspRingStretch=1.4,
                  coreForceMax=30.,coreForceMedian=10.,activeCoreCorners=2,minVolumeRatio=.6)
    result.update(updates)
    return result


def observation(grid,neighbors=None):
    return SimpleNamespace(neighbors=grid.neighbors if neighbors is None else neighbors,grip_points=4,
                           jaw_load_n=(.3,.2),nodes=grid.positions.copy())


class DamageTests(unittest.TestCase):
    def test_events_distinguish_blade_tear_and_air_links(self):
        grid=fixture();monitor=DamageMonitor(grid)
        after=grid.neighbors.copy()
        for a,b in ((0,1),(1,2),(2,3)):after[a,1]=after[b,0]=-1
        r=monitor.assess(observation(grid,after),metrics(tearEdges=1,materialTearEdges=1,interiorTearEdges=1))
        self.assertEqual((r['new_broken_edges'],r['new_material_broken_edges'],r['new_material_blade_edges']), (3,2,1))
        self.assertEqual(r['new_tear_location']['interior'],1)
        again=monitor.assess(observation(grid,after),metrics(tearEdges=1,materialTearEdges=1,interiorTearEdges=1))
        self.assertEqual(again['new_tear_edges'],0)

    def test_counter_without_physical_break_is_rejected(self):
        grid=fixture()
        with self.assertRaisesRegex(ValueError,'actually severed'):
            DamageMonitor(grid).assess(observation(grid),metrics(tearEdges=1))

    def test_rollback_without_restore_baseline_is_rejected(self):
        grid=fixture();monitor=DamageMonitor(grid)
        monitor.assess(observation(grid),metrics(brokenCoreCorners=1,brokenGripPoints=2))
        with self.assertRaisesRegex(ValueError,'rolled backwards'):
            monitor.assess(observation(grid),metrics())

    def test_real_units_and_unavailable_contact_load_are_explicit(self):
        grid=fixture()
        r=DamageMonitor(grid).assess(observation(grid),metrics(brokenCoreCorners=1,brokenGripPoints=2))
        self.assertEqual(r['anchors']['core_force_ema_max_n'],.3)
        self.assertEqual(r['anchors']['active_core_force_ema_median_n'],.1)
        self.assertEqual(r['anchors']['new_broken_material_points'],2)
        self.assertIsNone(r['jaw_tangential_load_n'])
        self.assertEqual(r['jaw_normal_load_n'],[.3,.2])

    def test_missing_or_nonfinite_metrics_are_not_invented(self):
        grid=fixture()
        for value in (None,metrics(coreForceMax=float('nan'))):
            with self.assertRaises((RuntimeError,ValueError)):
                DamageMonitor(grid).assess(observation(grid),value)

    def test_tear_permission_does_not_hide_disabled_blade_or_detached_scope(self):
        grid=fixture();after=grid.neighbors.copy();after[0,1]=after[1,0]=-1
        for permitted,disabled_failure in ((0,True),(1,False)):
            quality=QualityMonitor(grid).assess(observation(grid,after),None,Command(scissors_cutting=False),
                Settings(),np.array([1.,0]),None,None,permitted_tear_edges=permitted)
            self.assertEqual('new_cut_while_disabled' in quality['numerical_failure_reasons'],disabled_failure)
            self.assertFalse(quality['within_declared_local_scope'])
            self.assertIn('detached_material_outside_local_task_scope',quality['scope_failure_reasons'])


if __name__=='__main__':unittest.main()
