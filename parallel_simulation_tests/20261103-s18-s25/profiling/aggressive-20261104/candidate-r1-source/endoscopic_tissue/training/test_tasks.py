"""Task, policy leakage, and frozen evaluation regressions on independent tiny graphs."""
from dataclasses import replace
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .actions import ActionBounds
from .data import EpisodeReader, EpisodeWriter
from .env import State, StepResult
from .experiment import group_key, validate_spec
from .policies import observation, Policy
from .provenance import json_hash
from .tasks import TaskGoal, TaskMonitor, recompute_episode

BOUNDS = ActionBounds((-1.,)*3+(0.,)+(-1.,)*6+(0.,), (1.,)*11, (1.,)*11)
ACTION = {'policy': [0.]*11, 'executed_target_si': [0.]*11, 'previous_target_si': [0.]*11}
WEIGHTS = {'progress': 1., 'outside_damage': 2., 'grasp': .1, 'slip_m': 0., 'time_s': 0., 'action': 0.}


def fixture():
    neighbors = np.full((4,6), -1, np.int32)
    # 0--1 is material; 0--2--1 is a support-only bridge; 1--3 stays in the body.
    neighbors[0,1], neighbors[1,0] = 1,0
    neighbors[0,3], neighbors[2,2] = 2,0
    neighbors[2,1], neighbors[1,2] = 1,2
    neighbors[1,5], neighbors[3,4] = 3,1
    return {'initial_neighbors': neighbors, 'material_mask': np.array([1,1,0,1]),
            'active_mask': np.ones(4), 'rest_positions_m': np.zeros((4,3))}


def sample(index=0, *, cut=False, support=False, valid=True, closure=0):
    neighbors = fixture()['initial_neighbors'].copy()
    if cut:
        neighbors[0,1] = neighbors[1,0] = -1
    if support:
        neighbors[0,3] = neighbors[2,2] = -1
    return State({'episode_id':'task-test', 'step_id':index, 'dt_s':.1, 't_sim_s':index*.1,
                  'native_tick':30+index, 'topology_revision':int(cut)+int(support),
                  'quality':{'checked_metrics_passed':valid, 'within_declared_local_scope':True},
                  'tools':{'grasper_hinge_m':[0,0,0], 'grasper_gap_m':.005, 'grip_points':99,
                           'grip_error_m':0, 'scissors_phase':3,
                           'scissors_loads':{'fractures':88}, 'scissors_closure':closure},
                  'grasp':{'geometric_loaded_hold':True, 'witness_retained_fraction':1.,
                           'witness_tangential_slip_m':0., 'bilateral_load':True,
                           'anchor_points':99, 'between_pads_material_nodes':2}},
                 {'positions_m':np.zeros((4,3)), 'estimated_velocity_m_s':np.zeros((4,3))},
                 {'revision':int(cut)+int(support), 'neighbors':neighbors})


class TaskTests(unittest.TestCase):
    def test_material_ids_are_strict_and_json_serializable(self):
        for edge in ((.9, 1.9), (False, True)):
            with self.subTest(edge=edge), self.assertRaises(ValueError):
                TaskGoal(kind='incision', target_edges=(edge,))
        for field in ('hold_nodes', 'region_nodes'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                TaskGoal(**{'kind':'hold', 'hold_nodes':(0,), field:(True,)})
        goal = TaskGoal(kind='incision', target_edges=((np.int32(0), np.int64(1)),))
        self.assertTrue(all(type(i) is int for i in goal.target_edges[0]))
        self.assertEqual(len(json_hash(goal.to_dict())), 64)

    def goal(self, **kw):
        return TaskGoal(**{'kind':'incision', 'target_edges':((0,1),), 'required_hold_s':.1,
                           'weights':WEIGHTS, **kw})

    def test_closed_scissors_without_fracture_are_not_success(self):
        monitor = TaskMonitor(self.goal(), fixture(), sample())
        outcome = monitor.evaluate(sample(), sample(1, closure=1), ACTION, BOUNDS)
        self.assertFalse(outcome['success'])
        self.assertEqual(outcome['metrics']['cut_progress'], 0)

    def test_one_sided_removed_edge_does_not_count_as_cut(self):
        after = sample(1, cut=True)
        after.topology['neighbors'][1,0] = 0
        outcome = TaskMonitor(self.goal(), fixture(), sample()).evaluate(sample(), after, ACTION, BOUNDS)
        self.assertFalse(outcome['success'])

    def test_material_separation_with_support_bridge_is_not_excision(self):
        monitor = TaskMonitor(self.goal(kind='excision', region_nodes=(0,)), fixture(), sample())
        first = sample(1, cut=True)
        outcome = monitor.evaluate(sample(), first, ACTION, BOUNDS)
        self.assertFalse(outcome['metrics']['material_connected_to_body'])
        self.assertTrue(outcome['metrics']['support_connected_to_body'])
        self.assertFalse(outcome['success'])
        outcome = monitor.evaluate(first, sample(2, cut=True, support=True), ACTION, BOUNDS)
        self.assertTrue(outcome['success'])

    def test_numerical_failure_wins_over_cut_success_and_no_bootstrap(self):
        outcome = TaskMonitor(self.goal(), fixture(), sample()).evaluate(sample(), sample(1, cut=True, valid=False), ACTION, BOUNDS)
        self.assertFalse(outcome['success'])
        self.assertIsNone(outcome['reward'])
        self.assertFalse(outcome['bootstrap_allowed'])

    def test_outside_damage_cancels_success_and_has_separate_reward_term(self):
        after = sample(1, cut=True)
        after.topology['neighbors'][1,5] = after.topology['neighbors'][3,4] = -1
        outcome = TaskMonitor(self.goal(), fixture(), sample()).evaluate(sample(), after, ACTION, BOUNDS)
        self.assertFalse(outcome['success'])
        self.assertEqual(outcome['failure'], 'outside_target_material_damage')
        self.assertEqual(outcome['reward_components']['outside_damage'], -1)

    def test_hold_needs_stable_material_displacement_and_actual_loaded_grasp(self):
        goal = self.goal(kind='hold', target_edges=(), hold_nodes=(0,), displacement_m=(0,.002,0))
        monitor = TaskMonitor(goal, fixture(), sample())
        first = sample(1)
        self.assertFalse(monitor.evaluate(sample(), first, ACTION, BOUNDS)['success'])
        second = sample(2)
        second.arrays['positions_m'][0,1] = .002
        self.assertTrue(monitor.evaluate(first, second, ACTION, BOUNDS)['success'])

    def test_opening_for_release_is_not_grasp_loss_failure(self):
        goal = self.goal(kind='release', target_edges=(), region_nodes=(0,), fail_on_grasp_loss=True)
        after = sample(1,cut=True,support=True)
        after.arrays['positions_m'][0,1] = .01
        after.metadata['grasp'].update(geometric_loaded_hold=False, bilateral_load=False, anchor_points=0,
                                        between_pads_material_nodes=0, witness_retained_fraction=0.)
        before = sample(cut=True, support=True)
        outcome = TaskMonitor(goal, fixture(), before).evaluate(before, after, ACTION, BOUNDS)
        self.assertTrue(outcome['success'])
        self.assertIsNone(outcome['failure'])

    def test_already_cut_or_invalid_material_goal_is_rejected(self):
        with self.assertRaises(ValueError):
            TaskMonitor(self.goal(), fixture(), sample(cut=True))
        with self.assertRaises(ValueError):
            TaskMonitor(self.goal(target_edges=((0,2),)), fixture(), sample())
        from .tasks import material_reference
        bound = self.goal(material_reference_sha256=material_reference(fixture()))
        changed = fixture()
        changed['rest_positions_m'][0,0] = .001
        with self.assertRaises(ValueError):
            TaskMonitor(bound,changed,sample())

    def test_offline_recompute_matches_serialized_rewards(self):
        goal = self.goal()
        initial, after = sample(), sample(1,cut=True)
        monitor = TaskMonitor(goal, fixture(), initial)
        task = monitor.evaluate(initial, after, ACTION, BOUNDS)
        from dataclasses import asdict
        context = {'episode_id':'task-test', 'goal':goal.to_dict(), 'action_bounds':asdict(BOUNDS)}
        context['context_sha256'] = json_hash(context)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'episode'
            with EpisodeWriter(path, context, initial, fixture()) as writer:
                writer.append(StepResult(initial,after,ACTION,True,False,'task_success',{},task=task))
            results = recompute_episode(EpisodeReader(path))
            self.assertEqual(results, [task])

    def test_policy_observation_hides_ground_truth_and_is_owned(self):
        state = sample()
        obs = observation(state, np.zeros(11))
        self.assertNotIn('positions_m', obs)
        self.assertNotIn('grip_points', obs['tools'])
        self.assertNotIn('fractures', obs['tools']['scissors_loads'])
        obs['tools']['grasper_hinge_m'][0] = 99
        self.assertEqual(state.metadata['tools']['grasper_hinge_m'][0], 0)

    def test_random_policy_is_reproducible_and_varies_scissors_angles(self):
        config = {'kind':'smooth_random'}
        a,b = Policy(config,np.zeros(11),8), Policy(config,np.zeros(11),8)
        obs = observation(sample(1),np.zeros(11))
        np.testing.assert_array_equal(a.act(obs), b.act(obs))
        self.assertTrue(np.any(a.act(obs)[7:10] != 0))

    def test_frozen_eval_detects_edits_and_training_overlap(self):
        case = {'id':'case-a','seed':1,'policy':{'kind':'hold'}}
        spec = {'schema':'tissue_experiment_v1','purpose':'evaluation','episodes':[case]}
        spec['frozen_sha256'] = json_hash(spec)
        validate_spec(spec)
        altered = copy.deepcopy(spec)
        altered['episodes'][0]['seed'] = 2
        with self.assertRaises(ValueError): validate_spec(altered)
        spec['excluded_training_groups'] = [group_key(case)]
        spec['frozen_sha256'] = json_hash({k:v for k,v in spec.items() if k!='frozen_sha256'})
        with self.assertRaises(ValueError): validate_spec(spec)

    def test_rgb_depth_warp_preserves_pixel_identity_and_rejects_background(self):
        import cv2
        from .vision import align_depth_to_endoscope
        depth = np.arange(1,17,dtype=np.float32).reshape(4,4)*.001
        u,v = np.meshgrid(np.arange(4),np.arange(4))
        # RGB samples bottom-up buffer; an upright identity image has inverted map y.
        lens_map = cv2.convertMaps(np.stack((u,3-v),axis=-1).astype(np.float32),None,cv2.CV_16SC2)
        aligned, valid = align_depth_to_endoscope(depth,lens_map,np.ones((4,4,3)),.0001,.5)
        np.testing.assert_array_equal(aligned,depth)
        self.assertTrue(valid.all())
        depth[1,2] = 0
        aligned, valid = align_depth_to_endoscope(depth,lens_map,np.ones((4,4,3)),.0001,.5)
        self.assertFalse(valid[1,2])
        self.assertEqual(aligned[1,2],0)

    def test_two_edge_crossing_and_separated_parallel_edges(self):
        from .cutting import closest_edge_points
        a,b = closest_edge_points([[[-1,0,0],[1,0,0]],[[0,-1,0],[0,1,0]]])
        np.testing.assert_array_equal(a,[0,0,0])
        np.testing.assert_array_equal(a,b)
        a,b = closest_edge_points([[[0,0,0],[1,0,0]],[[0,0,.001],[1,0,.001]]])
        self.assertAlmostEqual(np.linalg.norm(a-b),.001)

    def test_real_grasper_asset_pad_layout_matches_existing_diagnostic(self):
        from types import SimpleNamespace
        from ..scenario import Sequence, scissors_settings
        from ..instrument_geometry import load_grasper
        from .grasp import between_pads
        settings = scissors_settings()
        pads = load_grasper(settings.jaw_length,settings.jaw_width,settings.instrument_model).bounds
        ref = Sequence.__new__(Sequence)
        ref.settings, ref.pads = settings,pads
        ref.jaw_frame = np.array([settings.jaw_axis,settings.jaw_normal,np.cross(settings.jaw_axis,settings.jaw_normal)])
        points = np.random.default_rng(7).uniform(-2,2,(10000,3))
        native = SimpleNamespace(hinge=np.zeros(3),jaw_gap=.2)
        expected,_ = ref.between_pads(points,native)
        tools={'grasper_axis':settings.jaw_axis,'grasper_normal':settings.jaw_normal,
               'grasper_hinge_m':[0,0,0],'grasper_gap_m':.002}
        np.testing.assert_array_equal(between_pads(points*.01,tools,pads*.01,settings.jaw_length*.01),expected)
        self.assertGreater(expected.sum(),0)


if __name__ == '__main__':
    unittest.main()
