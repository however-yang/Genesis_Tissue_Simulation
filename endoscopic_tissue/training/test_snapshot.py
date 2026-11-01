"""Snapshot ownership, corruption rejection, branch isolation and lineage tests."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ..native_backend import Observation, Settings
from .actions import ActionBounds
from .data import EpisodeReader, EpisodeWriter
from .env import EnvironmentConfig, NativeEnvironment, SceneConfig
from .lineage import episode_lineage, validate_lineages
from .physics import native_defaults
from .provenance import json_hash, sha256
from .snapshot import EnvSnapshot, encode_python, decode_python
from .tasks import TaskGoal, TaskMonitor, recompute_episode
from .test_contract import state as fixture_state, result


class MemoryBackend:
    loads = 0
    def __init__(self, native):
        self.native = native
    def save(self):
        return self.native.device_state
    def load(self, data):
        MemoryBackend.loads += 1
        self.native.device_state = data


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        dll = self.root/'fixture.dll'
        dll.write_bytes(b'CPU snapshot fixture, never a native physics DLL')
        config = EnvironmentConfig(str(dll), sha256(dll), ActionBounds((0.,)*11, (1.,)*11, (1.,)*11))
        self.env = env = NativeEnvironment(config)
        s = fixture_state(4)
        grid = SimpleNamespace(positions=s.arrays['positions_m']*100,
            neighbors=s.topology['neighbors'], inside=np.ones(3,np.int32),
            active=np.ones(3,np.int32), pinned=np.array([1,0,0],np.int32))
        env.grid, env.settings, env.native_parameters = grid, Settings(), native_defaults()
        env.scene_config = SceneConfig()
        env.native = SimpleNamespace(tick=34, neighbors=grid.neighbors, topology_revision=0,
            dll_path=dll, device_state=b'physical history', close=lambda: None)
        env.state, env.step_id, env.episode_id = s, 4, s.metadata['episode_id']
        env.observed = Observation(grid.positions.copy(), grid.neighbors, np.zeros(3,np.int32),
            np.zeros(3), 1., 2, 0., np.zeros(10,np.float32), np.zeros(3))
        env.previous_action = np.full(11,.5)
        env.rng = np.random.default_rng(5)
        env.quality = SimpleNamespace(previous_neighbors=grid.neighbors, components=1)
        env.grasp = SimpleNamespace(ids=np.array([1,2],np.int32), initial_local=np.ones((2,3)), initial_world=np.zeros((2,3)))
        env.damage = env.failure_metrics = env.task = env.last_surface = None
        env._topology_revision, env._topology = 0, s.topology
        env.preparation, env.done = [], False
        env.context = {'episode_id':env.episode_id, 'dll_sha256':sha256(dll), 'seed':5}
        env.context['context_sha256'] = json_hash(env.context)
        self.mock = patch('endoscopic_tissue.training.snapshot.SnapshotBackend',MemoryBackend)
        self.mock.start()
        MemoryBackend.loads = 0

    def tearDown(self):
        self.mock.stop()
        self.temp.cleanup()

    def test_codec_preserves_array_identity_values_and_writeability_without_pickle(self):
        a = np.arange(24,dtype=np.float32).reshape(8,3)[::2]
        a.flags.writeable = False
        encoded = encode_python({'one':a,'two':(a,self.env.observed),'rng':self.env.rng.bit_generator.state})
        decoded = decode_python(encoded)
        self.assertIs(decoded['one'],decoded['two'][0])
        self.assertFalse(decoded['one'].flags.writeable)
        np.testing.assert_array_equal(a,decoded['one'])
        self.assertIsInstance(decoded['two'][1],Observation)
        with self.assertRaises(TypeError):
            encode_python(np.array([object()],dtype=object))
        with self.assertRaises(ValueError):
            decode_python(encoded+b'extra')

    def test_snapshot_owns_bytes_and_each_restore_has_new_episode_but_same_root(self):
        snap = self.env.snapshot()
        with self.assertRaises(FrozenInstanceError):
            snap.native_compressed = b'changed'
        exposed = snap.metadata
        exposed['parent_episode_id'] = 'mutated-copy'
        self.env.observed.nodes[:] = 99
        self.env.previous_action[:] = 1
        first, a = self.env.restore(snap,branch_label='a')
        self.assertEqual(a['parent_episode_id'],'cpu-fixture')
        self.assertEqual(a['reset_lineage_id'],'cpu-fixture')
        self.assertEqual(a['branch_from_step'],4)
        self.assertEqual(first.metadata['step_id'],0)
        self.assertEqual(first.metadata['native_tick'],34)
        np.testing.assert_array_equal(self.env.previous_action,.5)
        np.testing.assert_array_equal(self.env.observed.nodes,400.)
        self.env.grasp.ids[:] = 0
        self.env.native.device_state = b'changed branch'
        _, b = self.env.restore(snap,branch_label='b')
        self.assertNotEqual(a['episode_id'],b['episode_id'])
        self.assertEqual(self.env.native.device_state,b'physical history')
        np.testing.assert_array_equal(self.env.grasp.ids,[1,2])
        self.assertIs(self.env.quality.previous_neighbors,self.env.observed.neighbors)
        child = self.env.snapshot()
        _, c = self.env.restore(child)
        self.assertEqual(c['parent_episode_id'],b['episode_id'])
        self.assertEqual(c['reset_lineage_id'],'cpu-fixture')

    def test_snapshot_file_checks_payload_abi_and_expected_physics(self):
        snap = self.env.snapshot()
        path = self.root/'one.lcssnap'
        snap.save(path)
        loaded = EnvSnapshot.load(path,dll_sha256=snap.metadata['dll_sha256'],state_abi=1,
                                  physics_config_sha256=snap.metadata['physics_config_sha256'])
        self.assertEqual(loaded.sha256,snap.sha256)
        with self.assertRaises(FileExistsError):
            snap.save(path)
        for values in ({'dll_sha256':'0'*64},{'state_abi':2},{'physics_config_sha256':'bad'}):
            with self.assertRaises(ValueError):
                EnvSnapshot.load(path,**{'dll_sha256':snap.metadata['dll_sha256'],
                    'physics_config_sha256':snap.metadata['physics_config_sha256'],**values})
        corrupt = bytearray(path.read_bytes())
        corrupt[-1] ^= 1
        path.write_bytes(corrupt)
        with self.assertRaises(ValueError):
            EnvSnapshot.load(path,dll_sha256=snap.metadata['dll_sha256'],
                             physics_config_sha256=snap.metadata['physics_config_sha256'])

    def test_physics_mismatch_and_midstep_snapshot_rejected_before_native_mutation(self):
        snap = self.env.snapshot()
        self.env.settings = replace(self.env.settings,young_pa=self.env.settings.young_pa+1)
        with self.assertRaises(ValueError):
            self.env.restore(snap)
        self.assertEqual(MemoryBackend.loads,0)
        self.env._step_in_progress = True
        with self.assertRaises(RuntimeError):
            self.env.snapshot()

    def test_rng_returns_to_snapshot_and_preparation_stage_does_not_change_physics(self):
        snap = self.env.snapshot()
        expected = self.env.rng.integers(0,2**32,size=10)
        self.env.scene_config = replace(self.env.scene_config,preparation='cut_ready')
        self.env.restore(snap)
        np.testing.assert_array_equal(expected,self.env.rng.integers(0,2**32,size=10))

    def test_branch_cannot_change_cut_permission_and_records_actual_episode_controls(self):
        snap = self.env.snapshot()
        original = self.env.config
        self.env.config = replace(original,cutting_enabled=True)
        with self.assertRaises(ValueError):
            self.env.restore(snap)
        self.assertEqual(MemoryBackend.loads,0)
        self.env.config = replace(original,diagnostic_continue=True,max_steps=33)
        _,context = self.env.restore(snap)
        self.assertTrue(context['diagnostic_continue'])
        self.assertEqual(context['max_steps'],33)
        self.assertFalse(context['cutting_enabled'])

    def test_branch_writer_starts_at_restored_state_and_cannot_append_parent_history(self):
        snap = self.env.snapshot()
        first, context = self.env.restore(snap)
        after = decode_python(encode_python(first))
        after.metadata.update(step_id=1,t_sim_s=first.metadata['dt_s'],native_tick=35)
        directory = self.root/'branch'
        with EpisodeWriter(directory,context,first,self.env.static_arrays()) as writer:
            with self.assertRaises(ValueError):
                writer.append(result(fixture_state(0),fixture_state(1)))
            writer.append(result(first,after))
        reader = EpisodeReader(directory)
        self.assertEqual(reader.lineage['reset_lineage_id'],'cpu-fixture')
        self.assertEqual(reader.manifest['states'],2)
        self.assertEqual(reader.manifest['normal_transitions'],1)
        self.assertEqual(len(list(reader.windows(1,allow_unvalidated_scope=True))),1)

    def test_vision_rebuilt_after_restore_and_missing_renderer_is_rejected(self):
        class Vision:
            manifest = {'fixture': 'CPU renderer', 'width': 2, 'height': 2}
            def capture(self,env,state):
                state.arrays['rgb'] = np.full((2,2,3),env.native.device_state[0],np.uint8)
                return 7.
        vision = Vision()
        self.env._snapshot_vision = vision
        self.env.state.arrays['rgb'] = np.zeros((2,2,3),np.uint8)
        self.env.native.surface = lambda:'new restored surface'
        snap = self.env.snapshot()
        self.assertNotIn('rgb',snap.python_state()['state'].arrays)
        self.env._snapshot_vision = None
        with self.assertRaises(ValueError):
            self.env.restore(snap)
        self.assertEqual(MemoryBackend.loads,0)
        self.env._snapshot_vision = vision
        vision.manifest = {**vision.manifest, 'width': 3}
        with self.assertRaisesRegex(ValueError,'vision configuration'):
            self.env.restore(snap)
        self.assertEqual(MemoryBackend.loads,0)
        vision.manifest = {**vision.manifest, 'width': 2}
        restored, context = self.env.restore(snap)
        self.assertEqual(self.env.last_surface,'new restored surface')
        self.assertTrue((restored.arrays['rgb'] == ord('p')).all())
        self.assertEqual(context['restore_vision_ms'],7.)
        self.assertEqual(context['vision'],vision.manifest)

    def test_inherited_task_history_is_preserved_and_offline_recomputable(self):
        env=self.env
        env.state.metadata['grasp']={'witness_retained_fraction':1.,'witness_tangential_slip_m':0.,
                                     'geometric_loaded_hold':True,'anchor_points':2}
        env.state.metadata['tools']={'grasper_gap_m':.001,'grasper_hinge_m':[0,0,0]}
        goal=TaskGoal('hold',hold_nodes=(1,2),displacement_m=(0,0,0),required_hold_s=1.)
        env.task=TaskMonitor(goal,env.static_arrays(),env.state)
        env.task.dwell_s=.25
        snap=env.snapshot()
        first,context=env.restore(snap)
        self.assertEqual(env.task.dwell_s,.25)
        after=decode_python(encode_python(first))
        after.metadata.update(step_id=1,t_sim_s=first.metadata['dt_s'],native_tick=35)
        action={'executed_target_si':[.5]*11,'previous_target_si':[.5]*11,'policy':[.5]*11}
        task=env.task.evaluate(first,after,action,env.config.bounds)
        directory=self.root/'task-branch'
        with EpisodeWriter(directory,context,first,env.static_arrays()) as writer:
            writer.append(result(first,after,action=action,task=task))
        rows=recompute_episode(EpisodeReader(directory))
        self.assertEqual(rows,[task])
        env.restore(snap,goal=replace(goal,required_hold_s=2.))
        self.assertEqual(env.task.dwell_s,0.)


class LineageTests(unittest.TestCase):
    def test_partial_or_self_parented_branch_rejected(self):
        with self.assertRaises(ValueError):
            episode_lineage({'episode_id':'a','parent_episode_id':'b'})
        with self.assertRaises(ValueError):
            episode_lineage({'episode_id':'a','reset_lineage_id':'root','parent_episode_id':'a',
                             'branch_from_step':2,'snapshot_sha256':'0'*64})

    def test_evaluation_branch_and_geometry_change_are_rejected(self):
        parent = {'episode_id':'root','reset_lineage_id':'root','parent_episode_id':None,
                  'geometry_group':'geometry','purpose':'collection'}
        branch = {'episode_id':'branch','reset_lineage_id':'root','parent_episode_id':'root',
                  'geometry_group':'geometry','purpose':'evaluation'}
        with self.assertRaisesRegex(ValueError,'snapshot branch'):
            validate_lineages([parent,branch])
        branch.update(purpose='collection',geometry_group='other')
        with self.assertRaisesRegex(ValueError,'physical geometry'):
            validate_lineages([parent,branch])


if __name__ == '__main__':
    unittest.main()
