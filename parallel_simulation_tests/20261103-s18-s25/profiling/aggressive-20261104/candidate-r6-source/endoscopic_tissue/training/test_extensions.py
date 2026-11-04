"""S18-S25 CPU contracts. These tests are not GPU physics/FPS acceptance."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import test_contract as records
from . import test_snapshot as snapshots
from . import test_tasks as tasks
from .data import EpisodeReader, EpisodeWriter
from .geometry import WallGeometry
from .instrument_collision import Box, box_distance, capsule_box_distance
from .outcomes import classify, step_events
from .policies import observation, recorded_observation, privileged_state, KINEMATICS_TOOL_FIELDS, Policy
from .pool_session import WorkerSession, scene_config
from .provenance import sha256
from .server_queue import submit, verify_request
from .visual_randomization import VisualRandomization


def tools():
    return dict(grasper_gap_m=.001,grasper_axis=[1,0,0],grasper_normal=[0,1,0],
        scissors_gap_m=.002,scissors_axis=[1,0,0],scissors_normal=[0,1,0],
        grasper_hinge_m=[999,999,999],scissors_hinge_m=[888,888,888],scissors_edges_m=[[777]*3]*4,
        jaw_load_n=[1e8,1e8],reaction_n=[1e8]*3,scissors_loads={'left_n':1e8,'right_n':1e8},
        grip_points=100,grip_error_m=1,scissors_phase=99,secret_diagnostic_n=9e9)


def quality(*, numerical=(), scope=(), components=1):
    return dict(checked_metrics_passed=not numerical,numerical_failure_reasons=list(numerical),
        within_declared_local_scope=not scope,scope_failure_reasons=list(scope),metrics={'material_components':components})


class OutcomeTests(unittest.TestCase):
    def test_valid_outcome_never_overrides_numerical_quality(self):
        from .outcomes import state_valid
        self.assertFalse(state_valid(dict(outcome={'valid_outcome':True},quality=quality(numerical=('invalid_material_tetrahedra',)))))

    def test_released_already_separated_fragment_has_distinct_damage_reason(self):
        out=classify(self.detached(),self.detached(),dict(anchors=dict(new_broken_material_points=2,remaining_material_points=0)))
        self.assertEqual(out['terminal_reason'],'damage:fragment_detached');self.assertTrue(out['valid_outcome'])

    def detached(self, **kwargs):
        return quality(scope=('detached_material_outside_local_task_scope',),components=2,**kwargs)

    def test_tear_is_valid_terminal_and_preserves_original_quality(self):
        q=self.detached(); original=copy.deepcopy(q)
        out=classify(q,quality(),{'new_material_tear_edges':2,'new_material_blade_edges':0})
        self.assertEqual(out['terminal_reason'],'damage:tear_separation')
        self.assertTrue(out['valid_outcome']);self.assertEqual(q,original)

    def test_numerical_failure_dominates_even_with_tears(self):
        out=classify(self.detached(numerical=('invalid_material_tetrahedra',)),quality(),{'new_material_tear_edges':2})
        self.assertFalse(out['valid_outcome']);self.assertTrue(out['terminal_reason'].startswith('numerical:'))

    def test_workspace_is_scope_not_numerical(self):
        out=classify(quality(numerical=('workspace_displacement_exceeded',)))
        self.assertEqual(out['terminal_reason'],'scope:workspace_displacement_exceeded');self.assertFalse(out['valid_outcome'])

    def test_mixed_or_unknown_separation_is_not_invented_damage(self):
        for metrics in ({},{'new_material_tear_edges':1,'new_material_blade_edges':1}):
            out=classify(self.detached(),quality(),metrics,cutting_enabled=True)
            self.assertFalse(out['valid_outcome']);self.assertTrue(out['terminal_reason'].startswith('scope:'))

    def test_blade_cut_is_distinct_from_tear(self):
        out=classify(self.detached(),quality(),{'new_material_blade_edges':2},cutting_enabled=True)
        self.assertEqual(out['material_separation'],'blade');self.assertTrue(out['valid_outcome'])
        self.assertFalse(out['terminal_reason'].startswith('damage:'))

    def test_valid_damage_final_state_survives_recording_and_reader(self):
        with tempfile.TemporaryDirectory() as d:
            before,after=records.state(0),records.state(1)
            after.metadata['quality']=self.detached()
            after.metadata['outcome']=classify(after.metadata['quality'],quality(),{'new_material_tear_edges':1})
            with EpisodeWriter(Path(d)/'ep',records.context(),before,{}) as w:
                w.append(records.result(before,after,terminated=True,reason='damage:tear_separation',valid_outcome=True))
            r=EpisodeReader(Path(d)/'ep')
            self.assertEqual(len(list(r.windows(1,allow_unvalidated_scope=True))),1)
            self.assertEqual(r.outcome_counts()['damage_terminals'],1)
            self.assertEqual(r.manifest['states'],2)

    def test_damage_is_task_failure_with_valid_reward(self):
        initial=tasks.sample(); goal=tasks.TaskGoal(kind='incision',target_edges=((0,1),),weights=tasks.WEIGHTS)
        monitor=tasks.TaskMonitor(goal,tasks.fixture(),initial)
        after=tasks.sample(1,cut=True)
        after.metadata['quality']=self.detached()
        after.metadata['outcome']=classify(after.metadata['quality'],quality(),{'new_material_tear_edges':1})
        result=monitor.evaluate(initial,after,tasks.ACTION,tasks.BOUNDS)
        self.assertTrue(result['physical_valid_for_task']);self.assertTrue(result['reward_valid'])
        self.assertEqual(result['failure'],'damage:tear_separation');self.assertFalse(result['success'])

    def test_grasp_loss_event_does_not_terminate_and_has_snapshot_history(self):
        before,after=tasks.sample(),tasks.sample(1)
        before.metadata['grasp'].update(witness_ids=[0],witness_retained_fraction=1.)
        after.metadata['grasp'].update(witness_ids=[0],witness_retained_fraction=.2)
        after.metadata['failure_physics']={'new_material_tear_edges':1,'new_material_blade_edges':2}
        action={'requested_command_native':{'scissors_cutting':True,'scissors_closure':1}}
        step_events(before,after,action)
        kinds={e['kind'] for e in after.metadata['events']}
        self.assertEqual(kinds,{'grasp_lost','first_tear_edge','cut_onset'})
        continued=copy.deepcopy(after);continued.metadata['step_id']=2
        step_events(after,continued,action)
        self.assertNotIn('first_tear_edge',[e['kind'] for e in continued.metadata['events']])
        self.assertIsNone(classify(quality())['terminal_reason'])


class ObservationTests(unittest.TestCase):
    def state(self,index=0):
        state=records.state(index);state.metadata['tools']=tools()
        state.metadata['previous_target_si']=np.linspace(0,1,11).tolist()
        return state

    def test_whitelist_has_no_force_truth_or_solved_position(self):
        s=self.state();out=observation(s,np.ones(11)*999,contract='tool_kinematics_v1')
        self.assertEqual(set(out['tools']),set(KINEMATICS_TOOL_FIELDS))
        np.testing.assert_array_equal(out['previous_target_si'],s.metadata['previous_target_si'])
        self.assertNotIn('positions_m',out);self.assertNotIn('scissors_hinge_m',out['tools'])
        self.assertNotIn('reaction_n',out['tools']);self.assertNotIn('jaw_load_n',out['tools'])
        out['tools']['grasper_axis'][0]=55;self.assertEqual(s.metadata['tools']['grasper_axis'][0],1)

    def test_legacy_contract_unchanged(self):
        out=observation(self.state(),np.zeros(11))
        self.assertEqual(out['observation_contract'],'simulated_tool_measurements_v1')
        self.assertIn('jaw_load_n',out['tools'])

    def test_tracking_requires_explicit_switch_and_no_force_policy(self):
        out=observation(self.state(),np.zeros(11),contract='tool_kinematics_v1',visual_tracking=True)
        self.assertEqual(out['visually_tracked_tools']['scissors_hinge_m'],[888]*3)
        with self.assertRaises(ValueError):Policy({'kind':'force_feedback','observation_contract':'tool_kinematics_v1'},np.zeros(11),0)
        with self.assertRaises(ValueError):observation(self.state(),np.zeros(11),contract='tool_kinematics_v1',privileged=True)

    def test_privileged_output_is_separate_owned_and_explicit(self):
        s=self.state();p=privileged_state(s,['positions_m','telemetry','solved_tools'])
        self.assertIn('jaw_load_n',p['solved_tools']);p['positions_m'][:]=99
        self.assertFalse(np.all(s.arrays['positions_m']==99))
        with self.assertRaises(ValueError):privileged_state(s,['all'])

    def test_recorded_observation_reproduces_command_and_tools(self):
        with tempfile.TemporaryDirectory() as d:
            s=self.state()
            with EpisodeWriter(Path(d)/'ep',records.context(),s,{}):pass
            r=EpisodeReader(Path(d)/'ep')
            out=recorded_observation(r,0,contract='tool_kinematics_v1')
            expected=observation(s,np.zeros(11),contract='tool_kinematics_v1')
            self.assertEqual(out['tools'],expected['tools'])
            np.testing.assert_array_equal(out['previous_target_si'],expected['previous_target_si'])


class PoolTests(unittest.TestCase):
    def test_explicit_reset_passes_configuration_and_invalidates_old_snapshots(self):
        from dataclasses import asdict
        from .provenance import json_hash
        f,s=self.session();saved=self.capture(s)
        selected={'scene':'wall-traction','physics':{'young_pa':12000},'contact_model':1}
        def reset(*,seed,scene_config,goal):
            f.env.scene_config=scene_config
            f.env.context.update(seed=seed,scene_config=asdict(scene_config),physics_config_sha256=json_hash(asdict(scene_config)))
            return f.env.state,f.env.context
        with patch.object(f.env,'reset',side_effect=reset) as method:
            out=s.execute('reset',dict(seed=7,scene_config=selected))
        self.assertEqual(method.call_args.kwargs['scene_config'].physics.young_pa,12000)
        self.assertEqual(out['context']['physics_config_sha256'],json_hash(asdict(f.env.scene_config)))
        with self.assertRaises(KeyError):s.execute('restore',{'snapshot_id':saved['snapshot_id']})

    def test_snapshot_restore_preserves_history_but_does_not_repeat_event(self):
        f,s=self.session()
        f.env.state.metadata.update(events=[dict(kind='first_tear_edge',step_id=4)],event_history={'tear_seen':True})
        saved=self.capture(s);s.execute('restore',{'snapshot_id':saved['snapshot_id']})
        self.assertEqual(f.env.state.metadata['events'],[])
        self.assertTrue(f.env.state.metadata['event_history']['tear_seen'])

    def session(self, **kwargs):
        f=snapshots.SnapshotTests();f.setUp();self.addCleanup(f.tearDown)
        f.env.state.metadata['tools']=tools()
        f.env.state.metadata['previous_target_si']=f.env.previous_action.tolist()
        worker=WorkerSession(f.env,f.root,record=False,**kwargs)
        self.addCleanup(worker.close)
        return f,worker

    def capture(self,worker):
        with patch.object(snapshots.MemoryBackend,'size',create=True,return_value=16), \
             patch('endoscopic_tissue.training.pool_session.psutil.virtual_memory',return_value=SimpleNamespace(available=64*1024**3)):
            return worker.execute('snapshot',{})

    def test_scene_config_per_reset_is_typed_and_carries_overrides(self):
        value=scene_config({'scene':'wall-traction','contact_model':1,'physics':{'young_pa':3000},'geometry':{'length_m':.026}})
        self.assertEqual(value.contact_model,1);self.assertEqual(value.physics.young_pa,3000)
        self.assertEqual(value.geometry.length_m,.026)
        with self.assertRaises(TypeError):scene_config({'unknown':1})

    def test_five_restores_preserve_state_and_branch_lineage(self):
        f,s=self.session(observation_contract='tool_kinematics_v1')
        snap=self.capture(s);ids=[]
        for i in range(5):
            f.env.observed.nodes[:]=999
            out=s.execute('restore',{'snapshot_id':snap['snapshot_id'],'branch_label':str(i)})
            np.testing.assert_array_equal(f.env.observed.nodes,400.)
            self.assertEqual(out['context']['branch_from_step'],4)
            self.assertEqual(out['context']['branch_label'],str(i))
            ids.append(out['episode_id'])
        self.assertEqual(len(set(ids)),5)

    def test_worker_ids_cannot_be_restored_by_another_worker(self):
        f,a=self.session();g,b=self.session();saved=self.capture(a)
        with self.assertRaises(KeyError):b.execute('restore',{'snapshot_id':saved['snapshot_id']})
        self.assertEqual(g.env.native.device_state,b'physical history')

    def test_cache_limit_and_drop_are_explicit(self):
        f,s=self.session();saved=self.capture(s);s.budget=s.reserved
        with self.assertRaises(MemoryError):self.capture(s)
        self.assertEqual(len(s.snapshots),1)
        s.execute('drop_snapshot',{'snapshot_id':saved['snapshot_id']})
        self.assertEqual(s.reserved,0)
        with self.assertRaises(KeyError):s.execute('restore',{'snapshot_id':saved['snapshot_id']})

    def test_observation_budget_rejects_large_privileged_payload(self):
        f,s=self.session(privileged_fields=['positions_m'],max_response_bytes=1)
        with self.assertRaises(MemoryError):s._observe(f.env.state)


class CollisionTests(unittest.TestCase):
    def box(self,x):return Box(np.array([x,0.,0.]),np.eye(3),np.ones(3)*.001)
    def test_separation_touch_overlap_and_containment(self):
        a=self.box(0)
        self.assertAlmostEqual(box_distance(a,self.box(.003)),.001)
        self.assertAlmostEqual(box_distance(a,self.box(.002)),0)
        self.assertLess(box_distance(a,self.box(.0017)),-.0002)
        self.assertAlmostEqual(box_distance(a,a),-.002)
    def test_distance_is_symmetric_under_rigid_rotation(self):
        a,b=self.box(0),self.box(.004)
        theta=.43;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        ra=Box(a.center@R,R,a.half);rb=Box(b.center@R,R,b.half)
        self.assertAlmostEqual(box_distance(a,b),box_distance(ra,rb),places=10)
        self.assertAlmostEqual(box_distance(rb,ra),.002,places=10)
    def test_capsule_crossing_and_clearance(self):
        a=self.box(0)
        self.assertLess(capsule_box_distance(np.array([-.01,0,0]),np.array([.01,0,0]),.001,a),-.0002)
        self.assertAlmostEqual(capsule_box_distance(np.array([-.01,.004,0]),np.array([.01,.004,0]),.001,a),.002)


class GeometryVisualQueueTests(unittest.TestCase):
    def test_pixel_support_composition_retains_affine_offset_and_invalid_mask(self):
        from .visual_labels import compose_pixel_supports
        ids=np.full((3,8),-1,np.int32);ids[:,0]=np.arange(3)
        weights=np.zeros((3,8));weights[:,0]=1
        bindings=dict(node_ids=ids,weights=weights,offset_m=np.array([[.1,0,0],[0,.2,0],[0,0,.3]]),
                      triangles=np.array([[0,1,2]]),valid=np.ones(3,bool))
        out=compose_pixel_supports(np.array([0,-1]),np.array([[.2,.3,.5],[0,0,0]]),bindings)
        self.assertEqual(out['valid'].tolist(),[True,False])
        self.assertAlmostEqual(out['weights'][0].sum(),1)
        np.testing.assert_allclose(out['offset_m'][0],[.02,.06,.15])
        bindings['valid'][1]=False
        self.assertFalse(compose_pixel_supports(np.array([0]),np.array([[.2,.3,.5]]),bindings)['valid'][0])

    def test_segmentation_preserves_class_zero_occlusion_and_restores_render_map(self):
        from .visual_labels import segmentation
        from types import SimpleNamespace as S
        background,tissue,grasper,scissors=object(),object(),object(),object()
        context=S(seg_node_map={background:np.array([1,2,3])},external_nodes={'tissue':tissue},
                  _scene=S(mesh_nodes=[background,tissue,grasper,scissors]))
        def render(**kwargs):
            self.assertEqual(context.seg_node_map[tissue].tolist(),[0,0,1])
            self.assertEqual(context.seg_node_map[background].tolist(),[0,0,0])
            return None,None,np.array([[0,1],[2,3]]),None
        import cv2
        xy=np.array([[[0,1],[1,1]],[[0,0],[1,0]]],np.float32)
        view=S(scene=S(visualizer=S(context=context)),tissue_meshes=[S(name='tissue'),None],
               grasper_jaws=[grasper],grasper_body=[],scissors_blades=[scissors],scissors_body=[],
               camera=S(render=render),lens_map=cv2.convertMaps(xy,None,cv2.CV_16SC2),vignette=np.ones((2,2,3)))
        labels=segmentation(view)
        np.testing.assert_array_equal(labels,[[0,1],[2,3]])
        self.assertEqual(set(context.seg_node_map),{background})
        np.testing.assert_array_equal(context.seg_node_map[background],[1,2,3])

    def test_geometry_removes_near_cut_support_without_new_pins(self):
        original=WallGeometry().build_grid(1050)
        candidate=WallGeometry(support_length_m=.003).build_grid(1050)
        self.assertLess(int(candidate.inside.sum()),int(original.inside.sum()))
        self.assertTrue(np.all(candidate.positions[candidate.pinned!=0,1] <= -.7+1e-6))
        unsupported=(candidate.positions[:,0]>.4)&(candidate.positions[:,1]<.7)
        self.assertFalse(candidate.inside[unsupported].any())
        self.assertEqual(WallGeometry().scene().stages,WallGeometry(support_length_m=.003).scene().stages)

    def test_visual_sampling_and_noise_restore_without_global_rng(self):
        config=VisualRandomization(camera_translation_m=.001,fov_delta_deg=3,rgb_noise_std=3,depth_noise_std_m=.0001)
        pose=((0,0,.1),(0,0,0));a=config.sample(17,pose)
        self.assertEqual(a,config.sample(17,pose));self.assertNotEqual(a,config.sample(18,pose))
        rgb=np.zeros((8,8,3),np.uint8);d=np.full((8,8),.1,np.float32);valid=np.ones((8,8),bool);valid[0]=False
        x=config.noise(rgb,d,valid,seed=17,native_tick=100)
        y=config.noise(rgb,d,valid,seed=17,native_tick=100)
        for u,v in zip(x,y):np.testing.assert_array_equal(u,v)
        self.assertTrue(np.all(x[1][0]==0));self.assertFalse(x[2][0].any())
        for value in (-1,np.nan,99):
            with self.assertRaises(ValueError):VisualRandomization(rgb_noise_std=value)

    def test_disabled_visual_noise_is_identity(self):
        a,b,c=np.zeros((2,2,3),np.uint8),np.ones((2,2)),np.ones((2,2),bool)
        x=VisualRandomization().noise(a,b,c,seed=1,native_tick=1)
        self.assertIs(x[0],a);self.assertIs(x[1],b);self.assertIs(x[2],c)

    def test_queue_submission_pins_spec_and_rejects_later_changes(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);lib=d/'dummy.so';lib.write_bytes(b'CPU-only fixture; never executed')
            spec=dict(schema='tissue_experiment_v1',purpose='collection',dll=str(lib),dll_sha256=sha256(lib),
                      episodes=[dict(id='one',seed=0,policy={'kind':'hold'})])
            job=submit(d/'queue',spec);request,loaded=verify_request(job)
            self.assertEqual(loaded,spec);self.assertEqual(json.loads((job/'status.json').read_text())['phase'],'queued')
            (job/'spec.json').write_text('{}')
            with self.assertRaises(ValueError):verify_request(job)




class FastTopologyTests(unittest.TestCase):
    def test_exact_predicates_for_welded_cut_soup_and_corruptions(self):
        import trimesh
        from ..native_backend import Surface
        from ..surface_diagnostics import topology_statistics
        from .topology_fast import topology_statistics_fast
        mesh=trimesh.creation.icosphere(subdivisions=2)
        for kind in ('closed','boundary','duplicate','reversed','degenerate','unreferenced'):
            vertices=mesh.vertices.astype(np.float32);faces=mesh.faces.copy()
            if kind=='boundary':faces=faces[1:]
            if kind=='duplicate':faces=np.vstack((faces,faces[:1]))
            if kind=='reversed':faces[0]=faces[0,::-1]
            if kind=='degenerate':vertices[0]=vertices[1]
            if kind=='unreferenced':vertices=np.vstack((vertices,[[123,456,789]]))
            anchors=np.zeros((len(vertices),4));anchors[:,3]=(vertices[:,0]>0)
            surface=Surface(vertices,np.zeros_like(vertices),faces,anchors)
            self.assertEqual(topology_statistics(surface),topology_statistics_fast(surface),kind)


class PoolWireConfigTests(unittest.TestCase):
    def test_full_dataclass_configuration_roundtrip_preserves_typed_limits(self):
        from dataclasses import asdict
        from .env import EnvironmentConfig,SceneConfig
        from .collect import local_bounds
        from .quality import QualityLimits
        from .pool_session import environment_config
        scene=SceneConfig(preparation='settled')
        original=EnvironmentConfig('/explicit/library','0'*64,local_bounds(scene),scene)
        restored=environment_config(asdict(original))
        self.assertIsInstance(restored.limits,QualityLimits)
        self.assertEqual(asdict(restored),asdict(original))


class RenderDeviceTests(unittest.TestCase):
    def test_uuid_binding_does_not_assume_egl_and_visible_cuda_indices_match(self):
        import os
        from .egl_device import bind
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-selected'},clear=False):
            with patch('endoscopic_tissue.training.egl_device.subprocess.run',return_value=SimpleNamespace(stdout='[{"gpu_uuid":"GPU-other","egl_index":0},{"gpu_uuid":"GPU-selected","egl_index":3}]')) as run:
                result=bind('GPU-selected')
                self.assertEqual(result['egl_index'],3)
                self.assertEqual(os.environ['EGL_DEVICE_ID'],'3')
                self.assertNotIn('CUDA_VISIBLE_DEVICES',run.call_args.kwargs['env'])
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'GPU-selected')

    def test_unmapped_uuid_fails_before_creating_a_renderer(self):
        from .egl_device import bind
        with patch('endoscopic_tissue.training.egl_device.subprocess.run',return_value=SimpleNamespace(stdout='[]')):
            with self.assertRaises(RuntimeError):bind('GPU-selected')

    def test_resource_guard_includes_graphics_processes(self):
        from .resources import gpu_sample
        xml='<nvidia_smi_log><gpu><uuid>GPU-selected</uuid><fb_memory_usage><total>81920 MiB</total><used>200 MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util><memory_util>0 %</memory_util></utilization><temperature><gpu_temp>40 C</gpu_temp></temperature><processes><process_info><pid>123</pid><type>G</type><used_memory>180 MiB</used_memory></process_info></processes></gpu></nvidia_smi_log>'
        with patch('endoscopic_tissue.training.resources.subprocess.run',return_value=SimpleNamespace(stdout=xml)):
            result=gpu_sample('GPU-selected')
        self.assertEqual(result['processes'][0]['pid'],123)
        self.assertEqual(result['processes'][0]['type'],'G')


if __name__=='__main__':unittest.main()
