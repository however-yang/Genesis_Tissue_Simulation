"""Bounded reset/step/record/restore checks, not a policy-training launch."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
GPU='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'
os.environ.update(CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path.insert(0,str(ROOT.parents[1]/'parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/python-gpu-check'))
from endoscopic_tissue import run,scenario
from scene_config import configure
configure(run,'frontal65',1.25)
# Explicit candidate-only registration BEFORE the training adapter imports its aliases.
scenario.SCISSORS_SCENES=dict(scenario.SCISSORS_SCENES,kidney=run.SCISSORS_SCENES['kidney'])
scenario.scissors_settings=run.scissors_settings
from endoscopic_tissue.training.env import NativeEnvironment,EnvironmentConfig,SceneConfig
from endoscopic_tissue.training.actions import ActionBounds,command_to_action
from endoscopic_tissue.training.data import EpisodeWriter,EpisodeReader
from kidney_training import KidneyEnvironment,EnvSnapshot
from endoscopic_tissue.training.acceptance_queue import GPUReservation
from envelope_audit import capture


def main():
    parser=argparse.ArgumentParser();parser.add_argument('label');parser.add_argument('--name',required=True);args=parser.parse_args()
    if not all(x.replace('-','').isalnum() for x in [args.label,args.name]):raise ValueError('Invalid label')
    output=ROOT/'outputs'/args.name;output.mkdir(exist_ok=False)
    controls=ROOT/'controls'/args.name;controls.mkdir(exist_ok=False)
    guard=GPUReservation(GPU,controls);env=None
    result=dict(passed=False,scope='65k kidney held-pose API smoke; no training and no arbitrary policy-domain claim')
    try:
        dll=ROOT/'binaries'/(args.label+'.so');digest=hashlib.sha256(dll.read_bytes()).hexdigest()
        assert digest==json.loads((ROOT/'results'/('build-'+args.label+'.json')).read_text())['sha256']
        native=ctypes.CDLL(str(dll))
        for name,values in {'LCS_SetCoreOptimizations':(43264,),'LCS_SetBatchQueryMode':(3,),
                            'LCS_SetQueryThreads':(64,),'LCS_SetNativeSchedule':(512,1024),'LCS_SetCoarseBoundCache':(1,)}.items():
            fn=getattr(native,name);fn.argtypes=[ctypes.c_int]*len(values);fn.restype=ctypes.c_int;assert fn(*values)==0
        targets=np.array([command_to_action(s.command) for s in run.SCISSORS_SCENES['kidney'].stages])
        lo,hi=targets.min(axis=0),targets.max(axis=0)
        pad=np.array([.001]*3+[0]+[.001]*3+[.01]*3+[0]);lo-=pad;hi+=pad
        lo[[3,10]]=0;hi[[3,10]]=1
        bounds=ActionBounds(tuple(lo),tuple(hi),tuple([.02]*3+[2]+[.02]*3+[.2]*3+[2]))
        config=EnvironmentConfig(str(dll),digest,bounds,scene=SceneConfig('kidney','held'),
                                 max_steps=6,full_surface_checks=True,gpu_topology_checks=True)
        env=KidneyEnvironment(config)
        state,context=env.reset(seed=65)
        assert context['material_node_count']==65688
        assert not capture(env.native.dll,len(env.grid.edges))['flags'].any()
        snap=env.snapshot();snap.save(output/'held-state.snapshot')
        loaded=EnvSnapshot.load(output/'held-state.snapshot',dll_sha256=digest,
                               physics_config_sha256=snap.metadata['physics_config_sha256'])
        assert loaded.sha256==snap.sha256
        from dataclasses import replace
        settings=env.settings
        env.settings=replace(settings,young_pa=settings.young_pa+1)
        try:
            env.restore(loaded)
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched material snapshot was accepted')
        finally:
            env.settings=settings
        try:
            EnvSnapshot.load(output/'held-state.snapshot',dll_sha256='0'*64,
                             physics_config_sha256=snap.metadata['physics_config_sha256'])
        except ValueError:
            pass
        else:
            raise AssertionError('Wrong native DLL snapshot was accepted')
        initial_id=context['episode_id']
        action=env.previous_action.copy()
        with EpisodeWriter(output/'episode',context,state,env.static_arrays(),env.preparation) as writer:
            for _ in range(6):
                guard({os.getpid()});step=env.step(action);writer.append(step)
                assert step.after is not None and step.after.metadata['quality']['checked_metrics_passed']
        reader=EpisodeReader(output/'episode',verify_hashes=True)
        assert reader.manifest['states']==7 and reader.manifest['normal_transitions']==6
        restored,branch=env.restore(loaded,branch_label='kidney-smoke')
        assert restored.metadata['step_id']==0 and branch['episode_id']!=initial_id
        replay=env.step(action);assert replay.after is not None
        _,new_context=env.reset(seed=66)
        assert new_context['episode_id'] not in (initial_id,branch['episode_id'])
        assert not capture(env.native.dll,len(env.grid.edges))['flags'].any()
        result.update(passed=True,dll_sha256=digest,material_nodes=65688,
                      transitions=6,states=7,verified_episode_hashes=True,
                      compressed_snapshot_bytes=(output/'held-state.snapshot').stat().st_size,
                      snapshot_file_roundtrip=True,restore_new_branch=True,independent_reset=True)
        result['mismatched_material_and_dll_rejected']=True
    except Exception as error:
        result['error']=str(error)
        if env is not None:result['reset_diagnostic']=env.last_reset_diagnostic
        raise
    finally:
        if env is not None:env.close()
        guard.close();(output/'api-smoke.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
