"""Behavioral tests for ownership, frame identity, and quality/render joining."""
import os,sys,time,threading,unittest
from pathlib import Path
from types import SimpleNamespace
p=Path(__file__).resolve().parents[1];sys.path.insert(0,str(p/'candidate-source'));os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from endoscopic_tissue.training.vision import SynchronizedVision
from endoscopic_tissue.training.env import NativeEnvironment

class HandoffTests(unittest.TestCase):
 def make_vision(self):
  vision=SynchronizedVision.__new__(SynchronizedVision);vision.parallel_same_frame=True;vision._prepared_frame=None
  def raw(env,state):
   state.arrays['rgb']=np.full((2,3,3),state.metadata['native_tick'],np.uint8);state.arrays['rgb'].flags.writeable=False
   state.metadata['vision']=dict(native_tick=state.metadata['native_tick'],step_id=state.metadata['step_id']);return 1.
  vision._capture_raw=raw;return vision
 def test_same_frame_and_retained_observation(self):
  vision=self.make_vision();env=SimpleNamespace(episode_id='a',step_id=10,native=SimpleNamespace(tick=18),settings=SimpleNamespace(dt=1/30))
  vision.prepare_frame(env);state=SimpleNamespace(arrays={},metadata=dict(step_id=10,native_tick=18));vision.capture(env,state);old=state.arrays['rgb'];assert (old==18).all()
  env.step_id=11;env.native.tick=19;vision.prepare_frame(env);other=SimpleNamespace(arrays={},metadata=dict(step_id=11,native_tick=19));vision.capture(env,other)
  assert (old==18).all() and (other.arrays['rgb']==19).all() and not np.shares_memory(old,other.arrays['rgb'])
 def test_mismatched_state_rejected(self):
  vision=self.make_vision();env=SimpleNamespace(episode_id='a',step_id=10,native=SimpleNamespace(tick=18),settings=SimpleNamespace(dt=1/30));vision.prepare_frame(env)
  with self.assertRaises(RuntimeError):vision.capture(env,SimpleNamespace(arrays={},metadata=dict(step_id=11,native_tick=19)))
 def test_unconsumed_frame_rejected(self):
  vision=self.make_vision();env=SimpleNamespace(episode_id='a',step_id=10,native=SimpleNamespace(tick=18),settings=SimpleNamespace(dt=1/30));vision.prepare_frame(env)
  with self.assertRaises(RuntimeError):vision.prepare_frame(env)
 def env_for_join(self,render_raises):
  env=NativeEnvironment.__new__(NativeEnvironment);env._quality_executor=None;env._step_in_progress=True;env.step_id=1;env.damage=None;env.observed=SimpleNamespace();env.config=SimpleNamespace(full_surface_checks=True)
  env.native=SimpleNamespace(material_quality=lambda:np.array([1.,0.]),surface=lambda:object())
  started=threading.Event();render=threading.Event();done=threading.Event()
  def assess(previous,command,material,tears):
   started.set();assert render.wait(1);time.sleep(.02);done.set();return dict(checked_metrics_passed=False),20.
  env._assess_cpu=assess
  def prepare(e):
   assert started.wait(1);render.set()
   if render_raises:raise ValueError('render failed')
  env._snapshot_vision=SimpleNamespace(parallel_same_frame=True,every=1,prepare_frame=prepare)
  return env,done
 def test_quality_join_and_failure_preserved(self):
  env,done=self.env_for_join(False)
  try:
   result,timings=env._assess(None,None);assert done.is_set() and not result['checked_metrics_passed'] and timings['same_frame_parallel']
  finally:env._quality_executor.shutdown()
 def test_render_failure_also_joins_quality(self):
  env,done=self.env_for_join(True)
  try:
   with self.assertRaisesRegex(ValueError,'render failed'):env._assess(None,None)
   assert done.is_set()
  finally:env._quality_executor.shutdown()
if __name__=='__main__':unittest.main()
