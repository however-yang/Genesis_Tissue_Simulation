"""Exercise the rare post-rejection evidence branch without changing a solver."""
import ast,json,os,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace as S
import numpy as np

class CaptureTests(unittest.TestCase):
 def make(self,directory,passed):
  out=Path(directory)/'worker-000';(out/'episode-000').mkdir(parents=True)
  result=S(after=S(metadata={'quality':{'checked_metrics_passed':passed,'within_declared_local_scope':True}}))
  class Stub:
   def replay_command(self,command):return result
   def close(self):pass
  def write_json(path,data):path.write_text(json.dumps(data,allow_nan=False))
  tree=ast.parse(Path(__file__).with_name('topology_capture_worker.py').read_text())
  main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  cls=next(n for n in main.body if isinstance(n,ast.ClassDef) and n.name=='RecordedEnvironment')
  namespace=dict(env_class=Stub,out=out,current_episode=['episode-000'],np=np,time=time,os=os,write_json=write_json)
  exec(compile(ast.Module(body=[cls],type_ignores=[]),'isolated-capture-branch','exec'),namespace)
  env=namespace['RecordedEnvironment']();env.native=S(tick=123)
  nodes=np.arange(12,dtype=np.float32).reshape(4,3);neighbors=np.full((4,6),-1,np.int32)
  env.observed=S(nodes=nodes,neighbors=neighbors,grip_flags=np.zeros(4,np.int32))
  env.grid=S(positions=nodes.copy(),inside=np.ones(4,np.int32),active=np.ones(4,np.int32),neighbors=neighbors.copy())
  env.last_surface=S(vertices=nodes.copy(),normals=np.ones_like(nodes),triangles=np.array([[0,1,2]],np.int32),anchors=np.zeros((4,4),np.float32))
  return env,out,result
 def test_original_rejection_and_arrays_preserved_once(self):
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
   env,out,result=self.make(d,False);before=env.observed.nodes.copy();self.assertIs(env.replay_command(None),result)
   folder=out/'episode-000/failure-tick-123'
   with np.load(folder/'state.npz',allow_pickle=False) as data:
    np.testing.assert_array_equal(data['nodes_cm'],before);np.testing.assert_array_equal(data['faces'],env.last_surface.triangles)
   self.assertFalse(json.loads((folder/'metadata.json').read_text())['quality']['checked_metrics_passed'])
   env.native.tick=124;env.replay_command(None);self.assertFalse((out/'episode-000/failure-tick-124').exists());np.testing.assert_array_equal(env.observed.nodes,before)
 def test_accepted_step_does_not_create_failure_evidence(self):
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
   env,out,result=self.make(d,True);self.assertIs(env.replay_command(None),result);self.assertEqual(list((out/'episode-000').iterdir()),[])
if __name__=='__main__':unittest.main()
