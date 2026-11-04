import json,os,sys,multiprocessing as mp,struct
from pathlib import Path
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'candidate-v5-source'))
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from endoscopic_tissue.training.shared_images import ImageBuffer,ImageWriter,ImageReader

class Connection:
 def __init__(self):self.messages=[]
 def send(self,x):self.messages.append(x)
 def recv(self):return self.messages.pop(0)
 def close(self):pass

def message(request,value=3):return dict(request=request,ok=True,observation=dict(rgb=np.full((8,12,3),value,np.uint8),depth_m=np.full((8,12),.1*value,np.float32),depth_valid=np.full((8,12),True,bool),step_id=request))
def writer_process(c,spec):
 writer=ImageWriter(c,spec)
 try:
  writer.send(message(1));writer.recv();writer.send(message(2,9))
 finally:writer.close()

def main():
 tests=[];owner=ImageBuffer(4096);name=owner.memory.name;connection=Connection();writer=ImageWriter(connection,owner.spec);reader=ImageReader(connection,owner)
 try:
  source=message(1);writer.send(source);reply=reader.recv()
  for k in ['rgb','depth_m','depth_valid']:assert np.array_equal(source['observation'][k],reply['observation'][k]);assert reply['observation'][k].flags.owndata
  tests.append('bit_exact_owned_arrays')
  old=reply['observation']['rgb'];old[:]=41;writer.send(message(2,9));reply2=reader.recv();assert (old==41).all() and (reply2['observation']['rgb']==9).all();tests.append('retained_observation_survives_next_frame')
  writer.send(message(3));struct.pack_into('<Q',owner.memory.buf,0,4)
  try:reader.recv();raise AssertionError('Stale frame accepted')
  except ValueError:pass
  tests.append('generation_rejected')
  wrong=message(5);wrong['observation']['depth_m']=wrong['observation']['depth_m'].astype(np.float64)
  try:writer.send(wrong);raise AssertionError('Precision silently changed')
  except ValueError:pass
  tests.append('wrong_dtype_rejected')
  large=message(6);large['observation']['rgb']=np.zeros((40,40,3),np.uint8)
  try:writer.send(large);raise AssertionError('Capacity not enforced')
  except MemoryError:pass
  tests.append('byte_budget_enforced')
 finally:writer.close();owner.close()
 try:leak=SharedMemory(name=name);leak.close();raise AssertionError('Shared memory leaked')
 except FileNotFoundError:pass
 tests.append('buffer_unlinked')
 owner=ImageBuffer(4096);ctx=mp.get_context('spawn');parent,child=ctx.Pipe();proc=ctx.Process(target=writer_process,args=(child,owner.spec))
 try:
  proc.start();child.close();reader=ImageReader(parent,owner);assert reader.poll(30);first=reader.recv();owner.memory.unlink();reader.send({'ack':1});assert reader.poll(30);second=reader.recv()
  assert (first['observation']['rgb']==3).all() and (second['observation']['rgb']==9).all();proc.join(30);assert proc.exitcode==0;tests.append('real_spawn_pipe_publication_and_ownership')
 finally:
  parent.close()
  if proc.is_alive():proc.terminate();proc.join()
  owner.close()
 result=dict(passed=True,checks=tests);(r/'results/shared-images-regression.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
