import json,sys,time
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'candidate-source'))
from endoscopic_tissue.training.quality_kernels import node_extrema,connected_material
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def reference(nodes,previous,rest,active,scale):
 used=active!=0
 return (bool(np.isfinite(nodes).all()),float((np.linalg.norm(nodes[used]-rest[used],axis=1)*.01).max(initial=0)),float((np.linalg.norm(nodes[used]-previous[used],axis=1)*scale).max(initial=0)) if len(previous) else 0.)
def old_components(neighbors,material):
 ids=np.flatnonzero(material);mapping=np.full(len(material),-1,np.int32);mapping[ids]=np.arange(len(ids))
 target=neighbors[ids,1::2];owners,slot=np.where(target>=0);target=target[owners,slot];keep=material[target];owners,target=owners[keep],mapping[target[keep]]
 graph=coo_matrix((np.ones(len(owners),np.uint8),(owners,target)),shape=(len(ids),len(ids))).tocsr()
 return int(connected_components(graph,directed=False,return_labels=False))
rng=np.random.default_rng(154);cases=0
for n in (0,1,7,100,20000):
 for j in range(20):
  rest=rng.normal(size=(n,3)).astype(np.float32);nodes=rest+rng.normal(0,.1,(n,3)).astype(np.float32);previous=rest.copy();active=rng.integers(0,2,n,dtype=np.int32)
  if n and j<3:nodes[-1,j]=[np.nan,np.inf,-np.inf][j]
  scale=.01/(1/30)
  a=reference(nodes,previous,rest,active,scale);b=node_extrema(nodes,previous,rest,active,np.float32(scale))
  assert np.array_equal(np.asarray(a),np.asarray(b),equal_nan=True),(n,j,a,b)
  cases+=1
components=0
for n in (0,1,7,100,10000):
 for j in range(10):
  material=rng.integers(0,2,n,dtype=np.int32).astype(bool);neighbors=rng.integers(-1,max(0,n),size=(n,6),dtype=np.int32)
  a=old_components(neighbors,material);b=connected_material(neighbors,material)
  assert a==b,(n,j,a,b);components+=1
from endoscopic_tissue.training.snapshot import decode_python
state=decode_python((r/'results/baseline-profile/render-state.bin').read_bytes());nodes=state['observed'].nodes
rest=nodes.copy();previous=(nodes+.001).astype(np.float32);active=np.ones(len(nodes),np.int32)
expected=reference(nodes,previous,rest,active,.3);actual=node_extrema(nodes,previous,rest,active,np.float32(.3));assert np.array_equal(expected,actual)
timings={}
for name,fn in [('numpy',lambda:reference(nodes,previous,rest,active,.3)),('fused',lambda:node_extrema(nodes,previous,rest,active,np.float32(.3)))]:
 values=[]
 for _ in range(50):
  start=time.perf_counter_ns();fn();values.append((time.perf_counter_ns()-start)/1e6)
 timings[name]={'mean_ms':float(np.mean(values)),'p50_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95))}
result=dict(passed=True,node_cases=cases,component_cases=components,real_nodes=len(nodes),exact_node_results=True,timings=timings)
(r/'results/quality-equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
