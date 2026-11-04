import sys,os,json,time
from pathlib import Path
p=Path(__file__).resolve().parents[1];os.sched_setaffinity(0,[25]);os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');sys.path.insert(0,str(p/'candidate-source'))
import numpy as np
from dataclasses import replace
from endoscopic_tissue.training.mesh_kernels import prepare_soup
from endoscopic_tissue.training.snapshot import decode_python
from endoscopic_tissue.grid import to_genesis,TO_GENESIS
state=decode_python((p.parent/'results/baseline-profile/render-state.bin').read_bytes());original=state['surface'];rng=np.random.default_rng(44)
def reference(s):
 triangles=s.triangles;wall=s.anchors[triangles[:,0],3]>.5;rest=s.anchors[:,:3].reshape(-1,3,3)
 projection=np.abs(np.cross(rest[:,1]-rest[:,0],rest[:,2]-rest[:,0])).argmax(axis=1);pos=to_genesis(s.vertices);ns=s.normals@TO_GENESIS.T;uvaxes=np.array(((1,2),(0,2),(0,1)));result=[]
 for part in (False,True):
  ids=triangles[wall==part].ravel();axes=np.repeat(projection[wall==part],3)
  uv=s.anchors[ids[:,None],uvaxes[axes]]/np.float32(1.5);result.append((pos[ids],ns[ids],uv,ids))
 return result
variants=[original]
for i in range(20):
 anchors=original.anchors.copy();anchors[:,:3]+=rng.uniform(-.02,.02,anchors[:,:3].shape).astype(np.float32)
 if i<3:anchors[:,3]=i%2
 variants.append(replace(original,vertices=original.vertices+rng.normal(0,.001,original.vertices.shape).astype(np.float32),anchors=anchors))
for i,s in enumerate(variants):
 expected=reference(s);actual=prepare_soup(s.vertices,s.normals,s.anchors,np.float32(1.5))
 for part in range(2):
  for k in range(4):assert np.array_equal(expected[part][k],actual[part][k]),(i,part,k,np.max(np.abs(expected[part][k]-actual[part][k])))
  if len(actual[part][0]):assert np.array_equal(actual[2][part],np.stack((expected[part][0].min(axis=0),expected[part][0].max(axis=0))))
timings={}
for name,fn in [('reference',lambda:reference(original)),('fused',lambda:prepare_soup(original.vertices,original.normals,original.anchors,np.float32(1.5)))]:
 values=[]
 for _ in range(40):
  t=time.perf_counter_ns();fn();values.append((time.perf_counter_ns()-t)/1e6)
 timings[name]=dict(mean_ms=float(np.mean(values)),p95_ms=float(np.percentile(values,95)))
r=dict(passed=True,variants=len(variants),all_attributes_bit_exact=True,bounds_bit_exact=True,timings=timings);(p/'results/mesh-equivalence.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
