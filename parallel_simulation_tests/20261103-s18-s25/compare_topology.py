import json,time
from pathlib import Path
import numpy as np,trimesh
from endoscopic_tissue.native_backend import Surface
from endoscopic_tissue.surface_diagnostics import topology_statistics
from endoscopic_tissue.training.topology_fast import topology_statistics_fast
rng=np.random.default_rng(7);cases=[];elapsed={'original':[],'candidate':[]}
for i in range(64):
 mesh=trimesh.creation.icosphere(subdivisions=3 if i%2 else 4);v=mesh.vertices.astype(np.float32);f=mesh.faces.copy()
 mode=i%8
 if mode==1:f=f[1:]
 if mode==2:f=np.concatenate((f,f[:3]))
 if mode==3:f[0]=f[0,::-1]
 if mode==4:v[0]=v[1]
 if mode==5:v += rng.normal(0,1e-6,v.shape).astype(np.float32)
 if mode==6:f=np.vstack((f,np.array([[0,0,1]])))
 if mode==7:v=np.vstack((v,np.array([[0,0,0]],np.float32)))
 # Real native surfaces are a triangle soup.
 if i%3==0:v=v[f].reshape(-1,3);f=np.arange(len(v)).reshape(-1,3)
 anchors=np.zeros((len(v),4));anchors[:,3]=(v[:,0]>0)
 s=Surface(v,np.zeros_like(v),f,anchors);out=[]
 for name,fn in [('original',topology_statistics),('candidate',topology_statistics_fast)]:
  began=time.perf_counter();out.append(fn(s));elapsed[name].append((time.perf_counter()-began)*1000)
 cases.append({'case':i,'equal':out[0]==out[1]})
 if out[0]!=out[1]:cases[-1]['values']=out
result={'scope':'CPU exact predicate comparison including corrupted meshes; no GPU/FPS claim','cases':cases,'passed':all(x['equal'] for x in cases),
 'mean_ms':{k:float(np.mean(v)) for k,v in elapsed.items()}}
Path('../results/topology-equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
assert result['passed']
