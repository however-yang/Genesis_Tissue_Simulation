import os,sys,json
from pathlib import Path
os.sched_setaffinity(0,[30,31]);os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');p=Path(__file__).resolve().parents[1];sys.path.insert(0,str(p/'python-gpu-check'))
import numpy as np
from endoscopic_tissue.training.snapshot import decode_python
out=p/'results/topology-mre';out.mkdir(exist_ok=True);records=[]
for item in json.loads((p/'results/failure-localization.json').read_text()):
 data=decode_python((p/'results'/item['run']/'failure-render-state.bin').read_bytes());s=data['surface'];s=s if hasattr(s,'vertices') else type('Surface',(),s)
 for i,edge in enumerate(item['nonmanifold_edges']):
  tri=np.asarray(edge['triangles'],np.int32);original=s.triangles[tri];vertices=s.vertices[original].reshape(-1,3);indices=np.arange(len(vertices),dtype=np.int32).reshape(-1,3);name=item['run']+'-edge'+str(i);path=out/(name+'.npz');np.savez_compressed(path,vertices_native_cm=vertices,triangles=indices,original_triangles=tri)
  with (out/(name+'.obj')).open('w') as f:
   f.write('# Four triangles incident to one exact edge. Coordinates: native cm.\n')
   for v in vertices:f.write('v '+' '.join(format(float(x),'.9g') for x in v)+'\n')
   for t in indices:f.write('f '+' '.join(str(int(x)+1) for x in t)+'\n')
  _,inverse=np.unique(vertices,axis=0,return_inverse=True);counts={}
  for t in inverse[indices]:
   for k in range(3):
    key=tuple(sorted((int(t[k]),int(t[(k+1)%3]))));counts[key]=counts.get(key,0)+1
  maximum=max(counts.values());assert maximum==4
  records.append(dict(run=item['run'],frame=item['frame_id'],npz=str(path.relative_to(p)),maximum_exact_edge_incidence=maximum,interpretation='static failure reproduction, not an approved surface; no rounding or welding tolerance required'))
(out/'manifest.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
