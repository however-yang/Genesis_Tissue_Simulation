"""CPU-only static reproduction of the rejected four-triangle edge."""
import argparse,json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1];a=argparse.ArgumentParser();a.add_argument('--case',type=int,default=0);args=a.parse_args();m=json.loads((p/'results/topology-mre/manifest.json').read_text());x=m[args.case];data=np.load(p/x['npz']);vertices=data['vertices_native_cm'];tri=data['triangles'];_,inverse=np.unique(vertices,axis=0,return_inverse=True);counts={}
for t in inverse[tri]:
 for k in range(3):
  edge=tuple(sorted((int(t[k]),int(t[(k+1)%3]))));counts[edge]=counts.get(edge,0)+1
bad=[dict(edge=list(k),incident_triangles=v) for k,v in counts.items() if v>2];assert any(x['incident_triangles']==4 for x in bad)
print(json.dumps(dict(failure_reproduced=True,physically_approved=False,exact_coordinates_no_rounding=True,nonmanifold_edges=bad),indent=2))
