import json,sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
W=Path(__file__).resolve().parent;R=W.parents[1];sys.path[:0]=[str(W),str(R)];from validate_strict import fixture
_,s,settings,_,_=fixture('wall-reference');g=s.build_grid(settings.density_kg_m3);z=np.load(R/'endoscopic_tissue/outputs/grasp-cut-validation-20261031/wall-repeat-2/failure-state.npz');nb=z['neighbors'];ids=np.flatnonzero(g.active);a=np.repeat(ids,6);b=nb[ids].ravel();k=b>=0;k&=g.active[np.maximum(b,0)]!=0;labels=connected_components(coo_matrix((np.ones(k.sum(),np.int8),(a[k],b[k])),shape=(len(nb),)*2),directed=False)[1]
i=65439;members=(g.active!=0)&(labels==labels[i]);out=dict(node=i,component_active=int(members.sum()),component_material=int(np.sum(members&(g.inside!=0))),component_pinned=int(np.sum(members&(g.pinned!=0))),grasp_points=int(np.sum(z['grip_flags'][members]!=0)),max_displacement_mm=float(np.max(np.linalg.norm(z['nodes_cm'][members]-g.positions[members],axis=1))*10));print(json.dumps(out,indent=2));(W/'old-wall-component.json').write_text(json.dumps(out,indent=2))
