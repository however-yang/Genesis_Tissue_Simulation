import sys,json,pickle,zlib,struct
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;R=W.parents[1];sys.path[:0]=[str(R),str(W)]
from endoscopic_tissue.instrument_geometry import load_scissors
from inspect_cut_case import swept
D=R/'endoscopic_tissue/outputs/grasp-cut-validation-20261031/wall-repeat-recorded';target='incise_3';obs=None
with (D/'replay.bin').open('rb') as f:
 header=pickle.load(f)
 while h:=f.read(4):
  n=struct.unpack('<I',h)[0];data=pickle.loads(zlib.decompress(f.read(n)))
  if data[1]==target:obs=data[3]
  elif obs is not None:break
assert obs is not None
s=obs.scissors;geo=load_scissors();d=np.load(D/(target+'.npz'));audit=json.loads((D/'wall-bridge-audit.json').read_text());path=next(r['path_node_ids'] for r in audit['stages'] if r['stage']==target);rows=[]
u=s.axis/np.linalg.norm(s.axis);v=s.normal/np.linalg.norm(s.normal);normal=np.cross(u,v);point=s.edges.reshape(-1,3)[2];L=geo.jaws.pad_length if hasattr(geo.jaws,'pad_length') else None
# Native jawLen is configured from source edge radius; recover angle from the
# known closed edge and its actual solved world endpoint, independent of jawLen.
a=geo.edges.reshape(2,2,3); world=s.edges.reshape(2,2,3);q=world[0,0]-s.hinge;now=np.arctan2(q@v,q@u)-np.arctan2(a[0,0,1],a[0,0,0]);op=np.pi/2
for i,j in zip(path,path[1:]):
 p0,p1=d['nodes_cm'][[i,j]];da=(p0-point)@normal;db=(p1-point)@normal;row=dict(ids=[i,j],signed_mm=[float(da*10),float(db*10)],plane_crossing=bool((da>0)!=(db>0)))
 if row['plane_crossing']:
  t=da/(da-db);p=p0+t*(p1-p0)-s.hinge;x=float(p@u);y=float(p@v);row.update(in_plane_mm=[x*10,y*10],radial_mm=float(np.hypot(x,y)*10),covered_even_with_90deg_open=swept(x,y,*a[0],1,now,op,.05) or swept(x,y,*a[1],-1,now,op,.05))
 rows.append(row)
out=dict(case=str(D),stage=target,phase=s.phase,hinge_cm=s.hinge.tolist(),closed_edges_cm=a.tolist(),now_angle_rad=float(now),path=rows,note='Final-state coverage only. This cannot prove whether a link was covered at earlier substeps. The 90 degree opening deliberately overbounds the real sweep; radial exclusion is independent of opening.')
(W/'old-wall-retained-link-coverage.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
