"""Full wall fixture plus read-only tracking of actual grid links and solved blades."""
import sys,json,ctypes as ct
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W));import validate_strict as v
out=Path(sys.argv[sys.argv.index('--output')+1]);rows=[];poses=[];links=[];meta=[];ids=None;grid=None;snapped=False
import interaction_fixtures
interaction_fixtures.install()
rim_saved=False
original=v.NativeTissue.step
def step(self,command):
 global ids,grid,snapped,rim_saved
 obs=original(self,command);grid=self.grid
 if not rim_saved:
  fn=getattr(self.dll,'LCS_SaveRimTrace',None)
  if fn:
   fn.argtypes=[ct.c_char_p];fn.restype=ct.c_int;code=fn(str(out/('rim-step-'+str(len(rows)+1)+'.bin')).encode())
   if code<0:raise RuntimeError(('rim snapshot',code))
   if code==0:
    reset=getattr(self.dll,'LCS_ResetRimTrace',None)
    if reset:
     reset.argtypes=[];reset.restype=ct.c_int;assert reset()==0
    rim_saved=reset is None;print('RIM_TRACE',len(rows)+1,flush=True)
 if ids is None:
  p=grid.positions;ids=np.flatnonzero((grid.active!=0)&(p[:,0]>=-.1)&(p[:,0]<=2.7)&(p[:,1]>=.55)&(p[:,1]<=1.6)&(p[:,2]>=1.4)&(p[:,2]<=2.2))
 rows.append(obs.nodes[ids].copy());links.append(obs.neighbors[ids].copy());s=obs.scissors;poses.append(np.concatenate([s.hinge,s.axis,s.normal,s.edges.ravel()]))
 displacement=np.linalg.norm(obs.nodes-grid.positions,axis=1)*10;support=(grid.active!=0)&(grid.inside==0);i=int(np.argmax(np.where(support,displacement,-1)))
 meta.append(dict(step=len(rows),cutting=command.scissors_cutting,phase=s.phase,fractures=s.loads.fractures,gap_cm=s.gap,left_n=s.loads.left_n,right_n=s.loads.right_n,max_support_node=i,max_support_displacement_mm=float(displacement[i]),max_material_displacement_mm=float(displacement[grid.inside!=0].max())))
 if displacement[i]>30 and not snapped:
  snapped=True;np.savez_compressed(out/('node-spike-'+str(len(rows))+'.npz'),nodes_cm=obs.nodes,neighbors=obs.neighbors,rest_cm=grid.positions,inside=grid.inside,active=grid.active)
  for name,suffix in [('LCS_SaveFragmentCase','.bin'),('LCS_SavePairedConstraints','.fmc'),('LCS_SaveSolverCase','.sol'),('LCS_SaveCutCase','.cut')]:
   fn=getattr(self.dll,name,None)
   if fn:fn.argtypes=[ct.c_char_p];fn.restype=ct.c_int;assert fn(str(out/('node-spike-'+str(len(rows))+suffix)).encode())==0
 if command.scissors_cutting and (len(rows)%30==0 or len(meta)==1 or meta[-1]['fractures']!=meta[-2]['fractures']):
  fn=getattr(self.dll,'LCS_SaveCutCase',None)
  if fn:
   fn.argtypes=[ct.c_char_p];fn.restype=ct.c_int;code=fn(str(out/('cut-'+str(len(rows))+'.cut')).encode());meta[-1]['cut_snapshot_status']=code
 return obs
v.NativeTissue.step=step
try:v.main()
finally:
 if rows:
  np.savez_compressed(out/'wall-track.npz',node_ids=ids,rest_cm=grid.positions[ids],inside=grid.inside[ids],nodes_cm=np.array(rows),neighbors=np.array(links),scissors_pose=np.array(poses))
  (out/'wall-track.json').write_text(json.dumps(meta,indent=2))

  for filename in ['manifest.json','report.json']:
   path=out/filename
   if path.exists():
    data=json.loads(path.read_text());target=data.get('metadata',data);target['note']=target.get('note','')+' Pipeline timing includes extra read-only wall tracking and snapshot overhead; use standalone timing for comparisons.';path.write_text(json.dumps(data,indent=2))
