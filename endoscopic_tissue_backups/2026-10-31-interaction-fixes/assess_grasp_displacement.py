import sys,json
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;R=W.parents[1];sys.path.insert(0,str(R))
from endoscopic_tissue.native_backend import Settings
from endoscopic_tissue.scenario import SCISSORS_SCENES,Sequence
from endoscopic_tissue.surface_diagnostics import load_state
def assess(folder):
 r=json.loads((folder/'report.json').read_text());settings=Settings(**r['metadata']['settings']);scene=SCISSORS_SCENES[r['metadata']['scene']]
 settled=np.load(folder/'settle.npz')['nodes_cm'];bite=np.array([],dtype=int);rows=[]
 for stage in r['stages']:
  name=stage['stage'];file=folder/(name+'.npz')
  obs,_=load_state(file);d=np.load(file);probe=Sequence(obs,settings,scene,d['rest_cm'],d['inside']);mask,_=probe.between_pads(obs.nodes,obs);ids=np.flatnonzero(mask & d['inside'].astype(bool))
  if name in ('close','reclose'):bite=ids.copy()
  def lift(index):return float(np.median(obs.nodes[index,1]-settled[index,1])*10) if len(index) else None
  rows.append(dict(stage=name,current_enclosed_material_nodes=len(ids),initial_bite_nodes=len(bite),initial_bite_lift_from_settled_mm=lift(bite),current_enclosed_lift_from_settled_mm=lift(ids),closed_reference_lift_mm=stage['end']['bite_lift_mm'],jaw_load_n=stage['end']['jaw_load_n'],anchors=stage['end']['grip_points']))
 result=dict(case=r['metadata']['case'],reference='Same material node positions after initial 30-step gravity settling, before tool contact. Current enclosed set may change as friction permits sliding; region membership is not a force measurement.',stages=rows)
 (folder/'grasp-displacement-audit.json').write_text(json.dumps(result,indent=2));print(folder.name,json.dumps([x for x in rows if x['stage'] in ('sustain','rehold','relax')]),flush=True)
if __name__=='__main__':
 for folder in map(Path,sys.argv[1:]):assess(folder)
