"""Describe the largest observed local difference without asserting a new tolerance."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    states=json.loads((ROOT/'state-comparisons.json').read_text())
    worst=max((r for r in states if 'max_material_node_delta_mm' in r),key=lambda r:r['max_material_node_delta_mm'])
    ref=ROOT/'runs/plain-1-r0/worker-000/episode-000/final-state.npz'
    with np.load(ref) as a,np.load(ROOT/worst['path']) as b:
        ids=np.flatnonzero(a['inside']);delta=np.linalg.norm((a['nodes_cm']-b['nodes_cm'])[ids]*10,axis=1)
        top=np.argsort(delta)[-10:][::-1]
        record={'path':worst['path'],'material_nodes':len(ids),'rms_delta_mm':float(np.sqrt(np.mean(delta**2))),
            'max_delta_mm':float(delta.max()),'node_counts_above_mm':{str(t):int((delta>t).sum()) for t in (.1,.5,1,2,4)},
            'top_differences':[{'node_id':int(ids[i]),'delta_mm':float(delta[i]),'rest_xyz_mm':(a['rest_cm'][ids[i]]*10).tolist(),
                               'incident_neighbor_entries_changed':int((a['neighbors'][ids[i]]!=b['neighbors'][ids[i]]).sum())} for i in top],
            'interpretation':'Descriptive localization only. Small global RMS does not certify local physical equivalence; topology and local force/deformation regressions remain needed.'}
    (ROOT/'largest-local-difference.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k!='top_differences'},indent=2))
if __name__=='__main__':main()
