"""Audit recorded envelope failures; distinguish tissue from nonmaterial support nodes."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from endoscopic_tissue.scenario import SCISSORS_SCENES
from endoscopic_tissue.surface_diagnostics import topology_statistics, load_state

@threadpool_limits.wrap(limits=1,user_api='blas')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case_dir',type=Path)
    a=p.parse_args()
    m=json.loads((a.case_dir/'manifest.json').read_text(encoding='utf8'))
    grid=SCISSORS_SCENES[m['scene']].build_grid(m['settings']['density_kg_m3'])
    d=np.load(a.case_dir/'failure-state.npz')
    obs,surface=load_state(a.case_dir/'failure-state.npz')
    displacement=np.linalg.norm(d['nodes_cm']-d['rest_cm'],axis=1)*10
    material=d['inside'].astype(bool)
    shell=grid.active.astype(bool)&~material
    out=dict(event=json.loads((a.case_dir/'failure-state.json').read_text()),
        material_nodes_over_60mm=int(np.count_nonzero(material&(displacement>60))),
        support_nodes_over_60mm=int(np.count_nonzero(shell&(displacement>60))),
        max_material_displacement_mm=float(displacement[material].max()),
        max_support_displacement_mm=float(displacement[shell].max()),
        surface_topology=topology_statistics(surface),
        note='This is the rejected next step, not included in trajectory or replay.')
    (a.case_dir/'failure-audit.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(out))
if __name__=='__main__':main()
