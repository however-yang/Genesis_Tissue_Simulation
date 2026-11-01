"""Bind an explicit screening goal in material coordinates, before any native run."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from endoscopic_tissue.scenario import SCISSORS_SCENES, scissors_settings
from endoscopic_tissue.training.provenance import json_hash, sha256
from endoscopic_tissue.training.tasks import TaskGoal, material_reference
from endoscopic_tissue.training.experiment import group_key

P = Path(__file__).resolve().parent
DLL = ROOT/'endoscopic_tissue/native/bin-contact-groups31/LiverCudaSim.dll'
scene = SCISSORS_SCENES['wall-traction']
grid = scene.build_grid(scissors_settings().density_kg_m3)
axis = scene.plane_axis
a = np.flatnonzero((grid.inside != 0) & (grid.neighbors[:,2*axis+1] >= 0))
b = grid.neighbors[a,2*axis+1]
p, q = grid.positions[a], grid.positions[b]
keep = ((grid.inside[b] != 0) & (p[:,axis] <= scene.plane) & (q[:,axis] > scene.plane)
        & np.all((p >= np.array(scene.cut_low)-.11) & (p <= np.array(scene.cut_high)+.11),axis=1))
edges = np.stack((a[keep],b[keep]),axis=1).tolist()
if not edges: raise RuntimeError('No material edges in declared goal corridor')
weights = dict(progress=1., outside_damage=1., grasp=0., slip_m=0., time_s=0., action=0.)
reference = material_reference(dict(rest_positions_m=grid.positions*.01,initial_neighbors=grid.neighbors,
                                    material_mask=grid.inside,active_mask=grid.active))
goal = TaskGoal(kind='incision',target_edges=tuple(map(tuple,edges)),required_progress=.25,required_hold_s=.1,
                material_reference_sha256=reference,
                weights=weights,label='screening_wall_material_corridor_unvalidated').to_dict()
spec = dict(schema='tissue_experiment_v1',purpose='collection',dll=str(DLL),dll_sha256=sha256(DLL),
            episode_timeout_s=600,min_free_disk_bytes=1024**3,episodes=[])
for i,policy in enumerate([{'kind':'hold'}, {'kind':'smooth_random'},
                           {'kind':'smooth_random','close_scissors':True},
                           {'kind':'force_feedback','close_scissors':True}]):
    spec['episodes'].append(dict(id=f'pilot-{i:02}',scene='wall-traction',preparation='cut_ready',seed=3100+i,
                                steps=90,cutting_enabled=i>=2,policy=policy,goal=goal))
(P/'policy-pilot.json').write_text(json.dumps(spec,indent=2)+'\n')
evaluate = {**spec,'purpose':'evaluation','excluded_training_groups':[group_key(c) for c in spec['episodes']],
            'episodes':[{**c,'id':c['id'].replace('pilot','eval'),'seed':c['seed']+100} for c in spec['episodes']]}
evaluate['frozen_sha256'] = json_hash(evaluate)
(P/'frozen-evaluation.json').write_text(json.dumps(evaluate,indent=2)+'\n')
vision = {**spec,'episodes':[dict(id='rgbd-wall',scene='wall-traction',preparation='held',seed=3500,
                                 steps=6,cutting_enabled=False,policy={'kind':'hold'},
                                 vision={'width':640,'height':480,'every_steps':1})]}
(P/'vision-pilot.json').write_text(json.dumps(vision,indent=2)+'\n')
friction = {**spec,'episodes':[dict(id='friction-preparation',scene='wall-traction',preparation='held',
                                   contact_model=1,seed=3600,steps=30,cutting_enabled=False,
                                   policy={'kind':'force_feedback'})]}
(P/'friction-pilot.json').write_text(json.dumps(friction,indent=2)+'\n')
print('Explicit material target edges:',len(edges))
