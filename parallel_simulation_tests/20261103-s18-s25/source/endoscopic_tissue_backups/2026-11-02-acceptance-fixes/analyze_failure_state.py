import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

folder = Path(sys.argv[1])
a = np.load(folder / 'last-completed.npz')
b = np.load(folder / 'first-numerical-failure.npz')
delta = np.linalg.norm(b['nodes'] - a['nodes'], axis=1) * 10
distance = np.linalg.norm(b['nodes'] - b['rest'], axis=1) * 10
owner, slot = np.where(b['neighbors'][:, 1::2] >= 0)
target = b['neighbors'][owner, 1 + slot * 2]
n = len(b['nodes'])
_, labels = connected_components(coo_matrix((np.ones(len(owner)), (owner, target)), shape=(n, n)), directed=False)
counts = np.bincount(labels)
top = []
for i in np.argsort(distance * (b['active'] != 0))[-12:][::-1]:
    top.append(dict(node=int(i), material=int(b['material'][i]), active=int(b['active'][i]),
                    rest_cm=b['rest'][i].tolist(), before_cm=a['nodes'][i].tolist(),
                    after_cm=b['nodes'][i].tolist(), displacement_mm=float(distance[i]),
                    move_mm=float(delta[i]), live_neighbors=b['neighbors'][i].tolist(),
                    component_size=int(counts[labels[i]])))
result = dict(before_tick=int(a['tick']), after_tick=int(b['tick']),
              max_material_displacement_mm=float(distance[b['material'] != 0].max()),
              max_active_displacement_mm=float(distance[b['active'] != 0].max()), top=top)
(folder / 'failure-node-analysis.json').write_text(json.dumps(result, indent=2), encoding='utf8')
print(json.dumps(result, indent=2))
