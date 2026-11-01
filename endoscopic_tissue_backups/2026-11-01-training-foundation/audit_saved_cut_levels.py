"""Read the original diagnostic snapshots without changing the simulation."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT/'endoscopic_tissue/outputs/interaction-fixes-20261031/r13-wall-rim-2'
rows = []
for path in sorted(CASE.glob('cut-*.cut'), key=lambda p: int(p.stem.split('-')[1])):
    raw = path.read_bytes()
    h = np.frombuffer(raw, '<i4', count=10)
    assert h[0] == 0x43555431 and h[9] == 18
    n, e, axis = int(h[1]), int(h[2]), int(h[6])
    assert n == int(np.prod(h[3:6]+1))
    offset = 40 + n*(3*4+3*4+6*4+4)
    encoded = np.frombuffer(raw, '<u4', count=n, offset=offset)
    present = encoded != 0x7F7F7F7F
    levels = encoded[present].view('<f4')
    sweep = np.frombuffer(raw, '<f4', count=54, offset=len(raw)-54*4).reshape(18, 3)
    cross = np.cross(sweep[11], sweep[12])
    rows.append(dict(step=int(path.stem.split('-')[1]), sha256=hashlib.sha256(raw).hexdigest(),
        node_count=n, grid_edges=e, column_axis=axis, phase=int(h[7]), entry_state=int(h[8]),
        set_columns=int(present.sum()), unique_levels=int(len(np.unique(levels))),
        min_rest_level_mm=float(levels.min()*10) if len(levels) else None,
        max_rest_level_mm=float(levels.max()*10) if len(levels) else None,
        rest_level_range_mm=float(np.ptp(levels)*10) if len(levels) else None,
        plane_normal=sweep[8].tolist(), plane_point_cm=sweep[9].tolist(),
        plane_normal_vs_actual_frame_error=float(np.linalg.norm(sweep[8]-cross))))
output = dict(source=str(CASE), snapshots=rows,
    interpretation='Different material-coordinate cut levels are retained across columns; no universal constant cut height.',
    limitations=['R13 diagnostics, not current default binary acceptance',
                 'Stored levels include propagated skin levels and end-link adjustment',
                 'Post-frame snapshots do not prove pointwise two-edge contact at each fracture substep',
                 'Fixed column-axis representation and straight fitted blade edges remain approximations'])
target = Path(__file__).parent/'saved-cut-level-audit.json'
target.write_text(json.dumps(output, indent=2), encoding='utf8')
print(json.dumps(rows, indent=2))
