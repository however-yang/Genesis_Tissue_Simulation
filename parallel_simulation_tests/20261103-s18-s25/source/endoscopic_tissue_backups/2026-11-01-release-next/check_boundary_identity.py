"""Show actual cut changes that a surface-only mask would conceal from caches."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue.grid import EDGE_CORNERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    with np.load(args.directory/'reconstruction-grid.npz') as g:
        flags = g['inside'][g['corners']] != 0
    air = np.zeros(len(flags), np.uint32)
    for bit, (a, b) in enumerate(EDGE_CORNERS):
        air |= (~(flags[:, a] & flags[:, b])).astype(np.uint32) << bit
    masks = []
    for step in (600, 840):
        with np.load(args.directory/f'native-topology-{step:04d}.npz') as d:
            masks.append(d['voxel_masks'] & 4095)
    effective = [np.where(m != 0, m | air, 0) for m in masks]
    changed = (masks[0] != masks[1]) & (masks[0] != 0)
    hidden = np.flatnonzero(changed & (effective[0] == effective[1]))
    result = dict(source='Frozen actual kidney-reference cuts at steps 600 and 840',
        changed_previously_cut_voxels=int(changed.sum()),
        changed_cuts_hidden_by_surface_only_key=len(hidden),
        examples=[dict(voxel=int(v), actual_before=int(masks[0][v]), actual_after=int(masks[1][v]),
                       shared_boundary_key=int(effective[0][v])) for v in hidden[:12]],
        correction='Use original actual cut mask for contact and grip identity; use boundary mask only for component interpolation.',
        geometric_mask_change_not_a_cut=False)
    assert len(hidden) > 0, 'This captured transition does not exercise the alias; do not claim coverage.'
    (args.directory/'boundary-identity-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
