"""Audit closed vertex neighborhoods in saved stages and every free-release frame.

Captures contain expanded render vertices, not native feature IDs. This uses the
same 1e-5 cm positional merge as the independent containment audit and labels that
limitation explicitly; the frozen-input audit separately verifies native IDs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from audit_frozen_boundary import audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory/'report.json').read_text())
    assert report['completed']
    free = json.loads((args.directory/'release-overlap-audit.json').read_text())
    states = [(s['stage'], s['end'], args.directory/(s['stage']+'.npz')) for s in report['stages']]
    states += [('free_release', s['step'], args.directory/f"state-{s['step']:04d}.npz") for s in free]
    rows = []
    for stage, step, path in states:
        with np.load(path) as data:
            mesh = trimesh.Trimesh(data['vertices_cm'], data['faces'], process=False)
        mesh.merge_vertices(digits_vertex=5)
        keys = np.arange(len(mesh.vertices), dtype=np.uint32) | np.uint32(0x80000000)
        result = audit(mesh.vertices, mesh.faces, keys)
        rows.append(dict(stage=stage, step=step, **result))
    summary = dict(states=len(rows), all_closed=all(r['watertight'] for r in rows),
                   all_winding_consistent=all(r['winding_consistent'] for r in rows),
                   states_with_invalid_links=sum(bool(r['vertex_link_failures']) for r in rows),
                   invalid_links_total=sum(len(r['vertex_link_failures']) for r in rows))
    output = dict(method='position-welded render mesh at 1e-5 cm; not native feature identity',
                  summary=summary, rows=rows)
    (args.directory/'vertex-link-audit.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
