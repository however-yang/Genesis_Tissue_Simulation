"""Independent float64 oracle on snapshots of the actual frozen GPU hierarchy."""
from pathlib import Path
import ctypes as ct
import hashlib
import json
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue import native_backend, run


def verify_snapshot(path):
    with path.open('rb') as file:
        def get(dtype, count):
            value = np.fromfile(file, dtype=dtype, count=count)
            assert len(value) == count, path
            return value
        magic, n, top, coarse = get('<i4', 4)
        assert magic == 0x414d4731 and 0 < n < 10_000_000 and 0 < top <= 8
        mask = (get('<u2', n) >> 7 & 1).astype(bool)
        h = get('<f4', 6*n).reshape(n, 6).astype(np.float64)
        parents, inverses = [None], [None]
        children = n
        for level in range(1, top + 1):
            count, size = get('<i4', 2)
            assert size == children
            parent = get('<i4', size)
            assert np.all((parent >= -1) & (parent < count))
            parents.append(parent)
            inverses.append(get('<f4', 9*count).reshape(count, 3, 3).astype(np.float64))
            children = count
        assert coarse == 3*children
        dense = get('<f4', coarse*coarse).reshape(coarse, coarse).astype(np.float64)
        inputs, outputs = [], []
        for q in range(2):
            inputs.append(get('<f4', 3*n).reshape(n, 3).astype(np.float64))
            outputs.append(get('<f4', 3*n).reshape(n, 3).astype(np.float64))
        assert not file.read(1)
    assert all(np.isfinite(x).all() for x in [h, dense, *inverses[1:], *inputs, *outputs])
    assert all(np.all(x[~mask] == 0) for x in inputs + outputs)
    # Independent cofactor inverse of the unchanged fine diagonal blocks.
    xx, yy, zz, xy, xz, yz = h.T
    a, b, c = yy*zz-yz*yz, xz*yz-xy*zz, xy*yz-xz*yy
    d, e, f = xx*zz-xz*xz, xy*xz-xx*yz, xx*yy-xy*xy
    det = xx*a+xy*b+xz*c
    valid = mask & (det > 1e-25)
    errors = []
    for r, gpu in zip(inputs, outputs):
        restricted = [r]
        for level in range(1, top+1):
            parent = parents[level]
            keep = parent >= 0
            size = len(inverses[level])
            restricted.append(np.stack([np.bincount(parent[keep], weights=restricted[-1][keep, axis], minlength=size)
                                        for axis in range(3)], axis=1))
        correction = (dense @ restricted[top].ravel()).reshape(-1, 3)
        for level in range(top-1, 0, -1):
            local = np.einsum('nij,nj->ni', inverses[level], restricted[level]) * .7
            parent = parents[level+1]
            keep = parent >= 0
            local[keep] += correction[parent[keep]]
            correction = local
        reference = np.zeros_like(r)
        reference[:, 0] = a*r[:, 0]+b*r[:, 1]+c*r[:, 2]
        reference[:, 1] = b*r[:, 0]+d*r[:, 1]+e*r[:, 2]
        reference[:, 2] = c*r[:, 0]+e*r[:, 1]+f*r[:, 2]
        reference[valid] *= (.7/det[valid])[:, None]
        reference[~valid] = 0
        keep = mask & (parents[1] >= 0)
        reference[keep] += correction[parents[1][keep]]
        error = float(np.linalg.norm(gpu-reference)/max(np.linalg.norm(reference), 1e-30))
        assert error < 1e-4, (path, error)
        errors.append(error)
    quadratic = [float(np.sum(r*z)) for r, z in zip(inputs, outputs)]
    assert all(value > 0 for value in quadratic), quadratic
    symmetry = abs(float(np.sum(inputs[0]*outputs[1]) - np.sum(inputs[1]*outputs[0]))) / np.sqrt(quadratic[0]*quadratic[1])
    assert symmetry < 1e-5, (path, symmetry)
    return dict(nodes=int(n), cg_nodes=int(mask.sum()), levels=int(top),
                relative_l2_error=errors, normalized_symmetry_error=float(symmetry),
                quadratic_forms=quadratic, fixed_output_nonzero=0,
                snapshot=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    output = Path(sys.argv[1]).resolve()
    opened, checked = run.Flow.open, native_backend.NativeTissue.check
    rows, ready = [], False
    sample_steps = {120, 180, 240, 450, 600, 700, 750, 800, 1005, 1110}

    def open_probe(self, tissue):
        global ready
        tissue.dll.LCS_ProbeAdditiveMg.argtypes = (ct.c_char_p, ct.c_uint)
        tissue.dll.LCS_ProbeAdditiveMg.restype = ct.c_int
        result = opened(self, tissue)
        ready = True
        return result

    def check_probe(self, name, *args):
        result = checked(self, name, *args)
        tick = self.tick + 1 if ready and name == 'LCS_Step' else -1
        if tick in sample_steps:
            path = output / f'additive-{tick:04d}.bin'
            assert self.dll.LCS_ProbeAdditiveMg(str(path).encode('utf8'), 1701+tick) == 0
            row = dict(step=tick, **verify_snapshot(path))
            rows.append(row)
            (output/'additive-checks.json').write_text(json.dumps(rows, indent=2), encoding='utf8')
            print('[ADDITIVE-CHECK] ' + json.dumps({k: v for k, v in row.items() if k not in ('snapshot', 'sha256')}), flush=True)
        return result

    run.Flow.open, native_backend.NativeTissue.check = open_probe, check_probe
    sys.argv = ['dev_run.py', '--output', str(output), '--native-only', '--scene', 'kidney', '--tool', 'scissors', *sys.argv[2:]]
    runpy.run_path(str(ROOT/'endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py'), run_name='__main__')
