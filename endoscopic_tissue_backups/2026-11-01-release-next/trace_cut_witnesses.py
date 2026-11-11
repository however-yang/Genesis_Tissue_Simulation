"""Save matched reconstruction inputs at real cutting states, for CPU replay."""
import ctypes as ct
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT)]
import trace_surface as trace

original = trace.timing.NativeTissue.step
output = trace.output
step_count = 0


def step(self, command):
    global step_count
    obs = original(self, command)
    step_count += 1
    if step_count not in (450, 600, 840):
        return obs
    if step_count == 450:
        grid = self.grid
        names = ('positions', 'inside', 'active', 'pinned', 'corners', 'occupied',
                 'intersections', 'offsets', 'counts', 'surface_edges', 'edges')
        np.savez_compressed(output/'reconstruction-grid.npz', dims=grid.dims,
            spacing=grid.spacing, cell_size=grid.cell_size,
            **{name: getattr(grid, name) for name in names})
    fn = self.dll.LCS_ReadCutWitnesses
    ip = ct.POINTER(ct.c_int)
    fn.argtypes = [ip, ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int,
                   ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p]
    fn.restype = ct.c_int
    sizes = np.zeros(9, np.int32)
    assert fn(sizes.ctypes.data_as(ip), None, 0, None, 0, None, 0, None, None) == 0
    assert tuple(sizes[6:]) == (32, 16, 36), sizes
    nodes, edges, points = map(int, sizes[3:6])
    cp = np.empty((max(points, 1), 32), np.uint8)
    ge = np.empty((edges, 4), np.int32)
    rot = np.empty((nodes, 3, 3), np.float32)
    cut_t, skin_t = (np.empty((nodes, 6), np.float32) for _ in range(2))
    ptr = lambda a: ct.c_void_p(a.ctypes.data)
    assert fn(sizes.ctypes.data_as(ip), ptr(cp), points, ptr(ge), edges,
              ptr(rot), nodes, ptr(cut_t), ptr(skin_t)) == 0
    voxels = int(np.prod(sizes[:3]))
    external = np.empty((voxels, 4), np.float32)
    cut_fp = np.empty((voxels, 8, 4), np.float32)
    feature_fn = self.dll.LCS_ReadFrozenFeatures
    feature_fn.argtypes, feature_fn.restype = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int], ct.c_int
    assert feature_fn(ptr(external), voxels, ptr(cut_fp), 8*voxels) == 0
    np.savez_compressed(output/f'cut-witnesses-{step_count:04d}.npz', sizes=sizes,
        cut_points=cp[:points], grid_edges=ge, rotations=rot, cut_t=cut_t, skin_t=skin_t,
        external_fp=external, cut_fp=cut_fp)
    print('READ_ONLY_CUT_WITNESSES', step_count, points, flush=True)
    return obs


trace.timing.NativeTissue.step = step
if __name__ == '__main__':
    trace.timing.main()
