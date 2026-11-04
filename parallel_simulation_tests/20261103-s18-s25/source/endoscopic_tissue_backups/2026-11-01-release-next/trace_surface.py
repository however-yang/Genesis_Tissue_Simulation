"""Capture native feature incidence on a fixed trajectory; diagnostic timings only."""
import ctypes as ct
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'endoscopic_tissue_backups/2026-10-31-interaction-fixes'), str(ROOT)]
import time_full_interaction as timing
import interaction_fixtures
timing.fixture = interaction_fixtures.fixture
output = Path(sys.argv[sys.argv.index('--output')+1])
original = timing.NativeTissue.step
count = 0


def step(self, command):
    global count
    obs = original(self, command)
    count += 1
    if count not in (450, 600, 840, 1320):
        return obs
    surface = self.surface()
    sizes = np.zeros(5, np.int32)
    fn = self.dll.LCS_ReadSurfaceTopology
    ip, up = ct.POINTER(ct.c_int), ct.POINTER(ct.c_uint)
    fn.argtypes, fn.restype = [ip, ip, ct.c_int, up, ct.c_int], ct.c_int
    assert fn(sizes.ctypes.data_as(ip), None, 0, None, 0) == 0
    ids, masks = np.empty(sizes[4], np.int32), np.empty(sizes[3], np.uint32)
    assert fn(sizes.ctypes.data_as(ip), ids.ctypes.data_as(ip), len(ids), masks.ctypes.data_as(up), len(masks)) == 0
    assert len(ids) == 3*len(surface.triangles)
    np.savez_compressed(output/f'native-topology-{count:04d}.npz', counts=sizes,
        feature_ids=ids.reshape(-1, 3).view(np.uint32), voxel_masks=masks,
        vertices_cm=surface.vertices, faces=surface.triangles, anchors=surface.anchors,
        nodes_cm=obs.nodes, neighbors=obs.neighbors)
    print('READ_ONLY_TOPOLOGY', count, flush=True)
    return obs


timing.NativeTissue.step = step
if __name__ == '__main__':
    timing.main()
