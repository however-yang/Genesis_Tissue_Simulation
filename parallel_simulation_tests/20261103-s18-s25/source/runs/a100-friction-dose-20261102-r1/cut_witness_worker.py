"""Retain exact cut records at stage boundaries of the unchanged strict script.

Read-only exports add diagnostic overhead. These timings are not FPS evidence.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.release.resolve(strict=True)
    sys.path[:0] = [str(root), str(root/'endoscopic_tissue_backups/2026-10-31-interaction-fixes')]
    import validate_strict as worker
    from endoscopic_tissue.training.data import atomic_json
    from endoscopic_tissue.training.provenance import sha256
    from functional_worker import selected_gpu_memory
    selected_gpu_memory()
    stages = worker.fixture('kidney-reference')[3]
    tick, boundaries = 0, {}
    for stage in stages:
        tick += stage.steps
        if stage.name in ('incise_1', 'incise_2', 'hold'):
            boundaries[tick] = stage.name
    original = worker.NativeTissue
    class Instrumented(original):
        def step(self, command):
            obs = super().step(command)
            if self.tick not in boundaries:
                return obs
            counts = np.zeros(9, np.int32)
            ip, fp, up = ct.POINTER(ct.c_int), ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint)
            read = self.dll.LCS_ReadCutWitnesses
            read.argtypes = [ip, ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int,
                             ct.c_void_p, ct.c_int, fp, fp]
            read.restype = ct.c_int
            assert read(counts.ctypes.data_as(ip), None, 0, None, 0, None, 0, None, None) == 0
            nc, ne, npnt = map(int, counts[3:6])
            assert tuple(counts[6:]) == (32, 16, 36)
            points = np.empty((npnt, 32), np.uint8)
            edges = np.empty((ne, 4), np.int32)
            rotations = np.empty((nc, 3, 3), np.float32)
            cuts, skin = np.empty((nc, 6), np.float32), np.empty((nc, 6), np.float32)
            assert read(counts.ctypes.data_as(ip), points.ctypes.data, npnt, edges.ctypes.data, ne,
                        rotations.ctypes.data, nc, cuts.ctypes.data_as(fp), skin.ctypes.data_as(fp)) == 0
            sizes = np.zeros(5, np.int32)
            topo = self.dll.LCS_ReadSurfaceTopology
            topo.argtypes, topo.restype = [ip, ip, ct.c_int, up, ct.c_int], ct.c_int
            assert topo(sizes.ctypes.data_as(ip), None, 0, None, 0) == 0
            tags, masks = np.empty(int(sizes[4]), np.int32), np.empty(int(sizes[3]), np.uint32)
            assert topo(sizes.ctypes.data_as(ip), tags.ctypes.data_as(ip), len(tags),
                        masks.ctypes.data_as(up), len(masks)) == 0
            boundary = np.empty_like(masks)
            read_boundary = self.dll.LCS_ReadMaterialBoundaryMask
            read_boundary.argtypes, read_boundary.restype = [up, ct.c_int], ct.c_int
            assert read_boundary(boundary.ctypes.data_as(up), len(boundary)) == 0
            surface = self.surface()
            stage = boundaries[self.tick]
            folder = args.output/'cut-witnesses'
            folder.mkdir(exist_ok=True)
            target = folder/(stage+'.npz')
            if target.exists():
                raise FileExistsError('Cut witnesses must be immutable')
            np.savez_compressed(target, counts=counts, points=points, edges=edges, rotations=rotations,
                cuts=cuts, skin=skin, feature_ids=tags, actual_masks=masks, boundary_masks=boundary,
                vertices=surface.vertices, triangles=surface.triangles, anchors=surface.anchors,
                positions=obs.nodes, rest=self.grid.positions, neighbors=obs.neighbors,
                inside=self.grid.inside, active=self.grid.active, native_tick=self.tick)
            atomic_json(folder/(stage+'.json'), dict(tick=self.tick, stage=stage,
                file_sha256=sha256(target), source_sha256=sha256(__file__),
                library_sha256=sha256(args.library), read_only_capture=True,
                performance_accepted=False, default_replaced=False))
            print('CUT WITNESS', stage, self.tick, npnt, flush=True)
            return obs
    worker.NativeTissue = Instrumented
    sys.argv = ['validate_strict', '--case', 'kidney-reference', '--dll', str(args.library),
                '--output', str(args.output)]
    try:
        worker.main()
    finally:
        worker.NativeTissue = original


if __name__ == '__main__':
    main()
