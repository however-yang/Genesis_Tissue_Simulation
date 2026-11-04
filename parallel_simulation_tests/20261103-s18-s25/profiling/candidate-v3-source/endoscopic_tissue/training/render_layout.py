"""Exact texture-layout cache with full anchor/index invalidation.

Positions and normals are never cached. Dynamic color callables are still run on
all updates. The cache is valid only when all material anchors, triangle indices,
and the texture scale are unchanged; equal vertex counts are not sufficient.
"""
import numpy as np

class SurfaceLayoutCache:
    def __init__(self):
        self.anchors = self.triangles = self.parts = None
        self.scale = None
        self.hits = self.misses = 0

    def get(self, surface, scale, uv_axes):
        anchors, triangles = surface.anchors, surface.triangles
        if (self.parts is not None and self.scale == scale
                and np.array_equal(self.anchors, anchors)
                and np.array_equal(self.triangles, triangles)):
            self.hits += 1
            return self.parts
        soup = triangles.size == len(surface.vertices) and np.array_equal(
            triangles.ravel(), np.arange(len(surface.vertices), dtype=np.int32))
        wall = anchors[triangles[:, 0], 3] > .5
        rest = anchors[:, :3].reshape(-1, 3, 3) if soup else anchors[triangles, :3]
        projection = np.abs(np.cross(rest[:, 1]-rest[:, 0],rest[:, 2]-rest[:, 0])).argmax(axis=1)
        parts = []
        for is_wall in (False, True):
            selected = wall == is_wall
            if not np.any(selected):
                parts.append(None)
                continue
            if soup:
                ids = triangles[selected].ravel()
                axes = np.repeat(projection[selected], 3)
                indices = np.arange(len(ids), dtype=np.int32).reshape(-1, 3)
            else:
                keys = triangles[selected]*3 + projection[selected,None]
                is_used = np.zeros(3*len(surface.vertices), bool)
                is_used[keys] = True
                used = np.flatnonzero(is_used)
                ids, axes = used//3, used%3
                indices = (np.cumsum(is_used)-1)[keys]
            texcoord = anchors[ids[:,None],uv_axes[axes]]/scale
            parts.append((ids,indices,texcoord))
        self.anchors, self.triangles = anchors.copy(), triangles.copy()
        self.scale, self.parts = scale, parts
        self.misses += 1
        return parts
