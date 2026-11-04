"""Isolated equivalent surface diagnostic; never changes the solved surface.

This candidate keeps five-decimal positional welding and the original area,
duplicate-face, edge multiplicity and winding criteria. It shares one edge
sort between multiplicity and winding instead of building a Trimesh cache.
Not connected to the default environment until replay equivalence is proved.
"""
import numpy as np
from trimesh.grouping import unique_rows, hashable_rows


def topology_statistics(surface):
    triangles = surface.vertices[surface.triangles]
    twice_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                        triangles[:, 2] - triangles[:, 0]), axis=1)
    # Trimesh first converts vertices to float64, then rounds this product.
    # Rounding the incoming float32 array directly is NOT equivalent.
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.triangles, dtype=np.int64)
    referenced = np.zeros(len(vertices), dtype=bool)
    referenced[faces] = True
    quantized = (vertices[referenced] * 100000).round().astype(np.int64)
    _, inverse = unique_rows(quantized, keep_order=True)
    mapping = np.zeros(len(vertices), dtype=np.int64)
    mapping[referenced] = inverse
    welded = mapping[faces]
    edges = welded[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    keys = hashable_rows(np.sort(edges, axis=1))
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    if not len(edges):
        counts = np.empty(0, dtype=np.int64)
        winding = False
    else:
        counts = np.diff(np.r_[starts, len(edges)])
        paired = starts[counts == 2]
        first, second = edges[order[paired]], edges[order[paired + 1]]
        # Match Trimesh: only edges with exactly two occurrences participate.
        # Multiplicity > 2 is independently reported as nonmanifold.
        winding = bool(np.equal(first[:, 1], second[:, 0]).all())
    face_keys = hashable_rows(np.sort(welded, axis=1))
    wall = surface.anchors[surface.triangles[:, 0], 3] > .5
    wall_vertices = triangles[wall].reshape(-1, 3)
    return {
        'triangles': len(triangles), 'wall_triangles': int(np.count_nonzero(wall)),
        'degenerate_triangles': int(np.count_nonzero(twice_area < 1e-8)),
        'positional_boundary_edges': int(np.count_nonzero(counts == 1)),
        'positional_nonmanifold_edges': int(np.count_nonzero(counts > 2)),
        'duplicate_triangles': int(len(face_keys) - len(np.unique(face_keys))),
        'winding_consistent': winding,
        'wall_bounds_native_mm': (np.array((wall_vertices.min(axis=0), wall_vertices.max(axis=0))) * 10).tolist()
            if len(wall_vertices) else None,
        'wall_extent_mm_not_incision_depth': (np.ptp(wall_vertices, axis=0) * 10).tolist()
            if len(wall_vertices) else None,
    }
