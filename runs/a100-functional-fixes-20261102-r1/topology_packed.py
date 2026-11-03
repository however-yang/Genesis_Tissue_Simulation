"""Exact integer-key topology candidate; no surface, solver, or criterion edits.

Position welding retains the reference float64 multiplication and rounding.
Integer tuple keys are packed only when provably injective in 63 bits; the
general Trimesh representation remains the fallback. Vertex labels need not
have the same numbering because every returned diagnostic is label invariant.
"""
import numpy as np
from trimesh.grouping import hashable_rows


def tuple_keys(rows):
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    if not len(rows):
        return np.empty(0, dtype=np.uint64)
    minima = [int(x) for x in rows.min(axis=0)]
    maxima = [int(x) for x in rows.max(axis=0)]
    widths = [(hi-lo).bit_length() for lo, hi in zip(minima, maxima)]
    if sum(widths) > 63:
        return hashable_rows(rows)
    keys = np.zeros(len(rows), dtype=np.uint64)
    shift = 0
    for axis, (lo, width) in enumerate(zip(minima, widths)):
        if width:
            # The width proof also guarantees this signed subtraction cannot
            # overflow; all resulting offsets are in [0, 2**63-1].
            offsets = (rows[:, axis]-lo).astype(np.uint64)
            keys |= offsets << shift
            shift += width
    return keys


def topology_statistics(surface):
    triangles = surface.vertices[surface.triangles]
    twice_area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                        triangles[:, 2]-triangles[:, 0]), axis=1)
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.triangles, dtype=np.int64)
    referenced = np.zeros(len(vertices), dtype=bool)
    referenced[faces] = True
    quantized = (vertices[referenced]*100000).round().astype(np.int64)
    _, inverse = np.unique(tuple_keys(quantized), return_inverse=True)
    mapping = np.zeros(len(vertices), dtype=np.int64)
    mapping[referenced] = inverse
    welded = mapping[faces]
    edges = welded[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    if len(vertices) <= 0xffffffff:
        low, high = edges.min(axis=1), edges.max(axis=1)
        keys = (low.astype(np.uint64) << 32) | high.astype(np.uint64)
    else:
        keys = hashable_rows(np.sort(edges, axis=1))
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    if not len(edges):
        counts = np.empty(0, dtype=np.int64)
        winding = bool(len(vertices))
    else:
        starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1])+1]
        counts = np.diff(np.r_[starts, len(edges)])
        paired = starts[counts == 2]
        winding = bool(np.equal(edges[order[paired], 1], edges[order[paired+1], 0]).all())
    face_keys = tuple_keys(np.sort(welded, axis=1))
    wall = surface.anchors[surface.triangles[:, 0], 3] > .5
    wall_vertices = triangles[wall].reshape(-1, 3)
    return {
        'triangles': len(triangles), 'wall_triangles': int(np.count_nonzero(wall)),
        'degenerate_triangles': int(np.count_nonzero(twice_area < 1e-8)),
        'positional_boundary_edges': int(np.count_nonzero(counts == 1)),
        'positional_nonmanifold_edges': int(np.count_nonzero(counts > 2)),
        'duplicate_triangles': int(len(face_keys)-len(np.unique(face_keys))),
        'winding_consistent': winding,
        'wall_bounds_native_mm': (np.array((wall_vertices.min(axis=0), wall_vertices.max(axis=0)))*10).tolist()
            if len(wall_vertices) else None,
        'wall_extent_mm_not_incision_depth': (np.ptp(wall_vertices, axis=0)*10).tolist()
            if len(wall_vertices) else None,
    }
