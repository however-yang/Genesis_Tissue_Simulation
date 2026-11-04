"""Exact positional topology diagnostic with packed integer edge/face keys.

Same 1e-5 native-unit vertex rounding and predicates as surface_diagnostics.
Only CPU bookkeeping changes; no caching across deformations or skipped checks.
"""
import numpy as np


def topology_statistics_fast(surface):
    if not len(surface.triangles) or len(surface.vertices) >= 2**21:
        from ..surface_diagnostics import topology_statistics
        return topology_statistics(surface)
    points=np.asarray(surface.vertices)
    triangles=points[surface.triangles]
    twice_area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    # Trimesh converts vertices to float64 before multiplying and rounding.
    rounded=np.column_stack([np.asarray(points,dtype=np.float64)*1e5]).round().astype(np.int64)
    from .topology_hash import count_topology
    boundary,nonmanifold,duplicates,winding=count_topology(rounded,np.asarray(surface.triangles,dtype=np.int64))
    wall=surface.anchors[surface.triangles[:,0],3]>.5
    wall_vertices=triangles[wall].reshape(-1,3)
    return dict(triangles=len(triangles),wall_triangles=int(wall.sum()),degenerate_triangles=int((twice_area<1e-8).sum()),
        positional_boundary_edges=int(boundary),positional_nonmanifold_edges=int(nonmanifold),
        duplicate_triangles=int(duplicates),winding_consistent=winding,
        wall_bounds_native_mm=(np.array((wall_vertices.min(axis=0),wall_vertices.max(axis=0)))*10).tolist() if len(wall_vertices) else None,
        wall_extent_mm_not_incision_depth=(np.ptp(wall_vertices,axis=0)*10).tolist() if len(wall_vertices) else None)
