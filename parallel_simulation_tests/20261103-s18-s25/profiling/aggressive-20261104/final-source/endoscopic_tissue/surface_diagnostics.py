"""Read and inspect solved surface checkpoints without initializing a renderer."""
import numpy as np
import trimesh

from .native_backend import Observation, Surface


def load_state(path):
    with np.load(path) as data:
        obs = Observation(data["nodes_cm"], data["neighbors"], data["grip_flags"], data["hinge_cm"],
                          float(data["jaw_gap_cm"]), int(data["grip_points"]), float(data["grip_error_m"]),
                          data["blade"], data["reaction_n"])
        surface = Surface(data["vertices_cm"], data["normals"], data["faces"], data["anchors"])
    return obs, surface


def topology_statistics(surface):
    triangles = surface.vertices[surface.triangles]
    twice_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                        triangles[:, 2] - triangles[:, 0]), axis=1)
    mesh = trimesh.Trimesh(surface.vertices, surface.triangles, process=False)
    # Positional welding is a diagnostic of the exported surface only. It does
    # not change the actual solver or claim that coincident crack sides are bonded.
    mesh.merge_vertices(digits_vertex=5)
    counts = np.bincount(mesh.edges_unique_inverse)
    face_keys = np.sort(mesh.faces, axis=1)
    wall = surface.anchors[surface.triangles[:, 0], 3] > .5
    wall_vertices = triangles[wall].reshape(-1, 3)
    return {
        "triangles": len(triangles), "wall_triangles": int(np.count_nonzero(wall)),
        "degenerate_triangles": int(np.count_nonzero(twice_area < 1e-8)),
        "positional_boundary_edges": int(np.count_nonzero(counts == 1)),
        "positional_nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "duplicate_triangles": int(len(face_keys) - len(np.unique(face_keys, axis=0))),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "wall_bounds_native_mm": (np.array((wall_vertices.min(axis=0), wall_vertices.max(axis=0))) * 10).tolist()
            if len(wall_vertices) else None,
        "wall_extent_mm_not_incision_depth": (np.ptp(wall_vertices, axis=0) * 10).tolist() if len(wall_vertices) else None,
    }


