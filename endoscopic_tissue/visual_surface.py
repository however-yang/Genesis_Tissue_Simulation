"""Per-triangle visual refinement; never welds across a crack or changes physics."""
import numpy as np

from .native_backend import Surface


def refine_surface(surface: Surface, divisions=4):
    if divisions < 1:
        raise ValueError("divisions must be positive")
    barycentric = []
    indices = {}
    for i in range(divisions + 1):
        for j in range(divisions + 1 - i):
            indices[i, j] = len(barycentric)
            barycentric.append((1 - (i + j) / divisions, i / divisions, j / divisions))
    local_faces = []
    for i in range(divisions):
        for j in range(divisions - i):
            local_faces.append((indices[i, j], indices[i + 1, j], indices[i, j + 1]))
            if i + j < divisions - 1:
                local_faces.append((indices[i + 1, j], indices[i + 1, j + 1], indices[i, j + 1]))
    weights = np.array(barycentric, dtype=np.float32)
    corners = surface.vertices[surface.triangles]
    normals = surface.normals[surface.triangles]
    normals = normals / np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-8)
    # Preserve the contact surface exactly; curvature belongs in the physical
    # mesh. Displacing only the display can push tissue through a solved jaw.
    vertices = np.einsum("ki,tij->tkj", weights, corners)
    interpolated_normals = np.einsum("ki,tij->tkj", weights, normals)
    interpolated_normals /= np.maximum(np.linalg.norm(interpolated_normals, axis=2, keepdims=True), 1e-8)
    anchors = np.einsum("ki,tij->tkj", weights, surface.anchors[surface.triangles])
    faces = np.array(local_faces, dtype=np.int32)[None] + np.arange(len(corners), dtype=np.int32)[:, None, None] * len(weights)
    return Surface(vertices.reshape(-1, 3), interpolated_normals.reshape(-1, 3),
                   faces.reshape(-1, 3), anchors.reshape(-1, 4))
