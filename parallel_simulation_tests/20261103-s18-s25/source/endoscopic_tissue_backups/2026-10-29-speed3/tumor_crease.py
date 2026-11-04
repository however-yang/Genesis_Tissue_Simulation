"""Measure how deeply the excised tumor's surface folds in: the largest distance of its surface vertices below the
convex hull of the held piece, from the snapshot of a stage after the second snip.

Usage: python tumor_crease.py <stage npz> [<stage npz> ...]

The held piece is the connected part of the surface mesh that contains the vertex nearest the jaw hinge. A smooth
excised tumor is nearly convex; a crease shows as a groove a millimetre or more below the hull.
"""
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull

for path in sys.argv[1:]:
    snapshot = np.load(path)
    hinge = snapshot["hinge_cm"]
    # The surface comes as separate triangles: weld the corners that share a position before following connectivity.
    vertices, welded = np.unique(np.round(snapshot["vertices_cm"], 6), axis=0, return_inverse=True)
    faces = welded.reshape(-1)[snapshot["faces"]]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices),) * 2)
    count, labels = connected_components(adjacency, directed=False)
    nearest = np.argmin(np.linalg.norm(vertices - hinge, axis=1))
    piece = vertices[labels == labels[nearest]]
    hull = ConvexHull(piece)
    # Each hull facet is a plane n.x + d <= 0 for the points inside; the depth of a point is its distance to the
    # nearest facet, and the deepest point lies at the bottom of the deepest groove.
    depth = (-(piece @ hull.equations[:, :3].T + hull.equations[:, 3])).min(axis=1)
    # A crease is a line of edges where the surface folds inward: two triangles whose normals turn towards each other
    # by more than 60 degrees across a shared edge, with the fold opening outward (concave seen from outside).
    held = faces[labels[faces[:, 0]] == labels[nearest]]
    corners = vertices[held]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    centers = corners.mean(axis=1)
    owners = {}
    for index, triangle in enumerate(held):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            owners.setdefault((min(triangle[a], triangle[b]), max(triangle[a], triangle[b])), []).append(index)
    fold_length = 0.0
    for (a, b), pair in owners.items():
        if len(pair) != 2:
            continue
        first, second = pair
        if np.dot(normals[first], normals[second]) < np.cos(np.radians(60)) and \
                np.dot(normals[first], centers[second] - centers[first]) > 0:
            fold_length += np.linalg.norm(vertices[a] - vertices[b])
    print(f"{path}: pieces {count}, held piece {len(piece)} vertices, extent {np.ptp(piece, axis=0).round(2)} cm, "
          f"deepest point below hull {depth.max() * 10:.2f} mm, 99th percentile {np.percentile(depth, 99) * 10:.2f} mm, "
          f"concave fold edges {fold_length * 10:.1f} mm")
