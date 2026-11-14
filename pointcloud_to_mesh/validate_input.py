import argparse
import sys
from pathlib import Path

import numpy as np

import open3d as o3d


class InputValidationError(ValueError):
    """An input geometry fails the requested conversion requirements."""


def read_input_geometry(source, input_type="auto"):
    """Read a triangle mesh or a point cloud with finite, distinct points."""
    source = Path(source)
    if not source.is_file():
        raise InputValidationError(f"Input file not found: {source}")
    if input_type != "pointcloud" and source.suffix.lower() in (".ply", ".obj", ".stl"):
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            mesh = o3d.io.read_triangle_mesh(str(source))
        if mesh.has_triangles():
            return mesh
    if input_type == "mesh" or source.suffix.lower() in (".obj", ".stl"):
        raise InputValidationError("The input does not contain a readable triangle mesh.")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        cloud = o3d.io.read_point_cloud(str(source), remove_nan_points=True, remove_infinite_points=True)
    cloud.remove_duplicated_points()
    if len(cloud.points) < 4:
        raise InputValidationError("At least four distinct finite points are required.")
    return cloud


def validate_mesh(mesh, *, require_volume=False):
    """Check triangle geometry and, for volume export, closed manifold topology."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        raise InputValidationError("The mesh contains no triangles.")
    if not np.isfinite(vertices).all():
        raise InputValidationError("Mesh vertex coordinates must be finite.")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise InputValidationError("Mesh triangle indices are outside the vertex array.")
    corners = vertices[triangles]
    areas = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise InputValidationError("The mesh contains degenerate triangles.")
    if require_volume:
        reasons = []
        if not mesh.is_edge_manifold(allow_boundary_edges=True):
            reasons.append("non-manifold edges")
        elif not mesh.is_edge_manifold(allow_boundary_edges=False):
            reasons.append("open boundary edges")
        if not mesh.is_vertex_manifold():
            reasons.append("non-manifold vertices")
        if mesh.is_self_intersecting():
            reasons.append("self-intersections")
        if reasons:
            raise InputValidationError(
                "Volume MSH requires a closed, manifold surface without self-intersections: " + ", ".join(reasons)
            )


def main():
    parser = argparse.ArgumentParser(description="Validate input geometry before point cloud or mesh conversion.")
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "data/input")
    parser.add_argument("--format", default="ply", help="Input extension to select in a directory, default: ply.")
    parser.add_argument("--input-type", choices=("auto", "pointcloud", "mesh"), default="auto")
    parser.add_argument("--msh-type", choices=("volume", "surface"), default="volume")
    args = parser.parse_args()
    if args.input.is_dir():
        sources = sorted(
            path for path in args.input.iterdir()
            if path.is_file() and path.suffix.lower() == f".{args.format.lower().removeprefix('.')}"
        )
        if not sources:
            parser.error(f"No matching input files found in {args.input}.")
    else:
        sources = [args.input]
    passed = 0
    skipped = 0
    pending = 0
    for source in sources:
        try:
            geometry = read_input_geometry(source, args.input_type)
            if isinstance(geometry, o3d.geometry.TriangleMesh):
                validate_mesh(geometry, require_volume=args.msh_type == "volume")
                print(f"[PASS] {source.name}")
            else:
                print(f"[PASS] {source.name}: {len(geometry.points)} distinct finite points")
                if args.msh_type == "volume":
                    pending += 1
                    print("  Volume eligibility will be checked after surface reconstruction by the main program.")
            passed += 1
        except InputValidationError as error:
            skipped += 1
            print(f"[SKIP] {source.name}: {error}", file=sys.stderr)
    print(f"Validation summary: {passed} passed, {skipped} skipped, {pending} awaiting reconstructed surface checks.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
