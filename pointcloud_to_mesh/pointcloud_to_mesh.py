import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="Reconstruct a triangle mesh from a surface point cloud.")
    parser.add_argument("--input", type=Path, required=True, help="PLY, PCD, XYZ, XYZN, XYZRGB or PTS point cloud.")
    parser.add_argument("--output", type=Path, required=True, help="Output mesh: PLY, OBJ or STL.")
    parser.add_argument("--method", choices=("poisson", "bpa"), default="poisson")
    parser.add_argument("--voxel", type=float, default=0.0, help="Downsampling spacing, in the input coordinate unit.")
    parser.add_argument("--denoise", action="store_true", help="Remove statistical outliers before reconstruction.")
    parser.add_argument("--depth", type=int, default=8, help="Poisson octree depth, usually 8-10.")
    parser.add_argument(
        "--trim-quantile", type=float, default=0.0, help="Poisson density trimming fraction, e.g. 0.01."
    )
    parser.add_argument("--radii", type=float, nargs="+", help="BPA ball radii, in the input coordinate unit.")
    parser.add_argument("--reestimate-normals", action="store_true", help="Replace the input normals with estimates.")
    parser.add_argument(
        "--normal-direction", type=float, nargs=3, help="Align normals with a direction for a local surface."
    )
    parser.add_argument("--view", action="store_true", help="Open a window showing the reconstructed mesh.")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Point cloud not found: {args.input}")
    if args.input.suffix.lower() not in (".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"):
        parser.error("Convert the input to PLY, PCD, XYZ, XYZN, XYZRGB or PTS first.")
    if args.output.suffix.lower() not in (".ply", ".obj", ".stl"):
        parser.error("The output must end in .ply, .obj or .stl.")
    if args.input.resolve() == args.output.resolve():
        parser.error("Use a different path for the output mesh and the input point cloud.")
    if not np.isfinite(args.voxel) or args.voxel < 0:
        parser.error("voxel must be finite and nonnegative.")
    if not 4 <= args.depth <= 12:
        parser.error("depth must be between 4 and 12.")
    if not 0 <= args.trim_quantile < 1:
        parser.error("trim-quantile must be between 0 and 1, excluding 1.")
    if args.radii is not None and (not np.isfinite(args.radii).all() or min(args.radii) <= 0):
        parser.error("All ball radii must be finite and positive.")
    if args.normal_direction is not None:
        direction = np.array(args.normal_direction)
        if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 0:
            parser.error("normal-direction must be finite and nonzero.")

    cloud = o3d.io.read_point_cloud(str(args.input), remove_nan_points=True, remove_infinite_points=True)
    cloud.remove_duplicated_points()
    initial_count = len(cloud.points)
    if args.voxel > 0:
        cloud = cloud.voxel_down_sample(args.voxel)
    if args.denoise and len(cloud.points) > 20:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if len(cloud.points) < 4:
        parser.error("At least four distinct points are required after preprocessing.")

    distances = np.asarray(cloud.compute_nearest_neighbor_distance())
    spacing = np.median(distances[distances > 0])
    if not np.isfinite(spacing) or spacing <= 0:
        parser.error("Cannot estimate point spacing. Check the coordinates and duplicate points.")

    estimated_normals = args.reestimate_normals or not cloud.has_normals()
    if estimated_normals:
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=spacing * 3, max_nn=30))
    normals = np.asarray(cloud.normals)
    if not np.isfinite(normals).all() or np.any(np.linalg.norm(normals, axis=1) <= 0):
        parser.error("Normals must be finite and nonzero. Check the input or use --reestimate-normals.")
    cloud.normalize_normals()
    if args.normal_direction is not None:
        cloud.orient_normals_to_align_with_direction(direction)
    elif estimated_normals:
        try:
            cloud.orient_normals_consistent_tangent_plane(min(30, len(cloud.points) - 1))
        except RuntimeError as error:
            raise RuntimeError(
                "Normal orientation failed. Supply oriented normals in PLY/XYZN. "
                "For a local surface, use --normal-direction X Y Z."
            ) from error

    print(f"Points: {initial_count} -> {len(cloud.points)}")
    print(f"Median point spacing: {spacing:g} (input coordinate unit)")
    print(f"Normals: {'estimated' if estimated_normals else 'from input'}")
    if args.method == "poisson":
        mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=args.depth)
        if args.trim_quantile > 0:
            density = np.asarray(density)
            mesh.remove_vertices_by_mask(density < np.quantile(density, args.trim_quantile))
    else:
        radii = sorted(args.radii) if args.radii is not None else [spacing * 1.5, spacing * 3, spacing * 6]
        print(f"BPA radii: {radii}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud,
            o3d.utility.DoubleVector(radii),
        )

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError("Reconstruction produced no triangles. Check normals, coverage and point spacing.")
    mesh.compute_vertex_normals()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(args.output), mesh):
        raise RuntimeError(f"Could not write mesh: {args.output}")
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    print(f"Watertight: {mesh.is_watertight()}")
    print(f"Saved: {args.output.resolve()}")
    if args.view:
        o3d.visualization.draw_geometries([mesh], window_name="Reconstructed mesh")


if __name__ == "__main__":
    main()
