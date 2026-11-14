import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np

try:
    import meshio
except ImportError:
    meshio = None

import open3d as o3d

try:
    import tetgen
except ImportError:
    tetgen = None

from validate_input import InputValidationError, read_input_geometry, validate_mesh


INPUT_FORMATS = ("ply", "pcd", "xyz", "xyzn", "xyzrgb", "pts", "obj", "stl")
OUTPUT_FORMATS = ("ply", "obj", "stl", "msh")


def conversion_paths(args, parser):
    """Resolve configured paths into input files and corresponding output files."""
    try:
        with args.config.open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        parser.error(f"Cannot read format config {args.config}: {error}")
    for section in ("input", "output"):
        if section not in config or not isinstance(config[section], dict):
            parser.error(f"Format config requires an [{section}] section.")
        for key in ("path", "format"):
            if key not in config[section] or not isinstance(config[section][key], str) or not config[section][key].strip():
                parser.error(f"Format config {section}.{key} must be a nonempty string.")
    input_format = config["input"]["format"].strip().lower().removeprefix(".")
    output_format = config["output"]["format"].strip().lower().removeprefix(".")
    if input_format not in INPUT_FORMATS:
        parser.error(f"Unsupported configured input format: {input_format}.")
    if output_format not in OUTPUT_FORMATS:
        parser.error(f"Unsupported configured output format: {output_format}.")
    config_root = args.config.resolve().parent
    input_path = args.input if args.input is not None else config_root / config["input"]["path"]
    output_path = args.output if args.output is not None else config_root / config["output"]["path"]
    if input_path.is_dir():
        sources = sorted(
            path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == f".{input_format}"
        )
        if not sources:
            parser.error(f"No .{input_format} files found in {input_path}.")
    elif input_path.is_file():
        if input_path.suffix.lower().removeprefix(".") not in INPUT_FORMATS:
            parser.error("Use a PLY/PCD/XYZ/XYZN/XYZRGB/PTS point cloud or a PLY/OBJ/STL mesh.")
        if args.input is None and input_path.suffix.lower() != f".{input_format}":
            parser.error("Configured input file extension must match input.format.")
        sources = [input_path]
    else:
        parser.error(f"Input file or directory not found: {input_path}")
    if output_path.is_dir() or not output_path.suffix:
        destinations = [output_path / f"{source.stem}.{output_format}" for source in sources]
    else:
        if len(sources) != 1:
            parser.error("Directory input requires an output directory.")
        if output_path.suffix.lower().removeprefix(".") not in OUTPUT_FORMATS:
            parser.error("The output must end in .ply, .obj, .stl or .msh.")
        if args.output is None and output_path.suffix.lower() != f".{output_format}":
            parser.error("Configured output file extension must match output.format.")
        destinations = [output_path]
    source_paths = {source.resolve() for source in sources}
    for destination in destinations:
        if destination.resolve() in source_paths:
            parser.error("Use different paths for output meshes and input files.")
        if args.tet_max_volume is not None and (destination.suffix.lower() != ".msh" or args.msh_type != "volume"):
            parser.error("tet-max-volume applies only to volume MSH output.")
    return list(zip(sources, destinations))


def tetrahedralize_surface(mesh, max_volume=None):
    """Return nodes, positive linear tetrahedra, and their outward boundary faces."""
    validate_mesh(mesh, require_volume=True)
    if max_volume is not None and (not np.isfinite(max_volume) or max_volume <= 0):
        raise ValueError("Maximum tetrahedron volume must be finite and positive.")
    if tetgen is None:
        raise RuntimeError("Volume MSH requires tetgen. Install the package requirements first.")
    requested_volume = max_volume
    for attempt in range(4):
        generator = tetgen.TetGen(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
        options = {"order": 1, "quality": True, "minratio": 2.0, "quiet": True}
        if requested_volume is not None:
            options.update(fixedvolume=True, maxvolume=requested_volume)
        try:
            nodes, elements, *_ = generator.tetrahedralize(**options)
        except RuntimeError as error:
            raise RuntimeError(f"TetGen could not generate a volume mesh: {error}") from error
        if max_volume is None:
            break
        candidate = np.asarray(nodes)[np.asarray(elements)]
        candidate_volumes = np.abs(np.einsum(
            "ij,ij->i", np.cross(candidate[:, 1] - candidate[:, 0], candidate[:, 2] - candidate[:, 0]),
            candidate[:, 3] - candidate[:, 0],
        )) / 6
        largest = np.max(candidate_volumes)
        if largest <= max_volume * (1 + 1e-8):
            break
        # TetGen optimization may leave elements above its requested target;
        # tighten the target and verify the actual exported element volumes.
        requested_volume *= 0.8 * max_volume / largest
    else:
        raise RuntimeError("TetGen did not meet tet-max-volume; no volume MSH was written.")
    nodes = np.asarray(nodes)
    elements = np.asarray(elements).copy()
    if elements.ndim != 2 or elements.shape[1] != 4 or len(elements) == 0:
        raise RuntimeError("TetGen did not produce linear four-node tetrahedra.")
    if not np.isfinite(nodes).all() or elements.min() < 0 or elements.max() >= len(nodes):
        raise RuntimeError("TetGen returned invalid nodes or element indices.")
    corners = nodes[elements]
    six_volumes = np.einsum(
        "ij,ij->i", np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        corners[:, 3] - corners[:, 0],
    )
    edge_lengths = np.linalg.norm(corners[:, 1:] - corners[:, :1], axis=2)
    tolerance = np.finfo(np.float64).eps * np.prod(edge_lengths, axis=1) * 64
    if not np.isfinite(six_volumes).all() or np.any(np.abs(six_volumes) <= tolerance):
        raise RuntimeError("TetGen produced degenerate tetrahedra; no volume MSH was written.")
    inverted = six_volumes < 0
    elements[inverted, :2] = elements[inverted, :2][:, ::-1]
    faces = elements[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    _, first_indices, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise RuntimeError("The volume mesh contains non-manifold tetrahedron faces.")
    boundary = faces[first_indices[counts == 1]]
    print(f"Volume mesh: {len(nodes)} nodes, {len(elements)} tetrahedra, {len(boundary)} boundary triangles")
    print(f"Minimum tetrahedron volume: {np.min(np.abs(six_volumes)) / 6:g} (input coordinate unit cubed)")
    print(f"Maximum tetrahedron volume: {np.max(np.abs(six_volumes)) / 6:g} (input coordinate unit cubed)")
    print("Degenerate tetrahedra: 0; inverted tetrahedra after orientation correction: 0")
    return nodes, elements, boundary


def write_mesh(mesh, output, *, msh_type="volume", tet_max_volume=None):
    """Export a surface or volume mesh, using Gmsh 2.2 ASCII for .msh files."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".msh":
        if meshio is None:
            raise RuntimeError("MSH export requires meshio. Install the package requirements first.")
        if msh_type not in ("surface", "volume"):
            raise ValueError("msh_type must be surface or volume.")
        if msh_type == "volume":
            points, elements, triangles = tetrahedralize_surface(mesh, tet_max_volume)
            cells = [("triangle", triangles), ("tetra", elements)]
            tags = [np.ones(len(triangles), dtype=np.int32), np.full(len(elements), 2, dtype=np.int32)]
            groups = {"surface": np.array([1, 2]), "volume": np.array([2, 3])}
        else:
            points = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            cells = [("triangle", triangles)]
            tags = [np.ones(len(triangles), dtype=np.int32)]
            groups = {"surface": np.array([1, 2])}
        msh = meshio.Mesh(
            points=points, cells=cells,
            cell_data={"gmsh:physical": tags, "gmsh:geometrical": tags},
            field_data=groups,
        )
        meshio.write(output, msh, file_format="gmsh22", binary=False)
    elif not o3d.io.write_triangle_mesh(str(output), mesh):
        raise RuntimeError(f"Could not write mesh: {output}")


def main():
    parser = argparse.ArgumentParser(description="Reconstruct a point cloud or export an existing triangle mesh.")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().with_name("format_config.toml"),
        help="Format config; relative paths inside it are resolved against its directory.",
    )
    parser.add_argument("--input", type=Path, help="Override the configured input file or directory.")
    parser.add_argument("--output", type=Path, help="Override the configured output file or directory.")
    parser.add_argument(
        "--input-type", choices=("auto", "pointcloud", "mesh"), default="auto",
        help="Auto preserves PLY faces and OBJ/STL meshes; pointcloud forces surface reconstruction.",
    )
    parser.add_argument(
        "--msh-type", choices=("volume", "surface"), default="volume",
        help="MSH element type: volume adds linear tetrahedra and boundary triangles (default).",
    )
    parser.add_argument(
        "--tet-max-volume", type=float,
        help="Maximum tetrahedron volume, in the input coordinate unit cubed; only for volume MSH.",
    )
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

    if args.tet_max_volume is not None:
        if not np.isfinite(args.tet_max_volume) or args.tet_max_volume <= 0:
            parser.error("tet-max-volume must be finite and positive.")
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

    paths = conversion_paths(args, parser)
    if any(destination.suffix.lower() == ".msh" for _, destination in paths):
        if meshio is None:
            parser.error("MSH export requires meshio. Install the package requirements first.")
        if args.msh_type == "volume" and tetgen is None:
            parser.error("Volume MSH requires tetgen. Install the package requirements first.")
    converted = 0
    skipped = 0
    failed = 0
    for source, destination in paths:
        print(f"Input: {source.resolve()}")
        try:
            convert_file(source, destination, args)
        except InputValidationError as error:
            skipped += 1
            print(f"[SKIP] {source.name}: {error}", file=sys.stderr)
        except RuntimeError as error:
            failed += 1
            print(f"[FAILED] {source.name}: {error}", file=sys.stderr)
        else:
            converted += 1
    print(f"Conversion summary: {converted} converted, {skipped} skipped, {failed} failed; {len(paths)} total.")
    return 0 if converted and not failed else 1


def convert_file(source, destination, args):
    """Convert one input file using the reconstruction and export options."""
    geometry = read_input_geometry(source, args.input_type)
    if isinstance(geometry, o3d.geometry.PointCloud):
        mesh = reconstruct_point_cloud(geometry, args)
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
    else:
        mesh = geometry
        print("Input: existing triangle mesh; preserving its coordinates and face connectivity")
    validate_mesh(mesh)
    mesh.compute_vertex_normals()
    write_mesh(mesh, destination, msh_type=args.msh_type, tet_max_volume=args.tet_max_volume)
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    print(f"Watertight: {mesh.is_watertight()}")
    if destination.suffix.lower() == ".msh":
        print(f"MSH: Gmsh 2.2 ASCII; {args.msh_type} mesh")
    print(f"Saved: {destination.resolve()}")
    if args.view:
        o3d.visualization.draw_geometries([mesh], window_name="Reconstructed mesh")


def reconstruct_point_cloud(cloud, args):
    initial_count = len(cloud.points)
    if args.voxel > 0:
        cloud = cloud.voxel_down_sample(args.voxel)
    if args.denoise and len(cloud.points) > 20:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if len(cloud.points) < 4:
        raise InputValidationError("At least four distinct points are required after preprocessing.")

    distances = np.asarray(cloud.compute_nearest_neighbor_distance())
    spacing = np.median(distances[distances > 0])
    if not np.isfinite(spacing) or spacing <= 0:
        raise InputValidationError("Cannot estimate point spacing. Check the coordinates and duplicate points.")

    estimated_normals = args.reestimate_normals or not cloud.has_normals()
    if estimated_normals:
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=spacing * 3, max_nn=30))
    normals = np.asarray(cloud.normals)
    if not np.isfinite(normals).all() or np.any(np.linalg.norm(normals, axis=1) <= 0):
        raise InputValidationError("Normals must be finite and nonzero. Check the input or use --reestimate-normals.")
    cloud.normalize_normals()
    if args.normal_direction is not None:
        cloud.orient_normals_to_align_with_direction(np.array(args.normal_direction))
    elif estimated_normals:
        try:
            cloud.orient_normals_consistent_tangent_plane(min(30, len(cloud.points) - 1))
        except RuntimeError as error:
            raise InputValidationError(
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

    return mesh


if __name__ == "__main__":
    sys.exit(main())
