import argparse
from pathlib import Path

import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic sphere point cloud for reconstruction.")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data/input/demo_points.ply")
    parser.add_argument("--points", type=int, default=2500)
    args = parser.parse_args()
    if args.points < 4:
        parser.error("At least four points are required.")

    o3d.utility.random.seed(7)
    sphere = o3d.geometry.TriangleMesh.create_sphere(resolution=30)
    sphere.compute_vertex_normals()
    cloud = sphere.sample_points_uniformly(args.points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(args.output), cloud, write_ascii=True):
        raise RuntimeError(f"Could not write point cloud: {args.output}")
    print(f"Synthetic sphere: {len(cloud.points)} surface points with normals, radius 1 coordinate unit")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
