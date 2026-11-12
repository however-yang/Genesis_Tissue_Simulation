import argparse
import sys

import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="Check Open3D and display a sphere mesh.")
    parser.add_argument("--no-window", action="store_true", help="Check mesh operations without opening a window.")
    args = parser.parse_args()

    mesh = o3d.geometry.TriangleMesh.create_sphere()
    mesh.compute_vertex_normals()
    print(f"Python: {sys.executable}")
    print(f"Open3D: {o3d.__version__}")
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    print(f"Vertex normals: {mesh.has_vertex_normals()}")

    if not args.no_window:
        o3d.visualization.draw_geometries(
            [mesh],
            window_name="Open3D environment check",
            width=960,
            height=720,
        )


if __name__ == "__main__":
    main()
