import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

import meshio
import open3d as o3d


SCRIPT = Path(__file__).resolve().parents[1] / "point_to_cloud_main.py"
VALIDATOR = SCRIPT.with_name("validate_input.py")


class MeshExportTests(unittest.TestCase):
    def run_config_cli(self, config, *options, cwd=None):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--config", str(config), *options],
            capture_output=True, text=True, timeout=120, cwd=cwd,
        )

    def run_cli(self, source, destination, *options):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(destination), *options],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def make_surface(self):
        points = np.array([
            [-12.3456789012345, 0.25, 7.125],
            [-10.125, 0.25, 7.125],
            [-12.3456789012345, 3.75, 7.125],
            [-12.3456789012345, 0.25, 11.5],
        ])
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
        surface = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(points), o3d.utility.Vector3iVector(faces)
        )
        return surface, points, faces

    def test_existing_mesh_msh_preserves_geometry_and_tags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            surface, points, faces = self.make_surface()
            source, destination = root / "surface.ply", root / "nested/surface.msh"
            self.assertTrue(o3d.io.write_triangle_mesh(str(source), surface))
            self.run_cli(source, destination, "--msh-type", "surface")
            text = destination.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("$MeshFormat\n2.2 0 8\n"))
            exported = meshio.read(destination)
            np.testing.assert_allclose(exported.points, points, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(exported.cells_dict["triangle"], faces)
            self.assertEqual(set(exported.cells_dict), {"triangle"})
            np.testing.assert_array_equal(exported.field_data["surface"], [1, 2])
            for tag in ("gmsh:physical", "gmsh:geometrical"):
                np.testing.assert_array_equal(exported.cell_data_dict[tag]["triangle"], np.ones(4))

    def test_pointcloud_poisson_and_bpa_export_msh(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sphere = o3d.geometry.TriangleMesh.create_sphere(resolution=20)
            sphere.compute_vertex_normals()
            o3d.utility.random.seed(7)
            cloud = sphere.sample_points_uniformly(500)
            source = root / "points.ply"
            self.assertTrue(o3d.io.write_point_cloud(str(source), cloud))
            for method in ("poisson", "bpa"):
                with self.subTest(method=method):
                    destination = root / f"{method}.msh"
                    self.run_cli(source, destination, "--method", method, "--depth", "5", "--msh-type", "surface")
                    exported = meshio.read(destination)
                    self.assertEqual(set(exported.cells_dict), {"triangle"})
                    self.assertGreater(len(exported.cells_dict["triangle"]), 0)
                    self.assertTrue(np.isfinite(exported.points).all())
                    self.assertGreaterEqual(exported.cells_dict["triangle"].min(), 0)
                    self.assertLess(exported.cells_dict["triangle"].max(), len(exported.points))

    def check_volume(self, exported):
        self.assertEqual(set(exported.cells_dict), {"triangle", "tetra"})
        elements = exported.cells_dict["tetra"]
        self.assertGreater(len(elements), 0)
        corners = exported.points[elements]
        volumes = np.einsum(
            "ij,ij->i", np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            corners[:, 3] - corners[:, 0],
        ) / 6
        self.assertTrue(np.isfinite(volumes).all())
        self.assertTrue((volumes > 0).all())
        all_faces = Counter(tuple(sorted(face)) for element in elements for face in combinations(element, 3))
        boundary = {face for face, count in all_faces.items() if count == 1}
        triangles = exported.cells_dict["triangle"]
        self.assertEqual({tuple(sorted(face)) for face in triangles}, boundary)
        self.assertEqual(len(triangles), len(boundary))
        np.testing.assert_array_equal(exported.field_data["surface"], [1, 2])
        np.testing.assert_array_equal(exported.field_data["volume"], [2, 3])
        return volumes

    def test_volume_mesh_covers_the_solid_and_respects_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            surface, points, _ = self.make_surface()
            source, destination = root / "solid.ply", root / "solid.msh"
            self.assertTrue(o3d.io.write_triangle_mesh(str(source), surface))
            self.run_cli(source, destination, "--tet-max-volume", "0.25")
            volumes = self.check_volume(meshio.read(destination))
            expected = abs(np.linalg.det((points[1:] - points[0]).T)) / 6
            self.assertAlmostEqual(volumes.sum(), expected, places=9)
            self.assertLessEqual(volumes.max(), 0.25 * (1 + 1e-8))

    def test_pointcloud_to_volume_msh(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sphere = o3d.geometry.TriangleMesh.create_sphere(resolution=20)
            sphere.compute_vertex_normals()
            o3d.utility.random.seed(7)
            cloud = sphere.sample_points_uniformly(500)
            source, destination = root / "points.ply", root / "solid.msh"
            self.assertTrue(o3d.io.write_point_cloud(str(source), cloud))
            self.run_cli(source, destination, "--method", "poisson", "--depth", "5")
            self.check_volume(meshio.read(destination))

    def test_open_surface_is_rejected_for_volume_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            surface, _, faces = self.make_surface()
            surface.triangles = o3d.utility.Vector3iVector(faces[:-1])
            source, destination = root / "open.ply", root / "invalid.msh"
            self.assertTrue(o3d.io.write_triangle_mesh(str(source), surface))
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(destination)],
                capture_output=True, text=True, timeout=120,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("closed, manifold surface", result.stderr)
            self.assertFalse(destination.exists())

    def test_existing_ply_output_still_works(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            surface, points, faces = self.make_surface()
            source, destination = root / "surface.ply", root / "copy.ply"
            self.assertTrue(o3d.io.write_triangle_mesh(str(source), surface))
            self.run_cli(source, destination, "--input-type", "mesh")
            exported = o3d.io.read_triangle_mesh(str(destination))
            np.testing.assert_allclose(np.asarray(exported.vertices), points, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(np.asarray(exported.triangles), faces)

    def test_config_batch_resolves_relative_paths_and_filters_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_root = root / "settings"
            input_root = config_root / "input"
            input_root.mkdir(parents=True)
            surface, points, _ = self.make_surface()
            for name in ("first.ply", "second.PLY"):
                self.assertTrue(o3d.io.write_triangle_mesh(str(input_root / name), surface))
            (input_root / "ignore.xyz").write_text("0 0 0\n", encoding="utf-8")
            config = config_root / "format_config.toml"
            config.write_text(
                '[input]\npath = "input"\nformat = "ply"\n[output]\npath = "output"\nformat = "msh"\n',
                encoding="utf-8",
            )
            result = self.run_config_cli(config, cwd=root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output_root = config_root / "output"
            self.assertEqual({path.name for path in output_root.iterdir()}, {"first.msh", "second.msh"})
            for name in ("first.msh", "second.msh"):
                exported = meshio.read(output_root / name)
                volumes = self.check_volume(exported)
                self.assertAlmostEqual(volumes.sum(), abs(np.linalg.det((points[1:] - points[0]).T)) / 6, places=9)

    def test_config_single_file_and_command_line_path_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            surface, points, faces = self.make_surface()
            source = root / "selected.ply"
            self.assertTrue(o3d.io.write_triangle_mesh(str(source), surface))
            config = root / "format_config.toml"
            config.write_text(
                '[input]\npath = "selected.ply"\nformat = "ply"\n[output]\npath = "output"\nformat = "ply"\n',
                encoding="utf-8",
            )
            result = self.run_config_cli(config)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            exported = o3d.io.read_triangle_mesh(str(root / "output/selected.ply"))
            np.testing.assert_allclose(np.asarray(exported.vertices), points, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(np.asarray(exported.triangles), faces)
            override = root / "override.obj"
            self.assertTrue(o3d.io.write_triangle_mesh(str(override), surface))
            result = self.run_config_cli(config, "--input", str(override))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / "output/override.ply").is_file())
            destination = root / "override.msh"
            result = self.run_config_cli(config, "--output", str(destination))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.check_volume(meshio.read(destination))

    def test_invalid_config_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "format_config.toml"
            for text in (
                '[input\n',
                '[input]\npath = "input"\nformat = "ply"\n',
                '[input]\npath = 42\nformat = "ply"\n[output]\npath = "output"\nformat = "msh"\n',
                '[input]\npath = "input"\nformat = "ply"\n[output]\npath = "output"\nformat = "invalid"\n',
            ):
                with self.subTest(config=text):
                    config.write_text(text, encoding="utf-8")
                    result = self.run_config_cli(config)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("error:", result.stderr)
                    self.assertFalse((root / "output").exists())

    def test_config_empty_input_and_source_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            input_root.mkdir()
            config = root / "format_config.toml"
            config.write_text(
                '[input]\npath = "input"\nformat = "ply"\n[output]\npath = "input"\nformat = "ply"\n',
                encoding="utf-8",
            )
            result = self.run_config_cli(config)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No .ply files", result.stderr)
            surface, _, _ = self.make_surface()
            source = input_root / "source.ply"
            self.assertTrue(o3d.io.write_triangle_mesh(str(source), surface))
            original = source.read_bytes()
            result = self.run_config_cli(config)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("different paths", result.stderr)
            self.assertEqual(source.read_bytes(), original)

    def test_pointcloud_normal_direction_with_new_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            x_coordinates, y_coordinates = np.meshgrid(np.arange(5), np.arange(5))
            points = np.column_stack((x_coordinates.reshape(-1), y_coordinates.reshape(-1), np.zeros(25)))
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            source, destination = root / "plane.ply", root / "plane_mesh.ply"
            self.assertTrue(o3d.io.write_point_cloud(str(source), cloud))
            self.run_cli(source, destination, "--method", "bpa", "--normal-direction", "0", "0", "1")
            exported = o3d.io.read_triangle_mesh(str(destination))
            self.assertGreater(len(exported.triangles), 0)
            np.testing.assert_allclose(np.asarray(exported.vertices)[:, 2], 0, rtol=0, atol=1e-12)

    def test_batch_skips_invalid_inputs_and_converts_later_valid_mesh(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root, output_root = root / "input", root / "output"
            input_root.mkdir()
            sparse = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
            self.assertTrue(o3d.io.write_point_cloud(str(input_root / "a_sparse.ply"), sparse))
            surface, points, faces = self.make_surface()
            self.assertTrue(o3d.io.write_triangle_mesh(str(input_root / "z_valid.ply"), surface))
            surface.triangles = o3d.utility.Vector3iVector(faces[:-1])
            self.assertTrue(o3d.io.write_triangle_mesh(str(input_root / "b_open.ply"), surface))
            original = {path.name: path.read_bytes() for path in input_root.iterdir()}
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--input", str(input_root), "--output", str(output_root)],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("[SKIP] a_sparse.ply", result.stderr)
            self.assertIn("[SKIP] b_open.ply", result.stderr)
            self.assertIn("open boundary edges", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn("1 converted, 2 skipped, 0 failed; 3 total", result.stdout)
            self.assertEqual({path.name for path in output_root.iterdir()}, {"z_valid.msh"})
            volumes = self.check_volume(meshio.read(output_root / "z_valid.msh"))
            self.assertAlmostEqual(volumes.sum(), abs(np.linalg.det((points[1:] - points[0]).T)) / 6, places=9)
            for path in input_root.iterdir():
                self.assertEqual(path.read_bytes(), original[path.name])

    def test_standalone_validator_reports_pointcloud_checks_and_open_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            self.assertTrue(o3d.io.write_point_cloud(str(root / "points.ply"), cloud))
            surface, _, faces = self.make_surface()
            surface.triangles = o3d.utility.Vector3iVector(faces[:-1])
            self.assertTrue(o3d.io.write_triangle_mesh(str(root / "open.ply"), surface))
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), "--input", str(root)],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("[PASS] points.ply", result.stdout)
            self.assertIn("1 passed, 1 skipped, 1 awaiting", result.stdout)
            self.assertIn("[SKIP] open.ply", result.stderr)
            self.assertEqual({path.name for path in root.iterdir()}, {"points.ply", "open.ply"})
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), "--input", str(root / "open.ply"), "--msh-type", "surface"],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("[PASS] open.ply", result.stdout)

    def test_standalone_validator_rejects_invalid_geometry_and_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            surface, points, faces = self.make_surface()
            overlapping = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.vstack((points, points + 0.2))),
                o3d.utility.Vector3iVector(np.vstack((faces, faces + 4))),
            )
            self.assertTrue(o3d.io.write_triangle_mesh(str(root / "self_intersection.ply"), overlapping))
            reflected = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.vstack((points, 2 * points[0] - points[1:]))),
                o3d.utility.Vector3iVector(np.vstack((faces, np.array([0, 4, 5, 6])[faces]))),
            )
            self.assertTrue(o3d.io.write_triangle_mesh(str(root / "non_manifold.ply"), reflected))
            degenerate_points = points.copy()
            degenerate_points[2] = (points[0] + points[1]) / 2
            surface.vertices = o3d.utility.Vector3dVector(degenerate_points)
            self.assertTrue(o3d.io.write_triangle_mesh(str(root / "degenerate.ply"), surface))
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), "--input", str(root)],
                capture_output=True, text=True, timeout=120,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("degenerate triangles", result.stderr)
            self.assertIn("non-manifold vertices", result.stderr)
            self.assertIn("self-intersections", result.stderr)
            self.assertIn("0 passed, 3 skipped", result.stdout)


if __name__ == "__main__":
    unittest.main()
