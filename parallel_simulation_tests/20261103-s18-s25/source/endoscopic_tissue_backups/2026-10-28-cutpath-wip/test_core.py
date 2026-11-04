"""CPU contracts that guard the cross-language topology and unit conventions."""
import ctypes
from dataclasses import replace
from itertools import product
import unittest

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh

from .grid import BLOCK_SIZE, build_grid, connectivity_table, EDGE_CORNERS, Tab, to_genesis
from .native_backend import (Blade, Command, CutDesc, Grasper, GraspParams, Observation, PhysicsDesc, ReconDesc,
                             ScissorsLoads, ScissorsPose, Surface, scissors_frame)
from .scenario import (FLAP_SCENE, SCISSORS_SCENES, SCISSORS_SPACING, WALL_SCENE, Sequence, interpolate,
                       scissors_settings, severed_edges, stage_start)
from .visual_surface import refine_surface
from .instrument_geometry import (SCISSORS_BACK_CLEARANCE, knife_frame, knife_geometry, load_davinci_wrist,
                                  load_grasper, load_scissors, scissors_body_depths)
from .surface_diagnostics import topology_statistics


class CoreTests(unittest.TestCase):
    def test_davinci_dimensions_and_wrist_contact_envelopes(self):
        grasper = load_grasper(1.99, .52)
        np.testing.assert_allclose(grasper.vertices[0][:, 0].max(), 1.99, atol=1e-6)
        np.testing.assert_allclose(np.ptp(grasper.vertices[0][:, 2]), .52, atol=1e-6)
        wrist = load_davinci_wrist()
        for mesh in wrist:
            delta = mesh.vertices.copy()
            delta[:, 0] = np.maximum(delta[:, 0], 0)
            self.assertLessEqual(np.linalg.norm(delta, axis=1).max(), grasper.shaft_radius)
        knife = knife_geometry(1.6)
        for axis in ((0, 0, 1), (2 ** -.5, 0, 2 ** -.5)):
            a, b = knife.handle_points(np.zeros(3), axis)
            direction = (a - b) / np.linalg.norm(a - b)
            normal = np.array((0, 1, 0)) - direction * direction[1]
            normal /= np.linalg.norm(normal)
            pose = np.stack((direction, normal, np.cross(direction, normal)))
            box = knife.boxes[1]
            basis = np.stack((box[6:9], box[9:12], np.cross(box[6:9], box[9:12])))
            for mesh in wrist:
                local = ((a + mesh.vertices @ pose) @ knife_frame(axis).T - box[:3]) @ basis.T
                self.assertTrue(np.all(np.abs(local) <= box[3:6] + 1e-6))

    def test_surface_audit_rejects_holes_flips_and_internal_duplicates(self):
        vertices = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float32)
        faces = np.array(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
        def audit(triangles):
            return topology_statistics(Surface(vertices, np.zeros_like(vertices), triangles, np.zeros((4, 4))))
        closed = audit(faces)
        self.assertEqual(closed['positional_boundary_edges'], 0)
        self.assertEqual(closed['positional_nonmanifold_edges'], 0)
        self.assertTrue(closed['winding_consistent'])
        self.assertEqual(audit(faces[:-1])['positional_boundary_edges'], 3)
        flipped = faces.copy()
        flipped[0] = flipped[0, ::-1]
        self.assertFalse(audit(flipped)['winding_consistent'])
        duplicated = audit(np.concatenate((faces, faces[:1, ::-1])))
        self.assertEqual(duplicated['duplicate_triangles'], 1)
        self.assertEqual(duplicated['positional_nonmanifold_edges'], 3)

    def test_abi_layout(self):
        for structure, expected in ((ReconDesc, 40), (PhysicsDesc, 52), (CutDesc, 60),
                                    (Grasper, 92), (GraspParams, 60), (Blade, 52)):
            self.assertEqual(ctypes.sizeof(structure), expected, structure.__name__)

    def test_connectivity_exhaustive(self):
        table = connectivity_table()
        # Independent graph traversal, including multiple cuts and disconnected corners.
        for mask in range(4096):
            visited, groups = set(), []
            for start in range(8):
                if start in visited:
                    continue
                group, pending = set(), [start]
                while pending:
                    node = pending.pop()
                    if node in group:
                        continue
                    group.add(node)
                    for bit, (a, b) in enumerate(EDGE_CORNERS):
                        if mask & (1 << bit) == 0:
                            if node == a:
                                pending.append(int(b))
                            elif node == b:
                                pending.append(int(a))
                groups.append(group)
                visited.update(group)
            self.assertEqual(table[mask, 0], len(groups))
            for label, group in enumerate(groups):
                self.assertTrue(all(table[mask, node + 1] == label for node in group))

    def test_grid_boundary_and_reciprocal_neighbors(self):
        grid = build_grid(0.6)
        self.assertEqual(grid.intersections.dtype.itemsize, 40)
        self.assertEqual(grid.bends.dtype.itemsize, 32)
        self.assertEqual(grid.counts.sum(), len(grid.intersections))
        self.assertTrue(np.any(grid.pinned))
        supported_cells = np.all(grid.active[grid.corners] != 0, axis=1)
        expected_mass = np.count_nonzero(supported_cells) * (grid.spacing * .01) ** 3 * 1000
        np.testing.assert_allclose(grid.mass.sum(), expected_mass, rtol=1e-6)
        self.assertTrue(np.all(grid.mass[grid.active == 0] == 0))
        for slot in range(6):
            nodes = np.flatnonzero(grid.neighbors[:, slot] >= 0)
            other = grid.neighbors[nodes, slot]
            np.testing.assert_array_equal(grid.neighbors[other, slot ^ 1], nodes)
            self.assertTrue(np.all(grid.active[nodes] & grid.active[other]))
        # Surface exactly on a lattice endpoint is valid (phi == 0 is outside).
        self.assertTrue(np.all((grid.intersections["t"] >= 0) & (grid.intersections["t"] <= 1)))
        distances = (np.abs(grid.intersections["position"]) - (3.0, 0.7, 2.0)).max(axis=1)
        np.testing.assert_allclose(distances, 0, atol=1e-6)

    def test_tab_surface_lies_on_box_union_and_hangs_free(self):
        half = np.array(BLOCK_SIZE) / 2
        for tab in (Tab((3.0, .3, -1.0), (4.2, .7, 1.0), attached=0), Tab((0.0, .7, 1.4), (2.5, 1.7, 1.8), attached=1)):
            with self.subTest(tab=tab):
                grid = build_grid(.2, density=1000, vertical_spacing=.05, tab=tab)
                lows = np.array((-half, tab.low), dtype=np.float32)
                highs = np.array((half, tab.high), dtype=np.float32)
                def distance(points):
                    centers, halves = (lows + highs) / 2, (highs - lows) / 2
                    return (np.abs(points[:, None, :] - centers) - halves).max(axis=2).min(axis=1)
                hit, normal = grid.intersections["position"], grid.intersections["normal"]
                np.testing.assert_allclose(distance(hit), 0, atol=1e-6)
                # Probing off the hit is ambiguous where faces meet; test each edge's own ends.
                a, b, inside_a = (grid.intersections[name] for name in ("a", "b", "inside_a"))
                interior = grid.positions[np.where(inside_a != 0, a, b)]
                exterior = grid.positions[np.where(inside_a != 0, b, a)]
                self.assertTrue(np.all(distance(interior) < 1e-6) and np.all(distance(exterior) > -1e-6))
                self.assertTrue(np.all(np.einsum("ij,ij->i", normal, exterior - interior) > 0))
                for axis, corner in product(range(3), (tab.low, tab.high)):
                    self.assertTrue(np.any(np.abs(hit[:, axis] - corner[axis]) < 1e-6))
                for slot in range(6):
                    nodes = np.flatnonzero(grid.neighbors[:, slot] >= 0)
                    np.testing.assert_array_equal(grid.neighbors[grid.neighbors[nodes, slot], slot ^ 1], nodes)
                pinned = grid.positions[grid.pinned != 0]
                np.testing.assert_allclose(pinned[:, 1], -.7, atol=1e-6)
                grown = (grid.active != 0) & (grid.positions[:, tab.attached] > half[tab.attached] + .1)
                self.assertGreater(np.count_nonzero(grown), 0)
                self.assertFalse(np.any(grid.pinned[grown]))
                supported_cells = np.all(grid.active[grid.corners] != 0, axis=1)
                np.testing.assert_allclose(grid.mass.sum(),
                                           np.count_nonzero(supported_cells) * np.prod(grid.cell_size * .01) * 1000,
                                           rtol=1e-5)
                inside = np.flatnonzero(grid.inside)
                rows = np.repeat(inside, 6)
                cols = grid.neighbors[inside].reshape(-1)
                keep = (cols >= 0) & (grid.inside[np.maximum(cols, 0)] != 0)
                graph = coo_matrix((np.ones(np.count_nonzero(keep)), (rows[keep], cols[keep])),
                                   shape=(len(grid.positions),) * 2)
                labels = connected_components(graph, directed=False)[1]
                self.assertEqual(len(np.unique(labels[inside])), 1)
        for tab in (Tab((3.0, .3, -1.0), (4.2, .7, 1.0), attached=1), Tab((0.0, .7, 1.4), (2.5, 1.7, 1.5), attached=1)):
            with self.assertRaises(ValueError):
                build_grid(.2, vertical_spacing=.05, tab=tab)

    def test_units_and_orientation(self):
        axes = to_genesis(np.eye(3))
        np.testing.assert_allclose(np.linalg.norm(axes, axis=1), 0.01)
        self.assertGreater(np.linalg.det(axes), 0)
        np.testing.assert_allclose(to_genesis(np.array([[0.0, 1.0, 0.0]])), ((0, 0, 0.01),))

    def test_depth_refinement_preserves_rest_lengths_and_mass(self):
        grid = build_grid(.2, vertical_spacing=.05)
        for slot in range(6):
            ids = np.flatnonzero(grid.neighbors[:, slot] >= 0)
            delta = grid.positions[grid.neighbors[ids, slot]] - grid.positions[ids]
            np.testing.assert_allclose(grid.rest_neighbors[ids, slot], delta, atol=3e-7)
        active = grid.positions[grid.active != 0]
        envelope_volume_m3 = np.prod((active.max(axis=0) - active.min(axis=0)) * .01)
        np.testing.assert_allclose(grid.mass.sum(), envelope_volume_m3 * 1000, rtol=2e-6)
        np.testing.assert_allclose((np.abs(grid.intersections["position"]) - (3, .7, 2)).max(axis=1),
                                   0, atol=1e-6)

    def test_knife_back_display_and_collision_are_identical(self):
        geometry = knife_geometry(1.6)
        center = np.array((.15, .7, -.6))
        for axis, frame in (((0, 0, 1), ((0, 0, 1), (0, 1, 0), (-1, 0, 0))),
                            ((1, 0, 0), ((1, 0, 0), (0, 1, 0), (0, 0, 1)))):
            mesh = geometry.blade_mesh(center, axis)
            local = (mesh.vertices - center) @ np.array(frame).T
            np.testing.assert_allclose(local.min(axis=0), geometry.boxes[0, :3] - geometry.boxes[0, 3:6], atol=1e-7)
            np.testing.assert_allclose(local.max(axis=0), geometry.boxes[0, :3] + geometry.boxes[0, 3:6], atol=1e-7)
        self.assertLess(geometry.boxes[0, 3], .8)

    def test_withdrawal_starts_at_accepted_pose(self):
        obs = Observation(np.zeros((2, 3)), np.full((2, 6), -1), np.ones(2), np.zeros(3),
                          .05, 1, 0., np.array((.15, -.1, -.6, 0, 0, 0, 0, 1, 0, 0)), np.zeros(3))
        old = Command(blade_center=(.15, -.15, -.6))
        start = stage_start("withdraw", old, obs)
        step = interpolate(start, Command(blade_center=(.15, 2.2, -.6)), 1 / 180)
        self.assertGreater(step.blade_center[1], obs.blade[1])
        self.assertEqual(stage_start("incise", old, obs), old)
        turned = interpolate(Command(), Command(scissors_pitch_deg=0, scissors_roll_deg=90, scissors_yaw_deg=-40), .5)
        np.testing.assert_allclose((turned.scissors_pitch_deg, turned.scissors_roll_deg, turned.scissors_yaw_deg),
                                   (Command().scissors_pitch_deg / 2, 45, -20))

    def test_cut_count_undirected(self):
        baseline = np.full((2, 6), -1, dtype=np.int32)
        baseline[0, 1], baseline[1, 0] = 1, 0
        self.assertEqual(severed_edges(baseline, np.full_like(baseline, -1)), 1)

    def test_visual_refinement_preserves_separated_cut_faces(self):
        vertices = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0),
                             (0, 0, 0.2), (1, 0, 0.2), (0, 1, 0.2)), dtype=np.float32)
        normals = np.tile((0, 0, 1), (6, 1)).astype(np.float32)
        faces = np.array(((0, 1, 2), (3, 4, 5)), dtype=np.int32)
        anchors = np.column_stack((vertices, np.ones(6))).astype(np.float32)
        fine = refine_surface(Surface(vertices, normals, faces, anchors), 4)
        self.assertEqual(len(fine.triangles), 32)
        np.testing.assert_allclose(fine.vertices[:15, 2], 0)
        np.testing.assert_allclose(fine.vertices[15:, 2], .2)
        self.assertTrue(np.all(fine.triangles[:16] < 15))
        self.assertTrue(np.all(fine.triangles[16:] >= 15))
        # Smooth shading must not move the surface into a touching instrument.
        normals[:3] = ((-.4, 0, 1), (.4, 0, 1), (0, .4, 1))
        anchors[:, 3] = 0
        fine = refine_surface(Surface(vertices, normals, faces, anchors), 4)
        np.testing.assert_allclose(fine.vertices[:15, 2], 0, atol=1e-7)
        np.testing.assert_allclose(fine.vertices[15:, 2], .2, atol=1e-7)

    def test_visible_jaws_match_contact_pads(self):
        geometry = load_grasper(1.2, .4)
        hinge = np.array((.5, 1.1, -.7))
        axis, normal = np.array((0, -.8, .6)), np.array((1., 0, 0))
        for gap in (0, .5, 1.0):
            for side, vertices, mask, bounds in zip((1, -1), geometry.posed_jaws(hinge, axis, normal, gap),
                                                   geometry.pad_masks, geometry.bounds):
                angle = np.arcsin(gap / (2 * geometry.length))
                rotation = np.stack((axis * np.cos(angle) + side * normal * np.sin(angle),
                                     normal * np.cos(angle) - side * axis * np.sin(angle),
                                     np.cross(axis, normal)))
                local = (vertices[mask] - hinge) @ rotation.T
                np.testing.assert_allclose(local.min(axis=0), bounds[:3] - bounds[3:], atol=1e-7)
                np.testing.assert_allclose(local.max(axis=0), bounds[:3] + bounds[3:], atol=1e-7)
        self.assertGreaterEqual(geometry.bounds[0, 1] - geometry.bounds[0, 4], 0)
        self.assertLessEqual(geometry.bounds[1, 1] + geometry.bounds[1, 4], 0)

    def test_scissor_edges_share_one_shear_face_and_lead_their_backs(self):
        geometry = load_scissors()
        np.testing.assert_allclose(geometry.edges[..., 2], geometry.edges[0, 0, 2])
        for name, scene in SCISSORS_SCENES.items():
            with self.subTest(scene=name):
                axis = scene.plane_axis
                for stage in scene.stages:
                    command = stage.command
                    width = np.cross(*scissors_frame(command.scissors_pitch_deg, command.scissors_roll_deg,
                                                     command.scissors_yaw_deg))
                    np.testing.assert_allclose(np.linalg.norm(width), 1, atol=1e-6)
                    np.testing.assert_allclose(command.blade_center[axis] + geometry.edges[0, 0, 2] * width[axis],
                                               scene.plane + scene.face_offset, atol=5e-4)
                grid = build_grid(SCISSORS_SPACING, vertical_spacing=scene.vertical_spacing, tab=scene.tab)
                layers = np.unique(grid.positions[:, axis])
                np.testing.assert_allclose(np.abs(layers - scene.plane).min(), grid.cell_size[axis] / 2, atol=1e-5)
        np.testing.assert_allclose(geometry.max_gap, 2 * geometry.jaws.length * np.sin(np.radians(40)), rtol=1e-4)
        self.assertLessEqual(geometry.max_gap, 1.7321 * geometry.jaws.length)
        along = np.linspace(0, 1, 30)[:, None]
        for side, vertices, faces, edge, back in zip((1, -1), geometry.jaws.vertices, geometry.jaws.faces,
                                                     geometry.edges, geometry.backs):
            points = edge[0] + along * (edge[1] - edge[0])
            # The flat shear face departs from the curved blades by up to 1 mm in width.
            self.assertLess(trimesh.proximity.closest_point(trimesh.Trimesh(vertices, faces, process=False), points)[1].max(),
                            .11)
            basis = np.stack((back[6:9], back[9:12], np.cross(back[6:9], back[9:12])))
            clearance = (np.abs((points - back[:3]) @ basis.T) - back[3:6]).max(axis=1)
            self.assertGreaterEqual(clearance.min(), SCISSORS_BACK_CLEARANCE - 1e-6)
            # Blunt backs face away from the other blade; each back spans its own blade's side of the shear face.
            self.assertGreater(side * (back[1] - edge[:, 1].mean()), 0)
            corners = back[:3] + (np.array(list(product((-1, 1), repeat=3))) * back[3:6]) @ basis
            slope = (edge[1, 2] - edge[0, 2]) / (edge[1, 0] - edge[0, 0])
            offset = corners[:, 2] - edge[0, 2] - (corners[:, 0] - edge[0, 0]) * slope
            self.assertTrue(np.all(side * offset >= -1e-6))
            self.assertGreater((side * offset).max(), .1)

    def test_scissor_body_overlap_ignores_far_samples(self):
        geometry = load_scissors()
        hinge = np.array((1, 2, 3), dtype=np.float32)
        axis, normal = (v.astype(np.float32) for v in scissors_frame(15, 30, -40))
        gap = .6
        sn = gap / (2 * geometry.jaws.length)
        cs = np.sqrt(1 - sn * sn)
        width = np.cross(axis, normal)
        frame = np.stack((axis * cs + normal * sn, normal * cs - axis * sn, width))
        # Two wide triangles across the axis, each with only its centroid inside a part: the centre of the first
        # blade's back, and the shaft axis 4 cm behind the hinge. Their vertices and edge midpoints lie far outside,
        # the first triangle within 0.2 mm of its plane across the width.
        wide = np.array(((0, 9, 0), (0, -4.5, 7.8), (0, -4.5, -7.8)))
        basis = np.stack((axis, normal, width))
        vertices = np.concatenate((hinge + geometry.backs[0, :3] @ frame + (wide * (1, 1, .0025)) @ basis,
                                   hinge - 4 * axis + wide @ basis)).astype(np.float32)
        triangles = np.arange(6).reshape(2, 3)
        near = scissors_body_depths(vertices, triangles, geometry, hinge, axis, normal, gap)
        np.testing.assert_allclose(near, (geometry.backs[0, 3:6].min(), geometry.jaws.shaft_radius), atol=1e-4)
        # Small triangles and loose vertices 8 to 40 cm from the hinge, beyond the 6 cm shaft, listed first.
        rng = np.random.default_rng(0)
        directions = rng.normal(size=(3000, 3))
        centres = hinge + directions / np.linalg.norm(directions, axis=1, keepdims=True) * rng.uniform(8, 40, (3000, 1))
        far = (centres[:, None] + rng.uniform(-.1, .1, (3000, 3, 3))).reshape(-1, 3)
        mixed = np.concatenate((far, centres, vertices)).astype(np.float32)
        mixed_triangles = np.concatenate((np.arange(len(far)).reshape(-1, 3), triangles + len(far) + len(centres)))
        self.assertEqual(scissors_body_depths(mixed, mixed_triangles, geometry, hinge, axis, normal, gap), near)

    def test_scissor_snips_need_two_loaded_blades_after_squeeze_and_cut_the_wound(self):
        # Per scene, two columns inside the cut box and one outside it, each with a
        # link one cell below the shear plane and one across it. The rows stay
        # linked through the outer column, so severing each inner column once
        # keeps one component, at whichever of its links the cut follows.
        surface = Surface(np.array(((3.5, .5, .1), (3.7, .3, .1), (3.6, .6, .1)), dtype=np.float32),
                          np.zeros((3, 3)), np.array(((0, 1, 2),)), np.ones((3, 4)))
        settings = scissors_settings()
        for (scene, columns, base, cell), case in product(
                ((FLAP_SCENE, (3.1, 2.9, 3.6), (0, .5, 0), .1), (WALL_SCENE, (1.6, .5, 2.4), (0, 0, 1.8), .1)),
                ("valid", "one_blade_light", "cut_before_fracture", "no_squeeze", "wound_left_uncut",
                 "column_cut_twice", "stray_skin_cut")):
            slot = 2 * scene.plane_axis + 1
            rest = np.tile(np.array(base, dtype=np.float32), (9, 1))
            rest[:, 0] = np.repeat(columns, 3)
            rest[:, scene.plane_axis] = np.tile(scene.plane + cell * np.array((-1.5, -.5, .5)), 3)
            neighbors = np.full((9, 6), -1, dtype=np.int32)
            for node in range(9):
                if node % 3 < 2:
                    neighbors[node, slot], neighbors[node + 1, slot - 1] = node + 1, node
                if node < 6:
                    neighbors[node, 1], neighbors[node + 3, 0] = node + 3, node

            def sever(links, *nodes):
                for node in nodes:
                    links[node, slot] = links[node + 1, slot - 1] = -1
                return links

            # The first snip severs the first column across the plane, the second the
            # last column one layer lower, as a cut following deformed tissue can.
            first, both_cut = sever(neighbors.copy(), 1), sever(neighbors.copy(), 1, 6)
            with self.subTest(tab=scene.tab, case=case):
                def observe(links, separation, left, right, fractures, fracture_right=.105):
                    hinge, axis = rest[7] - scene.advance * np.array((.9, 0, 0)), np.array((scene.advance, 0, 0.))
                    edges = np.array([[hinge + .4 * axis + (0, side * separation / 2, 0),
                                       hinge + 1.2 * axis + (0, side * separation / 2, 0)] for side in (1, -1)])
                    scissors = ScissorsPose(hinge, axis, np.array((0, 1., 0)), .5, .01, edges, left + right, 0, 2,
                                            ScissorsLoads(left, right, .11 * bool(fractures),
                                                          fracture_right * bool(fractures), .2, fractures,
                                                          min(left, right) > 0))
                    return Observation(rest + (0, .2, 0), links, np.ones(9), hinge, .05, 1, 0., np.zeros(10),
                                       np.zeros(3), scissors)
                obs = observe(neighbors.copy(), 1.0, 0, 0, 0)
                obs.nodes[:] = rest
                inside = np.ones(9)
                if case == "stray_skin_cut":
                    inside[3:6] = 0
                sequence = Sequence(obs, settings, scene, rest, inside)
                sequence.finish_stage("close", obs, surface)
                lifted = observe(neighbors.copy(), 1.0, 0, 0, 0)
                sequence.finish_stage("lift", lifted, surface)
                sequence.finish_stage("blade_approach", lifted, surface)
                # The second snip carries each defect; the first one is valid.
                both = 0. if case == "no_squeeze" else .05
                # A stray skin cut severs the outer column, here outside the material, 2 mm and more from the wound.
                second = {"wound_left_uncut": first, "column_cut_twice": sever(both_cut.copy(), 7),
                          "stray_skin_cut": sever(both_cut.copy(), 4)}.get(case, both_cut)
                for stage, before, after, squeezing, early, fractures in (
                        ("incise_1", neighbors, first, .05, neighbors, 0),
                        ("incise_2", first, second, both, second if case == "cut_before_fracture" else first, 1)):
                    for links, separation, left, right, count in (
                            (before, .5, .15, 0, fractures), (before, .4, .05, squeezing, fractures),
                            (early, .3, .05, squeezing, fractures), (after, .25, .05, .05, fractures + 1)):
                        light = case == "one_blade_light" and stage == "incise_2"
                        obs = observe(links.copy(), separation, left, right, count, .05 if light else .105)
                        sequence.record_step(stage, obs, 0.)
                    sequence.finish_stage(stage, obs, surface)
                    for idle in (("withdraw_1", "release_1", "reposition_2") if stage == "incise_1" else ("withdraw", "release")):
                        sequence.record_step(idle, obs, 0.)
                        sequence.finish_stage(idle, obs, surface)
                if case == "valid":
                    sequence.finish_stage("hold", obs, surface)
                    m = sequence.metrics
                    self.assertTrue(m.acceptance_passed)
                    self.assertEqual([snip.stage for snip in m.snips], ["incise_1", "incise_2"])
                    np.testing.assert_allclose(m.two_blade_squeeze_m, .002, atol=1e-6)
                    self.assertEqual((m.material_components, m.uncut_columns, m.multiply_cut_columns), (1, 0, 0))
                else:
                    with self.assertRaisesRegex(RuntimeError, "two-blade compression"):
                        sequence.finish_stage("hold", obs, surface)
                    m = sequence.metrics
                    culprit = {"one_blade_light": m.fracture_right_n < .1, "cut_before_fracture": m.edges_cut_without_fracture > 0,
                               "no_squeeze": m.two_blade_squeeze_m < .0005, "wound_left_uncut": m.uncut_columns > 0,
                               "column_cut_twice": m.multiply_cut_columns > 0, "stray_skin_cut": m.stray_skin_cuts > 0}
                    self.assertEqual([name for name, failed in culprit.items() if failed], [case])

    def test_friction_grasp_follows_the_pressed_bite(self):
        settings = replace(scissors_settings(), contact_model=1)
        axis = np.array(settings.jaw_axis) / np.linalg.norm(settings.jaw_axis)
        frame = np.stack((axis, settings.jaw_normal, np.cross(axis, settings.jaw_normal)))
        hinge = np.array((1.3, 2.9, 1.8))
        # Jaw-frame points (axis, opening normal, width): three material nodes between the pads of jaws 1 mm open at
        # the tips, one beyond the tips, one outside a pad face, and a shell node between the pads.
        local = np.array(((1.8, 0, 0), (1.9, .03, .1), (1.7, -.03, -.2), (2.1, 0, 0), (1.8, .1, 0), (1.8, 0, .1)))
        inside = np.array((1, 1, 1, 1, 1, 0))
        surface = Surface(np.array(((3.5, .5, .1), (3.7, .3, .1), (3.6, .6, .1)), dtype=np.float32),
                          np.zeros((3, 3)), np.array(((0, 1, 2),)), np.ones((3, 4)))
        blade_hinge, blade_axis = np.array((5, 1, 2.6)), np.array((-1., 0, 0))
        edges = np.array([[blade_hinge + reach * blade_axis + (0, side * .1, 0) for reach in (.4, 1.2)]
                          for side in (1, -1)])
        scissors = ScissorsPose(blade_hinge, blade_axis, np.array((0, 1., 0)), .5, .01, edges, 0, 0, 0,
                                ScissorsLoads(0, 0, 0, 0, 0, 0, False))

        def observe(points, gap, offset=(0, 0, 0), loads=(.2, .6)):
            return Observation(hinge + offset + points @ frame, np.full((6, 6), -1), np.zeros(6), hinge + offset,
                               gap, 0, 0., np.zeros(10), np.zeros(3), scissors, loads)

        rest = observe(local, 1.).nodes
        # Three nodes lie between the pads, yet tissue pressing one pad only, or both with the light touch of jaws
        # closing onto it, is no bite.
        for loads in ((0, .6), (.02, .02)):
            sequence = Sequence(observe(local, 1.), settings, WALL_SCENE, rest, inside)
            with self.assertRaisesRegex(RuntimeError, "no material was captured"):
                sequence.finish_stage("close", observe(local, .1, loads=loads), surface)
            self.assertEqual(sequence.metrics.bite_nodes, 3)
        sequence = Sequence(observe(local, 1.), settings, WALL_SCENE, rest, inside)
        sequence.finish_stage("close", observe(local, .1), surface)
        m = sequence.metrics
        self.assertTrue(m.grasp_acquired)
        self.assertEqual(m.bite_nodes, 3)
        lifted = (0, .2, 0)
        sequence.finish_stage("lift", observe(local, .1, lifted), surface)
        np.testing.assert_allclose(m.lift_m, .002, atol=1e-6)
        # Squeezing the bite thinner moves it across the pads only.
        sequence.record_step("lift", observe(local * (1, .5, 1), .05, lifted), 0.)
        np.testing.assert_allclose(m.grip_slip_m, 0, atol=1e-9)
        self.assertEqual((m.held_bite_nodes, m.grip_lost_after_capture), (3, False))
        # Sliding 1.5 mm toward the tips takes one bite node past them.
        sequence.record_step("hold", observe(local + (.15, 0, 0), .1, lifted), 0.)
        np.testing.assert_allclose(m.grip_slip_m, .0015, atol=1e-6)
        self.assertEqual((m.held_bite_nodes, m.grip_lost_after_capture), (2, False))
        # At 2 mm two of them are past the tips: the median bite node has left the pads.
        sequence.record_step("hold", observe(local + (.2, 0, 0), .1, lifted), 0.)
        self.assertEqual((m.held_bite_nodes, m.grip_lost_after_capture), (1, True))

    def test_completed_stroke_does_not_hide_bad_contact_or_cut_geometry(self):
        neighbors = np.full((2, 6), -1, dtype=np.int32)
        neighbors[0, 1], neighbors[1, 0] = 1, 0
        vertices = np.array(((0., 0, 0), (1, 0, 0), (0, 1, 0)))
        normals = np.tile((0., 0, 1), (3, 1))
        faces = np.array(((0, 1, 2),), dtype=np.int32)
        for case in ("valid", "body_overlap", "zero_area_wall"):
            with self.subTest(case=case):
                obs = Observation(np.zeros((2, 3)), neighbors.copy(), np.ones(2), np.zeros(3),
                                  .05, 1, 0., np.zeros(10), np.zeros(3))
                surface = Surface(vertices.copy(), normals, faces, np.zeros((3, 4)))
                sequence = Sequence(obs)
                sequence.finish_stage("close", obs, surface)
                obs.nodes[:, 1] += .2
                sequence.finish_stage("lift", obs, surface)
                sequence.finish_stage("blade_approach", obs, surface)
                obs.neighbors[:] = -1
                surface.anchors[:, 3] = 1
                if case == "zero_area_wall":
                    surface.vertices[2] = surface.vertices[1]
                sequence.record_step("incise", obs, .0004 if case == "body_overlap" else 0.)
                sequence.finish_stage("incise", obs, surface)
                if case == "valid":
                    sequence.finish_stage("hold", obs, surface)
                    self.assertTrue(sequence.metrics.acceptance_passed)
                else:
                    with self.assertRaisesRegex(RuntimeError, "Interaction quality failed"):
                        sequence.finish_stage("hold", obs, surface)
                    self.assertFalse(sequence.metrics.acceptance_passed)
        obs.nodes[:, 1] += 7
        with self.assertRaisesRegex(RuntimeError, "60 mm operating envelope"):
            sequence.record_step("hold", obs, 0.)


if __name__ == "__main__":
    unittest.main()
