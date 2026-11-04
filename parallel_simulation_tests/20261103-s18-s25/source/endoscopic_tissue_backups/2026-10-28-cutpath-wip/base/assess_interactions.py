"""Repeat actual native interactions and retain independent physical evidence.

This is an assessment harness, not a change to the scene's physical parameters.
Run in a separate process; never connects to an existing Unity session.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .grid import METERS_PER_UNIT, build_grid
from .native_backend import Command, NativeTissue, Observation, Settings, Surface
from .scenario import STAGES, drive_command, interpolate, stage_start


@dataclass
class Assessment:
    name: str
    spacing_mm: float
    vertical_spacing_mm: float = .5
    status: str = "incomplete"
    stopped_stage: str = "initial"
    failure: str | None = None
    steps: int = 0
    release_steps: int = 0
    elapsed_s: float = 0.0
    captured_surface_patches: int = 0
    tracked_material_nodes: int = 0
    lift_median_mm: float = 0.0
    lift_min_mm: float = 0.0
    lift_max_mm: float = 0.0
    grip_lost_after_capture: bool = False
    max_grip_error_mm: float = 0.0
    unexpected_cuts_before_incision: int = 0
    cut_edges_at_incision_end: int = 0
    observed_cut_edges: int = 0
    cut_edges_after_withdrawal: int = 0
    healed_directed_connections: int = 0
    asymmetric_cut_connections: int = 0
    cut_wall_triangles: int = 0
    degenerate_wall_triangles: int = 0
    max_degenerate_wall_triangles: int = 0
    new_edges_during_withdrawal: int = 0
    blade_y_at_incision_end_mm: float = 0.0
    blade_y_after_withdrawal_mm: float = 0.0
    max_surface_triangle_edge_mm: float = 0.0
    wall_bounds_mm: list | None = None
    blade_peak_force_n: float = 0.0
    max_surface_penetration_mm: float = 0.0
    max_knife_body_penetration_mm: float = 0.0
    max_node_displacement_mm: float = 0.0
    max_exterior_displacement_mm: float = 0.0
    max_interior_displacement_mm: float = 0.0
    material_components_before: int = 0
    material_components_after: int | None = None
    release_checked: bool = False
    remaining_grip_patches_after_release: int | None = None
    tracked_nodes_drop_after_release_mm: float | None = None


def component_sizes(neighbors, selected):
    """Connected components of stable original interior material IDs."""
    visited = np.zeros(len(selected), dtype=bool)
    sizes = []
    for start in np.flatnonzero(selected):
        if visited[start]:
            continue
        visited[start] = True
        pending, size = [int(start)], 0
        while pending:
            current = pending.pop()
            size += 1
            for neighbor in neighbors[current]:
                if neighbor >= 0 and selected[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    pending.append(int(neighbor))
        sizes.append(size)
    return sorted(sizes, reverse=True)


def save_state(path: Path, obs: Observation, surface: Surface):
    np.savez_compressed(path, nodes_cm=obs.nodes, neighbors=obs.neighbors, grip_flags=obs.grip_flags,
                        hinge_cm=obs.hinge, jaw_gap_cm=obs.jaw_gap, blade=obs.blade,
                        grip_points=obs.grip_points, grip_error_m=obs.grip_error_m,
                        reaction_n=obs.reaction_n, vertices_cm=surface.vertices,
                        normals=surface.normals, faces=surface.triangles, anchors=surface.anchors)


def assess(output: Path, spacing, release=False, settings=Settings(), vertical_spacing=.05):
    output.mkdir(parents=True, exist_ok=False)
    result = Assessment(output.name, spacing * 10, vertical_spacing * 10)
    (output / "settings.json").write_text(json.dumps(asdict(settings), indent=2), encoding="utf8")
    (output / "stages.json").write_text(json.dumps([asdict(stage) for stage in STAGES], indent=2), encoding="utf8")
    grid = build_grid(spacing, density=settings.density_kg_m3, vertical_spacing=vertical_spacing)
    (output / "grid.json").write_text(json.dumps({"cell_size_cm": grid.cell_size.tolist(), "dims": grid.dims}), encoding="utf8")
    started = time.monotonic()
    contact_log = []
    try:
        with NativeTissue(grid, settings) as tissue:
            obs = tissue.observe()
            baseline = obs.neighbors.copy()
            previous_neighbors = baseline.copy()
            interior = grid.inside != 0
            result.material_components_before = len(component_sizes(baseline, interior))
            previous = Command()
            tracked = np.zeros(len(grid.positions), dtype=bool)
            captured_nodes = obs.nodes.copy()
            captured = False
            for stage in STAGES:
                result.stopped_stage = stage.name
                previous = stage_start(stage.name, previous, obs)
                print(f"{output.name}: {stage.name}", flush=True)
                for step in range(stage.steps):
                    requested = interpolate(previous, stage.command, (step + 1) / stage.steps)
                    obs = tissue.step(drive_command(requested, obs, settings))
                    result.steps += 1
                    depth_mm = tissue.contact_depth_m() * 1000
                    knife_depth_mm = tissue.knife_body_depth_m() * 1000
                    result.max_surface_penetration_mm = max(result.max_surface_penetration_mm, depth_mm)
                    result.max_knife_body_penetration_mm = max(result.max_knife_body_penetration_mm, knife_depth_mm)
                    displacement = np.linalg.norm(obs.nodes - grid.positions, axis=1) * 10
                    result.max_node_displacement_mm = max(result.max_node_displacement_mm, float(displacement.max()))
                    exterior = (grid.active != 0) & (grid.inside == 0)
                    result.max_exterior_displacement_mm = max(result.max_exterior_displacement_mm,
                                                               float(displacement[exterior].max()))
                    result.max_interior_displacement_mm = max(result.max_interior_displacement_mm,
                                                               float(displacement[interior].max()))
                    result.observed_cut_edges = int(np.count_nonzero(
                        (baseline[:, 1::2] >= 0) & (obs.neighbors[:, 1::2] < 0)))
                    if stage.name in ("withdraw", "hold"):
                        result.new_edges_during_withdrawal = max(result.new_edges_during_withdrawal,
                            result.observed_cut_edges - result.cut_edges_at_incision_end)
                    if result.observed_cut_edges:
                        cut_surface = tissue.surface()
                        tri = cut_surface.vertices[cut_surface.triangles]
                        wall = cut_surface.anchors[cut_surface.triangles[:, 0], 3] > .5
                        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
                        result.max_degenerate_wall_triangles = max(result.max_degenerate_wall_triangles,
                                                                  int(np.count_nonzero(wall & (area < 1e-8))))
                        result.max_surface_triangle_edge_mm = max(result.max_surface_triangle_edge_mm,
                            float(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max()) * 10)
                    contact_log.append({"step": result.steps, "stage": stage.name,
                                        "penetration_mm": depth_mm, "grip_points": obs.grip_points,
                                        "jaw_gap_mm": obs.jaw_gap * 10, "knife_body_penetration_mm": knife_depth_mm,
                                        "cut_edges": result.observed_cut_edges, "blade_center_cm": obs.blade[:3].tolist(),
                                        "blade_phase": int(obs.blade[7])})
                    if result.max_node_displacement_mm > 60:
                        save_state(output / "failed.npz", obs, tissue.surface())
                        raise RuntimeError("Material left the 60 mm operating envelope")
                    result.healed_directed_connections += int(np.count_nonzero(
                        (baseline >= 0) & (previous_neighbors < 0) & (obs.neighbors >= 0)))
                    previous_neighbors = obs.neighbors.copy()
                    if captured:
                        result.grip_lost_after_capture |= obs.grip_points == 0
                        result.max_grip_error_mm = max(result.max_grip_error_mm, obs.grip_error_m * 1000)
                    result.blade_peak_force_n = max(result.blade_peak_force_n,
                                                    float(np.linalg.norm(obs.blade[3:6]) * METERS_PER_UNIT))
                surface = tissue.surface()
                save_state(output / f"{stage.name}.npz", obs, surface)
                if stage.name == "close":
                    tracked = obs.grip_flags != 0
                    captured_nodes = obs.nodes.copy()
                    result.captured_surface_patches = obs.grip_points
                    result.tracked_material_nodes = int(np.count_nonzero(tracked))
                    captured = obs.grip_points > 0 and np.any(tracked)
                    if not captured:
                        raise RuntimeError("No material captured at the end of closure")
                if stage.name == "lift":
                    lift = (obs.nodes[tracked, 1] - captured_nodes[tracked, 1]) * 10
                    result.lift_median_mm = float(np.median(lift))
                    result.lift_min_mm, result.lift_max_mm = float(lift.min()), float(lift.max())
                    if result.lift_median_mm < 1:
                        raise RuntimeError("Tracked material failed to rise by 1 mm")
                if stage.name == "blade_approach":
                    result.unexpected_cuts_before_incision = int(np.count_nonzero(
                        (baseline[:, 1::2] >= 0) & (obs.neighbors[:, 1::2] < 0)))
                if stage.name == "incise":
                    result.blade_y_at_incision_end_mm = float(obs.blade[1] * 10)
                    result.cut_edges_at_incision_end = int(np.count_nonzero(
                        (baseline[:, 1::2] >= 0) & (obs.neighbors[:, 1::2] < 0)))
                previous = stage.command

            removed = (baseline >= 0) & (obs.neighbors < 0)
            result.cut_edges_after_withdrawal = int(np.count_nonzero(removed[:, 1::2]))
            result.blade_y_after_withdrawal_mm = float(obs.blade[1] * 10)
            for slot in (1, 3, 5):
                owner = np.flatnonzero(removed[:, slot])
                other = baseline[owner, slot]
                result.asymmetric_cut_connections += int(np.count_nonzero(obs.neighbors[other, slot ^ 1] >= 0))
            result.material_components_after = len(component_sizes(obs.neighbors, interior))
            wall = surface.anchors[surface.triangles[:, 0], 3] > .5
            triangles = surface.vertices[surface.triangles[wall]]
            result.cut_wall_triangles = len(triangles)
            if len(triangles):
                twice_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                                    triangles[:, 2] - triangles[:, 0]), axis=1)
                result.degenerate_wall_triangles = int(np.count_nonzero(twice_area < 1e-8))
                result.wall_bounds_mm = (np.array((triangles.min(axis=(0, 1)),
                                                  triangles.max(axis=(0, 1)))) * 10).tolist()
            if release:
                result.release_checked = True
                before_release = obs.nodes.copy()
                opened = Command(previous.hinge, 0, previous.blade_center)
                for step in range(30):
                    obs = tissue.step(interpolate(previous, opened, (step + 1) / 30))
                    result.release_steps += 1
                for _ in range(30):
                    obs = tissue.step(opened)
                    result.release_steps += 1
                result.remaining_grip_patches_after_release = obs.grip_points
                result.tracked_nodes_drop_after_release_mm = float(np.median(
                    before_release[tracked, 1] - obs.nodes[tracked, 1]) * 10)
                save_state(output / "released.npz", obs, tissue.surface())
            checks = (("lift", result.lift_median_mm >= 1), ("grip_retained", not result.grip_lost_after_capture),
                ("no_early_cut", result.unexpected_cuts_before_incision == 0),
                ("cut", result.cut_edges_at_incision_end > 0),
                ("no_healing", result.healed_directed_connections == 0),
                ("symmetric_edges", result.asymmetric_cut_connections == 0),
                ("cut_wall", result.cut_wall_triangles > 0),
                ("nondegenerate_wall", result.max_degenerate_wall_triangles == 0),
                ("withdrawal_does_not_cut", result.new_edges_during_withdrawal == 0),
                ("knife_body_contact", result.max_knife_body_penetration_mm <= .2),
                ("material_in_workspace", result.max_node_displacement_mm <= 60),
                ("release", not release or result.remaining_grip_patches_after_release == 0))
            failures = [name for name, passed in checks if not passed]
            result.status = "failed" if failures else "passed"
            result.failure = ", ".join(failures) if failures else None
    except (RuntimeError, ValueError, OSError) as ex:
        result.status, result.failure = "failed", str(ex)
    finally:
        result.elapsed_s = time.monotonic() - started
        (output / "contacts.jsonl").write_text("\n".join(json.dumps(row) for row in contact_log), encoding="utf8")
        (output / "assessment.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf8")
        print(f"{result.name}: {result.status}, lift={result.lift_median_mm:.3f} mm, "
              f"cut={result.cut_edges_after_withdrawal}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--spacing", type=float, default=.2)
    parser.add_argument("--vertical-spacing", type=float, default=.05)
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--single-spacing", type=float)
    parser.add_argument("--isolated", action="store_true", help="Start each repetition in a fresh Python process")
    parser.add_argument("--refine-passes", type=int, default=8, choices=range(9))
    parser.add_argument("--without-knife-body", action="store_true", help="Diagnostic ablation of knife body contact")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.single_spacing is not None:
        result = assess(args.output, args.single_spacing, release=args.release, vertical_spacing=args.vertical_spacing,
                        settings=Settings(contact_refine_passes=args.refine_passes, has_knife_body=not args.without_knife_body))
        if result.status != "passed":
            raise SystemExit(1)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    if args.isolated:
        results = []
        for i in range(args.repeats):
            output = args.output / f"baseline-{i + 1}"
            completed = subprocess.run([sys.executable, "-m", "endoscopic_tissue.assess_interactions",
                            "--output", str(output), "--single-spacing", str(args.spacing),
                            "--refine-passes", str(args.refine_passes), "--vertical-spacing", str(args.vertical_spacing)] +
                           (["--release"] if args.release else []) +
                           (["--without-knife-body"] if args.without_knife_body else []), check=False)
            if completed.returncode not in (0, 1):
                raise subprocess.CalledProcessError(completed.returncode, completed.args)
            results.append(Assessment(**json.loads((output / "assessment.json").read_text(encoding="utf8"))))
    else:
        settings = Settings(contact_refine_passes=args.refine_passes, has_knife_body=not args.without_knife_body)
        results = [assess(args.output / f"baseline-{i + 1}", args.spacing, release=args.release or i == 0,
                          settings=settings, vertical_spacing=args.vertical_spacing)
                   for i in range(args.repeats)]
    summary = {"cases": [asdict(result) for result in results],
               "session_mode": "fresh_process_per_case" if args.isolated else "reinitialize_in_same_process",
               "dll_sha256": hashlib.sha256((Path(__file__).parent / "native/bin/LiverCudaSim.dll").read_bytes()).hexdigest()}
    valid = [args.output / result.name / "hold.npz" for result in results
             if result.spacing_mm == args.spacing * 10 and result.status == "passed"]
    comparisons = []
    if len(valid) > 1:
        with np.load(valid[0]) as first:
            reference = first["nodes_cm"]
            for path in valid[1:]:
                with np.load(path) as later:
                    delta = np.linalg.norm(later["nodes_cm"] - reference, axis=1) * 10
                    comparisons.append({"reference": str(valid[0]), "case": str(path),
                                        "rms_grid_nodes_mm": float(np.sqrt(np.mean(delta ** 2))),
                                        "max_grid_node_error_mm": float(delta.max())})
    summary["repeat_comparison"] = comparisons
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    if any(result.status != "passed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
