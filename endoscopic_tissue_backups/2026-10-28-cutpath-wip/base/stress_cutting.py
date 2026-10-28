"""Exercise repeated incisions while retaining one grasp, with optional Genesis recording."""
import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import trimesh

from .assess_interactions import save_state
from .grid import build_grid
from .native_backend import Command, NativeTissue, Settings
from .rendering import EndoscopicView
from .probe_cut_topology import material_cut_edges
from .scenario import STAGES, Sequence, Stage, drive_command, interpolate, severed_edges, stage_start


@dataclass
class StrokeResult:
    number: int
    before_edges: int = 0
    incised_edges: int = 0
    after_edges: int = 0
    new_edges: int = 0
    before_material_edges: int = 0
    new_material_edges: int = 0
    withdrawal_edges: int = 0
    min_blade_y_mm: float = 100.0
    wall_triangles: int = 0
    boundary_edges: int = 0
    nonmanifold_edges: int = 0
    duplicate_triangles: int = 0
    winding_consistent: bool = True
    completed: bool = False


@dataclass
class StressResult:
    status: str = "running"
    failures: list[str] = field(default_factory=list)
    strokes: list[StrokeResult] = field(default_factory=list)
    steps: int = 0
    lift_mm: float = 0.0
    max_knife_depth_mm: float = 0.0
    max_node_displacement_mm: float = 0.0
    max_grip_error_mm: float = 0.0
    max_wall_degenerate: int = 0
    max_surface_degenerate: int = 0
    max_triangle_edge_mm: float = 0.0
    max_blade_force_n: float = 0.0
    min_volume_ratio: float = 1.0
    max_invalid_tets: int = 0
    healed_connections: int = 0
    grip_lost: bool = False
    elapsed_s: float = 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dll", type=Path)
    parser.add_argument("--yaw", type=float, default=0)
    parser.add_argument("--substeps", type=int, default=3)
    parser.add_argument("--pattern", choices=("deepen", "parallel", "retrace"), default="deepen")
    parser.add_argument("--cycles", type=int, default=3, choices=range(1, 4))
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--debug-cut", action="store_true")
    parser.add_argument("--render-every", type=int, default=3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    args = parser.parse_args()
    if not np.isfinite(args.yaw) or args.render_every < 1 or args.substeps < 1:
        parser.error("yaw must be finite; render-every and substeps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    angle = np.deg2rad(args.yaw)
    settings = Settings(substeps=args.substeps, blade_axis=(float(np.sin(angle)), 0.0, float(np.cos(angle))))
    grid = build_grid(.2, density=settings.density_kg_m3, vertical_spacing=.05)
    result = StressResult()
    view, video = None, None
    started = time.monotonic()
    dll = args.dll or Path(__file__).parent / "native/bin/LiverCudaSim.dll"
    dll_sha256 = hashlib.sha256(dll.read_bytes()).hexdigest()
    (args.output / "settings.json").write_text(json.dumps(asdict(settings), indent=2), encoding="utf8")
    plan = list(STAGES[:4])
    hinge = STAGES[3].command.hinge
    for cycle in range(args.cycles):
        x = .15 + (.6 * cycle if args.pattern == "parallel" else 0)
        bottom = (.15, -.15, -.35)[cycle] if args.pattern == "deepen" else -.15
        result.strokes.append(StrokeResult(cycle + 1))
        plan.extend((
            Stage(f"approach_{cycle + 1}", 90, Command(hinge, 1, (x, 1.05, -.6))),
            Stage(f"incise_{cycle + 1}", 100, Command(hinge, 1, (x, bottom, -.6))),
            Stage(f"withdraw_start_{cycle + 1}", 25, Command(hinge, 1, (x, bottom + .25, -.6))),
            Stage(f"pause_inside_{cycle + 1}", 12, Command(hinge, 1, (x, bottom + .25, -.6))),
            Stage(f"withdraw_end_{cycle + 1}", 155, Command(hinge, 1, (x, 2.2, -.6))),
            Stage(f"clear_{cycle + 1}", 20, Command(hinge, 1, (x, 2.2, -.6))),
        ))
    (args.output / "stages.json").write_text(json.dumps([asdict(stage) for stage in plan], indent=2), encoding="utf8")
    try:
        with NativeTissue(grid, settings, dll_path=dll) as tissue, (args.output / "trajectory.jsonl").open("w", encoding="utf8") as log:
            obs = tissue.observe()
            sequence = Sequence(obs)
            previous_neighbors = obs.neighbors.copy()
            previous = Command()
            if args.video:
                view = EndoscopicView(settings, args.width, args.height)
                video = cv2.VideoWriter(str(args.output / "endoscopic_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                        1 / (settings.dt * args.render_every), (args.width, args.height))
                if not video.isOpened():
                    raise RuntimeError("Could not create video")
            for stage in plan:
                role, _, suffix = stage.name.rpartition("_")
                stroke = result.strokes[int(suffix) - 1] if suffix.isdigit() else None
                if stroke is not None and role == "approach":
                    stroke.before_edges = int(severed_edges(sequence.initial_neighbors, obs.neighbors))
                    material = material_cut_edges(grid, tissue.cut_edge_fractions())
                    removed = (grid.neighbors >= 0) & (obs.neighbors < 0)
                    stroke.before_material_edges = int(np.count_nonzero((removed & material)[:, 1::2]))
                if role.startswith("withdraw"):
                    previous = stage_start("withdraw", previous, obs)
                target = stage.command
                if role == "withdraw_start":
                    target = Command(hinge, 1, tuple(obs.blade[:3] + (0, .25, 0)))
                elif role == "pause_inside":
                    previous = target = Command(hinge, 1, tuple(obs.blade[:3]))
                print(stage.name, flush=True)
                for step in range(stage.steps):
                    requested = interpolate(previous, target, (step + 1) / stage.steps)
                    obs = tissue.step(drive_command(requested, obs, settings))
                    surface = tissue.surface()
                    quality = tissue.material_quality()
                    result.min_volume_ratio = min(result.min_volume_ratio, float(quality[0]))
                    result.max_invalid_tets = max(result.max_invalid_tets, int(quality[1]))
                    result.steps += 1
                    count = int(severed_edges(sequence.initial_neighbors, obs.neighbors))
                    depth = tissue.knife_body_depth_m() * 1000
                    if depth > result.max_knife_depth_mm and depth > .05:
                        save_state(args.output / "peak-knife-overlap.npz", obs, surface)
                        (args.output / "peak-knife-overlap.json").write_text(json.dumps({
                            "step": result.steps, "stage": stage.name, "depth_mm": depth}, indent=2), encoding="utf8")
                    result.max_knife_depth_mm = max(result.max_knife_depth_mm, depth)
                    result.max_node_displacement_mm = max(result.max_node_displacement_mm,
                        float(np.linalg.norm(obs.nodes - grid.positions, axis=1).max()) * 10)
                    result.healed_connections += int(np.count_nonzero((sequence.initial_neighbors >= 0)
                        & (previous_neighbors < 0) & (obs.neighbors >= 0)))
                    previous_neighbors = obs.neighbors.copy()
                    if sequence.metrics.grasp_acquired:
                        result.grip_lost |= obs.grip_points == 0
                        result.max_grip_error_mm = max(result.max_grip_error_mm, obs.grip_error_m * 1000)
                    triangles = surface.vertices[surface.triangles]
                    wall = surface.anchors[surface.triangles[:, 0], 3] > .5
                    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                                  triangles[:, 2] - triangles[:, 0]), axis=1)
                    result.max_surface_degenerate = max(result.max_surface_degenerate, int(np.count_nonzero(area < 1e-8)))
                    result.max_wall_degenerate = max(result.max_wall_degenerate, int(np.count_nonzero(wall & (area < 1e-8))))
                    result.max_triangle_edge_mm = max(result.max_triangle_edge_mm,
                        float(np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2).max()) * 10)
                    result.max_blade_force_n = max(result.max_blade_force_n, float(np.linalg.norm(obs.blade[3:6])) * .01)
                    if stroke is not None:
                        stroke.min_blade_y_mm = min(stroke.min_blade_y_mm, float(obs.blade[1]) * 10)
                        if role.startswith("withdraw") or role in ("pause_inside", "clear"):
                            stroke.withdrawal_edges = max(stroke.withdrawal_edges, count - stroke.incised_edges)
                    log.write(json.dumps({"step": result.steps, "time_s": result.steps * settings.dt,
                        "stage": stage.name, "grip_points": obs.grip_points, "grip_error_mm": obs.grip_error_m * 1000,
                        "blade_center_cm": obs.blade[:3].tolist(), "phase": int(obs.blade[7]), "cut_edges": count,
                        "knife_depth_mm": depth, "min_volume_ratio": float(quality[0]), "invalid_tets": int(quality[1]),
                        "wall_degenerate": int(np.count_nonzero(wall & (area < 1e-8))),
                        "node_displacement_mm": result.max_node_displacement_mm}) + "\n")
                    if args.debug_cut and role == "incise" and step % 2 == 0:
                        save_state(args.output / f"debug-{result.steps:04d}.npz", obs, surface)
                    if result.max_node_displacement_mm > 60:
                        save_state(args.output / "failed.npz", obs, surface)
                        np.save(args.output / "cut-fractions.npy", tissue.cut_edge_fractions())
                        raise RuntimeError("Material left the 60 mm operating envelope")
                    if view is not None and result.steps % args.render_every == 0:
                        view.update(surface, obs)
                        frame, _ = view.capture(f"{args.yaw:+g} deg / {stage.name}", result.steps * settings.dt, count, obs.grip_points)
                        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
                save_state(args.output / f"{stage.name}.npz", obs, surface)
                if view is not None:
                    view.update(surface, obs)
                    frame, _ = view.capture(f"{args.yaw:+g} deg / {stage.name}", result.steps * settings.dt, count, obs.grip_points)
                    frame.save(args.output / f"{stage.name}.png")
                if stroke is None:
                    sequence.finish_stage(stage.name, obs, surface)
                    result.lift_mm = sequence.metrics.lift_m * 1000
                elif role == "approach" and count != stroke.before_edges:
                    result.failures.append(f"stroke {stroke.number}: cut during approach")
                elif role == "incise":
                    stroke.incised_edges = count
                    stroke.new_edges = count - stroke.before_edges
                    material = material_cut_edges(grid, tissue.cut_edge_fractions())
                    removed = (grid.neighbors >= 0) & (obs.neighbors < 0)
                    stroke.new_material_edges = int(np.count_nonzero((removed & material)[:, 1::2])) - stroke.before_material_edges
                elif role == "clear":
                    stroke.after_edges = count
                    stroke.wall_triangles = int(np.count_nonzero(wall))
                    mesh = trimesh.Trimesh(surface.vertices, surface.triangles, process=False)
                    mesh.merge_vertices(digits_vertex=5)
                    edge_counts = np.bincount(mesh.edges_unique_inverse)
                    stroke.boundary_edges = int(np.count_nonzero(edge_counts == 1))
                    stroke.nonmanifold_edges = int(np.count_nonzero(edge_counts > 2))
                    keys = np.sort(mesh.faces, axis=1)
                    stroke.duplicate_triangles = int(len(keys) - len(np.unique(keys, axis=0)))
                    stroke.winding_consistent = bool(mesh.is_winding_consistent)
                    stroke.completed = True
                    print(asdict(stroke), flush=True)
                previous = target
            np.save(args.output / "cut-fractions.npy", tissue.cut_edge_fractions())
            for stroke in result.strokes:
                if (args.pattern != "retrace" or stroke.number == 1) and stroke.new_material_edges == 0:
                    result.failures.append(f"stroke {stroke.number}: no new incision")
                if stroke.withdrawal_edges:
                    result.failures.append(f"stroke {stroke.number}: cut during withdrawal")
                if (stroke.boundary_edges or stroke.nonmanifold_edges or stroke.duplicate_triangles
                        or not stroke.winding_consistent):
                    result.failures.append(f"stroke {stroke.number}: invalid reconstructed topology")
            if result.max_knife_depth_mm > .2:
                result.failures.append("knife body overlap > 0.2 mm")
            if result.max_surface_degenerate:
                result.failures.append("degenerate reconstructed surface")
            if result.grip_lost or result.lift_mm < 1:
                result.failures.append("grasp/lift failed")
            if result.max_invalid_tets or result.min_volume_ratio < .149:
                result.failures.append("invalid or collapsed live tetrahedra")
            if result.healed_connections:
                result.failures.append("cut connections healed")
            result.status = "failed" if result.failures else "passed"
    except (RuntimeError, ValueError, OSError) as error:
        result.status = "failed"
        result.failures.append(str(error))
    finally:
        if video is not None:
            video.release()
        if view is not None:
            view.close()
        result.elapsed_s = time.monotonic() - started
        report = {"result": asdict(result), "settings": asdict(settings), "yaw_degrees": args.yaw,
                  "pattern": args.pattern, "cell_size_cm": grid.cell_size.tolist(),
                  "dll_sha256": dll_sha256}
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
        print(result.status, result.failures, flush=True)
    if result.status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
