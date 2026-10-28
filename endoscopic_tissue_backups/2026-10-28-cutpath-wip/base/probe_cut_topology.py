"""Record material/air cut ownership before diagnosing a topology change."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .assess_interactions import save_state
from .grid import build_grid
from .native_backend import Command, NativeTissue, Settings
from .scenario import STAGES, Stage, drive_command, interpolate, severed_edges, stage_start


def material_cut_edges(grid, fractions):
    skin = np.full_like(fractions, -1)
    intersections = grid.intersections
    a, b = intersections["a"], intersections["b"]
    delta = grid.positions[b] - grid.positions[a]
    axis = np.abs(delta).argmax(axis=1)
    slot = 2 * axis + (delta[np.arange(len(a)), axis] > 0)
    skin[a, slot] = intersections["t"]
    skin[b, slot ^ 1] = 1 - intersections["t"]
    neighbor = np.maximum(grid.neighbors, 0)
    inside_a, inside_b = grid.inside[:, None] != 0, grid.inside[neighbor] != 0
    return ((inside_a & inside_b) | ((inside_a != inside_b) & (skin >= 0) & (skin <= 1)
            & np.where(inside_a, fractions <= skin, fractions >= skin)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed", action="store_true", help="Cut a fixed specimen to isolate reconstruction")
    parser.add_argument("--vertical-spacing", type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = Settings()
    grid = build_grid(.2, density=settings.density_kg_m3, vertical_spacing=args.vertical_spacing)
    if args.fixed:
        grid.pinned[:] = grid.active
    stages = STAGES if not args.fixed else (
        Stage("approach", 40, Command(blade_center=(.15, 1.05, -.6))),
        Stage("incise", 100, Command(blade_center=(.15, -.15, -.6))),
        Stage("withdraw", 180, Command(blade_center=(.15, 2.2, -.6))),
        Stage("hold", 30, Command(blade_center=(.15, 2.2, -.6))))
    rows = []
    with NativeTissue(grid, settings) as tissue:
        obs = tissue.observe()
        baseline = obs.neighbors.copy()
        previous = Command()
        for stage in stages:
            previous = stage_start(stage.name, previous, obs)
            for step in range(stage.steps):
                requested = interpolate(previous, stage.command, (step + 1) / stage.steps)
                obs = tissue.step(drive_command(requested, obs, settings))
                fractions = tissue.cut_edge_fractions()
                removed = (baseline >= 0) & (obs.neighbors < 0)
                material = material_cut_edges(grid, fractions)
                displacement = np.linalg.norm(obs.nodes - grid.positions, axis=1) * 10
                row = {"step": len(rows) + 1, "stage": stage.name,
                       "cut_edges": int(severed_edges(baseline, obs.neighbors)),
                       "air_cut_edges": int(np.count_nonzero((removed & ~material)[:, 1::2])),
                       "max_node_displacement_mm": float(displacement.max()),
                       "max_inside_displacement_mm": float(displacement[grid.inside != 0].max()),
                       "blade_center_cm": obs.blade[:3].tolist(), "phase": int(obs.blade[7]),
                       "knife_depth_mm": tissue.knife_body_depth_m() * 1000}
                rows.append(row)
                if displacement.max() > 60:
                    break
            surface = tissue.surface()
            save_state(args.output / f"{stage.name}.npz", obs, surface)
            np.save(args.output / f"{stage.name}-cut-fractions.npy", fractions)
            triangles = surface.vertices[surface.triangles]
            wall = surface.anchors[surface.triangles[:, 0], 3] > .5
            area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
            print(stage.name, row, "walls", np.count_nonzero(wall), "degenerate", np.count_nonzero(wall & (area < 1e-8)), flush=True)
            if displacement.max() > 60:
                break
            previous = stage.command
    (args.output / "trajectory.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf8")
    (args.output / "settings.json").write_text(json.dumps(asdict(settings), indent=2), encoding="utf8")
    (args.output / "grid.json").write_text(json.dumps({"fixed": args.fixed, "cell_size_cm": grid.cell_size.tolist(),
                                                     "dims": grid.dims}), encoding="utf8")


if __name__ == "__main__":
    main()
