"""Push the non-cutting knife back into the tissue side while its edge stays below it."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .assess_interactions import save_state
from .fixation import add_fixation_arguments, use_fixation
from .grid import build_grid
from .native_backend import Command, NativeTissue, Settings
from .scenario import interpolate, severed_edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--without-knife-body", action="store_true")
    add_fixation_arguments(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = Settings(start_force_n=10, cut_force_n=1, has_knife_body=not args.without_knife_body)
    rows = []
    grid = build_grid(density=settings.density_kg_m3)
    # Fix the obstacle for this collision oracle: a freely deforming tissue
    # can move out of the way, so tool position alone would not prove blocking.
    grid.pinned[:] = grid.active
    use_fixation(parser, args, grid)
    with NativeTissue(grid, settings) as tissue:
        baseline = tissue.observe().neighbors.copy()
        previous = Command()
        stages = (("settle", 20, previous),
                  ("outside", 30, Command(blade_center=(4, 2.2, 0))),
                  ("lower", 40, Command(blade_center=(4, -.85, 0))),
                  ("press_back", 60, Command(blade_center=(2.6, -.85, 0))),
                  ("hold", 20, Command(blade_center=(2.6, -.85, 0))))
        for name, steps, command in stages:
            for step in range(steps):
                obs = tissue.step(interpolate(previous, command, (step + 1) / steps))
                rows.append({"stage": name, "position_cm": obs.blade[:3].tolist(),
                             "force_n": float(np.linalg.norm(obs.blade[3:6]) * .01),
                             "cut_pressure_n": float(obs.blade[6] * .01),
                             "body_depth_mm": tissue.knife_body_depth_m() * 1000,
                             "cut_edges": int(severed_edges(baseline, obs.neighbors))})
            save_state(args.output / f"{name}.npz", obs, tissue.surface())
            previous = command
    report = {"settings": asdict(settings), "final_center_cm": obs.blade[:3].tolist(),
              "max_body_depth_mm": max(row["body_depth_mm"] for row in rows),
              "max_force_n": max(row["force_n"] for row in rows),
              "max_cut_pressure_n": max(row["cut_pressure_n"] for row in rows),
              "cut_edges": rows[-1]["cut_edges"]}
    report["accepted"] = bool(obs.blade[0] > 2.85 and report["max_force_n"] > .01
                              and report["max_cut_pressure_n"] < .001 and report["cut_edges"] == 0
                              and report["max_body_depth_mm"] <= .2)
    (args.output / "trajectory.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
