"""Verify capture, lift, retention and release without making a cutting claim."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .assess_interactions import save_state
from .grid import build_grid
from .native_backend import Command, NativeTissue, Settings
from .scenario import STAGES, Sequence, drive_command, interpolate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=.2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = Settings()
    grid = build_grid(args.spacing, density=settings.density_kg_m3)
    peak_error_mm = 0.
    with NativeTissue(grid, settings) as tissue:
        obs = tissue.observe()
        sequence = Sequence(obs)
        previous = Command()
        for stage in STAGES[:4]:
            for step in range(stage.steps):
                requested = interpolate(previous, stage.command, (step + 1) / stage.steps)
                obs = tissue.step(drive_command(requested, obs, settings))
                sequence.record_step(stage.name, obs, tissue.knife_body_depth_m())
                peak_error_mm = max(peak_error_mm, obs.grip_error_m * 1000)
            surface = tissue.surface()
            save_state(args.output / f"{stage.name}.npz", obs, surface)
            sequence.finish_stage(stage.name, obs, surface)
            previous = stage.command
        before = obs.nodes.copy()
        patches = obs.grip_points
        opened = Command(previous.hinge, 0, previous.blade_center)
        for step in range(60):
            obs = tissue.step(interpolate(previous, opened, min(1, (step + 1) / 30)))
        save_state(args.output / "released.npz", obs, tissue.surface())
        drop_mm = float(np.median(before[sequence.tracked, 1] - obs.nodes[sequence.tracked, 1]) * 10)
    accepted = (sequence.metrics.grasp_acquired and sequence.metrics.lift_m >= .001
                and not sequence.metrics.grip_lost_after_capture and obs.grip_points == 0 and drop_mm > 1)
    report = {"accepted": accepted, "scope": "grasp_lift_release_only", "spacing_mm": args.spacing * 10,
              "settings": asdict(settings), "captured_patches": patches,
              "lift_mm": sequence.metrics.lift_m * 1000, "peak_grip_error_mm": peak_error_mm,
              "grip_lost_during_lift": sequence.metrics.grip_lost_after_capture,
              "released_patches": obs.grip_points, "drop_after_release_mm": drop_mm}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2), flush=True)
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
