"""Inspect actual cut geometry and render before/after at an identical camera pose."""
import argparse
import json
from pathlib import Path

import numpy as np

from .grid import to_genesis
from .native_backend import Settings
from .rendering import EndoscopicView
from .surface_diagnostics import load_state, topology_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-stage", default="blade_approach")
    parser.add_argument("--after-stage", default="hold")
    parser.add_argument("--offset", type=float, nargs=3, default=(.030, .024, .028),
                        help="Camera position relative to the cut centre, Genesis metres")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Preserve the actual checkpoint time/count in the diagnostic image HUD.
    # These images change only the camera and display subdivision, not physics.
    stage_records = {}
    for line in (args.input / "trajectory.jsonl").read_text(encoding="utf8").splitlines():
        record = json.loads(line)
        stage_records[record["stage"]] = record
    before_obs, before = load_state(args.input / f"{args.before_stage}.npz")
    after_obs, after = load_state(args.input / f"{args.after_stage}.npz")
    wall = after.anchors[after.triangles[:, 0], 3] > .5
    wall_points = after.vertices[after.triangles[wall]].reshape(-1, 3)
    if not len(wall_points):
        raise RuntimeError("No cut walls to inspect")
    target = to_genesis((wall_points.min(axis=0) + wall_points.max(axis=0))[None] / 2)[0]
    settings_path = args.input / "settings.json"
    settings_data = (json.loads(settings_path.read_text(encoding="utf8")) if settings_path.is_file()
                     else json.loads((args.input / "report.json").read_text(encoding="utf8"))["settings"])
    view = EndoscopicView(Settings(**settings_data), 1280, 960)
    try:
        view.camera.set_pose(pos=target + args.offset, lookat=target)
        for label, stage, obs, surface, subdivisions in (
            ("before-cut", args.before_stage, before_obs, before, 4),
            ("after-cut-raw", args.after_stage, after_obs, after, 1),
            ("after-cut-refined", args.after_stage, after_obs, after, 4),
        ):
            view.update(surface, obs, divisions=subdivisions)
            record = stage_records[stage]
            image, depth = view.capture(label, record["time_s"], record["cut_edges"], obs.grip_points)
            image.save(args.output / f"{label}.png")
            np.save(args.output / f"{label}-depth-m.npy", depth)
    finally:
        view.close()
    statistics = {"before": topology_statistics(before), "after": topology_statistics(after),
                  "note": "Geometric surface diagnostics, not a calibrated wound-width/depth measurement. Native y is up."}
    (args.output / "geometry.json").write_text(json.dumps(statistics, indent=2), encoding="utf8")
    print(json.dumps(statistics, indent=2), flush=True)


if __name__ == "__main__":
    main()
