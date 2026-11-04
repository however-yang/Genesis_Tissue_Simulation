"""Record every solved frame and regional motion; this is not a timing run."""
from dataclasses import replace
from pathlib import Path
import json
import pickle
import runpy
import struct
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from endoscopic_tissue import run

output = Path(sys.argv[1]).resolve()
args = sys.argv[2:]
scene = args[args.index("--scene") + 1] if "--scene" in args else "kidney"
original = run.Flow.run


class Capture:
    def __init__(self, stream, motion):
        self.stream, self.motion = stream, motion
        self.frames = 0
        self.previous = self.masks = None

    def after_step(self, flow, tissue, surface):
        if self.masks is None:
            rest = flow.grid.positions
            material = flow.grid.inside != 0
            tumor = material & (flow.scene.shape.tumor_weight(rest, 1e-3) > .5)
            kidney = material & ~tumor
            stump = kidney & (np.hypot(rest[:, 0], rest[:, 2]) < 1.0) & (rest[:, 1] > flow.scene.shape.half_extents[1] - .6)
            self.masks = dict(tumor=tumor, kidney=kidney, stump=stump)
            np.savez_compressed(output / "motion-masks.npz", rest=rest, **self.masks)
        obs = flow.obs
        row = dict(step=flow.step, stage=flow.stage, jaw_gap_mm=obs.jaw_gap * 10,
                   jaw_load_n=list(obs.jaw_load_n), grip_points=obs.grip_points,
                   grip_error_um=obs.grip_error_m * 1e6)
        for name, mask in self.masks.items():
            nodes = obs.nodes[mask]
            row[name + "_com_cm"] = nodes.mean(axis=0).tolist()
            if self.previous is not None:
                velocity = (nodes - self.previous[mask]) / flow.settings.dt * 10
                speed = np.linalg.norm(velocity, axis=1)
                row[name + "_rms_speed_mm_s"] = float(np.sqrt(np.mean(speed ** 2)))
                row[name + "_p99_speed_mm_s"] = float(np.percentile(speed, 99))
                relative = velocity - velocity.mean(axis=0)
                row[name + "_relative_rms_speed_mm_s"] = float(np.sqrt(np.mean(np.sum(relative ** 2, axis=1))))
        self.previous = obs.nodes.copy()
        self.motion.write(json.dumps(row) + "\n")
        surface = tissue.surface() if surface is None else surface
        obs = replace(obs, nodes=np.empty((0, 3), np.float32),
                      neighbors=np.empty((0, 6), np.int32), grip_flags=np.empty(0, np.int32))
        frame = zlib.compress(pickle.dumps((flow.step, flow.stage, surface, obs), protocol=5), 1)
        self.stream.write(struct.pack("<I", len(frame)))
        self.stream.write(frame)
        self.frames += 1
        return True

    def after_stage(self, flow, stage, surface):
        self.stream.flush()
        self.motion.flush()
        print(f"[CAPTURE] {self.frames} exact frames", flush=True)


def captured(self, tissue, trajectory, sink):
    assert sink is None, "Capture runs must use --native-only"
    with (output / "replay.bin").open("wb") as stream, (output / "motion.jsonl").open("w") as motion:
        pickle.dump((self.settings, scene, "exact solved surface; simulation-time replay"), stream, protocol=5)
        capture = Capture(stream, motion)
        capture.after_step(self, tissue, None)
        return original(self, tissue, trajectory, capture)


run.Flow.run = captured
sys.argv = ["dev_run.py", "--output", str(output), "--headless", "--native-only", *args]
runpy.run_path(str(ROOT / "endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py"), run_name="__main__")
