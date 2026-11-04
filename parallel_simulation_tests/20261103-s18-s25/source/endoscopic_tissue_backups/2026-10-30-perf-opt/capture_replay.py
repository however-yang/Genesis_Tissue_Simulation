"""Capture solved surfaces without creating OpenGL in the CUDA process.

This is a diagnostic recording run, not a performance measurement. Frames
contain the exact reconstructed geometry, normals and instrument poses.
"""
from dataclasses import replace
from pathlib import Path
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
    def __init__(self, stream):
        self.stream = stream
        self.frames = 0

    def after_step(self, flow, tissue, surface):
        surface = tissue.surface() if surface is None else surface
        # The renderer only reads solved tool poses; don't duplicate the full
        # volume and connectivity in every display frame.
        obs = replace(flow.obs, nodes=np.empty((0, 3), np.float32),
                      neighbors=np.empty((0, 6), np.int32), grip_flags=np.empty(0, np.int32))
        data = zlib.compress(pickle.dumps((flow.step, flow.stage, surface, obs), protocol=5), 1)
        self.stream.write(struct.pack("<I", len(data)))
        self.stream.write(data)
        self.frames += 1
        return True

    def after_stage(self, flow, stage, surface):
        self.stream.flush()
        print(f"[CAPTURE] {self.frames} exact frames", flush=True)


def captured(self, tissue, trajectory, sink):
    assert sink is None, "Capture runs must use --native-only"
    with (output / "replay.bin").open("wb") as stream:
        pickle.dump((self.settings, scene, "exact solved surface; simulation-time replay"), stream, protocol=5)
        capture = Capture(stream)
        capture.after_step(self, tissue, None)
        return original(self, tissue, trajectory, capture)


run.Flow.run = captured
sys.argv = ["dev_run.py", "--output", str(output), "--headless", "--native-only", *args]
runpy.run_path(str(ROOT / "endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py"), run_name="__main__")
