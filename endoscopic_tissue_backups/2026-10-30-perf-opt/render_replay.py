"""Render a trusted local recording in a process that never opens the native DLL."""
from collections import defaultdict
import importlib.util
import hashlib
import json
from pathlib import Path
import pickle
import struct
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from endoscopic_tissue.rendering import EndoscopicView
from endoscopic_tissue.scenario import SCISSORS_SCENES

source, output = Path(sys.argv[1]), Path(sys.argv[2])
if len(sys.argv) > 3:
    spec = importlib.util.spec_from_file_location("endoscopic_tissue._replay_renderer", sys.argv[3])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Frozen renderer code still uses the project's original asset directory.
    module.__file__ = str(ROOT / "endoscopic_tissue/rendering.py")
    EndoscopicView = module.EndoscopicView
output.mkdir(parents=True, exist_ok=False)
rows, count, last_stage = [], 0, None
with source.open("rb") as stream:
    settings, scene_name, provenance = pickle.load(stream)
    scene = SCISSORS_SCENES[scene_name]
    view = EndoscopicView(settings, 1280, 960, supersample=1, kidney=scene.shape, camera=scene.shot)
    video = cv2.VideoWriter(str(output / "interactions.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                            1 / settings.dt, (1280, 960))
    if not video.isOpened():
        raise RuntimeError("Could not open MP4 writer")
    try:
        while prefix := stream.read(4):
            step, stage, surface, obs = pickle.loads(zlib.decompress(stream.read(struct.unpack("<I", prefix)[0])))
            started = time.perf_counter()
            view.update(surface, obs)
            updated = time.perf_counter()
            frame = view.render()[0]
            rendered = time.perf_counter()
            rows.append(dict(step=step, stage=stage, update_ms=1000*(updated-started),
                             render_ms=1000*(rendered-updated),
                             frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest()))
            video.write(frame)
            if last_stage is not None and stage != last_stage:
                cv2.imwrite(str(output / f"{last_stage}.png"), last_frame)
                print(f"[REPLAY] {last_stage} at frame {count}", flush=True)
            # Lossless frames permit pixel-level verification of renderer changes.
            if step % 30 == 0:
                cv2.imwrite(str(output / f"frame-{step:04d}.png"), frame)
            last_frame, last_stage = frame, stage
            count += 1
        cv2.imwrite(str(output / f"{last_stage}.png"), last_frame)
    finally:
        video.release()
        view.close()
        (output / "render-timing.json").write_text(json.dumps(dict(
            source=str(source), provenance=provenance, fps=1/settings.dt, frames=count,
            note="Playback rate follows simulated time, not measured execution speed", rows=rows), indent=2))
print(f"[REPLAY] completed {count} frames", flush=True)
