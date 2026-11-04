"""Live Windows demo and isolated Linux comparison using the same scene code."""
import argparse
import ctypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
GPU = "GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa"
SOURCE = PROJECT / "parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/python-gpu-check"
CANDIDATE = PROJECT / "parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("baseline", "mesh65", "frontal65"), default="frontal65")
    parser.add_argument("--lateral-mm", type=float, choices=(1.225, 1.25, 1.275, 1.3), default=1.25)
    parser.add_argument("--name", help="Unique name under this isolated directory's outputs")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="CPU-only scene and asset checks")
    parser.add_argument("--guard", action="store_true", help="Reserve only server A100 GPU1; refuse busy GPU")
    args = parser.parse_args()
    if args.name and (not args.name.replace("-", "").replace("_", "").isalnum() or len(args.name) > 80):
        parser.error("Use letters, digits, hyphens or underscores in --name")
    if args.guard and os.name == "nt":
        parser.error("--guard is for server GPU1; the Windows live window uses the local GPU")
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be nonnegative")
    os.environ.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if os.name != "nt":
        os.environ.update(CUDA_VISIBLE_DEVICES=GPU, PYOPENGL_PLATFORM="egl")
    sys.path.insert(0, str(SOURCE))
    from endoscopic_tissue import run
    from scene_config import configure
    metadata, view_options = configure(run, args.preset, args.lateral_mm)
    grid = run.SCISSORS_SCENES["kidney"].build_grid(run.scissors_settings().density_kg_m3)
    metadata.update(material_nodes=int(grid.inside.sum()), active_nodes=int(grid.active.sum()),
                    lattice_nodes=len(grid.positions), cell_size_mm=(grid.cell_size * 10).tolist(),
                    mass_kg=float(grid.mass.sum()), python_source=str(SOURCE),
                    source_script_sha256=sha(ROOT / "scene_config.py"),
                    server_gpu=GPU if os.name != "nt" else None)
    del grid
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if args.inspect:
        return 0
    name = args.name or args.preset + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = ROOT / "outputs" / name
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Resource logs are separate from the leaf run.main creates exclusively.
    controls_dir = ROOT / "controls" / name
    controls_dir.mkdir(parents=True, exist_ok=False)
    guard = None
    try:
        if args.guard:
            from endoscopic_tissue.training.acceptance_queue import GPUReservation
            guard = GPUReservation(GPU, controls_dir)
        if os.name != "nt" and not args.native_only:
            from endoscopic_tissue.training.egl_device import bind
            metadata["egl_binding"] = bind(GPU)
        if os.name == "nt":
            windows = ROOT.parent / "windows-live-20261104"
            dll = windows / "build/lib/LiverCudaSim.dll"
            expected = json.loads((windows / "build-manifest.json").read_text(encoding="utf-8"))["dll_sha256"]
        else:
            dll = CANDIDATE / "native/binaries/runtime-core.so"
            expected = json.loads((CANDIDATE / "delivery/manifest.json").read_text())["library_sha256"]
        if sha(dll) != expected:
            raise ValueError("Native DLL differs from the recorded candidate build")
        native = ctypes.CDLL(str(dll))
        for name, values in {"LCS_SetCoreOptimizations": (43264,), "LCS_SetBatchQueryMode": (3,),
                             "LCS_SetQueryThreads": (64,), "LCS_SetNativeSchedule": (512, 1024),
                             "LCS_SetCoarseBoundCache": (1,)}.items():
            fn = getattr(native, name)
            fn.argtypes, fn.restype = [ctypes.c_int] * len(values), ctypes.c_int
            if fn(*values) != 0:
                raise RuntimeError("Candidate configuration rejected: " + name)
        metadata.update(dll=str(dll), dll_sha256=expected, core_mode=43264,
                        native_physics_source_modified=False,
                        timing_definition="Synchronous headless demo RGB frame, including step/readback, surface and recording submit, renderer and video write; excludes stage saves and final async drain. Not Windows displayed FPS or full training RGB-D.")
        base_view = run.EndoscopicView

        def view(*vargs, **kwargs):
            kwargs.setdefault("supersample", 1)
            kwargs.update(gpu_postprocess=True, gpu_aligned_depth=True, fused_mesh=True,
                          persistent_soup=True, **view_options)
            return base_view(*vargs, **kwargs)

        run.EndoscopicView = view
        measurements = []
        frame = {}
        original_step, original_check = run.NativeTissue.step, run.NativeTissue.check

        def check(self, name, *values):
            if name != "LCS_Step":
                return original_check(self, name, *values)
            begin = time.perf_counter()
            result = original_check(self, name, *values)
            frame["native_ms"] = (time.perf_counter() - begin) * 1000
            return result

        def step(self, command):
            if guard:
                guard({os.getpid()})
            frame["start"] = time.perf_counter()
            result = original_step(self, command)
            frame["step_readback_ms"] = (time.perf_counter() - frame["start"]) * 1000
            return result

        run.NativeTissue.check, run.NativeTissue.step = check, step
        original_record = run.Recording.after_step

        def record(self, flow, tissue, surface):
            render_start = time.perf_counter()
            result = original_record(self, flow, tissue, surface)
            end = time.perf_counter()
            measurements.append(dict(step=flow.step, stage=flow.stage, native_ms=frame["native_ms"],
                                     step_readback_ms=frame["step_readback_ms"],
                                     render_write_ms=(end - render_start) * 1000,
                                     frame_ms=(end - frame["start"]) * 1000))
            return result

        run.Recording.after_step = record
        options = ["--scene", "kidney", "--dll", str(dll), "--output", str(output),
                   "--width", "1280", "--height", "960", "--render-every", "1"]
        if args.headless:
            options.append("--headless")
        if args.native_only:
            options.append("--native-only")
        if args.steps is not None:
            options.extend(("--steps", str(args.steps)))
        if args.viewer:
            options.append("--viewer")
        # Keep the camera centred throughout the run; no final automatic zoom.
        sys.argv = [str(Path(__file__)), *options]
        try:
            run.main()
        finally:
            if output.exists():
                (output / "scene-config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                if measurements:
                    import numpy as np
                    summary = {}
                    for key in ("native_ms", "step_readback_ms", "render_write_ms", "frame_ms"):
                        values = np.array([row[key] for row in measurements])
                        summary[key] = dict(mean=float(values.mean()), p95=float(np.percentile(values, 95)), max=float(values.max()))
                    summary["headless_demo_hz"] = 1000 / summary["frame_ms"]["mean"]
                    summary["frames"] = len(measurements)
                    (output / "timing.json").write_text(json.dumps(dict(summary=summary, frames=measurements), indent=2) + "\n")
    finally:
        if guard:
            guard.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
