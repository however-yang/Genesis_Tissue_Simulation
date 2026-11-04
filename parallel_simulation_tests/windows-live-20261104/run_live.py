"""Windows live window for the published runtime-core candidate, in isolation.

The existing demo controls the grasper and scissors. This is not the A100
training RGB-D benchmark and does not certify Windows physics or performance.
"""
import ctypes
from datetime import datetime
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = PROJECT / "parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/python-gpu-check"
DLL = ROOT / "build/lib/LiverCudaSim.dll"


def main():
    arguments = sys.argv[1:]
    checking = arguments == ["--check"]
    if any(a == "--dll" or a.startswith("--dll=") or a == "--output" or a.startswith("--output=") for a in arguments):
        raise ValueError("The launcher pins its candidate DLL and creates a unique output directory.")
    os.environ.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    sys.path.insert(0, str(SOURCE))
    # Help imports the existing runner without loading a native session.
    if "--help" in arguments or "-h" in arguments:
        from endoscopic_tissue import run
        sys.argv = [str(Path(__file__)), "--help"]
        return run.main()
    from prepare_build import published_bytes
    record = json.loads((ROOT / "source-manifest.json").read_text(encoding="utf-8"))
    build = json.loads((ROOT / "build-manifest.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256(DLL.read_bytes()).hexdigest()
    if actual != build["dll_sha256"]:
        raise ValueError("The DLL differs from this Windows build record")
    for name, expected in record["python_files"].items():
        published_bytes(SOURCE / name, expected)
    native = ctypes.CDLL(str(DLL))
    controls = {
        "LCS_SetCoreOptimizations": (43264,),
        "LCS_SetBatchQueryMode": (3,),
        "LCS_SetQueryThreads": (64,),
        "LCS_SetNativeSchedule": (512, 1024),
        "LCS_SetCoarseBoundCache": (1,),
    }
    for name, values in controls.items():
        function = getattr(native, name)
        function.argtypes = [ctypes.c_int] * len(values)
        function.restype = ctypes.c_int
        result = function(*values)
        if result != 0:
            raise RuntimeError(f"{name} rejected {values}: {result}")
    from endoscopic_tissue import run
    metadata = {
        "source_commit": record["commit"], "python_source": str(SOURCE),
        "runner": str(Path(run.__file__).resolve()), "dll": str(DLL),
        "dll_sha256": actual, "native_controls": controls,
        "renderer": "published V8 with GPU postprocess, aligned depth, fused mesh, persistent soup",
        "existing_demo_checks_preserved": True,
        "query_compaction": "default 0 (optional setter absent from this published library)",
        "a100_training_benchmark_reproduced": False,
        "windows_physics_acceptance": "not established by this launcher",
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if checking:
        print("PASS: source hashes, Windows DLL load, native controls and runner import; no simulation initialized.")
        return 0
    output = ROOT / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    # Existing run.main creates this leaf with exist_ok=False.
    output.parent.mkdir(parents=True, exist_ok=True)
    options = ["--dll", str(DLL), "--output", str(output)]
    if not any(a == "--scene" or a.startswith("--scene=") for a in arguments):
        options += ["--scene", "kidney"]
    if "--native-only" not in arguments and "--demo" not in arguments and "--tool" not in arguments:
        options.append("--demo")
    run.EndoscopicView = partial(run.EndoscopicView, gpu_postprocess=True,
                                gpu_aligned_depth=True, fused_mesh=True, persistent_soup=True)
    sys.argv = [str(Path(__file__)), *options, *arguments]
    try:
        return run.main()
    finally:
        if output.exists():
            (output / "windows-launch.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
