"""Content-addressed provenance; old binary/source equivalence is never inferred."""
import ctypes
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

from .. import native_backend as abi

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def pinned_dll(path, expected_sha256):
    path = Path(path).resolve(strict=True)
    if len(expected_sha256) != 64 or sha256(path) != expected_sha256.lower():
        raise ValueError("DLL SHA-256 does not match the explicitly pinned version")
    return path


def manifest(native, settings, scene_config, bounds):
    package = ROOT / "endoscopic_tissue"
    sources = [package / p for p in ("native_backend.py", "scenario.py", "grid.py", "kidney.py",
                                     "instrument_geometry.py", "surface_diagnostics.py")]
    sources += sorted((package / "training").glob("*.py"))
    assets = sorted(p for p in (package / "assets/instruments").rglob("*") if p.is_file())
    files = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources + assets}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    sizes = {name: {"native": int(native.functions[name]()), "python": ctypes.sizeof(structure)}
             for name, structure in (("LCS_PhysicsAbiVersion", abi.PhysicsDesc),
                                     ("LCS_GraspToolAbiVersion", abi.Grasper),
                                     ("LCS_PressureBladeAbiVersion", abi.Blade),
                                     ("LCS_ScissorsAbiVersion", abi.ScissorsControl),
                                     ("LCS_ScissorsResultAbiVersion", abi.ScissorsResult))}
    return {"schema": "native_training_transition_v1", "git_commit": commit,
            "dll_path": str(native.dll_path.resolve()), "dll_sha256": sha256(native.dll_path),
            "binary_source_binding": "unverified_prebuilt_binary_kept_as_independent_baseline",
            "current_adapter_and_asset_sha256": files, "abi": sizes,
            "contact_model_version": int(native.functions["LCS_ContactModelVersion"]()),
            "settings": asdict(settings), "scene_config": scene_config,
            "action_bounds": asdict(bounds), "coordinate_axes": "native_world_xyz",
            "units": {"position": "m", "angle": "rad", "force": "N", "time": "s"},
            "observation_completeness": "partial; internal velocity and contact/fracture history not fully exported",
            "velocity_source": "finite_difference_of_accepted_positions_not_internal_solver_velocity",
            "domain_validation": "unvalidated; interface checks do not approve a physics training domain",
            "physical_defaults_changed": False}
