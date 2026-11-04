"""Stage the published native sources for an isolated RTX 3090 Windows build."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
CANDIDATE = PROJECT / "parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def published_bytes(path, expected):
    data = path.read_bytes()
    if digest(data) == expected:
        return data
    normalized = data.replace(b"\r\n", b"\n")
    if digest(normalized) == expected:
        return normalized
    raise ValueError("Published source hash mismatch: " + str(path))


def main():
    manifest = json.loads((CANDIDATE / "delivery/manifest.json").read_text(encoding="utf-8"))
    baseline = json.loads((CANDIDATE / "baseline.json").read_text(encoding="utf-8"))
    python_root = PROJECT / "parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/python-gpu-check"
    sources = {}
    for name, expected in manifest["native_source_sha256"].items():
        data = published_bytes(CANDIDATE / name, expected)
        sources[str(Path(name).relative_to("native/source"))] = data
    for name, expected in baseline["baseline_python_files"].items():
        published_bytes(python_root / name, expected)
    # Only the compilation target changes. All solver sources remain identical.
    cmake = sources["CMakeLists.txt"]
    old = b'CUDA_ARCHITECTURES "80"'
    if cmake.count(old) != 1:
        raise ValueError("Unexpected CMake architecture configuration")
    sources["CMakeLists.txt"] = cmake.replace(old, b'CUDA_ARCHITECTURES "86"')
    destination = ROOT / "native"
    destination.mkdir(exist_ok=False)
    for name, data in sources.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    record = {
        "commit": commit,
        "candidate": str(CANDIDATE.relative_to(PROJECT)),
        "python_source": str(python_root.relative_to(PROJECT)),
        "native_files": {name: digest(data) for name, data in sources.items()},
        "python_files": baseline["baseline_python_files"],
        "source_verification": "Published SHA256: exact bytes, or CRLF-to-LF when Git changed line endings",
        "only_native_source_change": 'CMake CUDA_ARCHITECTURES "80" -> "86"',
        "architecture": "sm_86",
        "core_mode": manifest["core_mode"],
        "windows_physics_acceptance": "not tested",
        "a100_performance_not_transferable": True,
    }
    (ROOT / "source-manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"native_files_verified": len(sources), "python_files_verified": len(record["python_files"]), "commit": commit}))


if __name__ == "__main__":
    main()
