"""Record only this independent local binary; never edit the A100 manifest."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

root = Path(__file__).resolve().parent
source = json.loads((root / "source-manifest.json").read_text(encoding="utf-8"))
for name, expected in source["native_files"].items():
    actual = hashlib.sha256((root / "native" / name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("Staged source changed: " + name)
dll = root / "build/lib/LiverCudaSim.dll"
data = dll.read_bytes()
if data[:2] != b"MZ":
    raise ValueError("Expected a Windows PE DLL")
record = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": source["commit"],
    "dll": str(dll), "dll_bytes": len(data),
    "dll_sha256": hashlib.sha256(data).hexdigest(),
    "platform": platform.platform(), "python": sys.version,
    "cuda_architecture": "sm_86", "cuda_toolkit": "12.5.40",
    "msvc_toolset": "14.29", "native_sources_verified": len(source["native_files"]),
    "physics_equivalence_validated": False,
    "interactive_fps_measured": False,
}
(root / "build-manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
