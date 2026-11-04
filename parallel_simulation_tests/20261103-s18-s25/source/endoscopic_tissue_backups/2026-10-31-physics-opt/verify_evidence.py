"""Verify this experiment's evidence and rollback; this is not a physical-quality pass."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

import cv2

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
DATA = ROOT / "endoscopic_tissue/outputs/physics-opt-20261031"
BASE = "fafb85a75e43e84de04c7d79e92e38c22c9849c8"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
require(branch == "dev", f"Unexpected branch: {branch}")
native = "endoscopic_tissue/native/"
sources = ["vendor/cuda/contact/surface_contact.cuh", "vendor/cuda/solver/material/unified_vertex.cuh",
           "vendor/cuda/xpbd/xpbd_solver.cu", "vendor/cuda/cutting/pressure_blade.cuh"]
protected = {}
for name in sources + ["bin/LiverCudaSim.dll", "bin-scissors/LiverCudaSim.dll"]:
    path = native + name
    original = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)
    current = (ROOT / path).read_bytes()
    if not name.endswith(".dll"):
        original, current = [x.replace(b"\r\n", b"\n") for x in (original, current)]
    require(current == original, f"Protected production file changed: {path}")
    protected[path] = hashlib.sha256(current).hexdigest()
require(sha(ROOT / native / "bin-scissors/LiverCudaSim.dll") ==
        "0d09f56a1f4acb813424cd1dee3233d89ce7ef5c0d45e64f079c7d7387912225", "Default DLL hash")
for patch in sorted((WORK / "patches").glob("*.patch")):
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=ROOT, check=True)

summary = read(DATA / "experiment-summary.json")
require(len(summary["runs"]) == 12, "Expected 12 normal timing runs")
require(all(r["state"] == "complete" and r["steps"] == 1110 for r in summary["runs"].values()), "Incomplete timing")
require((WORK / "results-summary.json").read_bytes() == (DATA / "experiment-summary.json").read_bytes(),
        "Archived timing summary differs from current results")
require(len(read(DATA / "fine-sections-02.json")) == 14, "Incomplete fine profiling")

probes = {}
for name in ("contact-equivalence-02", "contact-control"):
    report = read(DATA / name / "report.json")
    rows = read(DATA / name / "contact-equivalence.json")
    require(report["steps"] == 1110 and len(rows) == 74, f"Incomplete frozen comparison: {name}")
    require(all(r[key] == 0 for r in rows for key in
                ("status", "different_counts", "different_records", "different_degrees", "different_members")),
            f"Frozen comparison failed: {name}")
    probes[name] = dict(probes=len(rows), queries=sum(r["queries"] for r in rows),
                       triangle_queries=sum(r["triangle_queries"] for r in rows),
                       valid_records=sum(r["valid_records"] for r in rows))
for name, last_step, differences in (("noinline-equivalence", 165, 2), ("rolled-equivalence", 435, 3)):
    rows = read(DATA / name / "contact-equivalence.json")
    require(rows[-1]["step"] == last_step and rows[-1]["different_records"] == differences,
            f"Missing rejected-candidate evidence: {name}")
for name in ("accumulate-equivalence.json", "partial-accumulate-equivalence.json"):
    rows = read(DATA / name)
    require(len(rows) == 16 and all(r["status"] == 0 and r["differences"] == 0 and r["patterns"] == 6 for r in rows),
            f"Exact reduction check failed: {name}")


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, (float, int)) or math.isfinite(value)


videos = {}
for tag in ("baseline", "retain", "coarse-election", "coarse-partial"):
    capture = DATA / f"{tag}-capture"
    report = read(capture / "report.json")
    require(report["steps"] == 1110, f"Incomplete capture {tag}")
    motion = [json.loads(line) for line in (capture / "motion.jsonl").read_text().splitlines()]
    require([r["step"] for r in motion] == list(range(1111)) and finite(motion), f"Motion data {tag}")
    rendering = read(DATA / f"{tag}-video/render-timing.json")
    require(rendering["frames"] == 1111 and [r["step"] for r in rendering["rows"]] == list(range(1111)),
            f"Incomplete rendering {tag}")
    paths = [DATA / f"{tag}-video/interactions.mp4"]
    if tag != "baseline":
        manifest = read(DATA / f"{tag}-review/manifest.json")
        require(manifest["frames"] == 1111 and manifest["fps"] == 30, f"Comparison manifest {tag}")
        require(read(DATA / f"{tag}-motion/motion-comparison.json")["frames"] == 1111, f"Motion comparison {tag}")
        paths.append(DATA / f"{tag}-review/comparison.mp4")
    for path in paths:
        video = cv2.VideoCapture(str(path))
        try:
            require(video.isOpened() and int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 1111, f"Video length: {path}")
            require(video.get(cv2.CAP_PROP_FPS) == 30, f"Video playback rate: {path}")
            video.set(cv2.CAP_PROP_POS_FRAMES, 1110)
            ok, frame = video.read()
            require(ok and frame is not None, f"Last frame cannot be decoded: {path}")
        finally:
            video.release()
        videos[str(path.relative_to(DATA))] = sha(path)
result = dict(evidence_checks_passed=True, branch=branch, protected_files=protected, timing_runs=12,
              timing_steps=13320, frozen_probes=probes, videos=videos,
              physical_quality_passed=False,
              note="All candidate changes rejected/rolled back. Evidence integrity passed; existing kidney quality failures remain.")
(DATA / "evidence-verification.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
