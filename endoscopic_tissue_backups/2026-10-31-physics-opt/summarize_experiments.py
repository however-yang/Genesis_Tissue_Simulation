"""Summarize normal graph runs; keep diagnostic timings and video playback separate."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "endoscopic_tissue/outputs/physics-opt-20261031"
groups = {
    "retained_plane": (["baseline-01", "baseline-02"], ["retain-01", "retain-02"]),
    "warp_election": (["baseline-03", "baseline-04"], ["coarse-election-01", "coarse-election-02"]),
    "two_stage_reduction": (["baseline-05", "baseline-06"], ["coarse-partial-01", "coarse-partial-02"]),
}
names = [name for pair in groups.values() for side in pair for name in side]
result = {"note": "Normal CUDA Graph runs, no concurrent GPU work or video rendering. Pipeline excludes stage-boundary file saving; not live display FPS.",
          "runs": {}, "comparisons": {}}


def averages(rows):
    return dict(pipeline_ms=statistics.mean(r["pipeline_ms"] for r in rows),
                core_ms=statistics.mean(r["calls_ms"]["LCS_Step"] for r in rows),
                repair_ms=statistics.mean(r["calls_ms"]["LCS_RefineContactSurface"] for r in rows),
                p95_pipeline_ms=sorted(r["pipeline_ms"] for r in rows)[int(.95 * (len(rows) - 1))])


baseline_settings = json.loads((DATA / "baseline-01/report.json").read_text())["settings"]
for name in names:
    folder = DATA / name
    if not (folder / "timing-steps.json").exists():
        result["runs"][name] = {"state": "pending"}
        continue
    rows = json.loads((folder / "timing-steps.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    if len(rows) != 1110 or report["steps"] != 1110:
        raise ValueError(f"Incomplete run: {name}")
    if report["settings"] != baseline_settings:
        raise ValueError(f"Changed settings: {name}")
    result["runs"][name] = dict(state="complete", steps=len(rows), status=report["status"],
        dll_sha256=report["dll_sha256"], metrics=report["metrics"], timing=averages(rows),
        stages={stage: averages([r for r in rows if r["stage"] == stage])
                for stage in dict.fromkeys(r["stage"] for r in rows)})
for label, (baseline, candidate) in groups.items():
    if any(result["runs"][n]["state"] != "complete" for n in baseline + candidate):
        result["comparisons"][label] = {"state": "pending"}
        continue
    result["comparisons"][label] = {"state": "complete", "baseline": baseline, "candidate": candidate}
    for key in ("pipeline_ms", "core_ms", "repair_ms"):
        a, b = [statistics.mean(result["runs"][n]["timing"][key] for n in group)
                for group in (baseline, candidate)]
        result["comparisons"][label][key] = dict(baseline=a, candidate=b, change_percent=100 * (b / a - 1))
default = ROOT / "endoscopic_tissue/native/bin-scissors/LiverCudaSim.dll"
result["default_dll_sha256"] = hashlib.sha256(default.read_bytes()).hexdigest()
(DATA / "experiment-summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result["comparisons"], indent=2))
