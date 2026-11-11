"""Reproduce the research counters and verify baseline protection; no GPU needed."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
DATA = ROOT / "endoscopic_tissue/outputs/module-research-20261031"


@dataclass
class WorkSummary:
    stage: str
    steps: int
    queries: int
    broad_pass: int
    valid: int
    witness: int
    blade_witness: int
    ccd_calls: int
    ccd_general: int
    ccd_translation: int
    ccd_iterations: int
    ccd_max: int
    ccd_hist: list[int]
    sum: list[int]


@dataclass
class ShadowSummary:
    stage: str
    steps: int
    comparisons: int
    closer_over_1e_5_cm: int
    farther_over_1e_5_cm: int
    point_change_over_1e_4_cm: int
    endpoint_skin_class_changed: int
    invalid_bary_or_distance: int
    max_distance_change_mm: float
    max_point_change_mm: float
    max_normal_cosine_loss: float


@dataclass
class CaptureSummary:
    metadata: dict
    all: WorkSummary | ShadowSummary
    stages: list[WorkSummary] | list[ShadowSummary]


def bit_float(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def summarize(rows, stage):
    total = [sum(row["counters"][i] for row in rows) for i in range(96)]
    total[69] = max(row["counters"][69] for row in rows)
    return WorkSummary(stage, len(rows), sum(total[:64:8]), sum(total[1:64:8]), sum(total[6:64:8]),
                       total[64], total[65], total[67], total[86], total[66], total[68], total[69],
                       total[70:86], total)


def summarize_shadow(rows, stage):
    counts = [sum(row["counters"][i] for row in rows) for i in range(87, 93)]
    maximum = [bit_float(max(row["counters"][i] for row in rows)) for i in range(93, 96)]
    return ShadowSummary(stage, len(rows), *counts, 10 * maximum[0], 10 * maximum[1], maximum[2])


def verify_rows(rows):
    assert len(rows) == 1110
    assert [row["step"] for row in rows] == list(range(1, 1111))
    for row in rows:
        values = row["counters"]
        assert len(values) == 96 and all(isinstance(value, int) and value >= 0 for value in values)
        assert sum(values[70:86]) == values[67]
        for offset in range(0, 64, 8):
            assert values[offset + 6] <= values[offset + 1] <= values[offset + 7] <= values[offset]
            assert sum(values[offset + 2:offset + 6]) == values[offset + 1]


def process_capture(name, function, destination):
    source = json.loads((DATA / name / "contact-work.json").read_text(encoding="utf8"))
    rows = source["rows"]
    verify_rows(rows)
    report = json.loads((DATA / name / "report.json").read_text(encoding="utf8"))
    assert report["steps"] == 1110 and report["metrics"]["completed"]
    assert not report["metrics"]["acceptance_passed"]
    assert hashlib.sha256(Path(source["metadata"]["dll"]).read_bytes()).hexdigest() == source["metadata"]["dll_sha256"]
    stages = list(dict.fromkeys(row["stage"] for row in rows))
    result = CaptureSummary(source["metadata"], function(rows, "all"),
                            [function([row for row in rows if row["stage"] == stage], stage) for stage in stages])
    serialized = json.dumps(asdict(result), indent=2)
    (WORK / destination).write_text(serialized, encoding="utf8")
    (DATA / destination).write_text(serialized, encoding="utf8")
    print(f"{name}: {len(rows)} steps, integer counter invariants passed; existing quality acceptance remains false")
    return result


def verify_production():
    native = ROOT / "endoscopic_tissue/native"
    expected = (
        ("bin-scissors/LiverCudaSim.dll", "0d09f56a1f4acb813424cd1dee3233d89ce7ef5c0d45e64f079c7d7387912225"),
        ("bin/LiverCudaSim.dll", "2632a926b385d9c9a81d5f7d7ebb75fe0501917a9cf83d0c19d3a29c62af961b"),
    )
    for path, digest in expected:
        assert hashlib.sha256((native / path).read_bytes()).hexdigest() == digest
    subprocess.run(["git", "diff", "--exit-code", "fac3c7c", "--", "endoscopic_tissue/native/vendor/cuda"], cwd=ROOT, check=True)
    for name in ("contact-work-counters.patch", "wrist-shadow-probe.patch"):
        subprocess.run(["git", "apply", "--check", str(WORK / name)], cwd=ROOT, check=True)
    print("Production CUDA source and both default DLLs unchanged; two standalone patches apply cleanly")


if __name__ == "__main__":
    process_capture("contact-work", summarize, "contact-work-summary.json")
    process_capture("wrist-shadow", summarize_shadow, "wrist-shadow-summary.json")
    verify_production()
