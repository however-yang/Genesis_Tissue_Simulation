"""Serial diagnostic, capture and full-replay review; never a timing benchmark."""
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
work = Path(__file__).resolve().parent
data = root / "endoscopic_tissue/outputs/wrist-opt-20261031"
perf = root / "endoscopic_tissue_backups/2026-10-30-perf-opt"
commands = [[str(perf / "run_batch.py"), str(work / "dedicated-quality-jobs.json")]]
for name in ("baseline", "initial", "dedicated"):
    commands.append([str(perf / "replay_single_blas.py"), str(data / (name + "-capture/replay.bin")), str(data / (name + "-video"))])
for name in ("initial", "dedicated"):
    commands.append([str(perf / "make_review.py"), str(data / "baseline-video/interactions.mp4"), str(data / (name + "-video/interactions.mp4")), str(data / (name + "-review")), name.upper()])
    commands.append([str(work / "review_motion.py"), str(data / "baseline-capture/motion.jsonl"), str(data / (name + "-capture/motion.jsonl")), str(data / (name + "-review"))])
results = []
for i, command in enumerate(commands):
    print(f"[REVIEW] {i+1}/{len(commands)} {Path(command[0]).name}", flush=True)
    with (work / f"review-job-{i+1}.log").open("w", encoding="utf8") as log:
        result = subprocess.run([sys.executable, "-X", "utf8", *command], cwd=root, stdout=log, stderr=subprocess.STDOUT)
    results.append(dict(args=command, exit_code=result.returncode))
    (work / "review-jobs.results.json").write_text(json.dumps(results, indent=2), encoding="utf8")
    if result.returncode:
        raise SystemExit(result.returncode)
print("[REVIEW] all complete", flush=True)
