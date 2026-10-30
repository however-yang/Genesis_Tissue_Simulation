"""Run an explicit list of experiments serially so they never compete for the GPU."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf8"))
results = []
for job in plan:
    output = ROOT / job["output"]
    log = output.with_suffix(".log")
    args = [sys.executable, "-X", "utf8", str(ROOT / job["script"]), str(output), *job["args"]]
    print(f"[BATCH] starting {output.name}", flush=True)
    with log.open("w", encoding="utf8") as stream:
        result = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    entry = dict(output=str(output), exit_code=result.returncode, status=report.get("status"),
                 steps=report.get("steps"), error=report.get("error"), metrics=report.get("metrics"))
    results.append(entry)
    Path(sys.argv[1]).with_suffix(".results.json").write_text(json.dumps(results, indent=2), encoding="utf8")
    print(f"[BATCH] {output.name}: exit={result.returncode}, steps={entry['steps']}, status={entry['status']}", flush=True)
    if not report_path.exists() and not job.get("continue_without_report",False):
        raise RuntimeError(f"No simulation report: inspect {log}")
