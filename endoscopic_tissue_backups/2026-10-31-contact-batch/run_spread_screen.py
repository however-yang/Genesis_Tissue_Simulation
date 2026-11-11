"""Run exact-contact screening before normal timings, then record actual interaction."""
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
work = Path(__file__).resolve().parent
data = root / "endoscopic_tissue/outputs/contact-spread-20261031"
wrist = root / "endoscopic_tissue_backups/2026-10-31-wrist-opt"
perf = root / "endoscopic_tissue_backups/2026-10-30-perf-opt"
results = []
def call(label, args, allowed=(0,)):
    print("[SPREAD] " + label, flush=True)
    with (work / ("spread-"+label+".log")).open("w",encoding="utf8") as log:
        result = subprocess.run([sys.executable,"-X","utf8",*map(str,args)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
    results.append(dict(label=label,args=list(map(str,args)),exit_code=result.returncode))
    (work/"spread-screen-results.json").write_text(json.dumps(results,indent=2),encoding="utf8")
    if result.returncode not in allowed: raise SystemExit(result.returncode)

rows=json.loads((data/"spread-equivalence/contact-equivalence.json").read_text())
assert len(rows)==270 and all(r["status"]==0 and all(r[k]==0 for k in ("different_counts","different_records","different_degrees","different_members")) for r in rows), "Frozen-input equivalence failed"
call("graph-baseline",[work/"measure_solver_graphs.py",data/"graph-baseline-1","--native-only","--scene","kidney","--tool","scissors","--dll",root/"endoscopic_tissue/native/bin-graph-baseline31/LiverCudaSim.dll"],(0,1))
assert json.loads((data/"graph-baseline-1/report.json").read_text())["steps"]==1110
call("timing",[perf/"run_batch.py",work/"spread-timing-jobs.json"])
for name in ("baseline-a1","spread-b1","spread-b2","baseline-a2"):
    assert len(json.loads((data/name/"timing-steps.json").read_text()))==1110
call("capture",[wrist/"capture_motion.py",data/"spread-capture","--scene","kidney","--tool","scissors","--dll",root/"endoscopic_tissue/native/bin-contact-spread31/LiverCudaSim.dll"],(0,1))
assert json.loads((data/"spread-capture/report.json").read_text())["steps"]==1110
call("render",[perf/"replay_single_blas.py",data/"spread-capture/replay.bin",data/"spread-video"])
call("comparison",[perf/"make_review.py",root/"endoscopic_tissue/outputs/wrist-opt-20261031/baseline-video/interactions.mp4",data/"spread-video/interactions.mp4",data/"spread-review","SPREAD_BATCH"])
call("motion",[wrist/"review_motion.py",root/"endoscopic_tissue/outputs/wrist-opt-20261031/baseline-capture/motion.jsonl",data/"spread-capture/motion.jsonl",data/"spread-review"])
print("[SPREAD] screening complete",flush=True)
