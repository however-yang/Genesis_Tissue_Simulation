"""Run display verification jobs serially; preserve every subprocess log."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT / "endoscopic_tissue/outputs/perf-opt-20261030"
replay=OUT / "baseline-capture/replay.bin"
frozen=HERE / "baseline_rendering.py"
jobs=[
    ("render_replay.py",replay,OUT / "baseline-video-repeat",frozen),
    ("render_replay.py",replay,OUT / "persistent-video-repeat"),
    ("render_cases.py",replay,OUT / "baseline-cases",frozen),
    ("render_cases.py",replay,OUT / "persistent-cases"),
    ("render_cases.py",replay,OUT / "baseline-knife-cases",frozen,"knife"),
    ("render_cases.py",replay,OUT / "persistent-knife-cases",ROOT / "endoscopic_tissue/rendering.py","knife"),
]
if "--blas" in sys.argv:
    candidate=HERE / "persistent_rendering_v2.py"
    jobs=[
        ("replay_single_blas.py",replay,OUT / "baseline-video-blas",frozen),
        ("replay_single_blas.py",replay,OUT / "persistent-video-blas",candidate),
        ("cases_single_blas.py",replay,OUT / "persistent-blas-cases",candidate),
        ("cases_single_blas.py",replay,OUT / "persistent-blas-knife-cases",candidate,"knife"),
    ]
results=[]
for script,source,output,*extra in jobs:
    print(f"[VERIFY] {output.name}",flush=True)
    with output.with_suffix(".log").open("w",encoding="utf8") as stream:
        result=subprocess.run([sys.executable,"-X","utf8",str(HERE/script),str(source),str(output),
                               *map(str,extra)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    results.append(dict(output=str(output),exit_code=result.returncode))
    manifest="display-blas-verification-jobs.json" if "--blas" in sys.argv else "display-verification-jobs.json"
    (OUT / manifest).write_text(json.dumps(results,indent=2))
    if result.returncode:
        raise RuntimeError(f"Verification failed: {output}")
print("[VERIFY] all display jobs finished",flush=True)
