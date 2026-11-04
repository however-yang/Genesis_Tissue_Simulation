"""Check recorded evidence; this does not certify the baseline's surgical realism."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT / "endoscopic_tissue/outputs/perf-opt-20261030"
HERE=Path(__file__).resolve().parent


def read(folder,name):
    return json.loads((OUT/folder/name).read_text())


baseline=read("baseline-video-v1","render-timing.json")["rows"]
assert len(baseline)==1111
for folder in ("persistent-video-v2","persistent-video-repeat","persistent-video-blas"):
    rows=read(folder,"render-timing.json")["rows"]
    assert len(rows)==len(baseline)
    assert all((a["step"],a["frame_sha256"])==(b["step"],b["frame_sha256"]) for a,b in zip(baseline,rows))
print("PASS: all optimized replays match all 1111 first-baseline raw frames")
videos=[hashlib.sha256((OUT/f/"interactions.mp4").read_bytes()).hexdigest() for f in
        ("baseline-video-v1","baseline-video-repeat","baseline-video-blas","persistent-video-v2","persistent-video-repeat","persistent-video-blas")]
assert len(set(videos))==1
print("PASS: six complete encoded videos are byte-identical")
for suffix in ("cases","knife-cases"):
    a,b=read("baseline-"+suffix,"hashes.json"),read("persistent-"+suffix,"hashes.json")
    assert len(a)==9 and a==b
print("PASS: 18 boundary-display checks match baseline pixels")
for suffix in ("cases","knife-cases"):
    assert read("baseline-"+suffix,"hashes.json")==read("persistent-blas-"+suffix,"hashes.json")
print("PASS: boundary-display checks also match with bounded BLAS threads")
for target in ("contact/surface_contact.cuh","xpbd/xpbd_solver.cu","cutting/pressure_blade.cuh"):
    path="endoscopic_tissue/native/vendor/cuda/"+target
    frozen=subprocess.run(["git","show","c64c1a02ac0b6f004a36e370f14a6bfae7e76091:"+path],
                          cwd=ROOT,check=True,capture_output=True).stdout
    assert frozen.replace(b"\r\n",b"\n")== (ROOT/path).read_bytes().replace(b"\r\n",b"\n")
assert hashlib.sha256((ROOT/"endoscopic_tissue/native/bin-scissors/LiverCudaSim.dll").read_bytes()).hexdigest()=="0d09f56a1f4acb813424cd1dee3233d89ce7ef5c0d45e64f079c7d7387912225"
print("PASS: native solver sources and default DLL remain the protected baseline")
assert (ROOT/"endoscopic_tissue/rendering.py").read_bytes()==(HERE/"persistent_rendering_v2.py").read_bytes()
print("PASS: the default renderer is the exact validated candidate")
for folder in ("threaded-blas-baseline",*[f"threaded-blas-persistent-{i}" for i in range(1,7)]):
    r=read(folder,"report.json")
    m=r["metrics"]
    assert r["steps"]==1110 and m["completed"]
    assert m["fractures"]==2 and not m["grip_lost_after_capture"]
    assert m["new_edges_during_withdrawal"]==0 and m["tumor_nodes_left"]==0
    assert m["max_blade_body_overlap_m"]<=.0002 and m["two_blade_squeeze_m"]>=.0005
    pools=read(folder,"threaded-timing.json")
    assert all(p["num_threads"]==1 for p in pools["thread_pools_inside"] if p["user_api"]=="blas")
    if "thread_pools_before" in pools:
        before={p["filepath"]:p["num_threads"] for p in pools["thread_pools_before"] if p["user_api"]=="blas"}
        after={p["filepath"]:p["num_threads"] for p in pools["thread_pools_after"] if p["user_api"]=="blas"}
        assert before==after
    print(f"{folder}: completed integrated run, original acceptance={m['acceptance_passed']}, stray skin cuts={m['stray_skin_cuts']}")
for folder in ("review-collision","review-rendering"):
    m=read(folder,"manifest.json")
    assert m["frames"]==1111 and m["fps"]==30
    assert (OUT/folder/"comparison.mp4").stat().st_size>0
print("PASS: both synchronized comparison videos contain the complete interaction sequence")
