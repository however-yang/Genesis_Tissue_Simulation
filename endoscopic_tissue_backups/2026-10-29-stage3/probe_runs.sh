# Runs the probed kidney flow (realism_probe.py) once per listed build, one after another, and deletes the stage
# snapshots except settle, close, lift and hold, which realism_summary.py reads.
# Usage: bash probe_runs.sh <prefix> <dll folder under endoscopic_tissue/native>...   (run k of a build is <prefix>-<k>-<build>)
P=$1; shift; O="D:/Simulation/genesis-world/endoscopic_tissue/outputs/stage3-runs"; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
N=D:/Simulation/genesis-world/endoscopic_tissue/native
cd /d/Simulation/genesis-world
k=0; for build in "$@"; do k=$((k+1))
  name="$P-$k-$build"
  PYTHONPATH=. timeout 1800 $PY endoscopic_tissue_backups/2026-10-29-stage3/realism_probe.py "$O/$name.jsonl" --scene kidney --native-only \
    --dll $N/$build/LiverCudaSim.dll --output "$O/$name" > "$O/$name.log" 2>&1
  for f in "$O/$name"/*.npz; do case "$(basename "$f")" in settle.npz|close.npz|lift.npz|hold.npz) ;; *) rm -f "$f";; esac; done
  echo "$name done"
done
