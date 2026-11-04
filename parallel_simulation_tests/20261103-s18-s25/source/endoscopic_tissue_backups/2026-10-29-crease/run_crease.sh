# Runs the kidney flow once per listed build with the per-node dump of debug_crease_dump.py at frames
# $FRAMES (default 760,840,1110), keeps the reposition_2, incise_2 and hold snapshots, and prints the crease of hold.
# Usage: bash run_crease.sh <prefix> <dll folder under endoscopic_tissue/native>...   (run k of a build is <prefix>-<k>-<build>)
P=$1; shift; O="D:/Simulation/genesis-world/endoscopic_tissue/outputs/crease-runs"; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
N=D:/Simulation/genesis-world/endoscopic_tissue/native; B=D:/Simulation/genesis-world/endoscopic_tissue_backups
cd /d/Simulation/genesis-world
k=0; for build in "$@"; do k=$((k+1))
  name="$P-$k-$build"; mkdir -p "$O/$name-dump"
  LCS_DEBUG_FRAMES=${FRAMES:-760,840,1110} LCS_DEBUG_DIR="$O/$name-dump" timeout 1800 $PY $B/2026-10-28-kidney/dev_run.py --scene kidney \
    --native-only --dll $N/$build/LiverCudaSim.dll --output "$O/$name" > "$O/$name.log" 2>&1
  for f in "$O/$name"/*.npz; do case "$(basename "$f")" in reposition_2.npz|incise_2.npz|hold.npz) ;; *) rm -f "$f";; esac; done
  echo "$name: $($PY $B/2026-10-29-speed3/tumor_crease.py "$O/$name/hold.npz" 2>&1 | sed 's/.*: pieces/pieces/')"
done
