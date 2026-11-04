# Runs a scissors flow once per listed build and prints scene_line.py's summary line of each run.
# Usage: bash scene_runs.sh <scene> <prefix> <dll folder under endoscopic_tissue/native>...   (run k is <prefix>-<k>-<build>)
S=$1; P=$2; shift 2; O=D:/Simulation/genesis-world/endoscopic_tissue/outputs/stage56-runs; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
N=D:/Simulation/genesis-world/endoscopic_tissue/native
cd /d/Simulation/genesis-world
k=0; for build in "$@"; do k=$((k+1)); name="$P-$k-$build"
  PYTHONPATH=. timeout 1200 $PY -m endoscopic_tissue.run --scene $S --native-only --dll $N/$build/LiverCudaSim.dll --output "$O/$name" > "$O/$name.log" 2>&1
  rm -f "${O:?}/${name:?}"/*.npz
  $PY endoscopic_tissue_backups/2026-10-30-stage56/scene_line.py "$O/$name"
done
