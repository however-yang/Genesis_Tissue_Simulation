# Runs the probed kidney flow through the blade approach (450 steps) once per listed build and prints grasp_line.py's
# line for each. Usage: bash quick_probe.sh <prefix> <dll folder under endoscopic_tissue/native>...
P=$1; shift; O=D:/Simulation/genesis-world/endoscopic_tissue/outputs/stage56-runs; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
N=D:/Simulation/genesis-world/endoscopic_tissue/native
cd /d/Simulation/genesis-world
k=0; for build in "$@"; do k=$((k+1)); name="$P-$k-$build"
  PYTHONPATH=. timeout 1200 $PY endoscopic_tissue_backups/2026-10-29-stage3/realism_probe.py "$O/$name.jsonl" --scene kidney --native-only \
    --steps 450 --dll $N/$build/LiverCudaSim.dll --output "$O/$name" > "$O/$name.log" 2>&1
  rm -f "${O:?}/${name:?}"/*.npz "${O:?}/${name:?}-masks.npz"
  $PY endoscopic_tissue_backups/2026-10-30-stage56/grasp_line.py "$O/$name.jsonl"
done
