# Runs the first 166 kidney steps (settle, approach, bite) with each listed build, keeping the stage snapshots, so that
# runs of one build and of two builds can be compared node by node. Usage: bash repeat_prefix.sh <prefix> <dll folder>...
P=$1; shift; O="D:/Simulation/genesis-world/endoscopic_tissue/outputs/speed2-runs"; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
cd /d/Simulation/genesis-world
k=0; for dll in "$@"; do k=$((k+1))
  PYTHONPATH=. timeout 1200 $PY endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py --scene kidney --native-only --steps 166 \
    --dll D:/Simulation/genesis-world/endoscopic_tissue/native/$dll/LiverCudaSim.dll --output "$O/$P-$k-$dll" > "$O/$P-$k-$dll.log" 2>&1
done
