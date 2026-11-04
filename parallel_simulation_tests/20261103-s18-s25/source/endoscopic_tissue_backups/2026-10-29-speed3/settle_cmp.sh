# Runs the first 30 kidney steps (the settle, the only stage where the whole kidney is free) with each listed build and
# keeps the settle snapshot, so the settled shapes can be compared node by node.
# Usage: bash settle_cmp.sh <prefix> <dll folder>...   ("base" runs the committed Python with bin-scissors)
P=$1; shift; O="D:/Simulation/genesis-world/endoscopic_tissue/outputs/speed2-runs"; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
N=D:/Simulation/genesis-world/endoscopic_tissue/native
k=0; for build in "$@"; do k=$((k+1))
  if [ "$build" = base ]; then root=/d/Simulation/base-py; dll=$N/bin-scissors/LiverCudaSim.dll; else root=/d/Simulation/genesis-world; dll=$N/$build/LiverCudaSim.dll; fi
  cd $root && PYTHONPATH=. timeout 1200 $PY endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py --scene kidney --native-only --steps 30 \
    --dll $dll --output "$O/$P-$k-$build" > "$O/$P-$k-$build.log" 2>&1
done
