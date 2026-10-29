# Per-call timing of two test builds on the kidney flow (working Python), keeping the node snapshots of the first stages
# so their positions can be compared. Usage: bash exact_pair.sh <prefix> <dll folder A> <dll folder B>
P=$1; O="D:/Simulation/genesis-world/endoscopic_tissue/outputs/speed2-runs"; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
T=D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-29-speed3/timing_calls.py
cd /d/Simulation/genesis-world
for spec in "a $2" "b $3"; do set -- $spec
  PYTHONPATH=. timeout 2400 $PY $T --scene kidney --native-only --dll D:/Simulation/genesis-world/endoscopic_tissue/native/$2/LiverCudaSim.dll --output "$O/$P-$1" > "$O/$P-$1.log" 2>&1
  find "$O/$P-$1" -name "*.npz" ! -name "settle.npz" ! -name "approach.npz" ! -name "bite.npz" -delete
done
