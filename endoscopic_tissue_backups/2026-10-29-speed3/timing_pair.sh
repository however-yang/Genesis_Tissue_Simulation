# Per-call timing of one kidney run of the committed default (committed Python) and one of a test build.
# Usage: bash timing_pair.sh <prefix> <dll folder under endoscopic_tissue/native> [base|new|both]
P=$1; DLL=D:/Simulation/genesis-world/endoscopic_tissue/native/$2/LiverCudaSim.dll; WHICH=${3:-both}
O="D:/Simulation/genesis-world/endoscopic_tissue/outputs/speed2-runs"; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
T=D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-29-speed3/timing_calls.py
BASE=D:/Simulation/genesis-world/endoscopic_tissue/native/bin-scissors/LiverCudaSim.dll
if [ "$WHICH" != new ]; then
  cd /d/Simulation/base-py && PYTHONPATH=. timeout 2400 $PY $T --scene kidney --native-only --dll $BASE --output "$O/$P-base" > "$O/$P-base.log" 2>&1
  rm -f "$O/$P-base"/*.npz
fi
if [ "$WHICH" != base ]; then
  cd /d/Simulation/genesis-world && PYTHONPATH=. timeout 2400 $PY $T --scene kidney --native-only --dll $DLL --output "$O/$P-new" > "$O/$P-new.log" 2>&1
  rm -f "$O/$P-new"/*.npz
fi
