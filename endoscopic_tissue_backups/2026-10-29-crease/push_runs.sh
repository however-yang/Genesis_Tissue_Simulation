# Runs push_run.py once per listed build with LCS_DEBUG_PUSH=$PUSH (default 0,0,-2,765,800) and dumps at $FRAMES,
# in parallel, and prints each run's notch series when all are done.
# Usage: bash push_runs.sh <prefix> <dll folder under endoscopic_tissue/native>...   (run of a build is <prefix>-<build>)
P=$1; shift; O="D:/Simulation/genesis-world/endoscopic_tissue/outputs/crease-runs"; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
N=D:/Simulation/genesis-world/endoscopic_tissue/native; B=D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-29-crease
cd /d/Simulation/genesis-world
names=()
for build in "$@"; do name="$P-$build"; names+=("$O/$name"); mkdir -p "$O/$name-dump"
  LCS_DEBUG_PUSH=${PUSH:-0,0,-2,765,800} LCS_DEBUG_FRAMES=${FRAMES:-330,760,799,802,805,810,820,840,900,1110} LCS_DEBUG_DIR="$O/$name-dump" \
    timeout 2400 $PY $B/push_run.py --scene kidney --native-only --dll $N/$build/LiverCudaSim.dll --output "$O/$name" > "$O/$name.log" 2>&1 &
  sleep 10
done
wait
$PY $B/notch_series.py "${names[@]}"
