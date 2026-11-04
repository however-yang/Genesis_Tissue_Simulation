# Runs the kidney flow through the end of the lift (330 steps) with each listed build and prints the jaw gap and grip
# error at the end of close and lift: the grasp equilibrium, which is deterministic before the first cut.
# Usage: bash lift_gap.sh <prefix> <dll folder>...   ("base" runs the committed Python with bin-scissors)
P=$1; shift; O="D:/Simulation/genesis-world/endoscopic_tissue/outputs/speed2-runs"; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
N=D:/Simulation/genesis-world/endoscopic_tissue/native
k=0; for build in "$@"; do k=$((k+1))
  if [ "$build" = base ]; then root=/d/Simulation/base-py; dll=$N/bin-scissors/LiverCudaSim.dll; else root=/d/Simulation/genesis-world; dll=$N/$build/LiverCudaSim.dll; fi
  cd $root && PYTHONPATH=. timeout 1200 $PY endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py --scene kidney --native-only --steps 330 \
    --dll $dll --output "$O/$P-$k-$build" > "$O/$P-$k-$build.log" 2>&1
  rm -f "$O/$P-$k-$build"/*.npz
  $PY -c "
import json
rows = [json.loads(line) for line in open(r'$O/$P-$k-$build/trajectory.jsonl')]
ends = {r['stage']: r for r in rows}
print('$build', ' '.join('%s gap %.3f mm error %.1f um' % (s, ends[s]['jaw_gap_cm'] * 10, ends[s]['grip_error_m'] * 1e6) for s in ('close', 'lift') if s in ends), flush=True)"
done
