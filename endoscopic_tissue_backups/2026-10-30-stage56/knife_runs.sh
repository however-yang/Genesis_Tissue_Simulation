# Runs the knife flow once per listed build and prints its status and last stage.
# Usage: bash knife_runs.sh <prefix> <dll folder under endoscopic_tissue/native>...
P=$1; shift; O=D:/Simulation/genesis-world/endoscopic_tissue/outputs/stage56-runs; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
cd /d/Simulation/genesis-world
k=0; for build in "$@"; do k=$((k+1)); n="$P-$k-$build"
  PYTHONPATH=. timeout 600 $PY -m endoscopic_tissue.run --tool knife --native-only --dll endoscopic_tissue/native/$build/LiverCudaSim.dll --output "$O/$n" > "$O/$n.log" 2>&1
  rm -f "${O:?}/${n:?}"/*.npz
  echo "$n $($PY -c "
import json; r = json.load(open(r'$O/$n/report.json')); m = r['metrics']; print(r['status'], r.get('last_stage'), 'overlap %.3f' % (m['max_knife_body_penetration_m'] * 1000), 'grip', m.get('final_grip_points'))")"
done
