# Runs the wall flow once per listed build and prints the report line: status, blade-back overlap, uncut columns, snips.
# Usage: bash wall_runs.sh <prefix> <dll folder under endoscopic_tissue/native>...
P=$1; shift; O="D:/Simulation/genesis-world/endoscopic_tissue/outputs/crease-runs"; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
N=D:/Simulation/genesis-world/endoscopic_tissue/native
cd /d/Simulation/genesis-world
k=0; for build in "$@"; do k=$((k+1)); name="$P-$k-$build"
  PYTHONPATH=. timeout 1200 $PY -m endoscopic_tissue.run --scene wall --native-only --dll $N/$build/LiverCudaSim.dll --output "$O/$name" > "$O/$name.log" 2>&1
  rm -f "$O/$name"/*.npz
  echo "$name: $($PY -c "
import json; r = json.load(open(r'$O/$name/report.json')); m = r['metrics']
print(r['status'], 'overlap %.3f' % (m['max_knife_body_penetration_m'] * 1000), 'uncut', m.get('uncut_columns'), 'snips', [(round(x['fracture_left_n'], 2), round(x['fracture_right_n'], 2), round(x['two_blade_squeeze_m'] * 1000, 1)) for x in m.get('snips', [])])")"
done
