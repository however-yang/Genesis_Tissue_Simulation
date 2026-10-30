# Runs the flap flow once per listed build and prints, per snip, the cut edges and the blade gap at the end of the
# incision, then the report status. Usage: bash flap_runs.sh <prefix> <dll folder under endoscopic_tissue/native>...
P=$1; shift; O="D:/Simulation/genesis-world/endoscopic_tissue/outputs/crease-runs"; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
N=D:/Simulation/genesis-world/endoscopic_tissue/native
cd /d/Simulation/genesis-world
k=0; for build in "$@"; do k=$((k+1)); name="$P-$k-$build"
  PYTHONPATH=. timeout 1200 $PY -m endoscopic_tissue.run --scene flap --native-only --dll $N/$build/LiverCudaSim.dll --output "$O/$name" > "$O/$name.log" 2>&1
  rm -f "$O/$name"/*.npz
  echo "$name: $(grep -A2 "incise_[123]$" "$O/$name.log" | grep -o "cut_edges=[0-9]*\|scissors gap=[0-9.]* mm" | tr '\n' ' ') $($PY -c "
import json; r = json.load(open(r'$O/$name/report.json')); m = r['metrics']; print(r['status'], 'uncut', m.get('uncut_columns'), 'overlap %.3f' % (m['max_knife_body_penetration_m'] * 1000))")"
done
