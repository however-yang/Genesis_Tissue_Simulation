# Interleaved wall (and flap) runs of the committed default and the new build, under the same conditions.
# Usage: bash wall_ab.sh <first index> <last index>
O="D:/Simulation/genesis-world/endoscopic_tissue/outputs/speed2-runs"; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
cd /d/Simulation/genesis-world
line() {  # name, dir
  $PY -c "
import json
r = json.load(open(r'$2/report.json')); m = r['metrics']
print('$1', r['status'], 'overlap %.3f' % (m['max_knife_body_penetration_m']*1000), 'uncut', m.get('uncut_columns'), 'stray', m.get('stray_skin_cuts'), 'pieces', m.get('material_components'), (r.get('error') or '')[:50], flush=True)"
}
for k in $(seq $1 $2); do for spec in "base bin-scissors" "new bin-speed7"; do set -- $spec; name=ab-$1-wall-$k
  PYTHONPATH=. timeout 1200 $PY -m endoscopic_tissue.run --scene wall --native-only --dll endoscopic_tissue/native/$2/LiverCudaSim.dll --output "$O/$name" > "$O/$name.log" 2>&1; line $name "$O/$name"; rm -f "$O/$name"/*.npz
  if [ $k -le 3 ]; then name=ab-$1-flap-$k
    PYTHONPATH=. timeout 1200 $PY -m endoscopic_tissue.run --scene flap --native-only --dll endoscopic_tissue/native/$2/LiverCudaSim.dll --output "$O/$name" > "$O/$name.log" 2>&1; line $name "$O/$name"; rm -f "$O/$name"/*.npz
  fi
done; done
