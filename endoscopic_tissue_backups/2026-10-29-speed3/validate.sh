# Validation batch for a test build: wall x10, flap x5, kidney x$3 (dev runner), wall-traction, knife, unit tests.
# Usage: bash validate.sh <prefix> <dll folder under endoscopic_tissue/native> <kidney runs>
P=$1; DLL=endoscopic_tissue/native/$2/LiverCudaSim.dll; K=${3:-6}
O="D:/Simulation/genesis-world/endoscopic_tissue/outputs/speed2-runs"; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
cd /d/Simulation/genesis-world
scene_line() {  # name, scene dir
  $PY -c "
import json
r = json.load(open(r'$2/report.json')); m = r['metrics']
snips = [(round(x['fracture_left_n'], 2), round(x['fracture_right_n'], 2), round(x['two_blade_squeeze_m']*1000, 1)) for x in m.get('snips', [])]
print('$1', r['status'], 'overlap %.3f mm' % (m['max_knife_body_penetration_m']*1000), 'squeeze %.2f' % (m['two_blade_squeeze_m']*1000), 'uncut', m.get('uncut_columns'), 'multi', m.get('multiply_cut_columns'), 'stray', m.get('stray_skin_cuts'), 'pieces', m.get('material_components'), m.get('lattice_pieces'), 'tumor', m.get('tumor_nodes_left'), 'snips', snips, 'elapsed %.0f' % r['elapsed_s'], (r.get('error') or '')[:60], flush=True)"
}
for k in $(seq 1 10); do
  PYTHONPATH=. timeout 1200 $PY -m endoscopic_tissue.run --scene wall --native-only --dll $DLL --output "$O/$P-wall-$k" > "$O/$P-wall-$k.log" 2>&1; scene_line $P-wall-$k "$O/$P-wall-$k"; rm -f "$O/$P-wall-$k"/*.npz
  if [ $k -le 5 ]; then PYTHONPATH=. timeout 1200 $PY -m endoscopic_tissue.run --scene flap --native-only --dll $DLL --output "$O/$P-flap-$k" > "$O/$P-flap-$k.log" 2>&1; scene_line $P-flap-$k "$O/$P-flap-$k"; rm -f "$O/$P-flap-$k"/*.npz; fi
  if [ $k -le $K ]; then PYTHONPATH=. timeout 2400 $PY endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py --scene kidney --native-only --dll $DLL --output "$O/$P-kidney-$k" > "$O/$P-kidney-$k.log" 2>&1; scene_line $P-kidney-$k "$O/$P-kidney-$k"; rm -f "$O/$P-kidney-$k"/*.npz; fi
done
PYTHONPATH=. timeout 1200 $PY -m endoscopic_tissue.run --scene wall-traction --native-only --dll $DLL --output "$O/$P-traction" > "$O/$P-traction.log" 2>&1; scene_line $P-wall-traction "$O/$P-traction"; rm -f "$O/$P-traction"/*.npz
PYTHONPATH=. timeout 1200 $PY -m endoscopic_tissue.run --tool knife --native-only --dll $DLL --output "$O/$P-knife" > "$O/$P-knife.log" 2>&1; $PY -c "
import json; r = json.load(open(r'$O/$P-knife/report.json')); print('$P-knife', r['status'], 'elapsed %.1f' % r['elapsed_s'], flush=True)"; rm -f "$O/$P-knife"/*.npz
PYTHONPATH=. $PY -m unittest endoscopic_tissue.test_core > "$O/$P-tests.log" 2>&1; echo "tests exit $?"; tail -2 "$O/$P-tests.log"
