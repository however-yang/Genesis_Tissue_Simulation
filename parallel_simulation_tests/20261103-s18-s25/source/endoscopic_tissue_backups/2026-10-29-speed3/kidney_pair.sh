# One kidney run of the committed default (committed Python) and one of a test build (working Python), back to back.
# Usage: bash kidney_pair.sh <prefix> <dll folder under endoscopic_tissue/native>
P=$1; DLL=D:/Simulation/genesis-world/endoscopic_tissue/native/$2/LiverCudaSim.dll
O="D:/Simulation/genesis-world/endoscopic_tissue/outputs/speed2-runs"; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
BASE=D:/Simulation/genesis-world/endoscopic_tissue/native/bin-scissors/LiverCudaSim.dll
cd /d/Simulation/base-py && PYTHONPATH=. timeout 2400 $PY endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py --scene kidney --native-only --dll $BASE --output "$O/$P-base" > "$O/$P-base.log" 2>&1
rm -f "$O/$P-base"/*.npz
cd /d/Simulation/genesis-world && PYTHONPATH=. timeout 2400 $PY endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py --scene kidney --native-only --dll $DLL --output "$O/$P-new" > "$O/$P-new.log" 2>&1
rm -f "$O/$P-new"/*.npz
for r in base new; do $PY -c "
import json; r = json.load(open(r'$O/$P-$r/report.json')); m = r['metrics']
print('$P-$r', r['status'], 'elapsed %.1f' % r['elapsed_s'], 'overlap %.3f' % (m['max_knife_body_penetration_m']*1000), 'pieces', m.get('material_components'), m.get('lattice_pieces'), 'tumor', m.get('tumor_nodes_left'), 'stray', m.get('stray_skin_cuts'), 'snips', [(round(x['fracture_left_n'],3), round(x['fracture_right_n'],3), round(x['two_blade_squeeze_m']*1000,2)) for x in m.get('snips', [])])"; done
