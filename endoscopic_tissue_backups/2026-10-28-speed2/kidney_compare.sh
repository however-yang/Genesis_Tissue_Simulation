# Interleaved kidney runs on the committed and half-precision builds. Keeps the snapshots of runs that end in more than
# two pieces, for finding where a fragment comes from.
O="D:/Simulation/genesis-world/endoscopic_tissue/outputs/speed2-runs"; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
cd /d/Simulation/genesis-world
for k in $(seq $1 $2); do for spec in "orig bin-orig" "half bin-half"; do set -- $spec; name=kcmp-$1-$k
  PYTHONPATH=. timeout 2400 $PY endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py --scene kidney --native-only --dll endoscopic_tissue/native/$2/LiverCudaSim.dll --output "$O/$name" > "$O/$name.log" 2>&1
  pieces=$($PY -c "
import json; r = json.load(open(r'$O/$name/report.json')); m = r['metrics']
print(m['lattice_pieces'])")
  $PY -c "
import json; r = json.load(open(r'$O/$name/report.json')); m = r['metrics']
print('$name', 'pieces', m['material_components'], m['lattice_pieces'], 'overlap %.3f' % (m['max_knife_body_penetration_m']*1000), 'stray', m['stray_skin_cuts'], 'elapsed %.0f' % r['elapsed_s'], 'snips', [(round(x['fracture_left_n'],2), round(x['fracture_right_n'],2), round(x['two_blade_squeeze_m']*1000,1)) for x in m['snips']], (r.get('error') or '')[:40], flush=True)"
  if [ "$pieces" = "2" ]; then rm -f "$O/$name"/*.npz; fi
done; done
