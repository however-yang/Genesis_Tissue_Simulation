# Interleaved wall runs over several DLL folders: wall_variants.sh <first> <last> <name:folder> ...
O="D:/Simulation/genesis-world/endoscopic_tissue/outputs/speed2-runs"; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
cd /d/Simulation/genesis-world
first=$1; last=$2; shift 2
for k in $(seq $first $last); do for spec in "$@"; do name=${spec%%:*}; folder=${spec#*:}
  PYTHONPATH=. timeout 1200 $PY -m endoscopic_tissue.run --scene wall --native-only --dll endoscopic_tissue/native/$folder/LiverCudaSim.dll --output "$O/wall-$name-$k" > "$O/wall-$name-$k.log" 2>&1
  $PY -c "
import json
r = json.load(open(r'$O/wall-$name-$k/report.json')); m = r['metrics']
steps = [json.loads(l)['knife_body_penetration_mm'] for l in open(r'$O/wall-$name-$k/trajectory.jsonl')]
print('$name-$k', r['status'], 'overlap %.3f mm' % (m['max_knife_body_penetration_m']*1000), 'steps>0.15', sum(s > .15 for s in steps), 'uncut', m['uncut_columns'], 'stray', m['stray_skin_cuts'], 'elapsed %.1f' % r['elapsed_s'], flush=True)"
  rm -f "$O/wall-$name-$k"/*.npz
done; done
