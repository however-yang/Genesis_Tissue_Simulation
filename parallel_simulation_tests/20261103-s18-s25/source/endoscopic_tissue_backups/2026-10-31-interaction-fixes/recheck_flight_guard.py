"""Recheck saved free-release states with the conservative volume witness."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import validate_strict as v
from detached_flight_guard import audit_detached_flight
for folder in map(Path,sys.argv[1:]):
 r=json.loads((folder/'report.json').read_text());_,scene,settings,_,_=v.fixture(r['metadata']['case']);grid=scene.build_grid(settings.density_kg_m3)
 assert r['healed_connections']==0 and r['cuts_while_disabled']==0
 final=np.load(folder/'open_retract.npz');rows=[json.loads(s) for s in (folder/'trajectory.jsonl').read_text().splitlines()];checks=[]
 for row in rows:
  if not row.get('absolute_envelope_exceeded'):continue
  assert row['stage'] in ('jaw_open','open_retract'),'Cannot substitute final topology during cutting'
  data=np.load(folder/('state-%04d.npz'%row['step']))
  obs=SimpleNamespace(nodes=data['nodes_cm'],neighbors=final['neighbors'],grip_flags=data['grip_flags'])
  allowed,detail=audit_detached_flight(grid,obs,np.linalg.norm(obs.nodes-grid.positions,axis=1)*10)
  checks.append(dict(step=row['step'],allowed=allowed,detail=detail));assert allowed,(folder.name,row['step'],detail)
 out=dict(case=folder.name,checks=checks,pass_all=all(x['allowed'] for x in checks),guard='Intact interior material cell + unpinned/unheld component + rigid-fit intrinsic displacement <=60 mm. Final neighbors are used only after cutting is disabled and no connections heal.')
 (folder/'conservative-flight-audit.json').write_text(json.dumps(out,indent=2));print(folder.name,len(checks),'saved event frames passed')
