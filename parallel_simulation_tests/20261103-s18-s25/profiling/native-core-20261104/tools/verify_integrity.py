import argparse,json,hashlib,time,os
from pathlib import Path
p=Path(__file__).resolve().parents[1];os.sched_setaffinity(0,[30,31]);parser=argparse.ArgumentParser();parser.add_argument('--candidate-run',required=True);a=parser.parse_args();m=json.loads((p/'manifest.json').read_text());root=p.parents[1];prof=p.parent
hashfile=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
original=json.loads((prof/'baseline-manifest.json').read_text());checks={}
for name,base,hashes in [('original_s18_s25_source',root/'source',original['source_sha256']),('frozen_original_baseline',prof/'baseline-source',original['source_sha256']),('previous_final_python',Path(m['baseline_source']),m['source_sha256'])]:
 bad=[rel for rel,h in hashes.items() if not (base/rel).is_file() or hashfile(base/rel)!=h];checks[name]=dict(files_checked=len(hashes),mismatches=bad)
checks['original_dll_unchanged']=hashfile(root/'native-s25/build/lib/libLiverCudaSim.so')==original['native_sha256']
checks['previous_final_dll_unchanged']=hashfile(Path(m['baseline_dll']))==m['baseline_dll_sha256']
checks['passed']=all(not v['mismatches'] for v in checks.values() if isinstance(v,dict)) and checks['original_dll_unchanged'] and checks['previous_final_dll_unchanged'];checks['checked_unix_s']=time.time();(p/'results/source-integrity.json').write_text(json.dumps(checks,indent=2));assert checks['passed'],checks
before=json.loads((p/'results/baseline-rgbd-initial/context.json').read_text());after=json.loads((p/'results'/a.candidate_run/'context.json').read_text());keys=['settings','canonical_settings','native_physics_parameters','scene_config','action_bounds','quality_limits','cutting_enabled','full_surface_checks','instrument_collision_checks','outcome_semantics','node_count','material_node_count','active_node_count','gpu_topology_checks']
match={k:before[k]==after[k] for k in keys};(p/'results/parameter-equivalence.json').write_text(json.dumps(match,indent=2));assert all(match.values()),match
print('source integrity and physical parameters',checks['passed'],all(match.values()))
