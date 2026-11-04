from pathlib import Path
import hashlib,json
h=Path(__file__).resolve().parent;b=h.parent
s=(b/'benchmark.py').read_text().replace("str(ROOT/'worker.py')","str(ROOT/'continuation/topology_capture_worker.py')")
(b/'topology_benchmark.py').write_text(s)
s=(b/'mps_benchmark.py').read_text().replace("str(ROOT/'benchmark.py')","str(ROOT/'topology_benchmark.py')")
(b/'topology_mps.py').write_text(s)
q=json.loads((h/'restore-profile-plan.json').read_text());q['purpose']='Independent restore timing and exact-order fragment-tree diagnostic; no physical adoption'
python=str(h.parents[2]/'envs/genesis-py311-cu125-v1/bin/python')
q['jobs'].append(dict(id='fragment-tree-equivalence',args=[python,'-B','-u',str(h/'probe_fragment_tree.py'),'--output','{attempt}/probe'],cwd=str(h),max_idle=300,max_wall=900,results=[dict(path='{attempt}/probe/result.json',require={'completed':True,'tests':16,'passed':True})]))
for name in ['probe_fragment_tree.py','fragment_tree_fused.cuh']:q['inputs_sha256'][str(h/name)]=hashlib.sha256((h/name).read_bytes()).hexdigest()
(h/'diagnostic-followup-plan.json').write_text(json.dumps(q,indent=2)+'\n')
(h/'topology-reproduction-manifest.json').write_text(json.dumps(dict(benchmark_sha256=hashlib.sha256((b/'topology_benchmark.py').read_bytes()).hexdigest(),mps_wrapper_sha256=hashlib.sha256((b/'topology_mps.py').read_bytes()).hexdigest(),worker_sha256=hashlib.sha256((h/'topology_capture_worker.py').read_bytes()).hexdigest(),changed_physics=False,scope='32 full wall three-snips repeat; capture first rejected full mesh after original quality assessment. Diagnostic run, not comparable controlled FPS.'),indent=2))
print('Prepared follow-up diagnostics and separate 32-worker topology-capture driver')
