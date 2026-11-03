from pathlib import Path
import json,hashlib,shutil
h=Path(__file__).resolve().parent;p=h.parents[2]
base=p/'runs/a100-cone-repair-20261102-r1/verify_cone_gpu.py'
s=base.read_text().replace('actual R8 CUDA algebra','R9 per-vertex candidate CUDA algebra')
s=s.replace("assert here == root/'runs/a100-cone-repair-20261102-r1'", "assert here == root/'parallel_simulation_tests/20261103-scaleup/continuation'")
s=s.replace("out = here/'algebra-check'", "out = Path(os.environ['ACCEPTANCE_OUTPUT'])")
s=s.replace('repair_cone.cuh','vertex_cone_algebra.cuh').replace("here/'source/vendor", "here/'r9-vertex-cone/source/vendor")
(h/'verify_vertex_cone_gpu.py').write_text(s)
shutil.copyfile(p/'runs/a100-functional-fixes-20261102-r1/snapshot_matrix.py',h/'snapshot_matrix.py')
plan=json.loads((h/'recovery-plan.json').read_text());plan['purpose']='R9 experimental per-vertex feasible direction: algebra and fail-fast 20-pair regrasp, original thresholds, no adoption'
plan['jobs']=[dict(id='r9-algebra',args=[str(p/'envs/genesis-py311-cu125-v1/bin/python'),'-B','-u',str(h/'verify_vertex_cone_gpu.py')],cwd=str(h),env={'ACCEPTANCE_OUTPUT':'{attempt}/algebra'},max_idle=300,max_wall=600,results=[dict(path='{attempt}/algebra/result.json',require={'cases':72},acceptance_field='passed')]),dict(id='r9-regrasp-matrix',args=[str(p/'envs/genesis-py311-cu125-v1/bin/python'),'-B','-u',str(h/'snapshot_matrix.py'),'--release',str(p/'releases/0643e10991cf'),'--library-dir',str(h/'r9-vertex-cone/build/lib'),'--fork','regrasp','--output','{attempt}/matrix'],cwd=str(p/'releases/0643e10991cf'),max_idle=900,max_wall=14400,allowed_exit_codes=[0,2],progress_globs=['{attempt}/matrix/**/*.json'],results=[dict(path='{attempt}/matrix/result.json',require={'required_pairs':20,'full_surface_checks':True},acceptance_field='passed')])]
paths=[h/'verify_vertex_cone_gpu.py',h/'snapshot_matrix.py',h/'r9-vertex-cone/source/vendor/cuda/solver/material/vertex_cone_algebra.cuh',h/'r9-vertex-cone/build/lib/libLiverCudaSim.so']
plan['inputs_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
(h/'r9-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print('Pinned R9 validation plan prepared')
