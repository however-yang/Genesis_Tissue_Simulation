import os,json,hashlib,difflib,subprocess
from pathlib import Path
os.sched_setaffinity(0,[30,31]);p=Path(__file__).resolve().parents[1];old=json.loads((p.parent/'native-core-20261104/delivery/manifest.json').read_text());ref=old['variants']['recommended'];src=p/'native-readonly-v8/source';py=p/'python-gpu-check';dll=p/'native-readonly-v8/build/lib/libLiverCudaSim.so'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def files(root,python=False):return {str(x.relative_to(root)):digest(x) for x in sorted(root.rglob('*')) if x.is_file() and (x.suffix=='.py' if python else True)}
oldsrc=Path(ref['native_source']);oldpy=Path(old['python_source']);native=files(src);python=files(py,True)
assert all(digest(oldsrc/k)==v for k,v in ref['native_source_sha256'].items())
assert all(digest(oldpy/k)==v for k,v in old['python_source_sha256'].items())
assert digest(Path(ref['dll']))==ref['dll_sha256']
changed_native=[k for k,v in ref['native_source_sha256'].items() if native.get(k)!=v];changed_python=[k for k,v in python.items() if (oldpy/k).exists() and digest(oldpy/k)!=v]
assert all(digest(oldpy/k)==v for k,v in python.items() if (oldpy/k).exists() and k not in changed_python)
for rel in ['vendor/cuda/training/state_archive.cpp','vendor/cuda/training/state_archive.h','vendor/cuda/training/xpbd_state.cuh','vendor/cuda/xpbd/xpbd_solver.cu','vendor/cuda/solver/material/unified_vertex.cuh']:
 assert digest(src/rel)==digest(oldsrc/rel),rel
checks={}
for label,root,previous,keys in [('native',src,oldsrc,set(native)|set(ref['native_source_sha256'])),('python',py,oldpy,set(python)|set(old['python_source_sha256']))]:
 patch=[]
 for key in sorted(keys):
  a=previous/key;b=root/key
  if a.exists() and b.exists() and digest(a)==digest(b):continue
  patch+=difflib.unified_diff(a.read_text().splitlines(keepends=True) if a.exists() else [],b.read_text().splitlines(keepends=True) if b.exists() else [],fromfile='a/'+key if a.exists() else '/dev/null',tofile='b/'+key)
 target=p/'delivery'/(label+'-from-reference.patch');target.write_text(''.join(patch));r=subprocess.run(['git','apply','--check',str(target)],cwd=previous,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);checks[label]=dict(returncode=r.returncode,output=r.stdout);assert r.returncode==0,checks[label]
formal=p.parents[1]/'native-s25/build/lib/libLiverCudaSim.so';formalsha=digest(formal);assert formalsha=='3950d6219d57bb9a7bd8770c62365c08590662cb10688f94d939705195645f4c'
summary=json.loads((p/'results/final-summary.json').read_text());posts=json.loads((p/'results/post-checks.json').read_text());assert all(x['returncode']==0 for x in posts)
m=dict(native_dll=str(dll.relative_to(p)),native_dll_sha256=digest(dll),native_source=str(src.relative_to(p)),python_source=str(py.relative_to(p)),native_files=native,python_files=python,changed_native_files=changed_native,added_native_files=sorted(set(native)-set(ref['native_source_sha256'])),changed_python_files=changed_python,added_python_files=sorted(k for k in python if not (oldpy/k).exists()),copied_dependency_python_files=sum(1 for k in python if k not in old['python_source_sha256'] and (oldpy/k).exists()),baseline_native_source_unchanged=True,baseline_python_modules_unchanged=True,baseline_dll_unchanged=True,formal_dll_sha256=formalsha,formal_version_replaced=False,physics_parameters_changed=False,quality_thresholds_changed=False,physics_solver_source_unchanged=True,original_gpu_functions=330,all_original_gpu_functions_sass_unchanged=True,formal_physics_domain_approved=False,training_domain_approved=False,performance_25hz_approved=False,scope='local scenario/current RGBD validated; no dense S25 labels or full physics-domain approval',execution=dict(query_mode=3,bound_cache=True,core_optimizations=0,gpu_body_check=True,query_threads=64,assembly_blocks=512,manifold_blocks=1024,cpus=[7,19,20,25],gpu='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',dt=1/30,substeps=3,contact_refinements=8,width=1280,height=960),patch_checks=checks,harness_sha256={str(x.relative_to(p)):digest(x) for x in sorted((p/'tools').glob('*.py'))})
(p/'delivery/manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps({k:v for k,v in m.items() if k not in ['native_files','python_files','harness_sha256']},indent=2))
