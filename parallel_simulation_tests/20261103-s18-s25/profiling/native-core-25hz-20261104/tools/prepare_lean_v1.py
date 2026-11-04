from pathlib import Path
import shutil,hashlib,json
prof=Path('/srv/disk2/hryang/genesis-exec/parallel_simulation_tests/20261103-s18-s25/profiling')
old=prof/'native-core-20261104'; new=prof/'native-core-25hz-20261104'
new.mkdir(exist_ok=False)
for d in ['tools','logs','results','tmp','delivery']: (new/d).mkdir()
base=new/'native-lean-v1';shutil.copytree(old/'native-all-queries-v2/source',base/'source')
for name in ['profile_all_queries.py','check_current_frame_final_cache.py','check_snapshot_roundtrip_cache.py','bind_sass.py','analyze_timeline.py']:
 shutil.copy2(old/'tools'/name,new/'tools'/name)
f=base/'source/vendor/cuda/contact/surface_contact.cuh';s=f.read_text();start=s.index('__global__ void k_FindSurfaceBatched(');end=s.index('static_assert(sizeof(SurfaceContact)',start)
k=s[start:end];assert k.count('SurfaceContact retained;')==1
k=k.replace('k_FindSurfaceBatched(', 'k_FindSurfaceLean(',1)
k=k.replace('SurfaceContact retained;bool keep=', 'const SurfaceContact* retained=nullptr;bool keep=',1)
k=k.replace('if(keep)retained=contacts[slot];','if(keep)retained=&contacts[slot];',1)
k=k.replace('retained=cache[slot];','retained=&cache[slot];',1).replace('retained=cache[s];','retained=&cache[s];',1)
rebind='for(int k=0;k<3;k++){retained.owner[k]=vertices[3*t+k].voxel;retained.mask[k]=vertices[3*t+k].mask;}'
assert rebind in k;k=k.replace(rebind,'// The original copy rebinds owner/mask here, but neither field is read below.\n                    // Output owners/masks are always rebuilt from the current vertices.',1)
clear='contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;'
pos=k.index('    // Rejected triangles have no live contact.')
clearpos=k.index(clear,pos);k=k[:clearpos]+k[clearpos:].replace(clear,'// Delay invalidation until the retained witness has been consumed.',1)
k=k.replace('retained.','retained->')
assert k.count('SurfaceContact out={};out.kind=kind;')==1
k=k.replace('SurfaceContact out={};out.kind=kind;', 'ContactRecord plane={};float3 stick=mkf3(0,0,0);',1)
mid=k.index('    SurfaceVertex v[3]=')
front=k[:mid].replace('out.plane','plane').replace('out.stick','stick')
front=front.replace('if(!plane.valid)return;', 'if(!plane.valid){'+clear+'return;}',1)
k=front+'    SurfaceContact out={};out.kind=kind;out.plane=plane;out.stick=stick;\n'+k[mid:]
assert 'retained.' not in k and 'SurfaceContact retained' not in k
s=s[:end]+k+s[end:]
s=s.replace('static int g_queryBatchMode=0,g_queryNodes=0;', 'static int g_coreOptimizations=0;\nstatic int g_queryBatchMode=0,g_queryNodes=0;',1)
needle='        k_FindSurfaceBatched<<<dim3((triangles+g_queryThreads-1)/g_queryThreads,kindCount),g_queryThreads,0,stream>>>(triangles,packed,bladeBatch,from,to,skin,\n            vertices,masks,prev,pos,contacts,degree,current,previous,body,thickness,minGap,stationary,cacheSlots,cache,bladePhase,stride,groupBase,w.ids,w.counts);'
assert needle in s
lean=needle.replace('k_FindSurfaceBatched','k_FindSurfaceLean')
s=s.replace(needle,'        if(g_coreOptimizations&1){\n'+lean+'\n        }else{\n'+needle+'\n        }',1);f.write_text(s)
f=base/'source/vendor/cuda/xpbd/xpbd_solver.cu';s=f.read_text();s+='\nextern "C" int xpbd_core_optimizations(int mode){if(mode<0 || mode>1 || g_ready)return -10001;g_coreOptimizations=mode;return 0;}\n';f.write_text(s)
f=base/'source/vendor/src/plugin_api.cpp';s=f.read_text();s+='\nextern "C" int xpbd_core_optimizations(int);\nLCS_API int LCS_SetCoreOptimizations(int mode){return xpbd_core_optimizations(mode);}\n';f.write_text(s)
for name in ['profile_all_queries.py','check_current_frame_final_cache.py','check_snapshot_roundtrip_cache.py']:
 f=new/'tools'/name;s=f.read_text()
 if name=='profile_all_queries.py':
  s=s.replace('a=p.parse_args()',"p.add_argument('--core-optimizations',type=int,default=0)\na=p.parse_args()",1)
  s=s.replace("native_control=ctypes.CDLL(str(a.dll))", "native_control=ctypes.CDLL(str(a.dll))\n native_control.LCS_SetCoreOptimizations.argtypes=[ctypes.c_int];native_control.LCS_SetCoreOptimizations.restype=ctypes.c_int\n assert native_control.LCS_SetCoreOptimizations(a.core_optimizations)==0\n result['core_optimizations']=a.core_optimizations",1)
 else:
  assert 'a=parser.parse_args()' in s
  s=s.replace('a=parser.parse_args()',"parser.add_argument('--core-optimizations',type=int,default=0);a=parser.parse_args()",1)
  marker='native=ctypes.CDLL(str(a.dll))';assert marker in s
  s=s.replace(marker,marker+"\n native.LCS_SetCoreOptimizations.argtypes=[ctypes.c_int];native.LCS_SetCoreOptimizations.restype=ctypes.c_int;assert native.LCS_SetCoreOptimizations(a.core_optimizations)==0",1)
 f.write_text(s)
s=(old/'tools/build_all_queries_v2.py').read_text().replace('native-all-queries-v2','native-lean-v1').replace('build-all-queries-v2','build-lean-v1');(new/'tools/build_lean_v1.py').write_text(s)
files={str(f.relative_to(old/'native-all-queries-v2/source')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (old/'native-all-queries-v2/source').rglob('*') if f.is_file()}
(new/'delivery/baseline-manifest.json').write_text(json.dumps(dict(native_source=str(old/'native-all-queries-v2/source'),dll=str(old/'native-all-queries-v2/build/lib/libLiverCudaSim.so'),dll_sha256=hashlib.sha256((old/'native-all-queries-v2/build/lib/libLiverCudaSim.so').read_bytes()).hexdigest(),files=files,python_source=str(prof/'aggressive-20261104/final-source'),target_frame_ms=40,formal_physics_domain_approved=False,training_domain_approved=False),indent=2))
(new/'README.md').write_text('# 原生核心 25 Hz 优化实验\n\n独立实验，不替换正式版本。基线为 native-core-20261104 的最终 mode=3 + bound-cache 版本。\n\n目标：完整 1280×960 当前帧 RGB-D，40 ms/帧；保持物理参数、接触、切割与质量检查。诊断和性能测量分开。\n')
print(new)
