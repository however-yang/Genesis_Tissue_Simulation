from pathlib import Path
import shutil
p=Path(__file__).resolve().parents[1];src=p/'native-query-v4/source';shutil.copytree(p/'native-query-v3/source',src)
f=src/'vendor/cuda/contact/surface_contact.cuh';s=f.read_text();s=s.replace('template<int MinBlocks> __global__ __launch_bounds__(64,MinBlocks) void k_FindSurfaceFast(', '__global__ void k_FindSurfaceFastSafe(',1)
a=s.index('            if(g_coreOptimizations&16){k_FindSurfaceFast<8>');b=s.index('        }else if(g_coreOptimizations&4){',a)
args='<<<dim3((triangles+g_queryThreads-1)/g_queryThreads,kindCount),g_queryThreads,0,stream>>>(triangles,packed,bladeBatch,from,to,skin,\n            vertices,masks,prev,pos,contacts,degree,current,previous,body,thickness,minGap,stationary,cacheSlots,cache,bladePhase,stride,groupBase,w.ids,w.counts);'
s=s[:a]+'            k_FindSurfaceFastSafe'+args+'\n'+s[b:];f.write_text(s)
f=src/'vendor/cuda/xpbd/xpbd_solver.cu';f.write_text(f.read_text().replace('mode>31 || g_ready','mode>15 || g_ready'))
s=(p/'tools/build_query_v3.py').read_text().replace('native-query-v3','native-query-v4').replace('build-query-v3','build-query-v4');(p/'tools/build_query_v4.py').write_text(s)
print(src)
