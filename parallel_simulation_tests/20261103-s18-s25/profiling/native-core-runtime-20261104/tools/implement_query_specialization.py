from pathlib import Path
import difflib,json
p=Path('/srv/disk2/hryang/genesis-exec/parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104')
root=p/'native/source'
base=p.parent/'native-core-25hz-20261104/native-readonly-v8/source'
patch=[]
for name in sorted({str(x.relative_to(base)) for x in base.rglob('*') if x.is_file()}|{str(x.relative_to(root)) for x in root.rglob('*') if x.is_file()}):
 a=base/name;b=root/name
 old=a.read_text() if a.exists() else '';new=b.read_text() if b.exists() else ''
 if old!=new:patch.extend(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),fromfile='a/'+name if a.exists() else '/dev/null',tofile='b/'+name if b.exists() else '/dev/null'))
(p/'delivery/retained-pre-query-v7.patch').write_text(''.join(patch))
f=root/'vendor/cuda/contact/surface_contact.cuh';s=f.read_text()
a=s.index('__global__ void k_FindSurfaceBatched(');b=s.index('static_assert(sizeof(SurfaceContact)',a)
kernel=s[a:b]
kernel=kernel.replace('__global__ void k_FindSurfaceBatched(', 'template<bool BladeBatch> __global__ void k_RuntimeSpecializedQuery(').replace('bool bladeBatch,ContactShape from','bool ignoredBladeBatch,ContactShape from')
needle='const unsigned char* groupBase,int* groupedIds,int* groupedCounts){'
assert kernel.count(needle)==1
kernel=kernel.replace(needle,needle+'\n    constexpr bool bladeBatch=BladeBatch;')
(root/'vendor/cuda/contact/query_specialized.cuh').write_text('// Original query formula; compile-time jaw/blade dispatch only.\n'+kernel)
s=s.replace('static_assert(sizeof(SurfaceContact)', '#include "query_specialized.cuh"\n\nstatic_assert(sizeof(SurfaceContact)',1)
old='''        k_FindSurfaceBatched<<<dim3((triangles+g_queryThreads-1)/g_queryThreads,kindCount),g_queryThreads,0,stream>>>(triangles,packed,bladeBatch,from,to,skin,
            vertices,masks,prev,pos,contacts,degree,current,previous,body,thickness,minGap,stationary,cacheSlots,cache,bladePhase,stride,groupBase,w.ids,w.counts);'''
assert s.count(old)==1
new='''        if(g_runtimeSpecializedQuery){
            if(bladeBatch)'''+old.strip().replace('k_FindSurfaceBatched','k_RuntimeSpecializedQuery<true>')+'''
            else '''+old.strip().replace('k_FindSurfaceBatched','k_RuntimeSpecializedQuery<false>')+'''
        }else '''+old.strip()
s=s.replace(old,new);s=s.replace('static int g_queryBatchMode=0,g_queryNodes=0;','static int g_queryBatchMode=0,g_queryNodes=0;\nstatic bool g_runtimeSpecializedQuery=false;');f.write_text(s)
f=root/'vendor/cuda/xpbd/xpbd_solver.cu';s=f.read_text();assert s.count('256|2048|8192|32768')==1;s=s.replace('256|2048|8192|32768','256|2048|8192|32768|65536');s=s.replace('g_runtimeCoreMode=mode;g_runtimeCoarseBlocks=', 'g_runtimeCoreMode=mode;g_runtimeSpecializedQuery=(mode&65536)!=0;g_runtimeCoarseBlocks=');f.write_text(s)
print(json.dumps({'experimental_bit':65536,'new_formula':False,'new_temporal_cache':False,'new_query_count':False}))
