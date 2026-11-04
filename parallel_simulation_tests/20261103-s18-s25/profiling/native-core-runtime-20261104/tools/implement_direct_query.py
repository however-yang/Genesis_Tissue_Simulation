"""Generate a strict-equivalence query candidate with GPU-resident output records."""
from pathlib import Path
import hashlib,json

p=Path('/srv/disk2/hryang/genesis-exec/parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104')
root=p/'native/source'
path=root/'vendor/cuda/contact/surface_contact.cuh'
original=path.read_text()
start=original.index('__global__ void k_FindSurfaceBatched(')
end=original.index('static_assert(sizeof(SurfaceContact)',start)
kernel=original[start:end]
assert kernel.count('SurfaceContact retained;bool keep=')==1
kernel=kernel.replace('k_FindSurfaceBatched(','k_FindSurfaceGpuRecords(',1)
kernel=kernel.replace('const unsigned char* groupBase,int* groupedIds,int* groupedCounts){',
    'const unsigned char* groupBase,int* groupedIds,int* groupedCounts,SurfaceContact* __restrict__ next){')
kernel=kernel.replace('SurfaceContact retained;bool keep=', 'const SurfaceContact* retained=nullptr;bool keep=')
kernel=kernel.replace('if(keep)retained=contacts[slot];', 'if(keep)retained=&contacts[slot];')
kernel=kernel.replace('retained=cache[slot];keep=true;', 'retained=&cache[slot];keep=true;')
kernel=kernel.replace('retained=cache[s];keep=true;', 'retained=&cache[s];keep=true;')
# Keep the original validity writes. Only count is read after that point from
# the retained record, and it is saved first. Old stencil/plane bytes remain
# untouched until the separate, stream-ordered commit kernel.
needle='    contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;\n    ContactRecord retainedPlane={};'
assert kernel.count(needle)==1
kernel=kernel.replace(needle,'    const int retainedCount=keep?retained->count:0;\n'+needle)
kernel=kernel.replace('retained.', 'retained->').replace('retained->count','retainedCount')
legacy='                    for(int k=0;k<3;k++){retained->owner[k]=vertices[3*t+k].voxel;retained->mask[k]=vertices[3*t+k].mask;}\n'
assert kernel.count(legacy)==1
kernel=kernel.replace(legacy,'                    // Unused retained owner/mask remap; output is bound to current vertices.\n')
assert 'retained->owner' not in kernel and 'retained->mask' not in kernel
kernel=kernel.replace('const int retainedCount=keep?retainedCount:0;', 'const int retainedCount=keep?retained->count:0;')
needle='SurfaceContact out={};out.kind=kind;';assert kernel.count(needle)==1
kernel=kernel.replace(needle,'SurfaceContact& out=next[slot];out=SurfaceContact{};out.kind=kind;')
needle='    contacts[slot]=out;\n';assert kernel.count(needle)==1;kernel=kernel.replace(needle,'')
needle='    SurfaceVertex v[3]={vertices[3*t],vertices[3*t+1],vertices[3*t+2]};\n';assert kernel.count(needle)==1
kernel=kernel.replace(needle,'')
loop='''    for(int k=0;k<3;k++){
        out.owner[k]=v[k].voxel;'''
assert kernel.count(loop)==1
kernel=kernel.replace(loop,'''    for(int k=0;k<3;k++){
        const SurfaceVertex& v=vertices[3*t+k];
        out.owner[k]=v.voxel;''')
kernel=kernel.replace('v[k].','v.')
assert 'SurfaceContact retained;' not in kernel and 'retained.' not in kernel
extra='''// Copy only emitted valid records, one warp per record. Rejected entries
// already received the original three validity writes in the query kernel.
// No next query or solver consumer can run before this stream dependency.
__global__ void k_CommitGpuQueryRecords(int triangles,const int* grouped,const int* counts,
    const SurfaceContact* __restrict__ next,SurfaceContact* __restrict__ contacts){
    const int lane=threadIdx.x&31,warp=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int stride=gridDim.x*blockDim.x/32,kind=blockIdx.y;
    for(int index=warp;index<counts[kind];index+=stride){
        int id=grouped[kind*triangles+index];
        const unsigned* source=reinterpret_cast<const unsigned*>(next+id);
        unsigned* destination=reinterpret_cast<unsigned*>(contacts+id);
        for(int word=lane;word<sizeof(SurfaceContact)/sizeof(unsigned);word+=32)destination[word]=source[word];
    }
}
static int g_directQueryMode=0;
'''
(root/'vendor/cuda/contact/direct_query.cuh').write_text('// Generated from the unchanged V8 query; arithmetic and witnesses retained.\n'+kernel+extra)
# Preserve the original production and diagnostic kernels for paired checking.
original=original[:end]+'#include "direct_query.cuh"\n\n'+original[end:]
original=original.replace('SurfaceContact *before=nullptr,*reference=nullptr;', 'SurfaceContact *before=nullptr,*reference=nullptr,*direct=nullptr;')
original=original.replace('queryPrivateFree(w.before);queryPrivateFree(w.reference);', 'queryPrivateFree(w.before);queryPrivateFree(w.reference);queryPrivateFree(w.direct);')
needle='e=queryPrivateAlloc(&w.counts,6*sizeof(int));if(e!=cudaSuccess)return e;'
assert original.count(needle)==1
original=original.replace(needle,needle+'\n        if(g_directQueryMode){e=queryPrivateAlloc(&w.direct,QUERY_MAX_SLOTS*sizeof(SurfaceContact));if(e!=cudaSuccess)return e;}')
needle='''        k_FindSurfaceBatched<<<dim3((triangles+g_queryThreads-1)/g_queryThreads,kindCount),g_queryThreads,0,stream>>>(triangles,packed,bladeBatch,from,to,skin,
            vertices,masks,prev,pos,contacts,degree,current,previous,body,thickness,minGap,stationary,cacheSlots,cache,bladePhase,stride,groupBase,w.ids,w.counts);'''
assert original.count(needle)==1
replacement='''        if(g_directQueryMode && w.direct){
            k_FindSurfaceGpuRecords<<<dim3((triangles+g_queryThreads-1)/g_queryThreads,kindCount),g_queryThreads,0,stream>>>(triangles,packed,bladeBatch,from,to,skin,
                vertices,masks,prev,pos,contacts,degree,current,previous,body,thickness,minGap,stationary,cacheSlots,cache,bladePhase,stride,groupBase,w.ids,w.counts,w.direct);
            k_CommitGpuQueryRecords<<<dim3(32,kindCount),128,0,stream>>>(triangles,w.ids,w.counts,w.direct,contacts);
        }else{\n'''+needle+'\n        }'
original=original.replace(needle,replacement)
path.write_text(original)
path=root/'vendor/cuda/xpbd/xpbd_solver.cu'
s=path.read_text();s=s.replace('if(mode&~256)return -10001;','if(mode&~(256|512))return -10001;')
s=s.replace('g_runtimeCoreMode=mode;return 0;','g_runtimeCoreMode=mode;g_directQueryMode=(mode&512)!=0;return 0;')
path.write_text(s)
print(json.dumps({'added':'direct_query.cuh','arithmetic_unchanged_by_generator':True,'mode':512,'workspace':'private derived CUDA buffer, excluded from physical snapshots'}))
