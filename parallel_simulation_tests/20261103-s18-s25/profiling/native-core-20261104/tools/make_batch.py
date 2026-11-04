"""Generate an isolated scheduling-only contact-query candidate.

The reference kernel remains verbatim. The batch clone changes only dispatch
and active-list destination. Results remain grouped by the original kind order.
"""
from pathlib import Path
p=Path(__file__).resolve().parents[1];s=p/'native-batched/source';h=s/'vendor/cuda/contact/surface_contact.cuh';text=h.read_text();begin=text.index('__global__ void k_FindSurface(');end=text.index('// Bounded grid-stride launch',begin);original=text[begin:end]
# Use the same body, cache, preload, CCD and interpolation arithmetic as before.
body=original[original.index('    int t='):]
setup='''    int kind=(packedKinds>>(3*blockIdx.y))&7;
    bool cacheComponents=bladeBatch && kind>=3;
    if(bladeBatch){
        if(kind<3){skin=0;cacheSlots=nullptr;cache=nullptr;}
        else bladePhase=nullptr;
    }
    const unsigned char* groupRejected=groupBase?groupBase+kind*((triangles+63)/64):nullptr;
    int* active=groupedIds+blockIdx.y*triangles;
    int* activeCount=groupedCounts+blockIdx.y;
'''
signature='''__global__ void k_FindSurfaceBatched(int triangles,unsigned packedKinds,bool bladeBatch,ContactShape from,ContactShape to,float skin,
    const SurfaceVertex* vertices,const unsigned* masks,const float3* prev,const float3* pos,
    SurfaceContact* contacts,int* degree,const float4* current,const float3* previous,
    const CoupledTool* body,float thickness,float minGap,bool stationary,
    const int* cacheSlots,const SurfaceContact* cache,const int* bladePhase,int stride,
    const unsigned char* groupBase,int* groupedIds,int* groupedCounts){
'''
helpers=(p/'tools/batch_helpers.cuh').read_text()
h.write_text(text[:end]+signature+setup+body+'\n'+helpers+'\n'+text[end:])
# Replace only the five-kind dispatch loop; all surrounding solve/cache calls stay.
f=s/'vendor/cuda/cutting/pressure_blade.cuh';raw=f.read_bytes().decode();normalized=raw.replace('\r\n','\n');a=normalized.index('    for(int index=0;index<bladeShapeCount();index++){',normalized.index('static void findBlade('));b=normalized.index('    cudaMemsetAsync(d_bladeBackCacheSlots',a);old=normalized[a:b]
new='''    unsigned packed=0;for(int index=0;index<bladeShapeCount();index++)packed|=(unsigned)bladeShapeKind(index)<<(3*index);
    if(g_surfaceCount>0 && canBatchQueries(g_surfaceCount,d_bladeBody,1)){
        launchFindSurfaceGroup(g_surfaceCount,packed,bladeShapeCount(),true,stream,
            contactShape(g_bladeInitial,0,0,0),contactShape(g_bladeCommand,0,0,0),g_contactRadius,
            d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,d_bladeContacts,d_bladeDegree,d_bladeIds,d_bladeCount,
            d_surfaceCurrent,d_surfacePrevious,d_bladeBody,0,0,stationary,d_bladeBackCacheSlots,d_bladeBackCache,
            d_bladePhase,BLADE_KIND_SLOTS,d_contactGroupRejected);
    }else{
'''+old+'''    }
'''
normalized=normalized[:a]+new+normalized[b:];f.write_bytes(normalized.replace('\n','\r\n').encode() if '\r\n' in raw else normalized.encode())
x=s/'vendor/cuda/xpbd/xpbd_solver.cu';text=x.read_text();needle='    g_N = particleCount; g_E = edgeCount;';assert text.count(needle)==1;text=text.replace(needle,needle+'\n    XK(allocateQueryGroups(g_N));')
needle='extern "C" void xpbd_shutdown()\n{';assert text.count(needle)==1;text=text.replace(needle,needle+'\n    freeQueryGroups();')
# APIs live after all existing host graph declarations/definitions.
text+='''
extern "C" int xpbd_batch_query_mode(int mode){
    if(mode<0 || mode>1 || g_ready)return -10001;
    g_queryBatchMode=mode;return 0;
}
extern "C" int xpbd_batch_validation(int enabled){
    cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess)return -(int)e;
    lcs_state::invalidate_xpbd_graphs();
    if(enabled){if(!g_ready || !g_queryBatchMode)return -10002;
        e=allocateQueryValidation(g_N);if(e!=cudaSuccess)return -(int)e;
        e=cudaMemset(d_queryValidationStats,0,16*sizeof(int));if(e!=cudaSuccess)return -(int)e;
        e=cudaMemset(d_queryValidationStats+11,0xff,sizeof(int));if(e!=cudaSuccess)return -(int)e;}
    g_queryValidate=enabled!=0;return 0;
}
extern "C" int xpbd_batch_validation_read(int* values){
    if(!values || !d_queryValidationStats)return -10003;
    cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess)return -(int)e;
    return -(int)cudaMemcpy(values,d_queryValidationStats,16*sizeof(int),cudaMemcpyDeviceToHost);
}
''';x.write_text(text)
f=s/'vendor/src/plugin_api.cpp';raw=f.read_bytes();raw+=b'''\nextern "C" int xpbd_batch_query_mode(int);\nextern "C" int xpbd_batch_validation(int);\nextern "C" int xpbd_batch_validation_read(int*);\nLCS_API int LCS_SetBatchQueryMode(int mode){return xpbd_batch_query_mode(mode);}\nLCS_API int LCS_BatchQueryValidate(int enabled){return xpbd_batch_validation(enabled);}\nLCS_API int LCS_BatchQueryValidationRead(int* values){return xpbd_batch_validation_read(values);}\n''';f.write_bytes(raw)
print('Generated blade-group candidate; original kernel and every query retained.')
