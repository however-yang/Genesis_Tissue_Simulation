// Diagnostic-only frozen-input comparison. Never included in a production build.
#include "reference_find_surface.cuh"

struct ContactProbeStorage {
    SurfaceContact *reference=nullptr,*candidate=nullptr;
    int *degreeReference=nullptr,*degreeCandidate=nullptr,*activeReference=nullptr,*activeCandidate=nullptr;
    int *countReference=nullptr,*countCandidate=nullptr,*seenReference=nullptr,*seenCandidate=nullptr,*statistics=nullptr;
    float4* current=nullptr;float3* previous=nullptr;
    ~ContactProbeStorage(){
        cudaFree(reference);cudaFree(candidate);cudaFree(degreeReference);cudaFree(degreeCandidate);
        cudaFree(activeReference);cudaFree(activeCandidate);cudaFree(countReference);cudaFree(countCandidate);
        cudaFree(seenReference);cudaFree(seenCandidate);cudaFree(statistics);cudaFree(current);cudaFree(previous);
    }
};
__global__ void k_ProbeContactRecords(int count,const SurfaceContact* reference,const SurfaceContact* candidate,int* statistics){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    if(reference[i].count!=candidate[i].count){atomicAdd(statistics,1);return;}
    if(!reference[i].count)return;
    atomicAdd(statistics+4,1);
    const unsigned* a=reinterpret_cast<const unsigned*>(reference+i);
    const unsigned* b=reinterpret_cast<const unsigned*>(candidate+i);
    for(int k=0;k<sizeof(SurfaceContact)/sizeof(unsigned);k++)if(a[k]!=b[k]){
        atomicAdd(statistics+1,1);atomicCAS(statistics+7,0,k+1);return;
    }
}
__global__ void k_ProbeContactLists(const int* active,const int* count,int* seen){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<*count;i+=blockDim.x*gridDim.x)atomicAdd(seen+active[i],1);
}
__global__ void k_ProbeContactIntegers(int count,const int* reference,const int* candidate,int* different){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count && reference[i]!=candidate[i])atomicAdd(different,1);
}

extern "C" __declspec(dllexport) int LCS_ProbeContactEquivalence(int* output){
    if(!output || g_surfaceCount<=0 || !d_coupledTool)return -1;
    ContactProbeStorage scratch;cudaError_t error=cudaSuccess;
    const int slots=BLADE_KIND_SLOTS*g_surfaceCount;
#define CONTACT_PROBE(call) do { error=(call);if(error!=cudaSuccess)return -(int)error; } while(false)
    CONTACT_PROBE(cudaMalloc(&scratch.reference,(size_t)slots*sizeof(SurfaceContact)));
    CONTACT_PROBE(cudaMalloc(&scratch.candidate,(size_t)slots*sizeof(SurfaceContact)));
    CONTACT_PROBE(cudaMalloc(&scratch.degreeReference,g_N*sizeof(int)));
    CONTACT_PROBE(cudaMalloc(&scratch.degreeCandidate,g_N*sizeof(int)));
    CONTACT_PROBE(cudaMalloc(&scratch.activeReference,slots*sizeof(int)));
    CONTACT_PROBE(cudaMalloc(&scratch.activeCandidate,slots*sizeof(int)));
    CONTACT_PROBE(cudaMalloc(&scratch.countReference,sizeof(int)));
    CONTACT_PROBE(cudaMalloc(&scratch.countCandidate,sizeof(int)));
    CONTACT_PROBE(cudaMalloc(&scratch.seenReference,slots*sizeof(int)));
    CONTACT_PROBE(cudaMalloc(&scratch.seenCandidate,slots*sizeof(int)));
    CONTACT_PROBE(cudaMalloc(&scratch.statistics,8*sizeof(int)));
    CONTACT_PROBE(cudaMalloc(&scratch.current,3*g_surfaceCount*sizeof(float4)));
    CONTACT_PROBE(cudaMalloc(&scratch.previous,3*g_surfaceCount*sizeof(float3)));
    CONTACT_PROBE(cudaMemset(scratch.statistics,0,8*sizeof(int)));
    k_UpdateSurfacePoints<<<Gr(3*g_surfaceCount),64>>>(3*g_surfaceCount,d_surfaceVertices,g_surfaceMasks,
        d_prevPos,d_pos,scratch.current,scratch.previous);
    int queries=0,triangles=0;
    for(int tool=0;tool<2;tool++){
        if(tool && (!g_blade.enabled || !d_bladeBody || g_bladeCapacity<g_surfaceCount))continue;
        const int stride=tool?BLADE_KIND_SLOTS:3;
        const SurfaceContact* source=tool?d_bladeContacts:d_surfaceContacts;
        const CoupledTool* body=tool?d_bladeBody:d_coupledTool;
        const float thickness=tool?0:g_jawThickness,minGap=tool?0:g_gp.jawGapClosed;
        const int shapes=tool?bladeShapeCount():3;
        for(int stationary=0;stationary<2;stationary++){
            const size_t bytes=(size_t)stride*g_surfaceCount*sizeof(SurfaceContact);
            CONTACT_PROBE(cudaMemcpy(scratch.reference,source,bytes,cudaMemcpyDeviceToDevice));
            CONTACT_PROBE(cudaMemcpy(scratch.candidate,source,bytes,cudaMemcpyDeviceToDevice));
            CONTACT_PROBE(cudaMemset(scratch.degreeReference,0,g_N*sizeof(int)));
            CONTACT_PROBE(cudaMemset(scratch.degreeCandidate,0,g_N*sizeof(int)));
            CONTACT_PROBE(cudaMemset(scratch.countReference,0,sizeof(int)));
            CONTACT_PROBE(cudaMemset(scratch.countCandidate,0,sizeof(int)));
            for(int index=0;index<shapes;index++){
                const int kind=tool?bladeShapeKind(index):index;
                if(tool && kind==5)continue; // Wrist is the intentionally changed search.
                const float skin=tool && kind<3?0:g_contactRadius;
                const int* cacheSlots=tool?(kind>=3?d_bladeBackCacheSlots:nullptr):d_contactCacheSlots;
                const SurfaceContact* cache=tool?(kind>=3?d_bladeBackCache:nullptr):d_contactCache;
                const int* phase=tool && kind<3?d_bladePhase:nullptr;
                k_FindSurfaceReference<<<Gr(g_surfaceCount),64>>>(g_surfaceCount,kind,ContactShape{},ContactShape{},skin,
                    d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,scratch.reference,scratch.degreeReference,
                    scratch.activeReference,scratch.countReference,scratch.current,scratch.previous,body,thickness,minGap,
                    stationary!=0,cacheSlots,cache,phase,stride,tool && kind>=3);
                k_FindSurface<<<Gr(g_surfaceCount),64>>>(g_surfaceCount,kind,ContactShape{},ContactShape{},skin,
                    d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,scratch.candidate,scratch.degreeCandidate,
                    scratch.activeCandidate,scratch.countCandidate,scratch.current,scratch.previous,body,thickness,minGap,
                    stationary!=0,cacheSlots,cache,phase,stride,tool && kind>=3);
                queries++;triangles+=g_surfaceCount;
            }
            k_ProbeContactRecords<<<Gr(stride*g_surfaceCount),64>>>(stride*g_surfaceCount,scratch.reference,scratch.candidate,scratch.statistics);
            k_ProbeContactIntegers<<<Gr(g_N),64>>>(g_N,scratch.degreeReference,scratch.degreeCandidate,scratch.statistics+2);
            k_ProbeContactIntegers<<<1,1>>>(1,scratch.countReference,scratch.countCandidate,scratch.statistics+3);
            CONTACT_PROBE(cudaMemset(scratch.seenReference,0,slots*sizeof(int)));
            CONTACT_PROBE(cudaMemset(scratch.seenCandidate,0,slots*sizeof(int)));
            k_ProbeContactLists<<<128,64>>>(scratch.activeReference,scratch.countReference,scratch.seenReference);
            k_ProbeContactLists<<<128,64>>>(scratch.activeCandidate,scratch.countCandidate,scratch.seenCandidate);
            k_ProbeContactIntegers<<<Gr(slots),64>>>(slots,scratch.seenReference,scratch.seenCandidate,scratch.statistics+3);
        }
    }
    CONTACT_PROBE(cudaGetLastError());
    CONTACT_PROBE(cudaMemcpy(output,scratch.statistics,8*sizeof(int),cudaMemcpyDeviceToHost));
    output[5]=queries;output[6]=triangles;
#undef CONTACT_PROBE
    return 0;
}
