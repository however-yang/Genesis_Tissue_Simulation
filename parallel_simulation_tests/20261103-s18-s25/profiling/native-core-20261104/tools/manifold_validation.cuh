// Diagnostic only: run the old 1024-block manifold against identical inputs.
// Compare quadrature and complete material witnesses canonically by owner/rank.
constexpr int MANIFOLD_COMPARE_RECORDS=65536;
struct ManifoldValidationWorkspace {
    SurfaceContact* reference=nullptr;int *ids=nullptr,*owners=nullptr,*count=nullptr,*gated=nullptr,*degree=nullptr;
    int *startRef=nullptr,*startActual=nullptr,*bad=nullptr;float* quadrature=nullptr;
};
static ManifoldValidationWorkspace g_manifoldValidation[2];
static cudaError_t allocateManifoldValidation(int nodes){
    if(g_manifoldValidation[0].reference)return cudaSuccess;cudaError_t e=cudaSuccess;
#define MANIFOLD_ALLOC(pointer,bytes) do{e=queryPrivateAlloc(&(pointer),(bytes));if(e!=cudaSuccess)return e;}while(0)
    for(auto& w:g_manifoldValidation){
        MANIFOLD_ALLOC(w.reference,MANIFOLD_COMPARE_RECORDS*sizeof(SurfaceContact));
        MANIFOLD_ALLOC(w.ids,MANIFOLD_COMPARE_RECORDS*sizeof(int));MANIFOLD_ALLOC(w.owners,MANIFOLD_COMPARE_RECORDS*sizeof(int));
        MANIFOLD_ALLOC(w.quadrature,MANIFOLD_COMPARE_RECORDS*sizeof(float));MANIFOLD_ALLOC(w.degree,nodes*sizeof(int));
        MANIFOLD_ALLOC(w.count,sizeof(int));MANIFOLD_ALLOC(w.gated,sizeof(int));MANIFOLD_ALLOC(w.bad,sizeof(int));
        MANIFOLD_ALLOC(w.startRef,QUERY_MAX_SLOTS*sizeof(int));MANIFOLD_ALLOC(w.startActual,QUERY_MAX_SLOTS*sizeof(int));
    }
#undef MANIFOLD_ALLOC
    return cudaSuccess;
}
static void freeManifoldValidation(){
    for(auto& w:g_manifoldValidation){
        queryPrivateFree(w.reference);queryPrivateFree(w.ids);queryPrivateFree(w.owners);queryPrivateFree(w.quadrature);
        queryPrivateFree(w.degree);queryPrivateFree(w.count);queryPrivateFree(w.gated);queryPrivateFree(w.bad);
        queryPrivateFree(w.startRef);queryPrivateFree(w.startActual);w={};
    }
}
__global__ void k_PrepareManifoldValidation(const int* sourceCount,const int* actualTotal,int ownerSlots,int* gated,int* total,int* bad,int* stats){
    if(threadIdx.x||blockIdx.x)return;*total=0;*bad=0;
    if(*actualTotal || *sourceCount<0 || *sourceCount>MANIFOLD_COMPARE_RECORDS/13 || ownerSlots>QUERY_MAX_SLOTS){*gated=0;*bad=-1;atomicAdd(stats+24,1);}
    else *gated=*sourceCount;
}
__global__ void k_ManifoldOwnerStarts(int capacity,int slots,const int* count,const int* owner,int* starts,int* bad,int* stats){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(*bad<0 || i>=*count)return;
    if(*count>capacity || owner[i]<0 || owner[i]>=slots){queryMismatch(bad,stats,23);return;}
    atomicMin(starts+owner[i],i);
}
__global__ void k_CompareManifoldRecords(int capacity,int slots,const SurfaceContact* ref,const int* refIds,const int* refOwners,const int* refCount,const float* refQuadrature,const int* refStarts,
    const SurfaceContact* actual,const int* ids,const int* owners,const int* count,const float* quadrature,const int* starts,int* bad,int* stats){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(*bad<0)return;
    if(i==0 && *refCount!=*count)queryMismatch(bad,stats,23);
    if(i>=*count)return;if(*count>capacity){if(i==0)queryMismatch(bad,stats,23);return;}
    int owner=owners[i];if(owner<0 || owner>=slots){queryMismatch(bad,stats,23);return;}
    int rank=i-starts[owner],j=refStarts[owner]+rank;
    if(rank<0 || rank>=13 || j<0 || j>=*refCount || refOwners[j]!=owner || ids[i]!=i || refIds[j]!=j){queryMismatch(bad,stats,23);return;}
    const unsigned* a=reinterpret_cast<const unsigned*>(ref+j);const unsigned* b=reinterpret_cast<const unsigned*>(actual+i);
    for(int k=0;k<(int)(sizeof(SurfaceContact)/sizeof(unsigned));k++)if(a[k]!=b[k]){queryMismatch(bad,stats,21);break;}
    if(__float_as_uint(refQuadrature[j])!=__float_as_uint(quadrature[i]))queryMismatch(bad,stats,22);
    atomicAdd(stats+18,1);
}
__global__ void k_CompareManifoldDegrees(int nodes,const int* ref,const int* actual,int* bad,int* stats){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(*bad<0)return;if(i<nodes && ref[i]!=actual[i])queryMismatch(bad,stats,20);
}
__global__ void k_FinishManifoldValidation(bool captured,const int* bad,int* stats){
    if(threadIdx.x||blockIdx.x)return;if(*bad<0)return;atomicAdd(stats+16,1);if(*bad)atomicAdd(stats+17,1);if(captured)atomicAdd(stats+19,1);
}
static inline void launchUnifiedManifold(cudaStream_t stream,const SurfaceContact* source,const int* active,const int* count,
    const SurfaceVertex* vertices,const float3* pos,const float3* previous,const CoupledTool* body,
    float thickness,float minGap,float skin,SurfaceContact* out,int* ids,int* total,int* owners,float* quadrature,int* degree,int stride,int ownerSlots){
    auto actual=[&](){k_UnifiedManifold<<<MANIFOLD_BLOCKS,64,0,stream>>>(source,active,count,vertices,pos,previous,body,thickness,minGap,skin,out,ids,total,owners,quadrature,degree,stride);};
    if(!g_queryValidate){actual();return;}
    ManifoldValidationWorkspace& w=g_manifoldValidation[stream?1:0];
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;cudaStreamIsCapturing(stream,&capture);
    k_PrepareManifoldValidation<<<1,1,0,stream>>>(count,total,ownerSlots,w.gated,w.count,w.bad,d_queryValidationStats);
    cudaMemcpyAsync(w.degree,degree,g_queryNodes*sizeof(int),cudaMemcpyDeviceToDevice,stream);
    k_UnifiedManifold<<<1024,64,0,stream>>>(source,active,w.gated,vertices,pos,previous,body,thickness,minGap,skin,w.reference,w.ids,w.count,w.owners,w.quadrature,w.degree,stride);
    actual();
    cudaMemsetAsync(w.startRef,0x7f,QUERY_MAX_SLOTS*sizeof(int),stream);cudaMemsetAsync(w.startActual,0x7f,QUERY_MAX_SLOTS*sizeof(int),stream);
    k_ManifoldOwnerStarts<<<256,128,0,stream>>>(MANIFOLD_COMPARE_RECORDS,ownerSlots,w.count,w.owners,w.startRef,w.bad,d_queryValidationStats);
    k_ManifoldOwnerStarts<<<256,128,0,stream>>>(MANIFOLD_COMPARE_RECORDS,ownerSlots,total,owners,w.startActual,w.bad,d_queryValidationStats);
    k_CompareManifoldRecords<<<(MANIFOLD_COMPARE_RECORDS+127)/128,128,0,stream>>>(MANIFOLD_COMPARE_RECORDS,ownerSlots,w.reference,w.ids,w.owners,w.count,w.quadrature,w.startRef,out,ids,owners,total,quadrature,w.startActual,w.bad,d_queryValidationStats);
    k_CompareManifoldDegrees<<<(g_queryNodes+127)/128,128,0,stream>>>(g_queryNodes,w.degree,degree,w.bad,d_queryValidationStats);
    k_FinishManifoldValidation<<<1,1,0,stream>>>(capture!=cudaStreamCaptureStatusNone,w.bad,d_queryValidationStats);
}
