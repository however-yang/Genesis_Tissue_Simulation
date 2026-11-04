// Scheduling-only workspace. Recomputed every group, never physical snapshot state.
template<class T> static cudaError_t queryPrivateAlloc(T** p,size_t bytes){return (::cudaMalloc)(reinterpret_cast<void**>(p),bytes);}
static void queryPrivateFree(void* p){if(p)(::cudaFree)(p);}
constexpr int QUERY_MAX_TRIANGLES=65536;
constexpr int QUERY_MAX_SLOTS=6*QUERY_MAX_TRIANGLES;
struct QueryGroupWorkspace {
    int* ids=nullptr;int* counts=nullptr;
    SurfaceContact *before=nullptr,*reference=nullptr;
    int *degreeBefore=nullptr,*degreeRef=nullptr,*activeBefore=nullptr,*activeRef=nullptr;
    int *countBefore=nullptr,*countRef=nullptr,*frequencies=nullptr,*bad=nullptr;
};
static QueryGroupWorkspace g_queryWorkspace[2];
static int g_queryBatchMode=0,g_queryNodes=0;
static bool g_queryValidate=false;
static cudaStream_t g_queryNonzeroStream=nullptr;
static int* d_queryValidationStats=nullptr;
static void freeQueryGroups(){
    cudaDeviceSynchronize();
    for(auto& w:g_queryWorkspace){
        queryPrivateFree(w.ids);queryPrivateFree(w.counts);queryPrivateFree(w.before);queryPrivateFree(w.reference);
        queryPrivateFree(w.degreeBefore);queryPrivateFree(w.degreeRef);queryPrivateFree(w.activeBefore);queryPrivateFree(w.activeRef);
        queryPrivateFree(w.countBefore);queryPrivateFree(w.countRef);queryPrivateFree(w.frequencies);queryPrivateFree(w.bad);w={};
    }
    queryPrivateFree(d_queryValidationStats);d_queryValidationStats=nullptr;
    g_queryValidate=false;g_queryNodes=0;g_queryNonzeroStream=nullptr;
}
static cudaError_t allocateQueryGroups(int nodes){
    g_queryNodes=nodes;if(!g_queryBatchMode)return cudaSuccess;
    for(auto& w:g_queryWorkspace){
        cudaError_t e=queryPrivateAlloc(&w.ids,QUERY_MAX_SLOTS*sizeof(int));if(e!=cudaSuccess)return e;
        e=queryPrivateAlloc(&w.counts,6*sizeof(int));if(e!=cudaSuccess)return e;
    }return cudaSuccess;
}
static cudaError_t allocateQueryValidation(int nodes){
    if(d_queryValidationStats)return cudaSuccess;cudaError_t e=queryPrivateAlloc(&d_queryValidationStats,16*sizeof(int));if(e!=cudaSuccess)return e;
#define QUERY_ALLOC(pointer,bytes) do{e=queryPrivateAlloc(&(pointer),(bytes));if(e!=cudaSuccess)return e;}while(0)
    for(auto& w:g_queryWorkspace){
        QUERY_ALLOC(w.before,QUERY_MAX_SLOTS*sizeof(SurfaceContact));QUERY_ALLOC(w.reference,QUERY_MAX_SLOTS*sizeof(SurfaceContact));
        QUERY_ALLOC(w.degreeBefore,nodes*sizeof(int));QUERY_ALLOC(w.degreeRef,nodes*sizeof(int));
        QUERY_ALLOC(w.activeBefore,QUERY_MAX_SLOTS*sizeof(int));QUERY_ALLOC(w.activeRef,QUERY_MAX_SLOTS*sizeof(int));
        QUERY_ALLOC(w.countBefore,sizeof(int));QUERY_ALLOC(w.countRef,sizeof(int));
        QUERY_ALLOC(w.frequencies,2*QUERY_MAX_SLOTS*sizeof(int));QUERY_ALLOC(w.bad,sizeof(int));
    }
#undef QUERY_ALLOC
    return cudaSuccess;
}
static bool canBatchQueries(int triangles,const CoupledTool* body,int flag){
    return (g_queryBatchMode&flag) && body && triangles>0 && triangles<=QUERY_MAX_TRIANGLES && g_queryWorkspace[0].ids;
}
// One CTA preserves the original kind grouping, including when counters are zero.
__global__ void k_AppendQueryKinds(int triangles,int kindCount,const int* grouped,const int* counts,int* active,int* activeCount){
    __shared__ int offsets[7],base;
    if(threadIdx.x==0){base=*activeCount;offsets[0]=0;for(int k=0;k<kindCount;k++)offsets[k+1]=offsets[k]+counts[k];*activeCount=base+offsets[kindCount];}
    __syncthreads();
    for(int k=0;k<kindCount;k++)for(int i=threadIdx.x;i<counts[k];i+=blockDim.x)active[base+offsets[k]+i]=grouped[k*triangles+i];
}
__device__ inline void queryMismatch(int* bad,int* stats,int category){atomicExch(bad,1);atomicAdd(stats+category,1);}
__global__ void k_CompareQueryRecords(int entries,const SurfaceContact* reference,const SurfaceContact* actual,int* bad,int* stats){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=entries)return;
    const SurfaceContact& a=reference[i];const SurfaceContact& b=actual[i];
    if(a.count!=b.count || a.kind!=b.kind || __float_as_uint(a.lambdaN)!=__float_as_uint(b.lambdaN))queryMismatch(bad,stats,8);
    if(a.count>0){
        atomicAdd(stats+2,1);const unsigned* x=reinterpret_cast<const unsigned*>(&a);const unsigned* y=reinterpret_cast<const unsigned*>(&b);
        for(int k=0;k<(int)(sizeof(SurfaceContact)/sizeof(unsigned));k++)if(x[k]!=y[k]){
            queryMismatch(bad,stats,5);
            if(atomicCAS(stats+11,-1,i)==-1){stats[12]=k;stats[13]=(int)x[k];stats[14]=(int)y[k];}
            break;
        }
    }
}
__global__ void k_CompareQueryDegrees(int nodes,const int* a,const int* b,int* bad,int* stats){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<nodes && a[i]!=b[i])queryMismatch(bad,stats,6);
}
__global__ void k_QueryFrequencies(int entries,unsigned packed,int kindCount,const int* a,const int* na,const int* b,const int* nb,int* freq,int* bad,int* stats){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i==0 && *na!=*nb)queryMismatch(bad,stats,7);
    int n0=*na,n1=*nb;
    if(n0<0||n0>entries||n1<0||n1>entries){if(i==0)queryMismatch(bad,stats,7);return;}
    if(i<n0){int id=a[i];if(id<0||id>=entries)queryMismatch(bad,stats,7);else atomicAdd(freq+id,1);}
    if(i<n1){int id=b[i];if(id<0||id>=entries)queryMismatch(bad,stats,7);else atomicAdd(freq+entries+id,1);}
    // Cross-kind order was ordered in the old launch loop, and stays ordered.
    if(i>0 && i<n1){
        int stride=(entries==0)?1:6; // this checker is currently used for blade slots
        if(b[i-1]%stride>b[i]%stride)queryMismatch(bad,stats,9);
    }
}
__global__ void k_CompareQueryFrequencies(int entries,const int* freq,int* bad,int* stats){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<entries && freq[i]!=freq[entries+i])queryMismatch(bad,stats,7);
}
__global__ void k_FinishQueryValidation(int kinds,bool captured,const int* bad,int* stats){
    if(threadIdx.x||blockIdx.x)return;atomicAdd(stats,1);atomicAdd(stats+4,kinds);if(*bad)atomicAdd(stats+1,1);if(captured)atomicAdd(stats+3,1);
}
static inline void launchFindSurfaceGroup(int triangles,unsigned packed,int kindCount,bool bladeBatch,cudaStream_t stream,
    ContactShape from,ContactShape to,float skin,const SurfaceVertex* vertices,const unsigned* masks,const float3* prev,const float3* pos,
    SurfaceContact* contacts,int* degree,int* active,int* activeCount,const float4* current,const float3* previous,
    const CoupledTool* body,float thickness,float minGap,bool stationary,const int* cacheSlots,const SurfaceContact* cache,
    const int* bladePhase,int stride,const unsigned char* groupBase){
    int index=stream?1:0;
    if(stream){if(!g_queryNonzeroStream)g_queryNonzeroStream=stream;else if(g_queryNonzeroStream!=stream)index=-1;}
    auto reference=[&](){
        for(int j=0;j<kindCount;j++){
            int kind=(packed>>(3*j))&7;bool back=bladeBatch&&kind>=3;
            k_FindSurface<<<(triangles+63)/64,64,0,stream>>>(triangles,kind,from,to,bladeBatch?(back?skin:0):skin,
                vertices,masks,prev,pos,contacts,degree,active,activeCount,current,previous,body,thickness,minGap,stationary,
                bladeBatch&&!back?nullptr:cacheSlots,bladeBatch&&!back?nullptr:cache,back?nullptr:bladePhase,stride,back,
                groupBase?groupBase+kind*((triangles+63)/64):nullptr);
        }
    };
    if(index<0){reference();return;}
    QueryGroupWorkspace& w=g_queryWorkspace[index];
    auto batched=[&](){
        cudaMemsetAsync(w.counts,0,6*sizeof(int),stream);
        k_FindSurfaceBatched<<<dim3((triangles+63)/64,kindCount),64,0,stream>>>(triangles,packed,bladeBatch,from,to,skin,
            vertices,masks,prev,pos,contacts,degree,current,previous,body,thickness,minGap,stationary,cacheSlots,cache,bladePhase,stride,groupBase,w.ids,w.counts);
        k_AppendQueryKinds<<<1,256,0,stream>>>(triangles,kindCount,w.ids,w.counts,active,activeCount);
    };
    if(!g_queryValidate){batched();return;}
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;cudaStreamIsCapturing(stream,&capture);
    const int entries=triangles*stride;
    cudaMemsetAsync(w.bad,0,sizeof(int),stream);
    cudaMemcpyAsync(w.before,contacts,entries*sizeof(SurfaceContact),cudaMemcpyDeviceToDevice,stream);
    cudaMemcpyAsync(w.degreeBefore,degree,g_queryNodes*sizeof(int),cudaMemcpyDeviceToDevice,stream);
    cudaMemcpyAsync(w.activeBefore,active,entries*sizeof(int),cudaMemcpyDeviceToDevice,stream);
    cudaMemcpyAsync(w.countBefore,activeCount,sizeof(int),cudaMemcpyDeviceToDevice,stream);
    reference();
    cudaMemcpyAsync(w.reference,contacts,entries*sizeof(SurfaceContact),cudaMemcpyDeviceToDevice,stream);
    cudaMemcpyAsync(w.degreeRef,degree,g_queryNodes*sizeof(int),cudaMemcpyDeviceToDevice,stream);
    cudaMemcpyAsync(w.activeRef,active,entries*sizeof(int),cudaMemcpyDeviceToDevice,stream);
    cudaMemcpyAsync(w.countRef,activeCount,sizeof(int),cudaMemcpyDeviceToDevice,stream);
    cudaMemcpyAsync(contacts,w.before,entries*sizeof(SurfaceContact),cudaMemcpyDeviceToDevice,stream);
    cudaMemcpyAsync(degree,w.degreeBefore,g_queryNodes*sizeof(int),cudaMemcpyDeviceToDevice,stream);
    cudaMemcpyAsync(active,w.activeBefore,entries*sizeof(int),cudaMemcpyDeviceToDevice,stream);
    cudaMemcpyAsync(activeCount,w.countBefore,sizeof(int),cudaMemcpyDeviceToDevice,stream);
    batched();
    k_CompareQueryRecords<<<(entries+127)/128,128,0,stream>>>(entries,w.reference,contacts,w.bad,d_queryValidationStats);
    k_CompareQueryDegrees<<<(g_queryNodes+127)/128,128,0,stream>>>(g_queryNodes,w.degreeRef,degree,w.bad,d_queryValidationStats);
    cudaMemsetAsync(w.frequencies,0,2*entries*sizeof(int),stream);
    k_QueryFrequencies<<<(entries+127)/128,128,0,stream>>>(entries,packed,kindCount,w.activeRef,w.countRef,active,activeCount,w.frequencies,w.bad,d_queryValidationStats);
    k_CompareQueryFrequencies<<<(entries+127)/128,128,0,stream>>>(entries,w.frequencies,w.bad,d_queryValidationStats);
    k_FinishQueryValidation<<<1,1,0,stream>>>(kindCount,capture!=cudaStreamCaptureStatusNone,w.bad,d_queryValidationStats);
}
