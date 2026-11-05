// Original fused material/contact kernel and reduction; different compiler register budget only.
template<int N,class Add> __device__ void blockAccumulateOccupancy(const void* target,float (&v)[N],Add add){
#ifdef ENDO_KIDNEY_PRECISE_COARSE
    blockAccumulateCache(target,v,add);
#else
    __shared__ unsigned long long first;__shared__ float sum[N];
    if(threadIdx.x==0)first=0;
    for(int j=threadIdx.x;j<N;j+=blockDim.x)sum[j]=0;
    __syncthreads();
    if(target)atomicCAS(&first,0ull,(unsigned long long)target);
    __syncthreads();
    if(target && (unsigned long long)target==first){
        auto group=cooperative_groups::coalesced_threads();
        for(int j=0;j<N;j++)v[j]=cooperative_groups::reduce(group,v[j],cooperative_groups::plus<float>());
        if(group.thread_rank()==0)for(int j=0;j<N;j++)atomicAdd(sum+j,v[j]);
    }else if(target && unifiedReduce(target,v))add(target,v);
    __syncthreads();
    if(threadIdx.x==0 && first)add((const void*)first,sum);
#endif
}
__device__ inline void coarseAccumulateBlockOccupancy(CoarseBlock* b,float3 g,const Symmetric3& H){
    float v[9]={g.x,g.y,g.z,H.xx,H.yy,H.zz,H.xy,H.xz,H.yz};
    blockAccumulateOccupancy(b,v,[](const void* target,const float* values){coarseAdd(target,values);});
}

template<bool IncludeContacts> __device__ __forceinline__ void coarseOccupancyBody(const UnifiedState* state,CoarseBlock* blocks,CoarseBoundTerm* terms,float* actual,CoarseBlock** targets){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    float3 g=mkf3(0,0,0);Symmetric3 H={};float scale=1;
    CoarseBlock* target=t<s.activeCount?coarseNodeTerms(s,s.activeNodes[t],blocks,g,H):
        coarseTetTermsCache(s,t-s.activeCount,blocks,false,g,H,scale,terms);
    if(actual && t>=s.activeCount && t-s.activeCount<s.nt){int tet=t-s.activeCount;targets[tet]=target;
        float v[9]={g.x,g.y,g.z,H.xx,H.yy,H.zz,H.xy,H.xz,H.yz};for(int j=0;j<9;j++)actual[9*tet+j]=v[j];}
    coarseAccumulateBlockOccupancy(target,g,H);
    if constexpr(IncludeContacts)coarseContactGripTerms(s,blocks,t,gridDim.x*blockDim.x);
}
template<int MinimumBlocks> __global__ __launch_bounds__(256,MinimumBlocks) void k_CoarseOccupancyBoundCache(const UnifiedState* state,CoarseBlock* blocks,CoarseBoundTerm* terms,float* actual,CoarseBlock** targets){
    coarseOccupancyBody<true>(state,blocks,terms,actual,targets);
}
#ifdef ENDO_KIDNEY_SPLIT_COARSE
// Keep the old material thread/reduction mapping and all per-term functions.
// Only the contact/grip striding moves to a separate launch on the same stream;
// the following solve still observes every complete coarse contribution.
template<int MinimumBlocks> __global__ __launch_bounds__(256,MinimumBlocks) void k_CoarseMaterialBoundCache(const UnifiedState* state,CoarseBlock* blocks,CoarseBoundTerm* terms,float* actual,CoarseBlock** targets){
    coarseOccupancyBody<false>(state,blocks,terms,actual,targets);
}
__global__ void k_CoarseContactBoundCache(const UnifiedState* state,CoarseBlock* blocks){
    coarseContactGripTerms(*state,blocks,blockIdx.x*blockDim.x+threadIdx.x,gridDim.x*blockDim.x);
}
#endif
