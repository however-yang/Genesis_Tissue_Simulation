// Retain the exact material functions, thread mapping, warp float reductions,
// block double sums and final block-to-float rounding of blockAccumulateCache.
// Only each block's first/root contribution is staged. Other component terms
// and all contact/grip terms keep the reference coarseAdd path.
__device__ inline void coarseAccumulateStaged(CoarseBlock* blocks,CoarseBlock* target,
    float3 g,const Symmetric3& H,CoarsePartial* output){
    __shared__ unsigned long long first;
    __shared__ double sum[9];
    if(threadIdx.x==0){first=~0ull;output->root=-1;}
    for(int j=threadIdx.x;j<9;j+=blockDim.x)sum[j]=0;
    __syncthreads();
    if(target)atomicMin(&first,(unsigned long long)target);
    __syncthreads();
    float values[9]={g.x,g.y,g.z,H.xx,H.yy,H.zz,H.xy,H.xz,H.yz};
    if(target && (unsigned long long)target==first){
        auto group=cooperative_groups::coalesced_threads();
        for(int j=0;j<9;j++)values[j]=cooperative_groups::reduce(group,values[j],cooperative_groups::plus<float>());
        if(group.thread_rank()==0)for(int j=0;j<9;j++)atomicAdd(sum+j,(double)values[j]);
    }else if(target && unifiedReduce(target,values))coarseAdd(target,values);
    __syncthreads();
    if(threadIdx.x==0 && first!=~0ull){
        output->root=(int)((CoarseBlock*)first-blocks);
        for(int j=0;j<9;j++)output->values[j]=(float)sum[j];
    }
}

template<int MinimumBlocks> __global__ __launch_bounds__(256,MinimumBlocks) void k_CoarseStagedBoundCache(
    const UnifiedState* state,CoarseBlock* blocks,CoarseBoundTerm* terms,CoarsePartial* partials,
    float* actual,CoarseBlock** targets){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    float3 g=mkf3(0,0,0);Symmetric3 H={};float scale=1;
    CoarseBlock* target=t<s.activeCount?coarseNodeTerms(s,s.activeNodes[t],blocks,g,H):
        coarseTetTermsCache(s,t-s.activeCount,blocks,false,g,H,scale,terms);
    if(actual && t>=s.activeCount && t-s.activeCount<s.nt){
        int tet=t-s.activeCount;targets[tet]=target;
        float values[9]={g.x,g.y,g.z,H.xx,H.yy,H.zz,H.xy,H.xz,H.yz};
        for(int j=0;j<9;j++)actual[9*tet+j]=values[j];
    }
    coarseAccumulateStaged(blocks,target,g,H,partials+blockIdx.x);
    coarseContactGripTerms(s,blocks,t,gridDim.x*blockDim.x);
}

__global__ void k_CoarseAddPartials(int count,CoarseBlock* blocks,const CoarsePartial* partials){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count || partials[i].root<0)return;
    const auto& p=partials[i];unsigned mask=__activemask();
    bool common=__all_sync(mask,p.root==__shfl_sync(mask,p.root,__ffs(mask)-1));
    auto group=cooperative_groups::coalesced_threads();
    for(int j=0;j<9;j++){
        double value=(double)p.values[j];
        if(common)value=cooperative_groups::reduce(group,value,cooperative_groups::plus<double>());
        if(!common || group.thread_rank()==0)atomicAdd(blocks[p.root].accumulated+j,value);
    }
}
