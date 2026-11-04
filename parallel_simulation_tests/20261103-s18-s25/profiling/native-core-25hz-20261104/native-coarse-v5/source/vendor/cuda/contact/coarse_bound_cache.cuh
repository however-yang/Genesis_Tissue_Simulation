// Derived coefficients used only between adjacent accumulate/solve/bound launches.
// Inputs pos/coarseWeight/live/component are unchanged across that dependency chain.
struct CoarseBoundTerm {float3 gJ;float J;CoarseBlock* target;};
static int g_coarseBoundCacheEnabled=0,g_coarseBoundCacheCapacity=0;
static CoarseBoundTerm* d_coarseBoundTerms[2]={};
static unsigned long long* d_coarseBoundStats=nullptr;
static unsigned long long* d_coarseUniformStats=nullptr;
static float* d_coarseActualValues[2]={};
static CoarseBlock** d_coarseActualTargets[2]={};
static cudaStream_t g_coarseBoundCacheStream=nullptr;
static cudaError_t ensureCoarseBoundCache(int tets){
    if(!g_coarseBoundCacheEnabled || tets<=0)return cudaSuccess;
    if(tets<=g_coarseBoundCacheCapacity)return cudaSuccess;
    if(g_coarseBoundCacheCapacity)return cudaErrorInvalidValue; // never free buffers referenced by live graphs
    for(auto& terms:d_coarseBoundTerms){cudaError_t e=queryPrivateAlloc(&terms,(size_t)tets*sizeof(CoarseBoundTerm));if(e!=cudaSuccess)return e;}
    g_coarseBoundCacheCapacity=tets;return cudaSuccess;
}
static cudaError_t ensureCoarseBoundStats(){
    if(g_coreOptimizations&16){if(!d_coarseUniformStats){cudaError_t e=queryPrivateAlloc(&d_coarseUniformStats,4*sizeof(unsigned long long));if(e!=cudaSuccess)return e;}cudaError_t e=cudaMemset(d_coarseUniformStats,0,4*sizeof(unsigned long long));if(e!=cudaSuccess)return e;}
    if(!d_coarseBoundStats){cudaError_t e=queryPrivateAlloc(&d_coarseBoundStats,8*sizeof(unsigned long long));if(e!=cudaSuccess)return e;}
    for(int k=0;k<2;k++){
        if(!d_coarseActualValues[k]){cudaError_t e=queryPrivateAlloc(&d_coarseActualValues[k],(size_t)9*g_coarseBoundCacheCapacity*sizeof(float));if(e!=cudaSuccess)return e;}
        if(!d_coarseActualTargets[k]){cudaError_t e=queryPrivateAlloc(&d_coarseActualTargets[k],(size_t)g_coarseBoundCacheCapacity*sizeof(CoarseBlock*));if(e!=cudaSuccess)return e;}
    }
    return cudaMemset(d_coarseBoundStats,0,8*sizeof(unsigned long long));
}
static void freeCoarseBoundCache(){
    cudaDeviceSynchronize();queryPrivateFree(d_coarseUniformStats);d_coarseUniformStats=nullptr;
    cudaDeviceSynchronize();for(auto& terms:d_coarseBoundTerms){queryPrivateFree(terms);terms=nullptr;}
    for(int k=0;k<2;k++){queryPrivateFree(d_coarseActualValues[k]);d_coarseActualValues[k]=nullptr;queryPrivateFree(d_coarseActualTargets[k]);d_coarseActualTargets[k]=nullptr;}
    queryPrivateFree(d_coarseBoundStats);d_coarseBoundStats=nullptr;g_coarseBoundCacheCapacity=0;g_coarseBoundCacheStream=nullptr;
}
template<int N,class Add> __device__ void blockAccumulateCache(const void* target,float (&v)[N],Add add){
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
}
template<int N,class Add> __device__ void blockAccumulateUniformFallback(const void* target,float (&v)[N],Add add){
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
}
// The common one-component CTA has a unique target regardless of atomicCAS
// winner. Discover that SAME target with one ballot/shuffle per warp instead
// of 256 contending 64-bit shared atomics. Mixed CTAs use the original reduction.
template<int N,class Add> __device__ void blockAccumulateUniform(const void* target,float (&v)[N],Add add,unsigned long long* stats){
    __shared__ unsigned long long candidates[8],first;
    __shared__ int uniform;
    __shared__ float sum[N];
    for(int j=threadIdx.x;j<N;j+=blockDim.x)sum[j]=0;
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    unsigned long long key=(unsigned long long)target;
    unsigned mask=__ballot_sync(0xffffffffu,target!=nullptr);
    unsigned long long leader=__shfl_sync(0xffffffffu,key,mask?__ffs(mask)-1:0);
    bool same=__all_sync(0xffffffffu,!target || key==leader);
    if(lane==0)candidates[warp]=same?leader:~0ull;
    __syncthreads();
    if(threadIdx.x==0){
        first=0;uniform=1;
        for(int w=0;w<8;w++){
            unsigned long long c=candidates[w];
            if(c==~0ull){uniform=0;continue;}
            if(c){if(first && first!=c)uniform=0;else first=c;}
        }
        if(stats){atomicAdd(stats,1ull);atomicAdd(stats+(uniform?(first?1:2):3),1ull);}
    }
    __syncthreads();
    if(!uniform){blockAccumulateUniformFallback(target,v,add);return;}
    if(!first)return;
    if(target){
        auto group=cooperative_groups::coalesced_threads();
        for(int j=0;j<N;j++)v[j]=cooperative_groups::reduce(group,v[j],cooperative_groups::plus<float>());
        if(group.thread_rank()==0)for(int j=0;j<N;j++)atomicAdd(sum+j,v[j]);
    }
    __syncthreads();
    if(threadIdx.x==0)add((const void*)first,sum);
}
__device__ inline void coarseAccumulateBlockUniform(CoarseBlock* b,float3 g,const Symmetric3& H,unsigned long long* stats){
    float v[9]={g.x,g.y,g.z,H.xx,H.yy,H.zz,H.xy,H.xz,H.yz};
    blockAccumulateUniform(b,v,[](const void* target,const float* values){coarseAdd(target,values);},stats);
}
__device__ inline void coarseAccumulateBlockCache(CoarseBlock* b,float3 g,const Symmetric3& H){
    float v[9]={g.x,g.y,g.z,H.xx,H.yy,H.zz,H.xy,H.xz,H.yz};
    blockAccumulateCache(b,v,[](const void* target,const float* values){coarseAdd(target,values);});
}

__device__ inline CoarseBlock* coarseTetTermsCache(const UnifiedState& s,int t,CoarseBlock* block,bool bound,float3& g,Symmetric3& H,float& scale,CoarseBoundTerm* terms){
    if(t>=s.nt)return nullptr;terms[t].target=nullptr;
    if(!s.live[t])return nullptr;
    int id[4];unifiedTetIds(s,t,id);if(s.coarseMode)block+=s.component[id[0]];float w[4];for(int j=0;j<4;j++)w[j]=s.coarseWeight[id[j]];
    float3 q=unifiedKuhn(s,t).b0*(w[1]-w[0])+unifiedKuhn(s,t).b1*(w[2]-w[0])+unifiedKuhn(s,t).b2*(w[3]-w[0]);
    if(dot3(q,q)<1e-20f)return nullptr;
    float3 a=s.pos[id[0]],e1=s.pos[id[1]]-a,e2=s.pos[id[2]]-a,e3=s.pos[id[3]]-a;
    float det=unifiedKuhn(s,t).metric[6],J=dot3(e1,cross3(e2,e3))*det;
    float3 gJ=(cross3(e2,e3)*(w[1]-w[0])+cross3(e3,e1)*(w[2]-w[0])+cross3(e1,e2)*(w[3]-w[0]))*det;
    terms[t].gJ=gJ;terms[t].J=J;terms[t].target=block;
    if(bound){float dj=dot3(gJ,block->step);if(!(dj<0))return nullptr;scale=fminf(1.f,fmaxf(0.f,.9f*(J-s.minimum)/(-dj)));return block;}
    float3 Fq=e1*dot3(unifiedKuhn(s,t).b0,q)+e2*dot3(unifiedKuhn(s,t).b1,q)+e3*dot3(unifiedKuhn(s,t).b2,q);
    float mu=unifiedKuhn(s,t).muV,lambda=unifiedKuhn(s,t).bulkV+mu/3.f;
    CompressionBarrier barrier=unifiedCompressionBarrier(J,s.minimum,lambda);
    g=Fq*mu+gJ*(lambda*(J-1)-mu+barrier.gradient);H.diagonal(mu*dot3(q,q));H.outer(gJ,lambda+barrier.curvature);
    unifiedStrainDamping(s,t,e1,e2,e3,q,g,H);
    for(int j=0;j<4;j++)for(int k=j+1;k<4;k++){
        float dw=w[j]-w[k];if(dw==0)continue;float3 e=s.pos[id[j]]-s.pos[id[k]];
        float l=len3(e),l0=unifiedRestLength(s,t,j,k),strain=fmaxf(0.f,l/l0-1.f);
        if(strain>0 && l>1e-12f){float s2=strain*strain,s4=s2*s2;float f=3*mu*s.hardening*s4*strain/l0,h=15*mu*s.hardening*s4/(l0*l0);float3 n=e*(1.f/l);
            g=g+n*(f*dw);H.diagonal(f/l*dw*dw);H.outer(n,(h-f/l)*dw*dw);}
    }
    return block;
}

__device__ inline CoarseBlock* coarseCachedBound(const UnifiedState& s,int t,CoarseBoundTerm* terms,float& scale){
    if(t>=s.nt)return nullptr;const CoarseBoundTerm& term=terms[t];CoarseBlock* block=term.target;if(!block)return nullptr;
    float dj=dot3(term.gJ,block->step);if(!(dj<0))return nullptr;
    scale=fminf(1.f,fmaxf(0.f,.9f*(term.J-s.minimum)/(-dj)));return block;
}
// The SAME production kernels emit optional diagnostics. Validation never swaps
// their numerical code for a differently compiled debug specialization.
template<bool Bound> __global__ void k_CompareCoarseCacheTerms(const UnifiedState* state,CoarseBlock* blocks,
    const float* values,CoarseBlock* const* targets,unsigned long long* stats){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    bool checked=t<s.nt,badTarget=false,badValue=false,active=false,nontrivial=false;
    if(checked){float3 rg=mkf3(0,0,0);Symmetric3 rh={};float scale=1;
        CoarseBlock* reference=coarseTetTerms(s,t,blocks,Bound,rg,rh,scale),*actual=targets[t];badTarget=reference!=actual;
        if(reference && actual){
            if constexpr(Bound){badValue=__float_as_uint(scale)!=__float_as_uint(values[t]);active=true;nontrivial=values[t]>0 && values[t]<1;}
            else {float r[9]={rg.x,rg.y,rg.z,rh.xx,rh.yy,rh.zz,rh.xy,rh.xz,rh.yz};
                for(int j=0;j<9;j++)badValue|=__float_as_uint(values[9*t+j])!=__float_as_uint(r[j]);}
        }
    }
    unsigned mask=__activemask(),c=__ballot_sync(mask,checked),p=__ballot_sync(mask,badTarget),v=__ballot_sync(mask,badValue);
    unsigned activeCount=__ballot_sync(mask,active),nontrivialCount=__ballot_sync(mask,nontrivial);
    if((threadIdx.x&31)==__ffs(mask)-1){int k=Bound?3:0;atomicAdd(stats+k,(unsigned long long)__popc(c));atomicAdd(stats+k+1,(unsigned long long)__popc(p));atomicAdd(stats+k+2,(unsigned long long)__popc(v));
        if constexpr(Bound){atomicAdd(stats+6,(unsigned long long)__popc(activeCount));atomicAdd(stats+7,(unsigned long long)__popc(nontrivialCount));}}
}
__global__ void k_CoarseAccumulateBoundCache(const UnifiedState* state,CoarseBlock* blocks,CoarseBoundTerm* terms,float* actual,CoarseBlock** targets){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    float3 g=mkf3(0,0,0);Symmetric3 H={};float scale=1;
    CoarseBlock* target=t<s.activeCount?coarseNodeTerms(s,s.activeNodes[t],blocks,g,H):
        coarseTetTermsCache(s,t-s.activeCount,blocks,false,g,H,scale,terms);
    if(actual && t>=s.activeCount && t-s.activeCount<s.nt){int tet=t-s.activeCount;targets[tet]=target;
        float v[9]={g.x,g.y,g.z,H.xx,H.yy,H.zz,H.xy,H.xz,H.yz};for(int j=0;j<9;j++)actual[9*tet+j]=v[j];}
    coarseAccumulateBlockCache(target,g,H);
    coarseContactGripTerms(s,blocks,t,gridDim.x*blockDim.x);
}
__global__ void k_CoarseAccumulateUniform(const UnifiedState* state,CoarseBlock* blocks,CoarseBoundTerm* terms,float* actual,CoarseBlock** targets,unsigned long long* stats){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    float3 g=mkf3(0,0,0);Symmetric3 H={};float scale=1;
    CoarseBlock* target=t<s.activeCount?coarseNodeTerms(s,s.activeNodes[t],blocks,g,H):
        coarseTetTermsCache(s,t-s.activeCount,blocks,false,g,H,scale,terms);
    if(actual && t>=s.activeCount && t-s.activeCount<s.nt){int tet=t-s.activeCount;targets[tet]=target;
        float v[9]={g.x,g.y,g.z,H.xx,H.yy,H.zz,H.xy,H.xz,H.yz};for(int j=0;j<9;j++)actual[9*tet+j]=v[j];}
    coarseAccumulateBlockUniform(target,g,H,stats);
    coarseContactGripTerms(s,blocks,t,gridDim.x*blockDim.x);
}
__global__ void k_CoarseBoundFromCache(const UnifiedState* state,CoarseBlock* blocks,CoarseBoundTerm* terms,float* actual,CoarseBlock** targets){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    float scale=1;CoarseBlock* target=coarseCachedBound(s,t,terms,scale);
    if(actual && t<s.nt){actual[t]=scale;targets[t]=target;}
    if(target)coarseBound(target,scale);
    coarseContactBoundTerms(s,blocks,t,gridDim.x*blockDim.x);
}
static CoarseBoundTerm* coarseCacheForStream(cudaStream_t stream){
    if(!g_coarseBoundCacheEnabled)return nullptr;
    if(stream){if(!g_coarseBoundCacheStream)g_coarseBoundCacheStream=stream;else if(g_coarseBoundCacheStream!=stream)return nullptr;}
    return d_coarseBoundTerms[stream?1:0];
}
static void launchCoarseAccumulateCache(int blocks,cudaStream_t stream,const UnifiedState* state,CoarseBlock* coarse){
    auto terms=coarseCacheForStream(stream);int k=stream?1:0;
    if(!terms)k_CoarseAccumulate<<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse);
    else {
        if(g_coreOptimizations&16){
        k_CoarseAccumulateUniform<<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,terms,g_queryValidate?d_coarseActualValues[k]:nullptr,g_queryValidate?d_coarseActualTargets[k]:nullptr,g_queryValidate?d_coarseUniformStats:nullptr);
        }else{
        k_CoarseAccumulateBoundCache<<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,terms,g_queryValidate?d_coarseActualValues[k]:nullptr,g_queryValidate?d_coarseActualTargets[k]:nullptr);
        }
        if(g_queryValidate)k_CompareCoarseCacheTerms<false><<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,d_coarseActualValues[k],d_coarseActualTargets[k],d_coarseBoundStats);
    }
}
static void launchCoarseBoundCache(int blocks,cudaStream_t stream,const UnifiedState* state,CoarseBlock* coarse){
    auto terms=coarseCacheForStream(stream);int k=stream?1:0;
    if(!terms)k_CoarseBound<<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse);
    else {
        k_CoarseBoundFromCache<<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,terms,g_queryValidate?d_coarseActualValues[k]:nullptr,g_queryValidate?d_coarseActualTargets[k]:nullptr);
        if(g_queryValidate)k_CompareCoarseCacheTerms<true><<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,d_coarseActualValues[k],d_coarseActualTargets[k],d_coarseBoundStats);
    }
}
