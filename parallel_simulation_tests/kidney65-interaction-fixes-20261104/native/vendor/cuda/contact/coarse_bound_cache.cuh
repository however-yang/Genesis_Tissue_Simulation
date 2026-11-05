// Derived coefficients used only between adjacent accumulate/solve/bound launches.
// Inputs pos/coarseWeight/live/component are unchanged across that dependency chain.
struct CoarseBoundTerm {float3 gJ;float J;CoarseBlock* target;};
static int g_coarseBoundCacheEnabled=0,g_coarseBoundCacheCapacity=0;
static CoarseBoundTerm* d_coarseBoundTerms[2]={};
static unsigned long long* d_coarseBoundStats=nullptr;
static float* d_coarseActualValues[2]={};
static CoarseBlock** d_coarseActualTargets[2]={};
static cudaStream_t g_coarseBoundCacheStream=nullptr;
#ifdef ENDO_KIDNEY_STAGED_COARSE
struct CoarsePartial {int root;float values[9];};
static CoarsePartial* d_coarsePartial[2]={};
static int g_coarsePartialCapacity=0;
#endif
static cudaError_t ensureCoarseBoundCache(int tets,int nodes){
    if(!g_coarseBoundCacheEnabled || tets<=0)return cudaSuccess;
    if(tets<=g_coarseBoundCacheCapacity)return cudaSuccess;
    if(g_coarseBoundCacheCapacity)return cudaErrorInvalidValue; // never free buffers referenced by live graphs
    for(auto& terms:d_coarseBoundTerms){cudaError_t e=queryPrivateAlloc(&terms,(size_t)tets*sizeof(CoarseBoundTerm));if(e!=cudaSuccess)return e;}
#ifdef ENDO_KIDNEY_STAGED_COARSE
    // Scheduling-only scratch, like d_coarseBoundTerms. Every launched block
    // writes its root marker and (if valid) all nine values before the reducer
    // on the same stream can read them. No cross-step or restored state is
    // consumed; native restore invalidates the execution graphs as before.
    int blocks=(nodes+tets+ACCUMULATE_THREADS-1)/ACCUMULATE_THREADS;
    for(auto& values:d_coarsePartial){cudaError_t e=queryPrivateAlloc(&values,(size_t)blocks*sizeof(CoarsePartial));if(e!=cudaSuccess)return e;}
    g_coarsePartialCapacity=blocks;
#endif
    g_coarseBoundCacheCapacity=tets;return cudaSuccess;
}
static cudaError_t ensureCoarseBoundStats(){
    if(!d_coarseBoundStats){cudaError_t e=queryPrivateAlloc(&d_coarseBoundStats,8*sizeof(unsigned long long));if(e!=cudaSuccess)return e;}
    for(int k=0;k<2;k++){
        if(!d_coarseActualValues[k]){cudaError_t e=queryPrivateAlloc(&d_coarseActualValues[k],(size_t)9*g_coarseBoundCacheCapacity*sizeof(float));if(e!=cudaSuccess)return e;}
        if(!d_coarseActualTargets[k]){cudaError_t e=queryPrivateAlloc(&d_coarseActualTargets[k],(size_t)g_coarseBoundCacheCapacity*sizeof(CoarseBlock*));if(e!=cudaSuccess)return e;}
    }
    return cudaMemset(d_coarseBoundStats,0,8*sizeof(unsigned long long));
}
static void freeCoarseBoundCache(){
    cudaDeviceSynchronize();for(auto& terms:d_coarseBoundTerms){queryPrivateFree(terms);terms=nullptr;}
#ifdef ENDO_KIDNEY_STAGED_COARSE
    for(auto& values:d_coarsePartial){queryPrivateFree(values);values=nullptr;}
    g_coarsePartialCapacity=0;
#endif
    for(int k=0;k<2;k++){queryPrivateFree(d_coarseActualValues[k]);d_coarseActualValues[k]=nullptr;queryPrivateFree(d_coarseActualTargets[k]);d_coarseActualTargets[k]=nullptr;}
    queryPrivateFree(d_coarseBoundStats);d_coarseBoundStats=nullptr;g_coarseBoundCacheCapacity=0;g_coarseBoundCacheStream=nullptr;
}
template<int N,class Add> __device__ void blockAccumulateCache(const void* target,float (&v)[N],Add add){
#ifdef ENDO_KIDNEY_PRECISE_COARSE
    __shared__ unsigned long long first;__shared__ double sum[N];
    if(threadIdx.x==0)first=~0ull;
    for(int j=threadIdx.x;j<N;j+=blockDim.x)sum[j]=0;
    __syncthreads();
    if(target)atomicMin(&first,(unsigned long long)target);
    __syncthreads();
    if(target && (unsigned long long)target==first){
        auto group=cooperative_groups::coalesced_threads();
        for(int j=0;j<N;j++)v[j]=cooperative_groups::reduce(group,v[j],cooperative_groups::plus<float>());
        if(group.thread_rank()==0)for(int j=0;j<N;j++)atomicAdd(sum+j,(double)v[j]);
    }else if(target && unifiedReduce(target,v))add(target,v);
    __syncthreads();
    if(threadIdx.x==0 && first!=~0ull){float values[N];for(int j=0;j<N;j++)values[j]=(float)sum[j];add((const void*)first,values);}
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
__global__ void k_CoarseBoundFromCache(const UnifiedState* state,CoarseBlock* blocks,CoarseBoundTerm* terms,float* actual,CoarseBlock** targets){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    float scale=1;CoarseBlock* target=coarseCachedBound(s,t,terms,scale);
    if(actual && t<s.nt){actual[t]=scale;targets[t]=target;}
    if(target)coarseBound(target,scale);
    coarseContactBoundTerms(s,blocks,t,gridDim.x*blockDim.x);
}
#include "coarse_occupancy.cuh"
#ifdef ENDO_KIDNEY_STAGED_COARSE
#include "coarse_staged.cuh"
#endif

static CoarseBoundTerm* coarseCacheForStream(cudaStream_t stream){
    if(!g_coarseBoundCacheEnabled)return nullptr;
    if(stream){if(!g_coarseBoundCacheStream)g_coarseBoundCacheStream=stream;else if(g_coarseBoundCacheStream!=stream)return nullptr;}
    return d_coarseBoundTerms[stream?1:0];
}
static void launchCoarseAccumulateCache(int blocks,cudaStream_t stream,const UnifiedState* state,CoarseBlock* coarse){
    auto terms=coarseCacheForStream(stream);int k=stream?1:0;
    if(!terms)k_CoarseAccumulate<<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse);
    else {
        if(g_runtimeCoarseBlocks==3){
#ifdef ENDO_KIDNEY_STAGED_COARSE
            if(d_coarsePartial[k] && blocks<=g_coarsePartialCapacity){
                k_CoarseStagedBoundCache<3><<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,terms,d_coarsePartial[k],g_queryValidate?d_coarseActualValues[k]:nullptr,g_queryValidate?d_coarseActualTargets[k]:nullptr);
                k_CoarseAddPartials<<<(blocks+31)/32,32,0,stream>>>(blocks,coarse,d_coarsePartial[k]);
            }else k_CoarseOccupancyBoundCache<3><<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,terms,g_queryValidate?d_coarseActualValues[k]:nullptr,g_queryValidate?d_coarseActualTargets[k]:nullptr);
#elif defined(ENDO_KIDNEY_SPLIT_COARSE)
            k_CoarseMaterialBoundCache<3><<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,terms,g_queryValidate?d_coarseActualValues[k]:nullptr,g_queryValidate?d_coarseActualTargets[k]:nullptr);
            k_CoarseContactBoundCache<<<128,64,0,stream>>>(state,coarse);
#else
            k_CoarseOccupancyBoundCache<3><<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,terms,g_queryValidate?d_coarseActualValues[k]:nullptr,g_queryValidate?d_coarseActualTargets[k]:nullptr);
#endif
        }
        else k_CoarseAccumulateBoundCache<<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,terms,g_queryValidate?d_coarseActualValues[k]:nullptr,g_queryValidate?d_coarseActualTargets[k]:nullptr);
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
