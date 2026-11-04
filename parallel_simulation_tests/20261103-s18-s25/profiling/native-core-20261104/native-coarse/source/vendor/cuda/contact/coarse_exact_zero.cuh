// Exact zero-weight shortcut; originals remain available for same-input comparison.
static int g_exactZeroCoarse=0;
static unsigned long long* d_coarseValidation=nullptr;
__device__ inline bool coarseWeightsDefinitelyConstant(const UnifiedState& s,int t){
    if(t<0 || t>=s.nt || !s.live[t])return false;
    int id[4];unifiedTetIds(s,t,id);float w=s.coarseWeight[id[0]];
    if(!isfinite(w) || w!=s.coarseWeight[id[1]] || w!=s.coarseWeight[id[2]] || w!=s.coarseWeight[id[3]])return false;
    const KuhnTet& k=unifiedKuhn(s,t);
    // OR is conservative: it may reject finite bases, but can never accept NaN/Inf.
    unsigned all=__float_as_uint(k.b0.x)|__float_as_uint(k.b0.y)|__float_as_uint(k.b0.z)|
        __float_as_uint(k.b1.x)|__float_as_uint(k.b1.y)|__float_as_uint(k.b1.z)|
        __float_as_uint(k.b2.x)|__float_as_uint(k.b2.y)|__float_as_uint(k.b2.z);
    return (all&0x7f800000u)!=0x7f800000u;
}
__device__ inline CoarseBlock* coarseTetTermsExactZero(const UnifiedState& s,int t,CoarseBlock* b,bool bound,float3& g,Symmetric3& H,float& scale){
    // The original computes q==0 then returns nullptr on precisely this subset.
    if(coarseWeightsDefinitelyConstant(s,t))return nullptr;
    return coarseTetTerms(s,t,b,bound,g,H,scale);
}
template<int N,class Add> __device__ void blockAccumulateExactZero(const void* target,float (&v)[N],Add add){
    __shared__ unsigned long long first;__shared__ float sum[N];
    if(threadIdx.x==0)first=0;
    for(int j=threadIdx.x;j<N;j+=blockDim.x)sum[j]=0;
    __syncthreads();
    if(target)atomicCAS(&first,0ull,(unsigned long long)target);
    __syncthreads();
    // Uniform CTA decision; no target means the original issues no atomic at all.
    if(!first)return;
    if(target && (unsigned long long)target==first){
        auto group=cooperative_groups::coalesced_threads();
        for(int j=0;j<N;j++)v[j]=cooperative_groups::reduce(group,v[j],cooperative_groups::plus<float>());
        if(group.thread_rank()==0)for(int j=0;j<N;j++)atomicAdd(sum+j,v[j]);
    }else if(target && unifiedReduce(target,v))add(target,v);
    __syncthreads();
    if(threadIdx.x==0 && first)add((const void*)first,sum);
}
__device__ inline void coarseValidateTerm(const UnifiedState& s,int t,CoarseBlock* b,bool bound,CoarseBlock* actual,float3 g,Symmetric3 H,float scale,unsigned long long* stats){
    bool checked=t>=0 && t<s.nt,zero=false,badTarget=false,badValue=false;
    if(checked){float3 rg=mkf3(0,0,0);Symmetric3 rh={};float rs=1;
        CoarseBlock* reference=coarseTetTerms(s,t,b,bound,rg,rh,rs);
        zero=coarseWeightsDefinitelyConstant(s,t);badTarget=reference!=actual;
        if(reference && actual){
            if(bound)badValue=__float_as_uint(rs)!=__float_as_uint(scale);
            else{float r[9]={rg.x,rg.y,rg.z,rh.xx,rh.yy,rh.zz,rh.xy,rh.xz,rh.yz};
                float a[9]={g.x,g.y,g.z,H.xx,H.yy,H.zz,H.xy,H.xz,H.yz};
                for(int j=0;j<9;j++)badValue|=__float_as_uint(r[j])!=__float_as_uint(a[j]);}
        }
    }
    unsigned mask=__activemask();unsigned c=__ballot_sync(mask,checked),z=__ballot_sync(mask,zero),p=__ballot_sync(mask,badTarget),v=__ballot_sync(mask,badValue);
    if((threadIdx.x&31)==__ffs(mask)-1){atomicAdd(stats,(unsigned long long)__popc(c));atomicAdd(stats+1,(unsigned long long)__popc(z));atomicAdd(stats+2,(unsigned long long)__popc(p));atomicAdd(stats+3,(unsigned long long)__popc(v));}
}
template<bool Validate> __global__ void k_CoarseAccumulateExactZero(const UnifiedState* state,CoarseBlock* block,unsigned long long* stats){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    float3 g=mkf3(0,0,0);Symmetric3 H={};float scale=1;
    CoarseBlock* target=t<s.activeCount?coarseNodeTerms(s,s.activeNodes[t],block,g,H):
        coarseTetTermsExactZero(s,t-s.activeCount,block,false,g,H,scale);
    if constexpr(Validate)coarseValidateTerm(s,t-s.activeCount,block,false,target,g,H,scale,stats);
    float v[9]={g.x,g.y,g.z,H.xx,H.yy,H.zz,H.xy,H.xz,H.yz};
    blockAccumulateExactZero(target,v,[](const void* ptr,const float* values){coarseAdd(ptr,values);});
    coarseContactGripTerms(s,block,t,gridDim.x*blockDim.x);
}
template<bool Validate> __global__ void k_CoarseBoundExactZero(const UnifiedState* state,CoarseBlock* block,unsigned long long* stats){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    float3 g=mkf3(0,0,0);Symmetric3 H={};float scale=1;
    CoarseBlock* target=coarseTetTermsExactZero(s,t,block,true,g,H,scale);
    if constexpr(Validate)coarseValidateTerm(s,t,block,true,target,g,H,scale,stats);
    if(target)coarseBound(target,scale);
    coarseContactBoundTerms(s,block,t,gridDim.x*blockDim.x);
}
static void launchCoarseAccumulateExact(int blocks,cudaStream_t stream,const UnifiedState* state,CoarseBlock* coarse){
    if(!g_exactZeroCoarse)k_CoarseAccumulate<<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse);
    else if(g_queryValidate)k_CoarseAccumulateExactZero<true><<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,d_coarseValidation);
    else k_CoarseAccumulateExactZero<false><<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,nullptr);
}
static void launchCoarseBoundExact(int blocks,cudaStream_t stream,const UnifiedState* state,CoarseBlock* coarse){
    if(!g_exactZeroCoarse)k_CoarseBound<<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse);
    else if(g_queryValidate)k_CoarseBoundExactZero<true><<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,d_coarseValidation);
    else k_CoarseBoundExactZero<false><<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,nullptr);
}
