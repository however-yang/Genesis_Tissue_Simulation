from pathlib import Path
import shutil
p=Path(__file__).resolve().parents[1];src=p/'native-coarse-v5/source';shutil.copytree(p/'native-tool-v2/source',src)
f=src/'vendor/cuda/contact/coarse_bound_cache.cuh';s=f.read_text()
a=s.index('template<int N,class Add> __device__ void blockAccumulateCache(');b=s.index('__device__ inline void coarseAccumulateBlockCache',a)
fallback=s[a:b].replace('blockAccumulateCache(', 'blockAccumulateUniformFallback(',1)
helper='''// The common one-component CTA has a unique target regardless of atomicCAS
// winner. Discover that SAME target with one ballot/shuffle per warp instead
// of 256 contending 64-bit shared atomics. Mixed CTAs use the original reduction.
static unsigned long long* d_coarseUniformStats=nullptr;
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
'''
s=s[:b]+fallback+helper+s[b:]
a=s.index('__global__ void k_CoarseAccumulateBoundCache(');b=s.index('__global__ void k_CoarseBoundFromCache(',a);kernel=s[a:b]
kernel=kernel.replace('k_CoarseAccumulateBoundCache(', 'k_CoarseAccumulateUniform(',1).replace('float* actual,CoarseBlock** targets){','float* actual,CoarseBlock** targets,unsigned long long* stats){',1).replace('coarseAccumulateBlockCache(target,g,H);','coarseAccumulateBlockUniform(target,g,H,stats);',1)
s=s[:b]+kernel+s[b:]
needle='        k_CoarseAccumulateBoundCache<<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,terms,g_queryValidate?d_coarseActualValues[k]:nullptr,g_queryValidate?d_coarseActualTargets[k]:nullptr);';assert needle in s
new=needle.replace('k_CoarseAccumulateBoundCache','k_CoarseAccumulateUniform').replace('d_coarseActualTargets[k]:nullptr);','d_coarseActualTargets[k]:nullptr,g_queryValidate?d_coarseUniformStats:nullptr);')
s=s.replace(needle,'        if(g_coreOptimizations&16){\n'+new+'\n        }else{\n'+needle+'\n        }',1)
# Move stat declaration before ensure/free functions that use it.
s=s.replace('static unsigned long long* d_coarseUniformStats=nullptr;\n','',1).replace('static unsigned long long* d_coarseBoundStats=nullptr;','static unsigned long long* d_coarseBoundStats=nullptr;\nstatic unsigned long long* d_coarseUniformStats=nullptr;',1)
s=s.replace('static cudaError_t ensureCoarseBoundStats(){','static cudaError_t ensureCoarseBoundStats(){\n    if(g_coreOptimizations&16){if(!d_coarseUniformStats){cudaError_t e=queryPrivateAlloc(&d_coarseUniformStats,4*sizeof(unsigned long long));if(e!=cudaSuccess)return e;}cudaError_t e=cudaMemset(d_coarseUniformStats,0,4*sizeof(unsigned long long));if(e!=cudaSuccess)return e;}',1)
s=s.replace('static void freeCoarseBoundCache(){','static void freeCoarseBoundCache(){\n    cudaDeviceSynchronize();queryPrivateFree(d_coarseUniformStats);d_coarseUniformStats=nullptr;',1);f.write_text(s)
f=src/'vendor/cuda/xpbd/xpbd_solver.cu';s=f.read_text().replace('mode>7 || g_ready','mode>31 || g_ready');s+='\nextern "C" int xpbd_coarse_uniform_stats(unsigned long long* out){if(!out || !d_coarseUniformStats)return -10003;cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess)return -(int)e;return -(int)cudaMemcpy(out,d_coarseUniformStats,4*sizeof(unsigned long long),cudaMemcpyDeviceToHost);}\n';f.write_text(s)
f=src/'vendor/src/plugin_api.cpp';s=f.read_text();s+='\nextern "C" int xpbd_coarse_uniform_stats(unsigned long long*);\nLCS_API int LCS_CoarseUniformStats(unsigned long long* out){return xpbd_coarse_uniform_stats(out);}\n';f.write_text(s)
s=(p/'tools/build_tool_v2.py').read_text().replace('native-tool-v2','native-coarse-v5').replace('build-tool-v2','build-coarse-v5');(p/'tools/build_coarse_v5.py').write_text(s)
print(src)
