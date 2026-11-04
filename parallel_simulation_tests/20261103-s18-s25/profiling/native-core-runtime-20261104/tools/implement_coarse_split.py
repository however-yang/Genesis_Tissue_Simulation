from pathlib import Path
import json

p=Path('/srv/disk2/hryang/genesis-exec/parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104')
root=p/'native/source'
source=root/'vendor/cuda/contact/coarse_bound_cache.cuh'
s=source.read_text()
start=s.index('template<int N,class Add> __device__ void blockAccumulateCache')
end=s.index('__device__ inline CoarseBlock* coarseTetTermsCache',start)
reduction=s[start:end].replace('blockAccumulateCache','blockAccumulateSplit').replace('coarseAccumulateBlockCache','coarseAccumulateBlockSplit')
start=s.index('__global__ void k_CoarseAccumulateBoundCache(')
end=s.index('__global__ void k_CoarseBoundFromCache(',start)
kernel=s[start:end].replace('k_CoarseAccumulateBoundCache','k_CoarseElasticSplit').replace('coarseAccumulateBlockCache','coarseAccumulateBlockSplit')
needle='    coarseContactGripTerms(s,blocks,t,gridDim.x*blockDim.x);\n';assert kernel.count(needle)==1
kernel=kernel.replace(needle,'')
contact='''__global__ void k_CoarseInteractionSplit(const UnifiedState* state,CoarseBlock* blocks){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    coarseContactGripTerms(s,blocks,t,gridDim.x*blockDim.x);
}
'''
(root/'vendor/cuda/contact/coarse_split.cuh').write_text('// Same material coefficients and interaction objective; separate register footprints.\n'+reduction+kernel+contact)
needle='static CoarseBoundTerm* coarseCacheForStream(cudaStream_t stream){';assert s.count(needle)==1
s=s.replace(needle,'#include "coarse_split.cuh"\n\n'+needle)
needle='''        k_CoarseAccumulateBoundCache<<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,terms,g_queryValidate?d_coarseActualValues[k]:nullptr,g_queryValidate?d_coarseActualTargets[k]:nullptr);'''
assert s.count(needle)==1
s=s.replace(needle,'''        if(g_coarseSplitMode){
            k_CoarseElasticSplit<<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,terms,g_queryValidate?d_coarseActualValues[k]:nullptr,g_queryValidate?d_coarseActualTargets[k]:nullptr);
            k_CoarseInteractionSplit<<<std::min(blocks,128),ACCUMULATE_THREADS,0,stream>>>(state,coarse);
        }else{\n'''+needle+'\n        }')
source.write_text(s)
source=root/'vendor/cuda/xpbd/xpbd_solver.cu';s=source.read_text()
s=s.replace('static int g_runtimeCoreMode=0;','static int g_runtimeCoreMode=0;\nstatic int g_coarseSplitMode=0;')
s=s.replace('if(mode&~(256|512))return -10001;','if(mode&~(256|512|1024))return -10001;')
s=s.replace('g_directQueryMode=(mode&512)!=0;return 0;','g_directQueryMode=(mode&512)!=0;g_coarseSplitMode=(mode&1024)!=0;return 0;')
source.write_text(s)
print(json.dumps({'mode':1024,'changed':'split coarse material and interaction kernels','material_formulas_changed':False,'iteration_counts_changed':False,'old_kernels_retained_for_comparison':True}))
