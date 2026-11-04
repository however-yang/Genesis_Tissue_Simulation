from pathlib import Path
import json
p=Path('/srv/disk2/hryang/genesis-exec/parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104')
root=p/'native/source';path=root/'vendor/cuda/contact/coarse_bound_cache.cuh';s=path.read_text()
a=s.index('template<int N,class Add> __device__ void blockAccumulateCache');b=s.index('__device__ inline CoarseBlock* coarseTetTermsCache',a)
helper=s[a:b].replace('blockAccumulateCache','blockAccumulateOccupancy').replace('coarseAccumulateBlockCache','coarseAccumulateBlockOccupancy')
a=s.index('__global__ void k_CoarseAccumulateBoundCache(');b=s.index('__global__ void k_CoarseBoundFromCache(',a)
kernel=s[a:b].replace('__global__ void k_CoarseAccumulateBoundCache(', 'template<int MinimumBlocks> __global__ __launch_bounds__(256,MinimumBlocks) void k_CoarseOccupancyBoundCache(').replace('coarseAccumulateBlockCache','coarseAccumulateBlockOccupancy')
(root/'vendor/cuda/contact/coarse_occupancy.cuh').write_text('// Original fused material/contact kernel and reduction; different compiler register budget only.\n'+helper+kernel)
needle='static CoarseBoundTerm* coarseCacheForStream(cudaStream_t stream){';assert s.count(needle)==1
s=s.replace(needle,'#include "coarse_occupancy.cuh"\n\n'+needle)
needle='''        k_CoarseAccumulateBoundCache<<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,terms,g_queryValidate?d_coarseActualValues[k]:nullptr,g_queryValidate?d_coarseActualTargets[k]:nullptr);''';assert s.count(needle)==1
args='(state,coarse,terms,g_queryValidate?d_coarseActualValues[k]:nullptr,g_queryValidate?d_coarseActualTargets[k]:nullptr);'
s=s.replace(needle,'        if(g_runtimeCoarseBlocks==3)k_CoarseOccupancyBoundCache<3><<<blocks,ACCUMULATE_THREADS,0,stream>>>'+args+'\n        else if(g_runtimeCoarseBlocks==4)k_CoarseOccupancyBoundCache<4><<<blocks,ACCUMULATE_THREADS,0,stream>>>'+args+'\n        else '+needle.strip())
path.write_text(s)
path=root/'vendor/cuda/xpbd/xpbd_solver.cu';s=path.read_text()
s=s.replace('static int g_runtimeCoreMode=0;','static int g_runtimeCoreMode=0;\nstatic int g_runtimeCoarseBlocks=0;')
s=s.replace('if(mode&~256)return -10001;','if(mode&~(256|2048|8192|16384))return -10001;\n    if((mode&8192) && (mode&16384))return -10001;')
s=s.replace('g_runtimeCoreMode=mode;return 0;','g_runtimeCoreMode=mode;g_runtimeCoarseBlocks=(mode&8192)?3:(mode&16384)?4:0;return 0;')
needle='static cudaError_t observeGrip(){';assert s.count(needle)==1
s=s.replace(needle,'#include "../grasp/runtime_grip_feedback.cuh"\n'+needle)
needle='''    if(g_embeddedGrip && g_grasping && g_gripCount>0){
        k_ReadGripObservation''';assert s.count(needle)==1
s=s.replace(needle,'''    if(g_embeddedGrip && g_grasping && g_gripCount>0){
        if(g_runtimeCoreMode&2048)return observeRuntimeGrip();
        k_ReadGripObservation''')
s=s.replace('    freeRuntimeObservation();','    freeRuntimeObservation();\n    freeRuntimeGripTerms();')
path.write_text(s)
print(json.dumps({'coarse_register_budget_modes':[8192,16384],'parallel_ordered_grip_readback':2048,'material_energy_changed':False,'iterations_changed':False,'precision_reduced':False}))
