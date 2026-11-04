from pathlib import Path
import shutil
p=Path(__file__).resolve().parents[1];src=p/'native-material-v6/source';shutil.copytree(p/'native-tool-v2/source',src)
f=src/'vendor/cuda/contact/coarse_bound_cache.cuh';s=f.read_text()
a=s.index('template<int N,class Add> __device__ void blockAccumulateCache(');b=s.index('__device__ inline CoarseBlock* coarseTetTermsCache(',a);helpers=s[a:b].replace('blockAccumulateCache','blockAccumulateGuard').replace('coarseAccumulateBlockCache','coarseAccumulateBlockGuard')
a=s.index('__device__ inline CoarseBlock* coarseTetTermsCache(');b=s.index('__device__ inline CoarseBlock* coarseCachedBound(',a);fn=s[a:b].replace('coarseTetTermsCache(', 'coarseTetTermsGuard(',1)
needle='float l=len3(e),l0=unifiedRestLength(s,t,j,k),strain=fmaxf(0.f,l/l0-1.f);';assert needle in fn
fn=fn.replace(needle,'float l=len3(e),l0=unifiedRestLength(s,t,j,k);\n        // For l<=l0, RN(l/l0)<=1 and the original positive-part strain is exactly zero.\n        // Keep the ORIGINAL division for every possibly tensile edge; no reciprocal approximation.\n        if(!(l>l0))continue;\n        float strain=fmaxf(0.f,l/l0-1.f);',1)
s=s[:b]+helpers+fn+s[b:]
a=s.index('__global__ void k_CoarseAccumulateBoundCache(');b=s.index('__global__ void k_CoarseBoundFromCache(',a);kernel=s[a:b].replace('k_CoarseAccumulateBoundCache(', 'k_CoarseAccumulateGuard(',1).replace('coarseTetTermsCache(', 'coarseTetTermsGuard(').replace('coarseAccumulateBlockCache(', 'coarseAccumulateBlockGuard(')
s=s[:b]+kernel+s[b:]
needle='        k_CoarseAccumulateBoundCache<<<blocks,ACCUMULATE_THREADS,0,stream>>>(state,coarse,terms,g_queryValidate?d_coarseActualValues[k]:nullptr,g_queryValidate?d_coarseActualTargets[k]:nullptr);';assert needle in s
s=s.replace(needle,'        if(g_coreOptimizations&16){\n'+needle.replace('k_CoarseAccumulateBoundCache','k_CoarseAccumulateGuard')+'\n        }else{\n'+needle+'\n        }',1);f.write_text(s)
f=src/'vendor/cuda/xpbd/xpbd_solver.cu';f.write_text(f.read_text().replace('mode>7 || g_ready','mode>31 || g_ready'))
s=(p/'tools/build_tool_v2.py').read_text().replace('native-tool-v2','native-material-v6').replace('build-tool-v2','build-material-v6');(p/'tools/build_material_v6.py').write_text(s)
print(src)
