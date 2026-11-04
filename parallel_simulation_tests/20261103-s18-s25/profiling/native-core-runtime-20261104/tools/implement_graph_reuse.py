from pathlib import Path
import difflib
import json

p=Path('/srv/disk2/hryang/genesis-exec/parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104')
root=p/'native/source'
file=root/'vendor/cuda/xpbd/xpbd_solver.cu'
s=file.read_text()
needle='static int g_interactionStats[16]={};'
assert s.count(needle)==1
helper='''// Host-only graph lifecycle optimization. The physics state and every device
// kernel are unchanged. It is opt-in, fixed before initialization, and falls
// back to the original rebuild on CUDA's documented update rejection.
static int g_runtimeCoreMode=0;
static unsigned long long g_runtimeGraphStats[8]={};
extern "C" LCS_COMBINED_EXPORT int LCS_SetRuntimeCoreMode(int mode){
    if(mode&~256)return -10001;
    g_runtimeCoreMode=mode;return 0;
}
extern "C" LCS_COMBINED_EXPORT int LCS_ReadRuntimeGraphStats(unsigned long long* out,int capacity){
    if(!out||capacity<8)return -1;
    for(int i=0;i<8;i++)out[i]=g_runtimeGraphStats[i];return 8;
}
static cudaError_t installRuntimeGraph(cudaGraphExec_t& executable,cudaGraph_t graph){
    if(executable && (g_runtimeCoreMode&256)){
        g_runtimeGraphStats[0]++;
        cudaGraphExecUpdateResultInfo info={};
        cudaError_t error=cudaGraphExecUpdate(executable,graph,&info);
        if(error==cudaSuccess && info.result==cudaGraphExecUpdateSuccess){g_runtimeGraphStats[1]++;return cudaSuccess;}
        if(error!=cudaSuccess && error!=cudaErrorGraphExecUpdateFailure)return error;
        g_runtimeGraphStats[2]++;g_runtimeGraphStats[5]=(unsigned long long)info.result;
        cudaGetLastError();
    }
    if(executable){
        cudaError_t error=cudaGraphExecDestroy(executable);if(error!=cudaSuccess)return error;
        executable=nullptr;g_runtimeGraphStats[4]++;
    }
    cudaError_t error=cudaGraphInstantiate(&executable,graph,0);
    if(error==cudaSuccess)g_runtimeGraphStats[3]++;
    return error;
}
'''
s=s.replace(needle,helper+'\n'+needle)
needle='''        if(exec)cudaGraphExecDestroy(exec);exec=nullptr;
        if(e==cudaSuccess)e=cudaGraphInstantiate(&exec,graph,0);'''
assert s.count(needle)==1
s=s.replace(needle,'''        if(e==cudaSuccess)e=installRuntimeGraph(exec,graph);''')
for name in ['g_predictedFragmentGraph','g_materialPreludeGraph']:
    needle=f'if({name})cudaGraphExecDestroy({name});{name}=nullptr;\n'
    pos=s.index(needle,s.index('static cudaError_t markPredictedFragmentContact' if name=='g_predictedFragmentGraph' else 'if(!g_materialPreludeGraph ||'))
    tail=f'e=cudaGraphInstantiate(&{name},graph,0);'
    end=s.index(tail,pos)+len(tail)
    s=s[:pos]+f'e=installRuntimeGraph({name},graph);'+s[end:]
file.write_text(s)
file=root/'vendor/src/plugin_api.cpp'
data=file.read_bytes();needle=b'LCS_API int LCS_SetCoreOptimizations(int mode){return mode==0?0:-10001;}'
assert data.count(needle)==1
data=data.replace(needle,b'extern "C" int LCS_SetRuntimeCoreMode(int mode);\nLCS_API int LCS_SetCoreOptimizations(int mode){return LCS_SetRuntimeCoreMode(mode);}')
file.write_bytes(data)
baseline=Path(json.loads((p/'baseline.json').read_text())['baseline_experiment'])/'native-readonly-v8/source'
patch=''
for relative in ['vendor/cuda/xpbd/xpbd_solver.cu','vendor/src/plugin_api.cpp']:
    patch+=''.join(difflib.unified_diff((baseline/relative).read_text().splitlines(True),(root/relative).read_text().splitlines(True),fromfile='a/'+relative,tofile='b/'+relative))
(p/'delivery/graph-reuse.patch').write_text(patch)
print('Applied opt-in host graph reuse; no CUDA kernel source changes')
