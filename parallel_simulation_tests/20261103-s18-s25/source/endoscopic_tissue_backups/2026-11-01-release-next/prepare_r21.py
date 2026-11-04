"""R21: batch the unchanged non-volumetric fragment prelude and pass into one graph."""
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
OLD = ROOT/'endoscopic_tissue_backups/2026-10-31-interaction-fixes/package'
PACKAGE = P/'package'
VARIANT = 'release-r21-fragment-submission'


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Patch must match exactly once: '+old[:120])
    return text.replace(old, new)


HELPERS = r'''
// R21: all original initializations/copies and contact passes in their original
// order. The existing finishFeasible remains an unconditional separate launch.
static cudaError_t launchFragmentPreparation(bool prepare,cudaStream_t stream){
    cudaMemsetAsync(d_fragmentLambda,0,6*g_N*sizeof(float),stream);
    cudaMemsetAsync(d_fragmentQueryCount,0,sizeof(int),stream);
    k_FragmentQueries<<<Gr(3*g_surfaceCount+g_N),64,0,stream>>>(g_surfaceCount,g_N,d_unifiedComponents,g_liveActive,d_surfaceVertices,
        d_fragmentSupported,d_graspComponents,d_invMass,d_fragmentQueries,d_fragmentQueryCount,1);
#ifdef ENDO_FRAGMENT_TRACE
    k_FragmentTrace<<<Gr(g_N),64,0,stream>>>(prepare?0:2,g_N,d_unifiedComponents,d_fragmentSupported,d_graspComponents,d_invMass,d_pos,d_prevPos,d_unifiedTarget,d_vel);
#endif
    if(prepare)cudaMemcpyAsync(d_fragmentBefore,d_pos,g_N*sizeof(float3),cudaMemcpyDeviceToDevice,stream);
    else k_UnsupportedFlight<<<Gr(g_N),64,0,stream>>>(g_N,d_fragmentSupported,d_unifiedComponents,d_graspComponents,d_invMass,d_unifiedTarget,d_pos);
    cudaMemcpyAsync(d_fragmentStart,d_pos,g_N*sizeof(float3),cudaMemcpyDeviceToDevice,stream);
    cudaMemsetAsync(d_surfaceDelta,0,g_N*sizeof(float3),stream);
    cudaMemsetAsync(d_fragmentDegree,0,g_N*sizeof(int),stream);
    beginFeasible(stream);
    return cudaGetLastError();
}
static cudaError_t captureFragmentCorrection(cudaStream_t stream,bool prepare,const float3* previous){
    cudaStreamCaptureStatus status;cudaGraph_t graph=nullptr;const cudaGraphNode_t* deps=nullptr;size_t count=0;
    cudaError_t e=cudaStreamGetCaptureInfo_v2(stream,&status,nullptr,&graph,&deps,&count);if(e!=cudaSuccess)return e;
    if(status!=cudaStreamCaptureStatusActive)return cudaErrorStreamCaptureUnsupported;
    cudaGraphConditionalHandle handle=0;e=cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault);if(e!=cudaSuccess)return e;
    const int* queries=d_fragmentQueryCount;void* args[]={&handle,&queries};
    cudaKernelNodeParams p={};p.func=(void*)k_FragmentPassCondition;p.gridDim=p.blockDim=dim3(1);p.kernelParams=args;
    cudaGraphNode_t condition=nullptr;e=cudaGraphAddKernelNode(&condition,graph,deps,count,&p);if(e!=cudaSuccess)return e;
    cudaGraphNodeParams branch={cudaGraphNodeTypeConditional};branch.conditional.handle=handle;branch.conditional.type=cudaGraphCondTypeIf;branch.conditional.size=1;
    cudaGraphNode_t node=nullptr;e=cudaGraphAddNode(&node,graph,&condition,1,&branch);if(e!=cudaSuccess)return e;
    e=cudaStreamBeginCaptureToGraph(g_fragmentRefreshStream,branch.conditional.phGraph_out[0],nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal);if(e!=cudaSuccess)return e;
    cudaError_t work=launchFragmentPasses(g_fragmentRefreshStream,prepare,previous);
    cudaGraph_t body=nullptr;e=cudaStreamEndCapture(g_fragmentRefreshStream,&body);
    if(work!=cudaSuccess)return work;if(e!=cudaSuccess)return e;
    return cudaStreamUpdateCaptureDependencies(stream,&node,1,cudaStreamSetCaptureDependencies);
}
static cudaError_t runFragmentPreparation(bool prepare){
    const float3* previous=prepare?d_fragmentBefore:d_prevPos;
    if(g_sectionProfile || !g_fragmentSubmissionEnabled){
        cudaError_t e=launchFragmentPreparation(prepare,0);if(e!=cudaSuccess)return e;
        return runFragmentPasses(prepare,previous);
    }
    FragmentSubmissionKey key{};key.pass=makeFragmentPassKey(previous);
    key.feasibleStart=d_feasibleStart;key.copyFeasible=d_feasibleStart && g_volumePasses>0;
    key.liveActive=g_liveActive;
    const int slot=prepare?1:0;cudaGraphExec_t& exec=g_fragmentSubmissionGraph[slot];
    if(!exec || memcmp(&key,&g_fragmentSubmissionKey[slot],sizeof(key))){
        cudaError_t e=cudaSuccess;if(!g_captureStream)e=cudaStreamCreate(&g_captureStream);if(e!=cudaSuccess)return e;
        e=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(e!=cudaSuccess)return e;
        cudaError_t work=launchFragmentPreparation(prepare,g_captureStream);
        if(work==cudaSuccess)work=captureFragmentCorrection(g_captureStream,prepare,previous);
        cudaGraph_t graph=nullptr;e=cudaStreamEndCapture(g_captureStream,&graph);
        if(work!=cudaSuccess){if(graph)cudaGraphDestroy(graph);return work;}
        if(e!=cudaSuccess){if(graph)cudaGraphDestroy(graph);return e;}
        if(exec)cudaGraphExecDestroy(exec);exec=nullptr;
        e=cudaGraphInstantiate(&exec,graph,0);cudaGraphDestroy(graph);if(e!=cudaSuccess)return e;
        g_fragmentSubmissionKey[slot]=key;
    }
    return cudaGraphLaunch(exec,0);
}
'''


def main():
    manifest = json.loads((OLD/'release-r19-volume-gate.json').read_text())
    manifest['variant'] = VARIANT
    manifest['status'] = 'isolated experiment; unchanged arithmetic and volume protection, pending complete equivalence, timing and interaction review; never default'
    manifest['parent_variant'] = 'release-r19-volume-gate'
    target_root = PACKAGE/'overrides'/VARIANT
    if target_root.exists():
        raise RuntimeError('Refuse to overwrite prepared candidate source')
    for item in manifest['files']:
        src = OLD/'overrides'/item.get('source_variant', 'release-r19-volume-gate')/item['path']
        if hashlib.sha256(src.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Parent hash mismatch: '+str(src))
        dest = target_root/item['path']
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dest)
        item.pop('source_variant',None)
    path = target_root/'vendor/cuda/xpbd/xpbd_solver.cu'
    text = path.read_text()
    text = once(text, 'static void beginFeasible(){', 'static void beginFeasible(cudaStream_t stream=0){')
    text = once(text,
        'cudaMemcpyAsync(d_feasibleStart,d_pos,(size_t)g_N*sizeof(float3),cudaMemcpyDeviceToDevice);',
        'cudaMemcpyAsync(d_feasibleStart,d_pos,(size_t)g_N*sizeof(float3),cudaMemcpyDeviceToDevice,stream);')
    text = once(text, 'static FragmentPassKey g_fragmentPassKey[3]={};', '''static FragmentPassKey g_fragmentPassKey[3]={};
struct FragmentSubmissionKey{FragmentPassKey pass;const void* feasibleStart;const void* liveActive;bool copyFeasible;};
static cudaGraphExec_t g_fragmentSubmissionGraph[2]={};
static FragmentSubmissionKey g_fragmentSubmissionKey[2]={};
static bool g_fragmentSubmissionEnabled=true;''')
    text = once(text, 'static void destroyFragmentPassGraphs(){', '''static void destroyFragmentPassGraphs(){
    for(int k=0;k<2;k++){if(g_fragmentSubmissionGraph[k])cudaGraphExecDestroy(g_fragmentSubmissionGraph[k]);g_fragmentSubmissionGraph[k]=nullptr;}''')
    start = text.index('static cudaError_t solveFragmentContact(bool prepare=false){')
    end = text.index('\n    e=finishFeasible();', start)
    old = text[start:end]
    text = text[:start]+HELPERS+'''static cudaError_t solveFragmentContact(bool prepare=false){
    if(!d_unifiedComponents || g_surfaceCount<=0)return cudaSuccess;
    cudaError_t e=allocateFragmentContact();if(e!=cudaSuccess)return e;
    e=runFragmentPreparation(prepare);if(e!=cudaSuccess)return e;'''+text[end:]
    text += '''
// Test-only host dispatch switch. The material/contact kernels are identical.
extern "C" __declspec(dllexport) void LCS_SetFragmentSubmissionGraph(int enabled){
    g_fragmentSubmissionEnabled=enabled!=0;
}
'''
    path.write_text(text,encoding='utf8',newline='\n')
    (P/'replaced-r19-fragment-prelude.txt').write_text(old,encoding='utf8')
    for item in manifest['files']:
        data=(target_root/item['path']).read_bytes()
        item['sha256']=hashlib.sha256(data).hexdigest();item['bytes']=len(data)
    (PACKAGE/(VARIANT+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    source=ROOT/'endoscopic_tissue_backups/2026-11-01-training-completion/package'
    shutil.copyfile(source/'CMakeLists.txt',PACKAGE/'CMakeLists.txt')
    builder=(source/'build_experiment.py').read_text().replace('release-r20-support-cache',VARIANT)
    (PACKAGE/'build_experiment.py').write_text(builder,encoding='utf8',newline='\n')
    (PACKAGE/'.gitattributes').write_text('* -text whitespace=cr-at-eol\noverrides/** -whitespace\n')
    print(json.dumps({'variant':VARIANT,'parent':manifest['parent_variant'],
                      'modified_parent_files':['vendor/cuda/xpbd/xpbd_solver.cu'],
                      'default_replaced':False,'physics_budget_changed':False},indent=2))


if __name__=='__main__': main()
