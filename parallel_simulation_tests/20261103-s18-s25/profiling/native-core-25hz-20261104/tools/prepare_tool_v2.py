from pathlib import Path
import shutil
p=Path('/srv/disk2/hryang/genesis-exec/parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104')
src=p/'native-tool-v2/source';shutil.copytree(p/'native-lean-v1/source',src)
f=src/'vendor/cuda/contact/surface_contact.cuh';s=f.read_text();start=s.index('__global__ void k_FindSurfaceLean(');end=s.index('static_assert(sizeof(SurfaceContact)',start);k=s[start:end]
k=k.replace('k_FindSurfaceLean(', 'k_FindSurfaceLeanRefs(',1)
needle='    SurfaceVertex v[3]={vertices[3*t],vertices[3*t+1],vertices[3*t+2]};';assert k.count(needle)==1;k=k.replace(needle,'',1)
startloop=k.index('    for(int k=0;k<3;k++){\n        out.owner[k]=v[k].voxel;');front=k[:startloop];tail=k[startloop:]
tail=tail.replace('for(int k=0;k<3;k++){','for(int k=0;k<3;k++){\n        const SurfaceVertex& v=vertices[3*t+k];',1).replace('v[k].','v.');k=front+tail;s=s[:end]+k+s[end:]
needle='        if(g_coreOptimizations&1){';assert needle in s
args='<<<dim3((triangles+g_queryThreads-1)/g_queryThreads,kindCount),g_queryThreads,0,stream>>>(triangles,packed,bladeBatch,from,to,skin,\n            vertices,masks,prev,pos,contacts,degree,current,previous,body,thickness,minGap,stationary,cacheSlots,cache,bladePhase,stride,groupBase,w.ids,w.counts);'
s=s.replace(needle,'        if(g_coreOptimizations&4){\n            k_FindSurfaceLeanRefs'+args+'\n        }else if(g_coreOptimizations&1){',1);f.write_text(s)
unified=(src/'vendor/cuda/solver/material/unified_vertex.cuh').read_text();a=unified.index('__global__ void k_UnifiedTool(');b=unified.index('__global__ void k_UnifiedTargets(',a);tool=unified[a:b]
tool=tool.replace('k_UnifiedTool(UnifiedState* state)','k_UnifiedToolCached(UnifiedState* state,const ToolContactTerm* terms)',1)
needle='SurfaceContact& c=s.contacts[s.contactIds[k]];float3 force;Symmetric3 H={};unifiedContact(s,c,force,H);';assert tool.count(needle)==1
tool=tool.replace(needle,'SurfaceContact& c=s.contacts[s.contactIds[k]];float3 force;Symmetric3 H={};\n        if(k<TOOL_CONTACT_CAPACITY){force=terms[k].force;H=terms[k].H;}else unifiedContact(s,c,force,H);',1)
header='''// Per-call contact laws evaluated over all SMs. The final 32-lane solve retains
// the original k=lane, lane+32,... summation and four-DOF Cholesky/guide solve.
// Scratch is derived, overwritten before every use, and excluded from snapshots.
constexpr int TOOL_CONTACT_CAPACITY=65536;
struct ToolContactTerm {float3 force;Symmetric3 H;};
struct ToolContactWorkspace {ToolContactTerm* terms=nullptr;CoupledTool *before=nullptr,*reference=nullptr;};
static ToolContactWorkspace g_toolContactWorkspace[2];
static unsigned long long* d_toolContactStats=nullptr;
static cudaStream_t g_toolContactStream=nullptr;
static cudaError_t ensureToolContactCache(){
    if(!(g_coreOptimizations&2))return cudaSuccess;
    for(auto& w:g_toolContactWorkspace)if(!w.terms){cudaError_t e=queryPrivateAlloc(&w.terms,TOOL_CONTACT_CAPACITY*sizeof(ToolContactTerm));if(e!=cudaSuccess)return e;}
    return cudaSuccess;
}
static cudaError_t ensureToolContactValidation(){
    if(!(g_coreOptimizations&2))return cudaSuccess;
    if(!d_toolContactStats){cudaError_t e=queryPrivateAlloc(&d_toolContactStats,5*sizeof(unsigned long long));if(e!=cudaSuccess)return e;}
    for(auto& w:g_toolContactWorkspace){
        if(!w.before){cudaError_t e=queryPrivateAlloc(&w.before,sizeof(CoupledTool));if(e!=cudaSuccess)return e;}
        if(!w.reference){cudaError_t e=queryPrivateAlloc(&w.reference,sizeof(CoupledTool));if(e!=cudaSuccess)return e;}
    }
    return cudaMemset(d_toolContactStats,0,5*sizeof(unsigned long long));
}
static void freeToolContactCache(){
    cudaDeviceSynchronize();for(auto& w:g_toolContactWorkspace){queryPrivateFree(w.terms);queryPrivateFree(w.before);queryPrivateFree(w.reference);w={};}
    queryPrivateFree(d_toolContactStats);d_toolContactStats=nullptr;g_toolContactStream=nullptr;
}
__global__ void k_EvaluateToolContacts(const UnifiedState* state,ToolContactTerm* terms){
    const UnifiedState& s=*state;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<min(*s.contactCount,TOOL_CONTACT_CAPACITY);k+=gridDim.x*blockDim.x){
        const SurfaceContact& c=s.contacts[s.contactIds[k]];float3 force;Symmetric3 H={};unifiedContact(s,c,force,H);
        terms[k].force=force;terms[k].H=H;
    }
}
__global__ void k_CompareToolContactTerms(const UnifiedState* state,const ToolContactTerm* terms,unsigned long long* stats){
    const UnifiedState& s=*state;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*s.contactCount;k+=gridDim.x*blockDim.x){
        if(k>=TOOL_CONTACT_CAPACITY){atomicAdd(stats+4,1ull);continue;}
        const SurfaceContact& c=s.contacts[s.contactIds[k]];float3 force;Symmetric3 H={};unifiedContact(s,c,force,H);
        ToolContactTerm ref={force,H};const unsigned *a=reinterpret_cast<const unsigned*>(&ref),*b=reinterpret_cast<const unsigned*>(terms+k);bool diff=false;
        for(int j=0;j<sizeof(ToolContactTerm)/sizeof(unsigned);j++)diff|=a[j]!=b[j];
        atomicAdd(stats+1,1ull);if(diff)atomicAdd(stats+2,1ull);
    }
}
__global__ void k_CopyCoupledTool(UnifiedState* state,CoupledTool* copy,bool restore){
    if(threadIdx.x||blockIdx.x)return;if(restore)*state->tool=*copy;else *copy=*state->tool;
}
__global__ void k_CompareCoupledTool(const UnifiedState* state,const CoupledTool* reference,unsigned long long* stats){
    if(threadIdx.x||blockIdx.x)return;static_assert(sizeof(CoupledTool)%sizeof(unsigned)==0,"word comparison");
    const unsigned* a=reinterpret_cast<const unsigned*>(state->tool),*b=reinterpret_cast<const unsigned*>(reference);bool diff=false;
    for(int j=0;j<sizeof(CoupledTool)/sizeof(unsigned);j++)diff|=a[j]!=b[j];
    atomicAdd(stats,1ull);if(diff)atomicAdd(stats+3,1ull);
}
'''
footer='''static void launchUnifiedToolCached(UnifiedState* state,cudaStream_t stream){
    int k=stream?1:0;
    if(stream){if(!g_toolContactStream)g_toolContactStream=stream;else if(g_toolContactStream!=stream)k=-1;}
    if(!(g_coreOptimizations&2) || k<0 || !g_toolContactWorkspace[k].terms){k_UnifiedTool<<<1,32,0,stream>>>(state);return;}
    auto& w=g_toolContactWorkspace[k];
    if(g_queryValidate){
        k_CopyCoupledTool<<<1,1,0,stream>>>(state,w.before,false);
        k_UnifiedTool<<<1,32,0,stream>>>(state);
        k_CopyCoupledTool<<<1,1,0,stream>>>(state,w.reference,false);
        k_CopyCoupledTool<<<1,1,0,stream>>>(state,w.before,true);
    }
    k_EvaluateToolContacts<<<128,64,0,stream>>>(state,w.terms);
    if(g_queryValidate)k_CompareToolContactTerms<<<128,64,0,stream>>>(state,w.terms,d_toolContactStats);
    k_UnifiedToolCached<<<1,32,0,stream>>>(state,w.terms);
    if(g_queryValidate)k_CompareCoupledTool<<<1,1,0,stream>>>(state,w.reference,d_toolContactStats);
}
'''
(src/'vendor/cuda/contact/tool_contact_cache.cuh').write_text(header+tool+footer)
f=src/'vendor/cuda/xpbd/xpbd_solver.cu';s=f.read_text();s=s.replace('#include "../contact/coarse_bound_cache.cuh"','#include "../contact/coarse_bound_cache.cuh"\n#include "../contact/tool_contact_cache.cuh"',1)
for state,stream in [('d_unified','g_captureStream'),('d_bladeState','g_captureStream')]:s=s.replace('k_UnifiedTool<<<1,32,0,'+stream+'>>>('+state+');','launchUnifiedToolCached('+state+','+stream+');')
s=s.replace('k_UnifiedTool<<<1,32>>>(d_bladeState);','launchUnifiedToolCached(d_bladeState,0);')
s=s.replace('cudaError_t e=ensureCoarseBoundCache(g_T);if(e!=cudaSuccess)return -(int)e;','cudaError_t e=ensureCoarseBoundCache(g_T);if(e!=cudaSuccess)return -(int)e;\n    e=ensureToolContactCache();if(e!=cudaSuccess)return -(int)e;',1)
s=s.replace('freeCoarseBoundCache();freeManifoldValidation();','freeToolContactCache();freeCoarseBoundCache();freeManifoldValidation();',1)
s=s.replace('e=allocateManifoldValidation(g_N);if(e!=cudaSuccess)return -(int)e;','e=allocateManifoldValidation(g_N);if(e!=cudaSuccess)return -(int)e;\n        e=ensureToolContactValidation();if(e!=cudaSuccess)return -(int)e;',1)
s=s.replace('if(mode<0 || mode>1 || g_ready)return -10001;g_coreOptimizations=mode;', 'if(mode<0 || mode>7 || g_ready)return -10001;g_coreOptimizations=mode;',1)
s+='\nextern "C" int xpbd_tool_contact_stats(unsigned long long* out){if(!out || !d_toolContactStats)return -10003;cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess)return -(int)e;return -(int)cudaMemcpy(out,d_toolContactStats,5*sizeof(unsigned long long),cudaMemcpyDeviceToHost);}\n';f.write_text(s)
f=src/'vendor/src/plugin_api.cpp';s=f.read_text();s+='\nextern "C" int xpbd_tool_contact_stats(unsigned long long*);\nLCS_API int LCS_ToolContactStats(unsigned long long* out){return xpbd_tool_contact_stats(out);}\n';f.write_text(s)
s=(p/'tools/build_lean_v1.py').read_text().replace('native-lean-v1','native-tool-v2').replace('build-lean-v1','build-tool-v2');(p/'tools/build_tool_v2.py').write_text(s)
f=p/'tools/profile_all_queries.py';s=f.read_text();needle="     result['native_validation']=dict(";assert needle in s
s=s.replace(needle,"     if a.core_optimizations&2:\n      tool_stats=(ctypes.c_ulonglong*5)();native_control.LCS_ToolContactStats.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)];native_control.LCS_ToolContactStats.restype=ctypes.c_int;assert native_control.LCS_ToolContactStats(tool_stats)==0\n      result['tool_validation']=dict(solves=tool_stats[0],contact_terms=tool_stats[1],term_differences=tool_stats[2],body_differences=tool_stats[3],uncached_terms=tool_stats[4],passed=tool_stats[0]>0 and tool_stats[1]>0 and sum(tool_stats[2:])==0);assert result['tool_validation']['passed'],result['tool_validation']\n"+needle,1)
f.write_text(s)
print(src)
