// Prove a whole-component correction is disabled by the ORIGINAL application
// guard. These hints never suppress a free, unsupported or unheld component.
static int* d_runtimeFixedRoots=nullptr;
static int* d_runtimeModeQueries=nullptr;
static float3* d_runtimeModeBefore=nullptr;
static unsigned long long* d_runtimeModeProof=nullptr;
static int g_runtimeModeCapacity=0;
static int g_runtimeModeGraphKey[3]={-1,-1,-1};
static unsigned long long g_runtimeModeHostStats[8]={};
static cudaError_t ensureRuntimeModeGates(){
    cudaError_t error=cudaSuccess;
    if(g_runtimeModeCapacity<g_N){
        queryPrivateFree(d_runtimeFixedRoots);d_runtimeFixedRoots=nullptr;g_runtimeModeCapacity=0;
        error=queryPrivateAlloc(&d_runtimeFixedRoots,(size_t)g_N*sizeof(int));
        if(error==cudaSuccess)g_runtimeModeCapacity=g_N;
    }
    if(error==cudaSuccess && !d_runtimeModeQueries)error=queryPrivateAlloc(&d_runtimeModeQueries,5*sizeof(int));
    if(error==cudaSuccess && g_queryValidate && !d_runtimeModeBefore)error=queryPrivateAlloc(&d_runtimeModeBefore,(size_t)g_N*sizeof(float3));
    if(error==cudaSuccess && g_queryValidate && !d_runtimeModeProof){
        error=queryPrivateAlloc(&d_runtimeModeProof,8*sizeof(unsigned long long));
        if(error==cudaSuccess)error=cudaMemset(d_runtimeModeProof,0,8*sizeof(unsigned long long));
    }
    return error;
}
static void freeRuntimeModeGates(){
    queryPrivateFree(d_runtimeFixedRoots);queryPrivateFree(d_runtimeModeQueries);
    queryPrivateFree(d_runtimeModeBefore);queryPrivateFree(d_runtimeModeProof);
    d_runtimeFixedRoots=d_runtimeModeQueries=nullptr;d_runtimeModeBefore=nullptr;d_runtimeModeProof=nullptr;
    g_runtimeModeCapacity=0;for(int& key:g_runtimeModeGraphKey)key=-1;
}
__global__ void k_RuntimeFixedRoots(const UnifiedState* state,int* roots){
    const auto& s=*state;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<max(s.activeCount,s.grips);k+=gridDim.x*blockDim.x){
        int i=unifiedActiveNode(s,k);
        if(i>=0){
            int root=s.component[i];
            if(root>=0 && root<s.n){
                // Match k_RigidCenters' actual degree condition.
                int flags=s.grips>0 && s.gripDegree[i]>0?1:0;
                if((!s.active || s.active[i]) && (s.invMass[i]<=0 || (s.attach && s.attach[i]>0)))flags|=2;
                if(flags)atomicOr(roots+root,flags);
            }
        }
        // The translation guard is set by valid points themselves, not by a
        // possibly stale degree. This matches coarseContactGripTerms exactly.
        if(k<s.grips && s.points[k].valid && s.points[k].count>0){
            int root=s.component[s.points[k].id[0]];
            if(root>=0 && root<s.n)atomicOr(roots+root,4);
        }
    }
}
__global__ void k_RuntimeNeededModes(const UnifiedState* state,const int* roots,int* queries){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0 || s.invMass[i]<=0)return;
    int root=s.component[i];int flags=root>=0 && root<s.n?roots[root]:0;
    // Rigid rotations can remain physical around an isolated anchor. Only
    // an existing persistent grip proves that original rotation is blocked.
    if(!(flags&1))atomicExch(queries+3,1);
    // A grip, pin, or ligament proves the final translational null mode's
    // original external flag is set before any position can be written.
    if(!(flags&6))atomicExch(queries+4,1);
}
static cudaError_t launchRuntimeModeQueries(cudaStream_t stream){
    cudaError_t error=cudaMemsetAsync(d_runtimeFixedRoots,0,(size_t)g_N*sizeof(int),stream);
    if(error==cudaSuccess)error=cudaMemsetAsync(d_runtimeModeQueries+3,0,2*sizeof(int),stream);
    if(error!=cudaSuccess)return error;
    k_RuntimeFixedRoots<<<Gr(g_activeNodes),64,0,stream>>>(d_unified,d_runtimeFixedRoots);
    k_RuntimeNeededModes<<<Gr(g_activeNodes),64,0,stream>>>(d_unified,d_runtimeFixedRoots,d_runtimeModeQueries);
    // Join the original three decisions into the SAME existing CPU wait.
    return cudaMemcpyAsync(d_runtimeModeQueries,d_substepQuery,3*sizeof(int),cudaMemcpyDeviceToDevice,stream);
}
__global__ void k_RuntimeProveNoPositionWrite(int count,const float3* before,const float3* after,
    unsigned long long* stats,int slot){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    if(i==0){atomicAdd(stats+slot,1ull);atomicAdd(stats+slot+2,(unsigned long long)count*3);}
    unsigned bad=(__float_as_uint(before[i].x)!=__float_as_uint(after[i].x))+
        (__float_as_uint(before[i].y)!=__float_as_uint(after[i].y))+
        (__float_as_uint(before[i].z)!=__float_as_uint(after[i].z));
    if(bad)atomicAdd(stats+slot+1,(unsigned long long)bad);
}
extern "C" LCS_COMBINED_EXPORT int LCS_ReadRuntimeModeGateStats(unsigned long long* host,unsigned long long* proof,int capacity){
    if(!host||!proof||capacity<8)return -1;
    for(int i=0;i<8;i++){host[i]=g_runtimeModeHostStats[i];proof[i]=0;}
    if(!d_runtimeModeProof)return 0;
    return -(int)cudaMemcpy(proof,d_runtimeModeProof,8*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
}
