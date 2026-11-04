// Parallel geometry evaluation followed by the ORIGINAL ordered double sums.
// This is diagnostic readback, not the Gaussian grasp force/constraint solve.
struct RuntimeGripTerm {double weight;float3 local,point;float tangent;int valid;};
static RuntimeGripTerm* d_runtimeGripTerms=nullptr;
static float* d_runtimeGripReference=nullptr;
static int* d_runtimeGripStats=nullptr;
static int g_runtimeGripCapacity=0;
static cudaError_t ensureRuntimeGripTerms(int count){
    if(count<=g_runtimeGripCapacity)return cudaSuccess;
    queryPrivateFree(d_runtimeGripTerms);d_runtimeGripTerms=nullptr;g_runtimeGripCapacity=0;
    cudaError_t error=queryPrivateAlloc(&d_runtimeGripTerms,(size_t)count*sizeof(RuntimeGripTerm));
    if(error==cudaSuccess)g_runtimeGripCapacity=count;
    return error;
}
static void freeRuntimeGripTerms(){
    queryPrivateFree(d_runtimeGripTerms);queryPrivateFree(d_runtimeGripReference);queryPrivateFree(d_runtimeGripStats);
    d_runtimeGripTerms=nullptr;d_runtimeGripReference=nullptr;d_runtimeGripStats=nullptr;g_runtimeGripCapacity=0;
}
__global__ void k_RuntimeGripTerms(int count,const GripPoint* points,const unsigned* masks,const int* active,
    const float3* pos,const CoupledTool* body,XpbdToolState fallback,float minGap,RuntimeGripTerm* output){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const GripPoint& p=points[i];RuntimeGripTerm& out=output[i];out.valid=observedGripValid(p,masks,active);if(!out.valid)return;
    XpbdToolState pose=body?body->pose:fallback;
    float3 x=p.offset;for(int j=0;j<p.count;j++)x=x+pos[p.id[j]]*p.weight[j];
    out.weight=1.0/p.complianceScale;out.local=p.local;out.point=x;
    float3 a,n,b;gripFrame(pose,a,n,b);
    float3 r=x-gripTarget(p,pose,minGap);r=r-n*dot3(r,n);out.tangent=len3(r);
}
__global__ void k_RuntimeGripOrderedSum(int count,const RuntimeGripTerm* terms,const CoupledTool* body,
    XpbdToolState fallback,float minGap,float* result){
    XpbdToolState pose=body?body->pose:fallback;
    double weight=0,lx=0,ly=0,lz=0,px=0,py=0,pz=0;
    int live=0;float tangent=0;
    for(int i=0;i<count;i++){
        const auto& p=terms[i];if(!p.valid)continue;++live;double w=p.weight;
        weight+=w;lx+=w*p.local.x;ly+=w*p.local.y;lz+=w*p.local.z;
        px+=w*p.point.x;py+=w*p.point.y;pz+=w*p.point.z;
        tangent=fmaxf(tangent,p.tangent);
    }
    float error=0;
    if(weight>0){
        GripPatch patch={};patch.local=mkf3(float(lx/weight),float(ly/weight),float(lz/weight));
        float3 centroid=mkf3(float(px/weight),float(py/weight),float(pz/weight));
        error=len3(centroid-patchTarget(patch,pose,minGap));
    }
    result[0]=error;result[1]=(float)live;result[2]=tangent;
}
__global__ void k_RuntimeCompareGrip(const float* actual,const float* reference,int* stats){
    stats[0]++;
    for(int i=0;i<3;i++)if(__float_as_uint(actual[i])!=__float_as_uint(reference[i])){
        stats[1]++;if(!stats[2]){stats[2]=i+1;stats[3]=__float_as_int(reference[i]);stats[4]=__float_as_int(actual[i]);}
    }
}
static cudaError_t observeRuntimeGrip(){
    cudaError_t error=ensureRuntimeGripTerms(g_gripCount);if(error!=cudaSuccess)return error;
    k_RuntimeGripTerms<<<(g_gripCount+63)/64,64>>>(g_gripCount,d_gripPoints,g_surfaceMasks,g_liveActive,d_pos,d_coupledTool,g_tool,g_gp.jawGapClosed,d_runtimeGripTerms);
    k_RuntimeGripOrderedSum<<<1,1>>>(g_gripCount,d_runtimeGripTerms,d_coupledTool,g_tool,g_gp.jawGapClosed,d_gripError);
    if(g_queryValidate){
        if(!d_runtimeGripReference){error=queryPrivateAlloc(&d_runtimeGripReference,3*sizeof(float));if(error!=cudaSuccess)return error;}
        if(!d_runtimeGripStats){error=queryPrivateAlloc(&d_runtimeGripStats,8*sizeof(int));if(error!=cudaSuccess)return error;error=cudaMemsetAsync(d_runtimeGripStats,0,8*sizeof(int));if(error!=cudaSuccess)return error;}
        k_ReadGripObservation<<<1,1>>>(g_gripCount,d_gripPoints,g_surfaceMasks,g_liveActive,d_pos,d_coupledTool,g_tool,g_gp.jawGapClosed,d_runtimeGripReference);
        k_RuntimeCompareGrip<<<1,1>>>(d_gripError,d_runtimeGripReference,d_runtimeGripStats);
    }
    return cudaGetLastError();
}
extern "C" LCS_COMBINED_EXPORT int LCS_ReadRuntimeGripStats(int* output,int capacity){
    if(!output || capacity<8)return -1;
    for(int i=0;i<8;i++)output[i]=0;
    if(!d_runtimeGripStats)return 0;
    return -(int)cudaMemcpy(output,d_runtimeGripStats,8*sizeof(int),cudaMemcpyDeviceToHost);
}
