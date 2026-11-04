// Included last in xpbd_solver.cu: opt-in diagnostics and overload release.
// No position, velocity, damping, friction or solver budget is changed.
static float3* d_trainingCoreRaw=nullptr;
static float* d_trainingCoreEMA=nullptr;
static int* d_trainingBroken=nullptr;
static unsigned* d_trainingBreakCounts=nullptr;
static float* d_trainingForceScratch=nullptr;
static TrainingMetrics* d_trainingMetrics=nullptr;

__global__ void k_TrainingScatterForce(int count,const GripPoint* points,float inverseH2,float3* force){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !points[i].valid)return;
    const auto& p=points[i];float3 f=p.lambda*inverseH2;
    for(int k=0;k<p.count;k++){
        int id=p.id[k];float w=p.weight[k];
        atomicAdd(&force[id].x,f.x*w);atomicAdd(&force[id].y,f.y*w);atomicAdd(&force[id].z,f.z*w);
    }
}
__global__ void k_TrainingForceEMA(int n,float h,float threshold,int enabled,const float3* raw,
    float* ema,const int* core,int* broken,unsigned* counts){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n || !core[i] || broken[i])return;
    ema[i]+=(len3(raw[i])-ema[i])*(-expm1f(-h));
    if(enabled && ema[i]>threshold){broken[i]=1;atomicAdd(counts,1u);}
}
// GripPoint is the actual finite attachment in the unified objective. Its
// supporting corners carry the distributed force. A failed support releases
// that witness permanently; never silently recapture it while the jaw is shut.
__global__ void k_TrainingReleasePoints(int count,GripPoint* points,const int* broken,unsigned* counts){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !points[i].valid)return;
    auto& p=points[i];for(int k=0;k<p.count;k++)if(p.weight[k]>1e-8f && broken[p.id[k]]){
        p.valid=0;atomicAdd(counts+1,1u);return;
    }
}
static void trainingSubstep(float h){
    if(!d_trainingCoreRaw || !g_embeddedGrip || !g_anchorAssist || !g_grasping || g_gripCount<=0)return;
    cudaMemsetAsync(d_trainingCoreRaw,0,(size_t)g_N*sizeof(float3));
    k_TrainingScatterForce<<<Gr(g_gripCount),64>>>(g_gripCount,d_gripPoints,1.f/(h*h),d_trainingCoreRaw);
    k_TrainingForceEMA<<<Gr(g_N),64>>>(g_N,h,g_gp.coreBreakForce,g_trainingFlags&2,d_trainingCoreRaw,
        d_trainingCoreEMA,d_graspCore,d_trainingBroken,d_trainingBreakCounts);
    if(g_trainingFlags&2)k_TrainingReleasePoints<<<Gr(g_gripCount),64>>>(g_gripCount,d_gripPoints,d_trainingBroken,d_trainingBreakCounts);
}
static void trainingCapture(){
    if(d_trainingCoreEMA)cudaMemset(d_trainingCoreEMA,0,(size_t)g_N*sizeof(float));
    if(d_trainingBroken)cudaMemset(d_trainingBroken,0,(size_t)g_N*sizeof(int));
}
static void trainingShutdown(){
    cudaFree(d_trainingCoreRaw);d_trainingCoreRaw=nullptr;
    cudaFree(d_trainingCoreEMA);d_trainingCoreEMA=nullptr;
    cudaFree(d_trainingBroken);d_trainingBroken=nullptr;
    cudaFree(d_trainingBreakCounts);d_trainingBreakCounts=nullptr;
    cudaFree(d_trainingForceScratch);d_trainingForceScratch=nullptr;
    cudaFree(d_trainingMetrics);d_trainingMetrics=nullptr;g_trainingFlags=0;
}
extern "C" int training_enable(int flags){
    if(!g_bound || g_hFrame>0 || flags<0 || flags>3)return XPBD_ERR_BAD_ARG;
    if(flags==0){g_trainingFlags=0;return 0;}
    if(d_trainingMetrics)return XPBD_ERR_BAD_ARG;
    cudaError_t e=cudaMalloc(&d_trainingCoreRaw,(size_t)g_N*sizeof(float3));
    if(e==cudaSuccess)e=cudaMalloc(&d_trainingCoreEMA,(size_t)g_N*sizeof(float));
    if(e==cudaSuccess)e=cudaMalloc(&d_trainingBroken,(size_t)g_N*sizeof(int));
    if(e==cudaSuccess)e=cudaMalloc(&d_trainingForceScratch,(size_t)g_N*sizeof(float));
    if(e==cudaSuccess)e=cudaMalloc(&d_trainingBreakCounts,2*sizeof(unsigned));
    if(e==cudaSuccess)e=cudaMalloc(&d_trainingMetrics,sizeof(TrainingMetrics));
    if(e!=cudaSuccess){trainingShutdown();return -(int)e;}
    cudaMemset(d_trainingBreakCounts,0,2*sizeof(unsigned));trainingCapture();
    int r=cut_training_enable();if(r){trainingShutdown();return r;}
    g_trainingFlags=flags;return -(int)cudaGetLastError();
}
__global__ void k_TrainingCollectEMA(int n,const int* core,const float* ema,float* scratch,TrainingMetrics* m){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    // Keep the overload peak observable after the corresponding point releases.
    // EMA resets at each new capture. Median covers currently active supports.
    if(ema[i]>0){
        unsigned mask=__activemask(),maximum=__reduce_max_sync(mask,__float_as_uint(ema[i]));
        if((threadIdx.x&31)==__ffs(mask)-1)atomicMax((unsigned*)&m->coreForceMax,maximum);
    }
    if(!core[i])return;
    unsigned k=atomicAdd(&m->activeCoreCorners,1u);scratch[k]=ema[i];
}
// Exact order statistic of the small active support set, not a histogram proxy.
__device__ float trainingSelect(float* x,int n,int k){
    int left=0,right=n-1;
    while(left<right){
        float pivot=x[(left+right)/2];int i=left,j=right;
        while(i<=j){while(x[i]<pivot)i++;while(x[j]>pivot)j--;if(i<=j){float t=x[i];x[i++]=x[j];x[j--]=t;}}
        if(k<=j)right=j;else if(k>=i)left=i;else break;
    }return x[k];
}
__global__ void k_TrainingFinishMetrics(float* scratch,const unsigned* counts,const float* quality,TrainingMetrics* m){
    if(threadIdx.x || blockIdx.x)return;
    int n=m->activeCoreCorners;
    if(n){float a=trainingSelect(scratch,n,n/2);m->coreForceMedian=n%2?a:.5f*(a+trainingSelect(scratch,n,n/2-1));}
    m->brokenCoreCorners=counts[0];m->brokenGripPoints=counts[1];
    m->minVolumeRatio=quality[0];m->invalidTets=(unsigned)quality[1];
}
extern "C" int training_read(TrainingMetrics* out){
    if(!out || !g_bound || !g_trainingFlags || !d_trainingMetrics)return XPBD_ERR_BAD_ARG;
    cudaMemsetAsync(d_trainingMetrics,0,sizeof(TrainingMetrics));
    // Same quality kernel/initial value as xpbd_get_material_quality.
    if(!d_toolClearance){cudaError_t e=cudaMalloc(&d_toolClearance,9*sizeof(float));if(e!=cudaSuccess)return -(int)e;}
    float quality[2]={1.f,0};
    cudaError_t e=cudaMemcpy(d_toolClearance,quality,sizeof(quality),cudaMemcpyHostToDevice);if(e!=cudaSuccess)return -(int)e;
    if(g_T>0)k_MaterialQuality<<<Gr(g_T),64>>>(g_T,d_tetIds,d_tetActive,d_tetBr0,d_tetBr1,d_tetBr2,d_pos,d_toolClearance);
    k_TrainingCollectEMA<<<Gr(g_N),64>>>(g_N,d_graspCore,d_trainingCoreEMA,d_trainingForceScratch,d_trainingMetrics);
    int r=cut_training_reduce(d_trainingMetrics);if(r)return r;
    k_TrainingFinishMetrics<<<1,1>>>(d_trainingForceScratch,d_trainingBreakCounts,d_toolClearance,d_trainingMetrics);
    return -(int)cudaMemcpy(out,d_trainingMetrics,sizeof(TrainingMetrics),cudaMemcpyDeviceToHost);
}
extern "C" int training_core_forces(int count,float* out){
    if(!out || count!=g_N || !d_trainingCoreEMA)return XPBD_ERR_BAD_ARG;
    return -(int)cudaMemcpy(out,d_trainingCoreEMA,(size_t)g_N*sizeof(float),cudaMemcpyDeviceToHost);
}
