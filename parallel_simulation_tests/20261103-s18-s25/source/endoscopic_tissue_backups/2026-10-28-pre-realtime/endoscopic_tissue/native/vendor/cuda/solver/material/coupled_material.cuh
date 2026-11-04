// GPU preconditioned conjugate gradients for coupled material motion.
// A positive strain-space curvature retains rigid motion in its nullspace;
// block Jacobi alone cannot recover it in a small fixed number of sweeps.
// The nonlinear step is accepted only by the exact energy/determinant search.
// Contacted components retain additional V18 coarse/witness sweeps. Free
// components need more Krylov directions to resolve a thin hinge's inertia.
static constexpr int MaterialFreeIterations=80;
static constexpr int MaterialLoadedIterations=32;
struct MaterialCGSum { double rho,next,denominator; };
__device__ inline void materialDotAdd(double* address,double value){
    unsigned mask=__activemask();auto key=(unsigned long long)address;
    if(__all_sync(mask,key==__shfl_sync(mask,key,__ffs(mask)-1))){
        auto group=cooperative_groups::coalesced_threads();
        value=cooperative_groups::reduce(group,value,cooperative_groups::plus<double>());
        if(group.thread_rank())return;
    }
    atomicAdd(address,value);
}
__global__ void k_MaterialCGClear(int n,MaterialCGSum* sums,bool initial){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    sums[i].next=sums[i].denominator=0;if(initial)sums[i].rho=0;
}
__global__ void k_MaterialCGBegin(const UnifiedState* state,const float3* r,const Symmetric3* diagonal,
    float3* direction,float3* step,MaterialCGSum* sums){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    step[i]=mkf3(0,0,0);direction[i]=mkf3(0,0,0);
    if(materialCgFixed(s,i))return;
    direction[i]=diagonal[i].solve(r[i]);
    materialDotAdd(&sums[s.component[i]].rho,unifiedDot(r[i],direction[i]));
}
__global__ void k_MaterialCGAdvance(const UnifiedState* state,float3* residual,const Symmetric3* diagonal,
    const float3* direction,const float3* product,float3* step,MaterialCGSum* sums,int iteration){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    if(materialCgFixed(s,i)||(iteration>=MaterialLoadedIterations && (s.graspComponents[s.component[i]]&7)))return;
    auto& q=sums[s.component[i]];float alpha=q.denominator>1e-30&&q.rho>0?(float)(q.rho/q.denominator):0;
    step[i]=step[i]+direction[i]*alpha;float3 r=residual[i]-product[i]*alpha;residual[i]=r;
    materialDotAdd(&q.next,unifiedDot(r,diagonal[i].solve(r)));
}
__global__ void k_MaterialCGDirection(const UnifiedState* state,const float3* residual,const Symmetric3* diagonal,
    float3* direction,const MaterialCGSum* sums,int iteration){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    if(materialCgFixed(s,i)||(iteration>=MaterialLoadedIterations && (s.graspComponents[s.component[i]]&7)))return;
    const auto& q=sums[s.component[i]];float beta=q.rho>1e-30?(float)(q.next/q.rho):0;
    direction[i]=diagonal[i].solve(residual[i])+direction[i]*beta;
}
__global__ void k_MaterialCGFinish(const UnifiedState* state,MaterialCGSum* sums,int iteration){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n||s.component[i]!=i||(s.active&&!s.active[i])||(iteration>=MaterialLoadedIterations && (s.graspComponents[i]&7)))return;
    sums[i].rho=sums[i].next;sums[i].next=sums[i].denominator=0;
}

__global__ void k_MaterialReady(UnifiedState* state,bool ready){state->coupledReady=ready;}

__global__ void k_MaterialRecoveryOnly(UnifiedState* state,bool only){state->recoveryOnly=only;}

__global__ void k_MaterialRecoveryRequested(const UnifiedState* state,int* requested){
 const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<s.n && s.component[i]==i && materialFreeRecovery(s,i))atomicExch(requested,1);
}
