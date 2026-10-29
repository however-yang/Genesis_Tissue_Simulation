// GPU preconditioned conjugate gradients for coupled material motion.
// A positive strain-space curvature retains rigid motion in its nullspace;
// block Jacobi alone cannot recover it in a small fixed number of sweeps.
// The nonlinear step is accepted only by the exact energy/determinant search.
// Contacted components retain additional V18 coarse/witness sweeps. Free
// components need more Krylov directions to resolve a thin hinge's inertia.
static constexpr int MaterialFreeIterations=80;
static constexpr int MaterialLoadedIterations=32;
// The PCG sums of a component for three consecutive iterations, by iteration modulo 3: p'Ap (denominator) and r'z
// (next) of iteration k. The r'z of iteration k-1 is the rho of iteration k (materialRho). The direction update of
// iteration k therefore runs inside the matrix-vector product of iteration k+1 (k_MaterialStencilProduct), and the
// advance of iteration k clears the slot of iteration k+1, with no kernel of their own.
struct MaterialCGSum { double denominator[3],next[3]; };
// The rho of an iteration; k_MaterialCGBegin stores the initial one as the r'z of iteration -1.
__device__ inline double materialRho(const MaterialCGSum& q,int iteration){return q.next[(iteration+2)%3];}
__device__ inline void materialDotAdd(double* address,double value){
    unsigned mask=__activemask();auto key=(unsigned long long)address;
    if(__all_sync(mask,key==__shfl_sync(mask,key,__ffs(mask)-1))){
        auto group=cooperative_groups::coalesced_threads();
        value=cooperative_groups::reduce(group,value,cooperative_groups::plus<double>());
        if(group.thread_rank())return;
    }
    atomicAdd(address,value);
}
__global__ void k_MaterialCGClear(int n,MaterialCGSum* sums){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)sums[i]={};
}
// The first preconditioned residual z, and a zero direction before it: iteration 0 then forms its direction z+0*0
// as every later one forms z+beta*p (k_MaterialStencilProduct).
__global__ void k_MaterialCGBegin(const UnifiedState* state,const float3* r,const Symmetric3* diagonal,
    float3* preconditioned,float3* direction,float3* step,MaterialCGSum* sums){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    step[i]=mkf3(0,0,0);direction[i]=mkf3(0,0,0);
    if(materialCgFixed(s,i))return;
    preconditioned[i]=diagonal[i].solve(r[i]);
    materialDotAdd(&sums[s.component[i]].next[2],unifiedDot(r[i],preconditioned[i]));
}
// The CG nodes are the nodes whose stencil (k_MaterialStencilActiveMask) holds its own slot 7: exactly those that
// materialCgFixed leaves free, which it evaluates once per solve instead of at every iteration. The advance keeps the
// preconditioned residual for the direction update.
__device__ inline bool materialCgNode(const unsigned short* stencil,int i){return stencil[i]>>7&1;}
__global__ void k_MaterialCGAdvance(const UnifiedState* state,const unsigned short* stencil,float3* residual,
    const Symmetric3* diagonal,const float3* direction,const float3* product,float3* step,float3* preconditioned,
    MaterialCGSum* sums,int iteration){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    // The slot of the next iteration held the sums of the one before last, which this iteration's product read last.
    if(coarseRoot(s,i)){sums[i].denominator[(iteration+1)%3]=0;sums[i].next[(iteration+1)%3]=0;}
    if(!materialCgNode(stencil,i)||(iteration>=MaterialLoadedIterations && (s.graspComponents[s.component[i]]&7)))return;
    auto& q=sums[s.component[i]];double rho=materialRho(q,iteration),denominator=q.denominator[iteration%3];
    float alpha=denominator>1e-30&&rho>0?(float)(rho/denominator):0;
    step[i]=step[i]+direction[i]*alpha;float3 r=residual[i]-product[i]*alpha;residual[i]=r;
    float3 z=diagonal[i].solve(r);preconditioned[i]=z;
    materialDotAdd(&q.next[iteration%3],unifiedDot(r,z));
}

__global__ void k_MaterialReady(UnifiedState* state,bool ready){state->coupledReady=ready;}

__global__ void k_MaterialRecoveryOnly(UnifiedState* state,bool only){state->recoveryOnly=only;}

// Flags whether some movable node belongs to a free component: one neither held, touched nor anchored. Only those
// nodes take part in the PCG iterations beyond MaterialLoadedIterations.
__global__ void k_MaterialFreeNodes(const UnifiedState* state,int* free){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    bool freeNode=i<s.n && s.invMass[i]>0 && (!s.active||s.active[i]) && !(s.graspComponents[s.component[i]]&7);
    if(__any_sync(0xffffffff,freeNode) && (threadIdx.x&31)==0)atomicExch(free,1);
}
__global__ void k_MaterialRecoveryRequested(const UnifiedState* state,int* requested){
 const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<s.n && s.component[i]==i && materialFreeRecovery(s,i))atomicExch(requested,1);
}
