// GPU preconditioned conjugate gradients for coupled material motion, preconditioned by one aggregation multigrid
// V-cycle (material_multigrid.cuh). A positive strain-space curvature retains rigid motion in its nullspace, which
// block Jacobi resolves only over many iterations: the coarse levels resolve it at once.
// The nonlinear step is accepted only by the exact energy/determinant search.
// Contacted components retain additional V18 coarse/witness sweeps. Free
// components need more Krylov directions to resolve a thin hinge's inertia.
// The budgets reach the quadratic-model decrease the earlier 80 and 32 block-Jacobi iterations reached, or more. The
// grasp equilibrium converges more slowly than that decrease: with 3 loaded iterations the jaws close slightly further
// than with the earlier 32, while 4 or more let an excised piece fold along the jaw plane once its neck is cut.
static constexpr int MaterialFreeIterations=7;
static constexpr int MaterialLoadedIterations=3;
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
// A zero step and a zero direction before the first: iteration 0 then forms its direction z+0*0 as every later one
// forms z+beta*p (k_MaterialStencilProduct). The first preconditioned residual and its r'z, stored as those of
// iteration -1, come from the V-cycle that follows (k_MgFinePost).
__global__ void k_MaterialCGBegin(const UnifiedState* state,float3* direction,float3* step){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;
    step[i]=mkf3(0,0,0);direction[i]=mkf3(0,0,0);
}
// The CG nodes are the nodes whose stencil (k_MaterialStencilActiveMask) holds its own slot 7: exactly those that
// materialCgFixed leaves free, which it evaluates once per solve instead of at every iteration. The V-cycle that
// follows the advance preconditions its residual and adds this iteration's r'z (k_MgFinePost).
__device__ inline bool materialCgNode(const unsigned short* stencil,int i){return stencil[i]>>7&1;}
__global__ void k_MaterialCGAdvance(const UnifiedState* state,const unsigned short* stencil,float3* residual,
    const float3* direction,const float3* product,float3* step,MaterialCGSum* sums,int iteration){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;
    // The slot of the next iteration held the sums of the one before last, which this iteration's product read last.
    if(coarseRoot(s,i)){sums[i].denominator[(iteration+1)%3]=0;sums[i].next[(iteration+1)%3]=0;}
    if(!materialCgNode(stencil,i)||(iteration>=MaterialLoadedIterations && (s.graspComponents[s.component[i]]&7)))return;
    auto& q=sums[s.component[i]];double rho=materialRho(q,iteration),denominator=q.denominator[iteration%3];
    float alpha=denominator>1e-30&&rho>0?(float)(rho/denominator):0;
    step[i]=step[i]+direction[i]*alpha;residual[i]=residual[i]-product[i]*alpha;
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
