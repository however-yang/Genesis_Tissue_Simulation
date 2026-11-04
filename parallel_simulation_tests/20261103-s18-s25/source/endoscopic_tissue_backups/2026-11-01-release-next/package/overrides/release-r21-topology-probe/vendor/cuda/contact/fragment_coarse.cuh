// Paired contact terms in the existing material coarse spaces. No rigid
// attachment: the colored material/contact sweeps keep every original DOF.
__device__ inline int fragmentGroupRoot(const int* parent,int i){int next=atomicAdd((int*)parent+i,0);while(next!=i){i=next;next=atomicAdd((int*)parent+i,0);}return i;}
__global__ void k_InitFragmentGroups(const UnifiedState* state){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<s.n && fragmentContactActive(s))s.fragmentGroups[i]=i;
}
__global__ void k_JoinFragmentGroups(const UnifiedState* state){
    const auto& s=*state;if(!fragmentContactActive(s))return;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*s.fragmentConstraintCount;k+=blockDim.x*gridDim.x){
        const auto& c=s.fragmentConstraints[k];int a=c.sourceRoot,b=c.targetRoot;
        while(true){a=fragmentGroupRoot(s.fragmentGroups,a);b=fragmentGroupRoot(s.fragmentGroups,b);if(a==b)break;
            int hi=max(a,b),lo=min(a,b);if(atomicCAS(s.fragmentGroups+hi,hi,lo)==hi)break;
        }
    }
}
__global__ void k_FlattenFragmentGroups(const UnifiedState* state){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<s.n && fragmentContactActive(s) && s.fragmentContacted[i]){
        s.fragmentGroups[i]=fragmentGroupRoot(s.fragmentGroups,i);
        // Both surfaces carry an external contact load. They must not enter
        // the free-body material recovery path, just as for instrument contact.
        atomicOr(s.graspComponents+i,2);
    }
}
struct FragmentNormalObjective{float gap,stiffness,shift,pressure;};
__device__ inline FragmentNormalObjective fragmentNormalObjective(const UnifiedState& s,const FragmentConstraint& c){
    float x=-c.offset,old=-c.offset;
    for(int j=0;j<c.count;j++){x+=dot3(s.pos[c.id[j]],c.normal)*c.weight[j];old+=dot3(s.previous[c.id[j]],c.normal)*c.weight[j];}
    float damping=c.inverseMass>0?2.f*sqrtf(c.stiffness/c.inverseMass)/s.h:0;
    float K=c.stiffness+damping,shift=damping*fmaxf(0.f,-old)/fmaxf(K,1e-20f);
    return {x,K,shift,fmaxf(0.f,-K*(x+shift))};
}
__device__ inline double fragmentNormalEnergy(const FragmentNormalObjective& law,float delta){
    double depth=fmax(0.,-double(law.gap)-delta-law.shift);return .5*law.stiffness*depth*depth;
}
__global__ void k_FragmentCoarseTerms(const UnifiedState* state,CoarseBlock* blocks){
    const auto& s=*state;if(!fragmentCoarseEnabled(s) || s.coarseMode==6 || s.coarseMode==7)return;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*s.fragmentConstraintCount;k+=blockDim.x*gridDim.x){
        const auto& c=s.fragmentConstraints[k];auto law=fragmentNormalObjective(s,c);if(!(law.pressure>0))continue;
        float wa=0,wb=0;for(int j=0;j<c.count;j++){float w=c.weight[j]*s.coarseWeight[c.id[j]];if(s.component[c.id[j]]==c.sourceRoot)wa+=w;else wb+=w;}
        float3 gradient=c.normal*(-law.pressure);
        if(!s.coarseMode){float w=wa+wb;Symmetric3 H={};H.outer(c.normal,law.stiffness*w*w);coarseAccumulate(blocks,gradient*w,H);}
        else{
            // A block diagonal majorizer includes the omitted cross block; the
            // force is the exact derivative in each component's coarse space.
            float sum=fabsf(wa)+fabsf(wb);Symmetric3 A={},B={};A.outer(c.normal,law.stiffness*fabsf(wa)*sum);B.outer(c.normal,law.stiffness*fabsf(wb)*sum);
            coarseAccumulate(blocks+c.sourceRoot,gradient*wa,A);coarseAccumulate(blocks+c.targetRoot,gradient*wb,B);
        }
    }
}
__global__ void k_FragmentCoarseFreeze(const UnifiedState* state,CoarseBlock* blocks){
    const auto& s=*state;if(!fragmentCoarseEnabled(s) || s.coarseMode==6 || s.coarseMode==7)return;
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(!s.coarseMode){if(k)return;blocks->center=blocks->step*blocks->scale;blocks->mass=blocks->scale;return;}
    int i=unifiedActiveNode(s,k);if(!coarseRoot(s,i) || !s.fragmentContacted[i])return;
    auto& b=blocks[i];b.center=b.step*b.scale;b.mass=b.scale;
    if(s.fragmentGroups[i]==i)s.fragmentLineSums[i]=mkf3(1,0,0);
}
__global__ void k_FragmentCoarseBound(const UnifiedState* state,CoarseBlock* blocks){
    const auto& s=*state;if(!fragmentCoarseEnabled(s) || s.coarseMode==6 || s.coarseMode==7)return;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*s.fragmentConstraintCount;k+=blockDim.x*gridDim.x){
        const auto& c=s.fragmentConstraints[k];float gap=-c.offset,change=0;
        for(int j=0;j<c.count;j++){
            int id=c.id[j];const auto& b=blocks[s.coarseMode?s.component[id]:0];
            gap+=dot3(s.pos[id],c.normal)*c.weight[j];change+=dot3(b.center,c.normal)*c.weight[j]*s.coarseWeight[id];
        }
        if(change>=0)continue;
        // A trust region reaches the contact plane and at most one existing
        // contact skin beyond it. It does not alter the energy or its force.
        float scale=fminf(1.f,fmaxf(0.f,(gap+.001f)/(-change)));
        if(!s.coarseMode)coarseBound(blocks,blocks->mass*scale);
        else atomicMin((unsigned*)&s.fragmentLineSums[s.fragmentGroups[c.sourceRoot]].x,__float_as_uint(scale));
    }
}
__global__ void k_FragmentCoarseBoundApply(const UnifiedState* state,CoarseBlock* blocks){
    const auto& s=*state;if(!fragmentCoarseEnabled(s) || !s.coarseMode || s.coarseMode==6 || s.coarseMode==7)return;
    int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(!coarseRoot(s,i) || !s.fragmentContacted[i])return;
    blocks[i].scale=blocks[i].mass*s.fragmentLineSums[s.fragmentGroups[i]].x;
}
