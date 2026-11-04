// Certify nonlinear material, pressure and anchor energy for patch updates.
// Blade-loaded patches keep their existing coupled solve. Unloaded active-volume
// recovery directions use the same energy check, including ligament supports.
// Pass 0 certifies the full step. Pass 1 evaluates the half/quarter candidates
// only for rejected components; kernel boundaries preserve the completed full
// energy reduction. The default evaluates all candidates for equivalence tests.
__device__ bool pressureLineActive(const UnifiedState& s){
    if(fragmentCoarseEnabled(s) && s.coarseMode>=1 && s.coarseMode<=5)return true;
    return (s.coarseMode==5 || (s.coarseMode>=1 && s.coarseMode<=4 && s.grips==0)) && (s.grips>0 || *s.contactCount>0 || (s.coarseMode>=1 && s.coarseMode<=4)) &&
        (!s.secondary || *s.secondary->contactCount==0);
}
__device__ void pressureLineAdd(CoarseBlock* b,float (&e)[3]){
#ifdef ENDO_KIDNEY_PRECISE_LINE
    double v[3]={e[0],e[1],e[2]};unsigned mask=__activemask();auto key=(unsigned long long)b;
    if(__all_sync(mask,key==__shfl_sync(mask,key,__ffs(mask)-1))){
        auto group=cooperative_groups::coalesced_threads();
        for(int k=0;k<3;k++)v[k]=cooperative_groups::reduce(group,v[k],cooperative_groups::plus<double>());
        if(group.thread_rank())return;
    }
    atomicAdd(&b->gradient.x,v[0]);atomicAdd(&b->gradient.y,v[1]);atomicAdd(&b->gradient.z,v[2]);
#else
    if(unifiedReduce(b,e)){atomicAdd(&b->gradient.x,e[0]);atomicAdd(&b->gradient.y,e[1]);atomicAdd(&b->gradient.z,e[2]);}
#endif
}
__device__ bool pressureLineZeroStep(const CoarseBlock& b){
    return b.scale==0 || (b.step.x==0 && b.step.y==0 && b.step.z==0);
}
__global__ void k_PressureLineClear(const UnifiedState* state,CoarseBlock* blocks){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);
    if(coarseRoot(s,i) && pressureLineActive(s))blocks[i].gradient=mkf3(0,0,0);
}
__device__ void pressureLineNode(const UnifiedState& s,CoarseBlock* blocks,int i,int pass){
    if(i>=s.n || !pressureLineActive(s) || (s.active&&!s.active[i]))return;
    CoarseBlock* b=blocks+s.component[i];float w=s.coarseWeight[i];
    if(pressureLineZeroStep(*b) || (pass==1 && b->gradient.x<=0))return;
    // A zero patch weight contributes only through a surviving cut-rim edge
    // whose other endpoint actually moves. All other energy differences are
    // identically zero; do not evaluate their three identical trial states.
    if(w==0){
        bool movingEdge=false;
        if(s.neighbors && s.edgeSupport)for(int a=1;a<6;a+=2){int j=s.neighbors[6*i+a];
            if(j>=0 && (!s.active||s.active[j]) && !s.edgeSupport[6*i+a] && s.coarseWeight[j]!=0)movingEdge=true;}
        if(!movingEdge)return;
    }
    float3 d=b->step*(b->scale*w);float inertia=s.invMass[i]>0?1.f/(s.invMass[i]*s.h*s.h):0,stiffness=s.attach?s.attach[i]:0;
    float3 g=(s.pos[i]-s.target[i])*inertia;if(stiffness>0){float viscous=unifiedSupportDamping(s,i);
        g=g+(s.pos[i]-s.attachRest[i])*stiffness+(s.pos[i]-s.previous[i])*viscous;stiffness+=viscous;}
    float values[3]={};for(int k=pass==1?1:0;k<(pass==0?1:3);k++){float a=ldexpf(1.f,-k);float3 q=d*a;
        double energy=unifiedDot(g,q)+.5*(inertia+stiffness)*unifiedDot(q,q);
        if(s.neighbors && s.edgeSupport)for(int axis=1;axis<6;axis+=2){
            int j=s.neighbors[6*i+axis];if(j<0 || (s.active&&!s.active[j]) || s.edgeSupport[6*i+axis])continue;
            float3 e=s.pos[i]-s.pos[j],delta=b->step*(b->scale*a*(w-s.coarseWeight[j]));
            float l0=len3(s.restNeighbors[6*i+axis]),old=len3(e)-l0,next=len3(e+delta)-l0;
            float K=.5f*(unifiedEdgeStiffness(s,i,axis/2)+unifiedEdgeStiffness(s,j,axis/2));energy+=.5*double(K)*(double(next)*next-double(old)*old);
        }
        values[k]=(float)energy;
    }pressureLineAdd(b,values);
}
__device__ void pressureLineTet(const UnifiedState& s,CoarseBlock* blocks,int t,int pass){
    if(t>=s.nt || !pressureLineActive(s) || !s.live[t])return;
    int id[4];unifiedTetIds(s,t,id);float w[4];for(int j=0;j<4;j++)w[j]=s.coarseWeight[id[j]];
    if(w[0]==w[1] && w[0]==w[2] && w[0]==w[3])return;
    CoarseBlock* b=blocks+s.component[id[0]];if(pressureLineZeroStep(*b) || (pass==1 && b->gradient.x<=0))return;
    float3 p[4],step[4];for(int j=0;j<4;j++)p[j]=s.pos[id[j]];float values[3]={};
    for(int k=pass==1?1:0;k<(pass==0?1:3);k++){
        // The change to the trial positions rounded to float, as in k_MaterialLineTets.
        float a=ldexpf(b->scale,-k);
        for(int j=0;j<4;j++)step[j]=(p[j]+b->step*(a*w[j]))-p[j];
        values[k]=unifiedTetEnergyChange(s,t,p,step);
    }
    pressureLineAdd(b,values);
}
__device__ float pressureFrictionEnergy(float distance,float stiffness,float pressure,float muS,float muD){
    float limit=muS*pressure;if(limit<=0 || stiffness<=0)return 0;
    float boundary=limit/stiffness;if(distance<=boundary)return .5f*stiffness*distance*distance;
    float slip=distance-boundary;return .5f*limit*boundary+pressure*(muD*slip+(muS-muD)*boundary*(1-boundary/distance));
}
__device__ void pressureLineContacts(const UnifiedState& s,CoarseBlock* blocks,int pass,int start,int stride){
    if(!pressureLineActive(s))return;
    for(int i=start;i<max(*s.contactCount,s.grips);i+=stride){
        if(i<s.grips && s.points[i].valid){
            const auto& a=s.points[i];CoarseBlock* b=blocks+s.component[a.id[0]];if(!(pass==1 && b->gradient.x<=0)){float w=0;
            for(int j=0;j<a.count;j++)w+=a.weight[j]*s.coarseWeight[a.id[j]];
            float3 error=unifiedGripPoint(a,s.pos)-gripTarget(a,s.tool->pose,s.minGap);float K=s.gripK/a.complianceScale,values[3]={};
            for(int k=pass==1?1:0;k<(pass==0?1:3);k++){float3 d=b->step*(ldexpf(b->scale,-k)*w);values[k]=(float)(K*(unifiedDot(error,d)+.5*unifiedDot(d,d)));}
            pressureLineAdd(b,values);}
        }
        if(i>=*s.contactCount)continue;
        const auto& c=s.contacts[s.contactIds[i]];CoarseBlock* b=blocks+s.component[c.id[0]];
        if(pressureLineZeroStep(*b) || (pass==1 && b->gradient.x<=0))continue;
        float w=0;for(int j=0;j<c.count;j++)w+=c.weight[j]*s.coarseWeight[c.id[j]];
        if(w==0)continue;
        float3 x=unifiedPoint(c,s.pos),normal;float C=unifiedGap(s,c,x,normal);
        float K=unifiedContactStiffness(s,c),D=unifiedImpactStiffness(s,c),effective=K+D;
        float previous=unifiedPreviousDepth(c,normal),compression=fmaxf(0.f,-C-D*previous/fmaxf(effective,1e-20f));
        float pressure=effective*compression;float3 slip=unifiedSlip(s,c,x,normal);
        double old=.5*double(effective)*compression*compression+pressureFrictionEnergy(len3(slip),K,pressure,s.muS,s.muD);float values[3]={};
        for(int k=pass==1?1:0;k<(pass==0?1:3);k++){
            float3 d=b->step*(ldexpf(b->scale,-k)*w);float depth=fmaxf(0.f,-C-dot3(d,normal)-D*previous/fmaxf(effective,1e-20f));
            float3 tangent=slip+d-normal*dot3(d,normal);
            values[k]=(float)(.5*double(effective)*depth*depth+pressureFrictionEnergy(len3(tangent),K,pressure,s.muS,s.muD)-old);
        }pressureLineAdd(b,values);
    }
}
// One pressure-line pass in one launch: the first activeCount threads add the energy change of an active node, the
// next nt that of a tet, and all of them then stride over the contacts and grips.

// The line search takes one common fraction for every component connected by
// the paired manifold. Its energy includes each paired constraint exactly once.
__device__ void fragmentPressureLineContacts(const UnifiedState& s,CoarseBlock* blocks,int pass,int start,int stride){
    if(!fragmentCoarseEnabled(s) || !pressureLineActive(s))return;
    for(int k=start;k<*s.fragmentConstraintCount;k+=stride){
        const auto& c=s.fragmentConstraints[k];auto& source=blocks[c.sourceRoot];if(pass==1 && source.gradient.x<=0)continue;
        auto law=fragmentNormalObjective(s,c);double old=fragmentNormalEnergy(law,0);float values[3]={};float change=0;
        for(int j=0;j<c.count;j++){int id=c.id[j];const auto& b=blocks[s.component[id]];change+=dot3(b.step,c.normal)*b.scale*s.coarseWeight[id]*c.weight[j];}
        for(int q=pass==1?1:0;q<(pass==0?1:3);q++)values[q]=(float)(fragmentNormalEnergy(law,ldexpf(change,-q))-old);
        pressureLineAdd(&source,values);
    }
}
__global__ void k_FragmentLineClear(const UnifiedState* state){
    const auto& s=*state;if(!fragmentCoarseEnabled(s) || !pressureLineActive(s))return;
    int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);
    if(coarseRoot(s,i) && s.fragmentContacted[i] && s.fragmentGroups[i]==i)s.fragmentLineSums[i]=mkf3(0,0,0);
}
__global__ void k_FragmentLineSum(const UnifiedState* state,const CoarseBlock* blocks,int pass){
    const auto& s=*state;if(!fragmentCoarseEnabled(s) || !pressureLineActive(s))return;
    int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(!coarseRoot(s,i) || !s.fragmentContacted[i])return;
    int group=s.fragmentGroups[i];float3 v=blocks[i].gradient;if(pass==1 && group!=i)v.x=0;
    atomicAdd(&s.fragmentLineSums[group].x,v.x);atomicAdd(&s.fragmentLineSums[group].y,v.y);atomicAdd(&s.fragmentLineSums[group].z,v.z);
}
__global__ void k_FragmentLineSet(const UnifiedState* state,CoarseBlock* blocks){
    const auto& s=*state;if(!fragmentCoarseEnabled(s) || !pressureLineActive(s))return;
    int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);
    if(coarseRoot(s,i) && s.fragmentContacted[i])blocks[i].gradient=s.fragmentLineSums[s.fragmentGroups[i]];
}
__global__ void k_PressureLine(const UnifiedState* state,CoarseBlock* blocks,int pass){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<s.activeCount)pressureLineNode(s,blocks,s.activeNodes[t],pass);else pressureLineTet(s,blocks,t-s.activeCount,pass);
    pressureLineContacts(s,blocks,pass,t,gridDim.x*blockDim.x);
    fragmentPressureLineContacts(s,blocks,pass,t,gridDim.x*blockDim.x);
}
// Isolated scheduling experiment: identical material/contact functions, energy
// candidates and double accumulation, with no contact temporaries in the
// material kernel's register allocation. Retain the old material thread mapping
// (including the node/tet boundary) and keep this disabled in default builds.
#ifdef ENDO_KIDNEY_SPLIT_PRESSURE
#ifdef ENDO_KIDNEY_SPECIALIZE_PRESSURE
template<int Pass> __global__ void k_PressureLineMaterialFixed(const UnifiedState* state,CoarseBlock* blocks){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<s.activeCount)pressureLineNode(s,blocks,s.activeNodes[t],Pass);else pressureLineTet(s,blocks,t-s.activeCount,Pass);
}
template<int Pass> __global__ void k_PressureLineContactFixed(const UnifiedState* state,CoarseBlock* blocks){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    pressureLineContacts(s,blocks,Pass,t,gridDim.x*blockDim.x);
    fragmentPressureLineContacts(s,blocks,Pass,t,gridDim.x*blockDim.x);
}
#endif
__global__ void k_PressureLineMaterial(const UnifiedState* state,CoarseBlock* blocks,int pass){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<s.activeCount)pressureLineNode(s,blocks,s.activeNodes[t],pass);else pressureLineTet(s,blocks,t-s.activeCount,pass);
}
__global__ void k_PressureLineContact(const UnifiedState* state,CoarseBlock* blocks,int pass){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    pressureLineContacts(s,blocks,pass,t,gridDim.x*blockDim.x);
    fragmentPressureLineContacts(s,blocks,pass,t,gridDim.x*blockDim.x);
}
#endif
static void launchPressureLine(cudaStream_t stream,int blocks,const UnifiedState* state,CoarseBlock* coarse,int pass){
#ifdef ENDO_KIDNEY_SPECIALIZE_PRESSURE
    if(pass==0){
        k_PressureLineMaterialFixed<0><<<blocks,64,0,stream>>>(state,coarse);
        k_PressureLineContactFixed<0><<<128,64,0,stream>>>(state,coarse);
    }else if(pass==1){
        k_PressureLineMaterialFixed<1><<<blocks,64,0,stream>>>(state,coarse);
        k_PressureLineContactFixed<1><<<128,64,0,stream>>>(state,coarse);
    }else{
        // Preserve the reference all-candidate path for diagnostic callers.
        k_PressureLineMaterial<<<blocks,64,0,stream>>>(state,coarse,pass);
        k_PressureLineContact<<<128,64,0,stream>>>(state,coarse,pass);
    }
#elif defined(ENDO_KIDNEY_SPLIT_PRESSURE)
    k_PressureLineMaterial<<<blocks,64,0,stream>>>(state,coarse,pass);
    k_PressureLineContact<<<128,64,0,stream>>>(state,coarse,pass);
#else
    k_PressureLine<<<blocks,64,0,stream>>>(state,coarse,pass);
#endif
}
__global__ void k_PressureLineSelect(const UnifiedState* state,CoarseBlock* blocks){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(!coarseRoot(s,i) || !pressureLineActive(s))return;
    auto& b=blocks[i];float scale=b.gradient.x<=0?1.f:b.gradient.y<=0?.5f:b.gradient.z<=0?.25f:0.f;b.scale*=scale;
}
