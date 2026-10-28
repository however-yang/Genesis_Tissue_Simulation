// Certify nonlinear material, pressure and anchor energy for patch updates.
// Blade-loaded patches keep their existing coupled solve. Unloaded active-volume
// recovery directions use the same energy check, including ligament supports.
// Pass 0 certifies the full step. Pass 1 evaluates the half/quarter candidates
// only for rejected components; kernel boundaries preserve the completed full
// energy reduction. The default evaluates all candidates for equivalence tests.
__device__ bool pressureLineActive(const UnifiedState& s){
    return (s.coarseMode==5 || (s.coarseMode>=1 && s.coarseMode<=4 && s.grips==0)) && (s.grips>0 || *s.contactCount>0 || (s.coarseMode>=1 && s.coarseMode<=4)) &&
        (!s.secondary || *s.secondary->contactCount==0);
}
__device__ void pressureLineAdd(CoarseBlock* b,float (&e)[3]){
    if(unifiedReduce(b,e)){atomicAdd(&b->gradient.x,e[0]);atomicAdd(&b->gradient.y,e[1]);atomicAdd(&b->gradient.z,e[2]);}
}
__device__ bool pressureLineZeroStep(const CoarseBlock& b){
    return b.scale==0 || (b.step.x==0 && b.step.y==0 && b.step.z==0);
}
__global__ void k_PressureLineClear(const UnifiedState* state,CoarseBlock* blocks){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<s.n && pressureLineActive(s))blocks[i].gradient=mkf3(0,0,0);
}
__global__ void k_PressureLineNodes(const UnifiedState* state,CoarseBlock* blocks,int pass=-1){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
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
__global__ void k_PressureLineTets(const UnifiedState* state,CoarseBlock* blocks,int pass=-1){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=s.nt || !pressureLineActive(s) || !s.live[t])return;
    const int* id=s.ids+4*t;float w[4];for(int j=0;j<4;j++)w[j]=s.coarseWeight[id[j]];
    if(w[0]==w[1] && w[0]==w[2] && w[0]==w[3])return;
    CoarseBlock* b=blocks+s.component[id[0]];if(pressureLineZeroStep(*b) || (pass==1 && b->gradient.x<=0))return;
    float3 p[4],q[4];for(int j=0;j<4;j++)p[j]=s.pos[id[j]];
    double old=unifiedTetEnergy(s,t,p);float values[3]={};
    for(int k=pass==1?1:0;k<(pass==0?1:3);k++){float a=ldexpf(b->scale,-k);for(int j=0;j<4;j++)q[j]=p[j]+b->step*(a*w[j]);values[k]=(float)(unifiedTetEnergy(s,t,q)-old);}
    pressureLineAdd(b,values);
}
__device__ float pressureFrictionEnergy(float distance,float stiffness,float pressure,float muS,float muD){
    float limit=muS*pressure;if(limit<=0 || stiffness<=0)return 0;
    float boundary=limit/stiffness;if(distance<=boundary)return .5f*stiffness*distance*distance;
    float slip=distance-boundary;return .5f*limit*boundary+pressure*(muD*slip+(muS-muD)*boundary*(1-boundary/distance));
}
__global__ void k_PressureLineContacts(const UnifiedState* state,CoarseBlock* blocks,int pass=-1){
    const auto& s=*state;if(!pressureLineActive(s))return;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<max(*s.contactCount,s.grips);i+=blockDim.x*gridDim.x){
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
        float pressure=effective*compression;float3 slip=x-c.previous-c.plane.motion-unifiedShift(c,s.tool);
        slip=slip-normal*dot3(slip,normal);
        double old=.5*double(effective)*compression*compression+pressureFrictionEnergy(len3(slip),K,pressure,s.muS,s.muD);float values[3]={};
        for(int k=pass==1?1:0;k<(pass==0?1:3);k++){
            float3 d=b->step*(ldexpf(b->scale,-k)*w);float depth=fmaxf(0.f,-C-dot3(d,normal)-D*previous/fmaxf(effective,1e-20f));
            float3 tangent=slip+d-normal*dot3(d,normal);
            values[k]=(float)(.5*double(effective)*depth*depth+pressureFrictionEnergy(len3(tangent),K,pressure,s.muS,s.muD)-old);
        }pressureLineAdd(b,values);
    }
}
__global__ void k_PressureLineSelect(const UnifiedState* state,CoarseBlock* blocks){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n || !pressureLineActive(s))return;
    auto& b=blocks[i];float scale=b.gradient.x<=0?1.f:b.gradient.y<=0?.5f:b.gradient.z<=0?.25f:0.f;b.scale*=scale;
}
