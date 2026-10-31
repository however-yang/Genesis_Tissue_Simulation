// Experimental 12-DOF component correction. Included only by the isolated trial patch.
// Four existing graded scalar modes; all fine-grid and tool DOFs remain available.
constexpr int JointModes=4,JointDofs=12,JointTerms=90;
struct JointSafe {float4 sum,grip;int count,grips,pinned;};
struct JointBlock {float terms[JointTerms],step[JointDofs],scale;MaterialLineSearch line;int disabled,accepted,zero;};
__device__ inline float jointW(float4 w,int k){return k==0?w.x:k==1?w.y:k==2?w.z:w.w;}
__device__ inline int jointH(int i,int j){if(i>j){int t=i;i=j;j=t;}return 12+i*(23-i)/2+j;}
__device__ inline float jointSym(const Symmetric3& h,int i,int j){return i==j?(i==0?h.xx:i==1?h.yy:h.zz):(i+j==1?h.xy:i+j==2?h.xz:h.yz);}
__device__ inline bool jointEligible(const UnifiedState& s,int i){return !s.coupledReady || (s.graspComponents[s.component[i]]&15);}
__device__ inline void jointFloat4Add(float4* p,float4 w){atomicAdd(&p->x,w.x);atomicAdd(&p->y,w.y);atomicAdd(&p->z,w.z);atomicAdd(&p->w,w.w);}
__device__ inline float4 jointBasis(const UnifiedState& s,int i){
    if(s.invMass[i]<=0 || !jointEligible(s,i))return make_float4(0,0,0,0);
    if(s.graspComponents[s.component[i]]&5){
        float3 d=s.pos[i]-(s.tool->pose.jawHinge+s.patchOffset);
        float length=fmaxf(s.patchFalloff,1e-5f),r=fmaxf(0.f,len3(d)-s.patchCoreRadius);
        float w=1.f/(1.f+r*r/(length*length));w*=w;
        float a=rsqrtf(length*length+dot3(d,d));
        return make_float4(w,w*d.x*a,w*d.y*a,w*d.z*a);
    }
    return make_float4(1.f,float(i%s.nx)/fmaxf(1.f,float(s.nx-1))-.5f,
        float((i/s.nx)%s.ny)/fmaxf(1.f,float(s.ny-1))-.5f,
        float(i/(s.nx*s.ny))/fmaxf(1.f,float(s.n/(s.nx*s.ny)-1))-.5f);
}
__global__ void k_JointBegin(UnifiedState* state,float4* weights,JointSafe* safe,JointBlock* block){
    const auto& s=*state;int k=blockIdx.x*blockDim.x+threadIdx.x,i=unifiedActiveNode(s,k);
    if(k==0){*s.safeJoined=0;state->coarseMode=1;}
    if(i<0)return;weights[i]=jointBasis(s,i);safe[i]={};s.safeCluster[i]=i;
    if(coarseRoot(s,i)){block[i]={};block[i].scale=1.f;}
}
__global__ void k_JointSafeAccumulate(const UnifiedState* state,const float4* weights,JointSafe* aggregate){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);
    if(i<0 || !*s.safeJoined)return;auto* a=aggregate+s.safeCluster[i];
    jointFloat4Add(&a->sum,weights[i]);atomicAdd(&a->count,1);
    if(s.invMass[i]<=0)atomicExch(&a->pinned,1);
    if(s.grips>0 && s.gripDegree[i]>0){jointFloat4Add(&a->grip,weights[i]);atomicAdd(&a->grips,1);}
}
__global__ void k_JointSafeFinish(const UnifiedState* state,float4* weights,const JointSafe* aggregate){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);
    if(i<0 || !*s.safeJoined)return;const auto& a=aggregate[s.safeCluster[i]];
    float4 w=a.grips?a.grip:a.sum;float f=a.pinned?0.f:1.f/fmaxf(1.f,float(a.grips?a.grips:a.count));
    weights[i]=make_float4(w.x*f,w.y*f,w.z*f,w.w*f);
}
__device__ inline void jointPullback(float (&v)[JointTerms],float4 w,float3 g,const Symmetric3& h){
    #pragma unroll
    for(int i=0;i<12;i++)v[i]+=jointW(w,i/3)*compF3(g,i%3);
    #pragma unroll
    for(int i=0;i<12;i++){
        #pragma unroll
        for(int j=i;j<12;j++)v[jointH(i,j)]+=jointW(w,i/3)*jointW(w,j/3)*jointSym(h,i%3,j%3);
    }
}
__device__ inline void jointAdd(const void* target,const float* values){
    auto* b=(JointBlock*)target;for(int j=0;j<JointTerms;j++)atomicAdd(b->terms+j,values[j]);
}
__device__ inline void jointWarpAdd(JointBlock* b,float (&v)[JointTerms]){if(unifiedReduce(b,v))jointAdd(b,v);}
__device__ inline JointBlock* jointNodeTerms(const UnifiedState& s,int i,const float4* weights,JointBlock* blocks,float (&v)[JointTerms]){
    if(!jointEligible(s,i))return nullptr;
    float4 w=weights[i];float inertia=s.invMass[i]>0?1.f/(s.invMass[i]*s.h*s.h):0;
    float3 g=(s.pos[i]-s.target[i])*inertia;Symmetric3 h={};h.diagonal(inertia);
    if(s.attach){float damping=unifiedSupportDamping(s,i);g=g+(s.pos[i]-s.attachRest[i])*s.attach[i]+(s.pos[i]-s.previous[i])*damping;h.diagonal(s.attach[i]+damping);}
    jointPullback(v,w,g,h);
    if(s.neighbors && s.edgeSupport)for(int a=1;a<6;a+=2){
        int j=s.neighbors[6*i+a];if(j<0 || (s.active&&!s.active[j]) || s.edgeSupport[6*i+a])continue;
        float4 z=weights[j],dw=make_float4(w.x-z.x,w.y-z.y,w.z-z.z,w.w-z.w);
        float3 e=s.pos[i]-s.pos[j];float l=len3(e),l0=len3(s.restNeighbors[6*i+a]);
        if(l>1e-10f){float3 n=e*(1.f/l);float strain=l-l0,K=.5f*(unifiedEdgeStiffness(s,i,a/2)+unifiedEdgeStiffness(s,j,a/2));
            Symmetric3 edge={};edge.outer(n,K);edge.diagonal(K*fmaxf(0.f,strain/l));jointPullback(v,dw,n*(K*strain),edge);}
    }
    return blocks+s.component[i];
}
__device__ inline JointBlock* jointTetTerms(const UnifiedState& s,int t,const float4* weights,JointBlock* blocks,float (&v)[JointTerms]){
    if(t>=s.nt || !s.live[t])return nullptr;int id[4];unifiedTetIds(s,t,id);
    float4 w[4];for(int k=0;k<4;k++)w[k]=weights[id[k]];
    const auto& rest=unifiedKuhn(s,t);float3 q[4];float norm=0;
    #pragma unroll
    for(int m=0;m<4;m++){q[m]=rest.b0*(jointW(w[1],m)-jointW(w[0],m))+rest.b1*(jointW(w[2],m)-jointW(w[0],m))+rest.b2*(jointW(w[3],m)-jointW(w[0],m));norm+=dot3(q[m],q[m]);}
    if(norm<1e-20f)return nullptr;
    float3 p[4];for(int k=0;k<4;k++)p[k]=s.pos[id[k]];
    float3 e1=p[1]-p[0],e2=p[2]-p[0],e3=p[3]-p[0];
    float3 cof[3]={cross3(e2,e3)*rest.metric[6],cross3(e3,e1)*rest.metric[6],cross3(e1,e2)*rest.metric[6]};
    float J=dot3(e1,cof[0]),mu=rest.muV,lambda=rest.bulkV+mu/3.f;
    CompressionBarrier barrier=unifiedCompressionBarrier(J,s.minimum,lambda);
    float3 f[3]={e1*rest.b0.x+e2*rest.b1.x+e3*rest.b2.x,e1*rest.b0.y+e2*rest.b1.y+e3*rest.b2.y,e1*rest.b0.z+e2*rest.b1.z+e3*rest.b2.z};
    float3 gj[4],gradient[4],r[4],gq[4],G[3]={};float damp=0;
    if(s.damping>0){float3 r0=s.previousInverse[3*t],r1=s.previousInverse[3*t+1],r2=s.previousInverse[3*t+2];
        G[0]=f[0]*r0.x+f[1]*r1.x+f[2]*r2.x;G[1]=f[0]*r0.y+f[1]*r1.y+f[2]*r2.y;G[2]=f[0]*r0.z+f[1]*r1.z+f[2]*r2.z;
        damp=.5f*mu*s.damping/s.h;
        #pragma unroll
        for(int m=0;m<4;m++){r[m]=r0*q[m].x+r1*q[m].y+r2*q[m].z;gq[m]=G[0]*r[m].x+G[1]*r[m].y+G[2]*r[m].z;}
    }
    #pragma unroll
    for(int m=0;m<4;m++){
        gj[m]=cof[0]*(jointW(w[1],m)-jointW(w[0],m))+cof[1]*(jointW(w[2],m)-jointW(w[0],m))+cof[2]*(jointW(w[3],m)-jointW(w[0],m));
        gradient[m]=(f[0]*q[m].x+f[1]*q[m].y+f[2]*q[m].z)*mu+gj[m]*(lambda*(J-1)-mu+barrier.gradient);
        if(damp>0){float3 dq=mkf3(dot3(G[0],gq[m]),dot3(G[1],gq[m]),dot3(G[2],gq[m]))-r[m];gradient[m]=gradient[m]+(G[0]*dq.x+G[1]*dq.y+G[2]*dq.z)*damp;}
        v[3*m]+=gradient[m].x;v[3*m+1]+=gradient[m].y;v[3*m+2]+=gradient[m].z;
    }
    // Full cross-mode PSD pullback, including the objective viscosity Jacobian.
    // This is a GN search curvature, not a claim of the exact indefinite Hessian.
    #pragma unroll
    for(int i=0;i<12;i++){
        #pragma unroll
        for(int j=i;j<12;j++){
            int m=i/3,n=j/3,a=i%3,b=j%3;
            float h=(a==b?mu*dot3(q[m],q[n]):0)+(lambda+barrier.curvature)*compF3(gj[m],a)*compF3(gj[n],b);
            if(damp>0){float ggt=compF3(G[0],a)*compF3(G[0],b)+compF3(G[1],a)*compF3(G[1],b)+compF3(G[2],a)*compF3(G[2],b);
                h+=damp*(dot3(r[m],r[n])*ggt+compF3(gq[n],a)*compF3(gq[m],b));}
            v[jointH(i,j)]+=h;
        }
    }
    if(s.hardening>0)for(int j=0;j<4;j++)for(int k=j+1;k<4;k++){
        float3 e=p[j]-p[k];float l=len3(e),l0=unifiedRestLength(s,t,j,k),strain=fmaxf(0.f,l/l0-1.f);
        if(strain>0 && l>1e-12f){float s2=strain*strain,s4=s2*s2,force=3*mu*s.hardening*s4*strain/l0,h=15*mu*s.hardening*s4/(l0*l0);float3 n=e*(1.f/l);
            Symmetric3 H={};H.diagonal(force/l);H.outer(n,h-force/l);float4 a=w[j],b=w[k];jointPullback(v,make_float4(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w),n*force,H);}
    }
    return blocks+s.component[id[0]];
}
__global__ void k_JointAccumulate(const UnifiedState* state,const float4* weights,JointBlock* blocks){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;float v[JointTerms]={};
    JointBlock* b=t<s.activeCount?jointNodeTerms(s,s.activeNodes[t],weights,blocks,v):jointTetTerms(s,t-s.activeCount,weights,blocks,v);
    blockAccumulate(b,v,[](const void* p,const float* values){jointAdd(p,values);});
}
__device__ inline float4 jointWitnessWeights(int count,const int* ids,const float* values,const float4* weights){
    float4 w=make_float4(0,0,0,0);for(int j=0;j<count;j++){float4 a=weights[ids[j]];float v=values[j];w.x+=v*a.x;w.y+=v*a.y;w.z+=v*a.z;w.w+=v*a.w;}return w;
}
__device__ inline bool jointWitnessComponent(const UnifiedState& s,int count,const int* ids,JointBlock* blocks){
    int root=s.component[ids[0]];bool mixed=false;for(int j=1;j<count;j++)mixed|=s.component[ids[j]]!=root;
    if(mixed)for(int j=0;j<count;j++)atomicExch(&blocks[s.component[ids[j]]].disabled,1);
    return !mixed;
}
__global__ void k_JointInteractions(const UnifiedState* state,const float4* weights,JointBlock* blocks){
    const auto& s=*state;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<max(*s.contactCount,s.grips);k+=blockDim.x*gridDim.x){
        if(k<*s.contactCount){const auto& c=s.contacts[s.contactIds[k]];
            if(jointWitnessComponent(s,c.count,c.id,blocks)){float3 g;Symmetric3 h={};unifiedContact(s,c,g,h);float v[JointTerms]={};
                jointPullback(v,jointWitnessWeights(c.count,c.id,c.weight,weights),g,h);jointWarpAdd(blocks+s.component[c.id[0]],v);}}
        if(k<s.grips && s.points[k].valid){const auto& a=s.points[k];
            if(jointWitnessComponent(s,a.count,a.id,blocks)){float stiffness=s.gripK/a.complianceScale;Symmetric3 h={};h.diagonal(stiffness);float v[JointTerms]={};
                jointPullback(v,jointWitnessWeights(a.count,a.id,a.weight,weights),(unifiedGripPoint(a,s.pos)-gripTarget(a,s.tool->pose,s.minGap))*stiffness,h);jointWarpAdd(blocks+s.component[a.id[0]],v);}}
    }
}
__global__ void k_JointSolve(const UnifiedState* state,JointBlock* blocks){
    int i=unifiedActiveNode(*state,blockIdx.x*blockDim.x+threadIdx.x);if(!coarseRoot(*state,i))return;auto& b=blocks[i];
    if(b.disabled)return;double L[12][12]={},rhs[12],scale[12],maximum=0,norm=0;
    for(int j=0;j<12;j++){maximum=fmax(maximum,double(b.terms[jointH(j,j)]));norm+=double(b.terms[j])*b.terms[j];}
    if(norm==0){b.zero=1;return;}if(!(maximum>0) || !isfinite(norm+maximum)){b.disabled=1;return;}
    // Diagonal equilibration plus numerical regularization of the search system only.
    for(int j=0;j<12;j++){scale[j]=sqrt(fmax(double(b.terms[jointH(j,j)]),maximum*1e-10));rhs[j]=-b.terms[j]/scale[j];}
    for(int j=0;j<12;j++)for(int k=0;k<=j;k++){
        double a=double(b.terms[jointH(j,k)])/(scale[j]*scale[k])+(j==k?1e-7:0);
        for(int n=0;n<k;n++)a-=L[j][n]*L[k][n];
        if(j==k){if(!(a>0) || !isfinite(a)){b.disabled=1;return;}L[j][k]=sqrt(a);}else L[j][k]=a/L[k][k];
    }
    for(int j=0;j<12;j++){for(int k=0;k<j;k++)rhs[j]-=L[j][k]*rhs[k];rhs[j]/=L[j][j];}
    for(int j=11;j>=0;j--){for(int k=j+1;k<12;k++)rhs[j]-=L[k][j]*rhs[k];rhs[j]/=L[j][j];}
    double slope=0;for(int j=0;j<12;j++){b.step[j]=float(rhs[j]/scale[j]);slope+=double(b.terms[j])*b.step[j];}
    if(!(slope<0) || !isfinite(slope)){b.disabled=1;for(int j=0;j<12;j++)b.step[j]=0;}
}
__global__ void k_JointDirection(const UnifiedState* state,const float4* weights,const JointBlock* blocks,float3* direction){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;
    const auto& b=blocks[s.component[i]];float3 d=mkf3(0,0,0);
    if(!b.disabled)for(int j=0;j<4;j++)d=d+mkf3(b.step[3*j],b.step[3*j+1],b.step[3*j+2])*jointW(weights[i],j);
    direction[i]=d;
}
__device__ inline void jointBound(JointBlock* b,float a){
    unsigned mask=__activemask();int first=__ffs(mask)-1;auto key=(unsigned long long)b;unsigned value=__float_as_uint(a);
    if(__all_sync(mask,key==__shfl_sync(mask,key,first))){value=__reduce_min_sync(mask,value);if((threadIdx.x&31)!=first)return;}
    atomicMin((unsigned*)&b->scale,value);
}
__global__ void k_JointVolumeBound(const UnifiedState* state,const float3* direction,JointBlock* blocks){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=s.nt || !s.live[t])return;
    int id[4];unifiedTetIds(s,t,id);JointBlock* b=blocks+s.component[id[0]];if(b->disabled||b->zero)return;
    float3 p[4],d[4];for(int j=0;j<4;j++){p[j]=s.pos[id[j]];d[j]=direction[id[j]];}
    TetVolumePath J=unifiedVolumePath(p[1]-p[0],p[2]-p[0],p[3]-p[0],d[1]-d[0],d[2]-d[0],d[3]-d[0],unifiedKuhn(s,t).metric[6]);
    float floor=fminf(s.minimum,J.j0);
    if(!materialLineFeasible(1.f,floor,J.j0,J.j1,J.j2,J.j3)){
        float lo=0,hi=1;for(int k=0;k<24;k++){float mid=.5f*(lo+hi);if(materialLineFeasible(mid,floor,J.j0,J.j1,J.j2,J.j3))lo=mid;else hi=mid;}
        jointBound(b,.9f*lo);
    }
}
__device__ inline float3 jointWitnessDirection(int count,const int* ids,const float* w,const float3* direction){float3 d=mkf3(0,0,0);for(int j=0;j<count;j++)d=d+direction[ids[j]]*w[j];return d;}
__global__ void k_JointContactBound(const UnifiedState* state,const float3* direction,JointBlock* blocks,bool blade){
    const auto& s=*state;for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*s.contactCount;k+=blockDim.x*gridDim.x){
        const auto& c=s.contacts[s.contactIds[k]];auto* b=blocks+s.component[c.id[0]];if(b->disabled)continue;
        float change=dot3(jointWitnessDirection(c.count,c.id,c.weight,direction),c.plane.normal);if(change>=0)continue;
        ContactShape shape=bodyContactShape(s.tool,s.tool->pose,c.kind,s.thickness,s.minGap);
        float feature=blade?s.tool->pose.shaftR:shape.capsule?shape.radius:fminf(shape.half.x,fminf(shape.half.y,shape.half.z));
        jointBound(b,fminf(1.f,.5f*feature/(-change)));
    }
}
__device__ inline bool jointTrialAccepted(const JointBlock& b,int k){return !(b.line.invalid&(1<<k)) && isfinite(b.line.energy[k]) && b.line.energy[k]<-1e-10*fmax(1.,fabs(b.line.energy[0]));}
__device__ inline bool jointLineActive(const JointBlock& b,int pass){return !b.disabled && !b.zero && b.scale>0 && !(pass && jointTrialAccepted(b,1));}
__device__ inline float3 jointTrialDelta(const UnifiedState& s,int i,const float3* direction,float alpha){return (s.pos[i]+direction[i]*alpha)-s.pos[i];}
__device__ inline void jointLineNode(const UnifiedState& s,int i,const float3* direction,JointBlock* blocks,int pass){
    auto* b=blocks+s.component[i];if(!jointLineActive(*b,pass))return;
    float3 p=s.pos[i],g=mkf3(0,0,0);float curvature=0;double energy=0;
    if(s.invMass[i]>0){float inertia=1.f/(s.invMass[i]*s.h*s.h);float3 r=p-s.target[i];g=r*inertia;curvature=inertia;energy=.5*inertia*unifiedDot(r,r);}
    if(s.attach){float K=s.attach[i],D=unifiedSupportDamping(s,i);float3 r=p-s.attachRest[i],v=p-s.previous[i];g=g+r*K+v*D;curvature+=K+D;energy+=.5*(K*unifiedDot(r,r)+D*unifiedDot(v,v));}
    double values[5]={pass?0:energy};
    for(int k=pass?2:1;k<(pass?5:2);k++){float3 d=jointTrialDelta(s,i,direction,ldexpf(b->scale,1-k));values[k]=unifiedDot(g,d)+.5*curvature*unifiedDot(d,d);}
    if(s.neighbors && s.edgeSupport)for(int a=1;a<6;a+=2){int j=s.neighbors[6*i+a];if(j<0 || (s.active&&!s.active[j]) || s.edgeSupport[6*i+a])continue;
        float3 e=p-s.pos[j];float l=len3(e),strain=l-len3(s.restNeighbors[6*i+a]),K=.25f*(unifiedEdgeStiffness(s,i,a/2)+unifiedEdgeStiffness(s,j,a/2));
        if(!pass)values[0]+=K*double(strain)*strain;
        for(int k=pass?2:1;k<(pass?5:2);k++){float alpha=ldexpf(b->scale,1-k);float3 d=jointTrialDelta(s,i,direction,alpha)-jointTrialDelta(s,j,direction,alpha);float growth=unifiedLengthChange(e,d,l);values[k]+=K*double(growth)*(2.f*strain+growth);}
    }
    materialLineAdd(&b->line,values);
}
__device__ inline void jointLineTet(const UnifiedState& s,int t,const float3* direction,JointBlock* blocks,int pass){
    if(t>=s.nt || !s.live[t])return;int id[4];unifiedTetIds(s,t,id);auto* b=blocks+s.component[id[0]];if(!jointLineActive(*b,pass))return;
    float3 p[4];for(int j=0;j<4;j++)p[j]=s.pos[id[j]];double values[5]={};
    for(int k=pass?2:1;k<(pass?5:2);k++){
        float3 d[4];float energy;for(int j=0;j<4;j++)d[j]=jointTrialDelta(s,id[j],direction,ldexpf(b->scale,1-k));
        values[k]=unifiedTetEnergyChange(s,t,p,d,&energy);if(!pass)values[0]=energy;
        // Certify the actual float-rounded displacement over its complete cubic path.
        TetVolumePath J=unifiedVolumePath(p[1]-p[0],p[2]-p[0],p[3]-p[0],d[1]-d[0],d[2]-d[0],d[3]-d[0],unifiedKuhn(s,t).metric[6]);
        if(!materialLineFeasible(1.f,fminf(s.minimum,J.j0),J.j0,J.j1,J.j2,J.j3))atomicOr(&b->line.invalid,1<<k);
    }
    materialLineAdd(&b->line,values);
}
// Normal potential of the unchanged capped blade / unilateral Kelvin-Voigt law.
__device__ inline double jointNormalPotential(const UnifiedState& s,const SurfaceContact& c,float gap,float3 n,float& pressure){
    float K=unifiedContactStiffness(s,c),D=unifiedImpactStiffness(s,c),depth=fmaxf(0.f,-gap);pressure=0;
    if(s.cutPhase && c.kind<3 && bladeYields(*s.cutPhase)){
        float cap=s.cutResistance*K/s.contactK,limit=K>0?cap/K:0;pressure=fminf(K*depth,cap);
        return depth<=limit?.5*double(K)*depth*depth:double(cap)*(depth-.5f*limit);
    }
    float effective=K+D,compression=fmaxf(0.f,depth-D*unifiedPreviousDepth(c,n)/fmaxf(effective,1e-20f));pressure=effective*compression;
    return .5*double(effective)*compression*compression;
}
__device__ inline double jointFrictionPotential(const UnifiedState& s,const SurfaceContact& c,float distance,float pressure){
    float K=unifiedContactStiffness(s,c),limit=s.muS*pressure;if(K<=0 || limit<=0)return 0;
    float boundary=limit/K;if(distance<=boundary)return .5*double(K)*distance*distance;
    float slip=distance-boundary;
    if(s.cutPhase && c.kind<3)return .5*double(limit)*boundary+double(s.muD)*pressure*slip;
    return .5*double(limit)*boundary+double(pressure)*(s.muD*slip+(s.muS-s.muD)*boundary*(1-boundary/distance));
}
__device__ inline void jointLineInteractions(const UnifiedState& s,const float3* direction,JointBlock* blocks,int pass,int start,int stride){
    for(int i=start;i<max(*s.contactCount,s.grips);i+=stride){
        if(i<s.grips && s.points[i].valid){const auto& a=s.points[i];auto* b=blocks+s.component[a.id[0]];
            if(jointLineActive(*b,pass)){float3 e=unifiedGripPoint(a,s.pos)-gripTarget(a,s.tool->pose,s.minGap);float K=s.gripK/a.complianceScale;double values[5]={pass?0:.5*K*unifiedDot(e,e)};
                for(int k=pass?2:1;k<(pass?5:2);k++){float3 d=mkf3(0,0,0);for(int j=0;j<a.count;j++)d=d+jointTrialDelta(s,a.id[j],direction,ldexpf(b->scale,1-k))*a.weight[j];values[k]=K*(unifiedDot(e,d)+.5*unifiedDot(d,d));}materialLineAdd(&b->line,values);}}
        if(i>=*s.contactCount)continue;const auto& c=s.contacts[s.contactIds[i]];auto* b=blocks+s.component[c.id[0]];if(!jointLineActive(*b,pass))continue;
        float3 x=unifiedPoint(c,s.pos),n;float gap=unifiedGap(s,c,x,n),pressure;
        double oldNormal=jointNormalPotential(s,c,gap,n,pressure);float3 slip=unifiedSlip(s,c,x,n);
        double oldFriction=jointFrictionPotential(s,c,len3(slip),pressure),values[5]={pass?0:oldNormal+oldFriction};
        // Freeze the normal and friction load within this correction, as in a contact descent linearization.
        // Coulomb friction is dissipative; this merit is not a global conservative potential.
        for(int k=pass?2:1;k<(pass?5:2);k++){float3 d=mkf3(0,0,0);for(int j=0;j<c.count;j++)d=d+jointTrialDelta(s,c.id[j],direction,ldexpf(b->scale,1-k))*c.weight[j];
            float unused;double normal=jointNormalPotential(s,c,gap+dot3(d,n),n,unused);float3 tangent=slip+d-n*dot3(d,n);
            values[k]=normal-oldNormal+jointFrictionPotential(s,c,len3(tangent),pressure)-oldFriction;
        }materialLineAdd(&b->line,values);
    }
}
__global__ void k_JointLine(const UnifiedState* state,const float3* direction,JointBlock* blocks,int pass){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<s.activeCount)jointLineNode(s,s.activeNodes[t],direction,blocks,pass);else jointLineTet(s,t-s.activeCount,direction,blocks,pass);
    jointLineInteractions(s,direction,blocks,pass,t,blockDim.x*gridDim.x);
    if(s.secondary)jointLineInteractions(*s.secondary,direction,blocks,pass,t,blockDim.x*gridDim.x);
}
__global__ void k_JointSelect(const UnifiedState* state,JointBlock* blocks){
    int i=unifiedActiveNode(*state,blockIdx.x*blockDim.x+threadIdx.x);if(!coarseRoot(*state,i))return;auto& b=blocks[i];
    if(!b.disabled && b.zero){b.accepted=1;b.scale=0;return;}
    if(!b.disabled)for(int k=1;k<5;k++)if(jointTrialAccepted(b,k)){b.accepted=1;b.scale=ldexpf(b.scale,1-k);return;}
    b.scale=0;
}
__global__ void k_JointApply(const UnifiedState* state,const float3* direction,const JointBlock* blocks){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0 || s.invMass[i]<=0)return;const auto& b=blocks[s.component[i]];
    if(b.accepted && b.scale>0)s.pos[i]=s.pos[i]+direction[i]*b.scale;
}
// Original four sweeps remain a component-local fallback; accepted components receive zero weights.
__global__ void k_JointFallbackWeights(const UnifiedState* state,const JointBlock* blocks){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i>=0 && blocks[s.component[i]].accepted)s.coarseWeight[i]=0;
}
