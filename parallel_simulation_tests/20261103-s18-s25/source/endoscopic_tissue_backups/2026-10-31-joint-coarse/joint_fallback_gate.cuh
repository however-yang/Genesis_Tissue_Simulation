// Experimental scheduling only. Kernel bodies copied from the protected baseline.
// Null gate retains original execution; a zero gate skips work whose weights are all zero.
__global__ void k_CoarseWeightsGated(const int* gate,UnifiedState* state,int mode=0){
    if(gate && !*gate)return;
    const UnifiedState& s=*state;int k=blockIdx.x*blockDim.x+threadIdx.x,i=unifiedActiveNode(s,k);
    if(k==0)state->coarseMode=mode;
    if(i<0)return;
    if(s.coupledReady && mode!=6 && (!(s.graspComponents[s.component[i]]&7) && (!(s.graspComponents[s.component[i]]&8) || mode<1 || mode>4))){s.coarseWeight[i]=0;return;}
    if((mode>0 && mode<5)||mode==6){
        if(mode<5 && (s.graspComponents[s.component[i]]&5)){
            // A local joint or jaw pressure must not drive constant/affine whole-
            // island modes before the deformable solve can distribute its load.
            // Use a broad, decaying basis only on the held/jaw-loaded component.
            // These are solve directions in the SAME energy, not position
            // targets or forces; free/cut pieces retain the original modes.
            float3 d=s.pos[i]-(s.tool->pose.jawHinge+s.patchOffset);
            float length=fmaxf(s.patchFalloff,1e-5f),r=fmaxf(0.f,len3(d)-s.patchCoreRadius);
            float w=1.f/(1.f+r*r/(length*length));
            w*=w; // Smooth noncompact tail without a near-rigid far-field mode.
            if(mode>1)w*=compF3(d,mode-2)*rsqrtf(length*length+dot3(d,d));
            state->coarseWeight[i]=s.invMass[i]>0?w:0;return;
        }
        float w=1;
        if(mode==2)w=float(i%s.nx)/fmaxf(1.f,float(s.nx-1))-.5f;
        if(mode==3)w=float((i/s.nx)%s.ny)/fmaxf(1.f,float(s.ny-1))-.5f;
        if(mode==4)w=float(i/(s.nx*s.ny))/fmaxf(1.f,float(s.n/(s.nx*s.ny)-1))-.5f;
        state->coarseWeight[i]=s.invMass[i]>0?w:0;return;
    }
    float3 center=s.tool->pose.jawHinge+s.patchOffset;
    float r=len3(s.pos[i]-center);
    // Only the captured core shares a rigid transport direction. The old
    // 1.5-jaw-length plateau transported a large tissue ball as one block
    // within the finite iteration budget; retain a broad but graded tail.
    float t=fminf(1.f,fmaxf(0.f,(r-s.patchCoreRadius)/fmaxf(s.patchFalloff,1e-5f)));
    s.coarseWeight[i]=s.invMass[i]>0?1-t*t*(3-2*t):0;
}
__global__ void k_InitSafeCoarseGated(const int* gate,const UnifiedState* state){
    if(gate && !*gate)return;
    const UnifiedState& s=*state;int k=blockIdx.x*blockDim.x+threadIdx.x,i=unifiedActiveNode(s,k);
    if(k==0)*s.safeJoined=0;
    if(i<0)return;s.safeCluster[i]=i;s.safeAggregate[i]={};
}
__global__ void k_JoinSafeCoarseGated(const int* gate,const UnifiedState* state){
    if(gate && !*gate)return;
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=s.nt || !s.live[t] || s.minimum<=0)return;
    int id[4];unifiedTetIds(s,t,id);
    float3 a=s.pos[id[0]];
    float J=dot3(s.pos[id[1]]-a,cross3(s.pos[id[2]]-a,s.pos[id[3]]-a))*unifiedKuhn(s,t).metric[6];
    if(J>s.minimum+fmaxf(1e-5f,1e-3f*s.minimum))return;
    *s.safeJoined=1;
    for(int k=1;k<4;k++)while(true){
        int x=unifiedRoot(s.safeCluster,id[0]),y=unifiedRoot(s.safeCluster,id[k]);
        if(x==y)break;int hi=max(x,y),lo=min(x,y);if(atomicCAS(s.safeCluster+hi,hi,lo)==hi)break;
    }
}
__global__ void k_FlattenSafeCoarseGated(const int* gate,const UnifiedState* state){
    if(gate && !*gate)return;
    const UnifiedState& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=0 && *s.safeJoined)s.safeCluster[i]=unifiedRoot(s.safeCluster,i);
}
__global__ void k_AccumulateSafeCoarseGated(const int* gate,const UnifiedState* state){
    if(gate && !*gate)return;
    const UnifiedState& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);
    if(i<0 || !*s.safeJoined)return;
    SafeCoarseCluster* c=s.safeAggregate+s.safeCluster[i];
    atomicAdd(&c->weight,s.coarseWeight[i]);atomicAdd(&c->count,1);
    if(s.invMass[i]<=0)atomicExch(&c->pinned,1);
    if(s.grips>0 && s.gripDegree[i]>0){atomicAdd(&c->gripWeight,s.coarseWeight[i]);atomicAdd(&c->grips,1);}
}
__global__ void k_FinishSafeCoarseGated(const int* gate,const UnifiedState* state){
    if(gate && !*gate)return;
    const UnifiedState& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);
    if(i<0 || !*s.safeJoined)return;
    const SafeCoarseCluster& c=s.safeAggregate[s.safeCluster[i]];
    s.coarseWeight[i]=c.pinned?0.f:c.grips?c.gripWeight/c.grips:c.weight/fmaxf(1.f,float(c.count));
}
__global__ void k_CoarseBeginGated(const int* gate,const UnifiedState* state,CoarseBlock* b){
    if(gate && !*gate)return;
    const UnifiedState& s=*state;int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(s.coarseMode){int i=unifiedActiveNode(s,k);if(coarseRoot(s,i)){b[i]={};b[i].scale=1;}return;}
    if(k)return;float inertia=1.f/(s.tool->inverseMass*s.h*s.h);
    *b={};b->gradient=(s.tool->pose.jawHinge-s.toolTarget)*inertia;b->xx=b->yy=b->zz=inertia;b->scale=1;
}
__global__ void k_CoarseAccumulateGated(const int* gate,const UnifiedState* state,CoarseBlock* block){
    if(gate && !*gate)return;
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    float3 g=mkf3(0,0,0);Symmetric3 H={};float scale=1;
    CoarseBlock* target=t<s.activeCount?coarseNodeTerms(s,s.activeNodes[t],block,g,H):
        coarseTetTerms(s,t-s.activeCount,block,false,g,H,scale);
    coarseAccumulateBlock(target,g,H);
    coarseContactGripTerms(s,block,t,gridDim.x*blockDim.x);
}
__global__ void k_CoarseSolveGated(const int* gate,const UnifiedState* state,CoarseBlock* block){
    if(gate && !*gate)return;
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(state->coarseMode){int i=unifiedActiveNode(*state,k);if(!coarseRoot(*state,i))return;block+=i;}else if(k)return;
    Symmetric3 H={block->xx,block->yy,block->zz,block->xy,block->xz,block->yz};
    const CoupledTool& b=*state->tool;
    if(state->coarseMode==7 && block->pins==-1){
        // Exact minimizer of inertial energy in the cylinder's rotational
        // symmetry. No normal contact or isotropic material work is possible
        // in this one-dimensional null space.
        float3 axis=contactUnit(b.pose.shaftB-b.pose.shaftA,mkf3(1,0,0));
        block->step=axis*atan2f(block->xy,block->xx);return;
    }
    if(!state->coarseMode && b.guided){
        float3 a=b.guideAxis;float den=H.xx*a.x*a.x+H.yy*a.y*a.y+H.zz*a.z*a.z+2*(H.xy*a.x*a.y+H.xz*a.x*a.z+H.yz*a.y*a.z);
        float d=den>1e-20f?-dot3(block->gradient,a)/den:0;
        float current=dot3(b.pose.jawHinge-b.guideOrigin,a);
        block->step=a*(fminf(b.guideLength,fmaxf(0.f,current+d))-current);
    }else block->step=H.solve(block->gradient)*(-1.f);
}
__global__ void k_CoarseBoundGated(const int* gate,const UnifiedState* state,CoarseBlock* block){
    if(gate && !*gate)return;
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    float3 g;Symmetric3 H;float scale=1;
    CoarseBlock* target=coarseTetTerms(s,t,block,true,g,H,scale);
    // The bound adds one value per warp, which the warp-level coarseBound sums at little cost (blockAccumulate).
    if(target)coarseBound(target,scale);
    coarseContactBoundTerms(s,block,t,gridDim.x*blockDim.x);
}
__global__ void k_CoarseApplyGated(const int* gate,const UnifiedState* state,const CoarseBlock* block){
    if(gate && !*gate)return;
    const UnifiedState& s=*state;int k=blockIdx.x*blockDim.x+threadIdx.x,i=unifiedActiveNode(s,k);
    if(s.coarseMode){if(i<0)return;block+=s.component[i];if(s.coarseMode==6 && block->external)return;}
    else if(i<0 && k)return;
    float3 d=block->step*block->scale;
    if(!isfinite(d.x+d.y+d.z))return;
    if(i>=0)s.pos[i]=s.pos[i]+d*s.coarseWeight[i];
    if(k==0 && !s.coarseMode)s.tool->pose=shiftTool(s.tool->pose,d);
}
__global__ void k_PressureLineClearGated(const int* gate,const UnifiedState* state,CoarseBlock* blocks){
    if(gate && !*gate)return;
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);
    if(coarseRoot(s,i) && pressureLineActive(s))blocks[i].gradient=mkf3(0,0,0);
}
__global__ void k_PressureLineGated(const int* gate,const UnifiedState* state,CoarseBlock* blocks,int pass){
    if(gate && !*gate)return;
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<s.activeCount)pressureLineNode(s,blocks,s.activeNodes[t],pass);else pressureLineTet(s,blocks,t-s.activeCount,pass);
    pressureLineContacts(s,blocks,pass,t,gridDim.x*blockDim.x);
}
__global__ void k_PressureLineSelectGated(const int* gate,const UnifiedState* state,CoarseBlock* blocks){
    if(gate && !*gate)return;
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(!coarseRoot(s,i) || !pressureLineActive(s))return;
    auto& b=blocks[i];float scale=b.gradient.x<=0?1.f:b.gradient.y<=0?.5f:b.gradient.z<=0?.25f:0.f;b.scale*=scale;
}
__global__ void k_BladeCoarseGated(const int* gate,const UnifiedState* state,const UnifiedState* blade,CoarseBlock* blocks,bool bound){
    if(gate && !*gate)return;
    const UnifiedState& s=*state;const UnifiedState& cut=*blade;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*cut.contactCount;k+=blockDim.x*gridDim.x){
        const SurfaceContact& c=cut.contacts[cut.contactIds[k]];
        CoarseBlock* b=s.coarseMode?blocks+s.component[c.id[0]]:blocks;
        if(s.coarseMode==6)atomicExch(&b->external,1);
        float w=0;for(int j=0;j<c.count;j++)w+=c.weight[j]*s.coarseWeight[c.id[j]];
        if(bound){float change=dot3(b->step,c.plane.normal)*w;if(change<0)coarseBound(b,fminf(1.f,.5f*cut.tool->pose.shaftR/(-change)));}
        else {float3 f;Symmetric3 H={};unifiedContact(cut,c,f,H);H.xx*=w*w;H.yy*=w*w;H.zz*=w*w;H.xy*=w*w;H.xz*=w*w;H.yz*=w*w;coarseAccumulate(b,f*w,H);}
    }
}
__global__ void k_JointFallbackWeightsGated(const int* gate,const UnifiedState* state,const JointBlock* blocks){
    if(gate && !*gate)return;
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i>=0 && blocks[s.component[i]].accepted)s.coarseWeight[i]=0;
}

__global__ void k_JointGateReset(int* gate){*gate=0;}
__global__ void k_JointGateCollect(const UnifiedState* state,const JointBlock* blocks,int* gate){int i=unifiedActiveNode(*state,blockIdx.x*blockDim.x+threadIdx.x);if(coarseRoot(*state,i)&&!blocks[i].accepted)atomicExch(gate,1);}
__global__ void k_JointGateFinish(UnifiedState* state,CoarseBlock* blocks,const int* gate){
 if(*gate)return;const auto& s=*state;int k=blockIdx.x*blockDim.x+threadIdx.x,i=unifiedActiveNode(s,k);if(k==0)state->coarseMode=4;if(i<0)return;
 s.coarseWeight[i]=0;if(coarseRoot(s,i)){blocks[i]={};blocks[i].scale=1;}
 // safeCluster/safeAggregate/safeJoined are not read again before k_InitSafeCoarse resets them.
}
