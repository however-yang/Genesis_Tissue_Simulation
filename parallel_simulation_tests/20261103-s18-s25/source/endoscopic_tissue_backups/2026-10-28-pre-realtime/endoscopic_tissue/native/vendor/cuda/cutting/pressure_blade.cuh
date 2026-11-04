// A second instrument shares the material objective with the grasper. Its
// normal reaction triggers fracture; after initiation the normal energy has
// a lower, finite yield force. No force exists at a separating contact.
static UnifiedState* d_bladeState=nullptr;
static CoupledTool* d_bladeBody=nullptr;
static SurfaceContact* d_bladeContacts=nullptr;
static SurfaceContact* d_bladeBackCache=nullptr;
static int* d_bladeBackCacheSlots=nullptr;
static int *d_bladeIds=nullptr,*d_bladeCount=nullptr,*d_bladeDegree=nullptr,*d_bladePhase=nullptr;
static SurfaceContact* d_bladeManifold=nullptr;
static int *d_bladeManifoldIds=nullptr,*d_bladeManifoldCount=nullptr,*d_bladeOwners=nullptr,*d_bladeEdgeCount=nullptr;
static float* d_bladeQuadrature=nullptr;
// squeezeGap: scissors jaw gap when both blades first loaded the tissue in the
// current pressing gesture, valid while squeezing is set. strokeGap: scissors
// jaw gap at which the current closing stroke began.
struct BladeHistory {float3 entryAxis;float freeTravel;int fractured,hadContact;int squeezing;float squeezeGap,strokeGap;};
static float3* d_bladeSweep=nullptr;

static BladeHistory* d_bladeHistory=nullptr;
extern "C" const int* xpbd_blade_had_contact_device(){
    return d_bladeHistory ? reinterpret_cast<const int*>(reinterpret_cast<const char*>(d_bladeHistory)+offsetof(BladeHistory,hadContact)) : nullptr;
}
static float* d_bladeResult=nullptr;
// Scissors loads: current left/right blade, left/right/total at the latest
// fracture, fracture count. Persists across frames until the scissors reset.
static float* d_scissorsLoads=nullptr;
static int g_bladeCapacity=0;
static XpbdBladeState g_blade={};
static XpbdScissorsState g_scissors={};
static bool g_scissorsMode=false;
static bool g_knifeBody=false;
static XpbdScissorsBackBox g_knifeBoxes[2]={};
// Scissors wrist capsule: front of the hinge, length behind it, radius (0: none).
static float g_scissorsWrist[3]={0,0,0};
static bool scissorsWrist(){return g_scissorsMode && g_scissorsWrist[2]>0;}
// Contact slots per surface triangle: knife or edges 0..2, backs 3..4, wrist 5.
constexpr int BLADE_KIND_SLOTS=6;
static int bladeShapeCount(){
    return g_scissorsMode?2+(g_scissors.backContact?2:0)+(scissorsWrist()?1:0):(g_knifeBody?3:1);
}
static int bladeShapeKind(int index){
    if(!g_scissorsMode)return index?index+2:0;
    return index<2 || (g_scissors.backContact && index<4)?index+1:5;
}
static void queryBladeDepth(int triangles,float* output){
    if(!g_blade.enabled || !d_bladeBody)return;
    for(int index=0;index<bladeShapeCount();index++)
        k_SurfaceDepth<<<Gr(triangles),64>>>(triangles,ContactShape{},g_surfaceExpanded,output,d_bladeBody,bladeShapeKind(index));
}
extern "C" int xpbd_knife_body_depth(int triangles,float* output){
    if(!output)return XPBD_ERR_BAD_ARG;*output=0;
    if(!g_knifeBody || !g_blade.enabled || !d_bladeBody || triangles<=0)return 0;
    if(!d_toolClearance){cudaError_t e=cudaMalloc(&d_toolClearance,9*sizeof(float));if(e!=cudaSuccess)return -(int)e;}
    float* scratch=d_toolClearance+8;cudaError_t e=cudaMemsetAsync(scratch,0,sizeof(float));if(e!=cudaSuccess)return -(int)e;
    for(int kind=3;kind<5;kind++)k_SurfaceDepth<<<Gr(triangles),64>>>(triangles,ContactShape{},g_surfaceExpanded,scratch,d_bladeBody,kind);
    return -(int)cudaMemcpy(output,scratch,sizeof(float),cudaMemcpyDeviceToHost);
}
extern "C" int xpbd_blade_sweep_count(){return g_blade.enabled?(g_scissorsMode?2:1):0;}
extern "C" const float3* xpbd_blade_sweep_device(){return g_blade.enabled?d_bladeSweep:nullptr;}
__global__ void k_BladeSweep(const CoupledTool* body,float3* sweep,const BladeHistory* history){
    if(body->scissors){for(int edge=0;edge<2;edge++){
        ContactShape a=bodyContactShape(body,body->previous,edge+1,0,0),b=bodyContactShape(body,body->pose,edge+1,0,0);
        sweep[4*edge]=a.a;sweep[4*edge+1]=a.b;sweep[4*edge+2]=b.a;sweep[4*edge+3]=b.b;
    }
        // The shear plane both edges lie in: its normal, then one of its points.
        // Then the blade frame: hinge, axis, opening normal, (tip gap, blade
        // length, tip gap where the stroke began), and both edges in the closed
        // hinge frame.
        float3 u=contactUnit(body->pose.jawAxis,mkf3(0,0,1)),n=contactUnit(body->pose.jawNormal,mkf3(0,1,0));
        sweep[8]=cross3(u,n);sweep[9]=sweep[6];
        sweep[10]=body->pose.jawHinge;sweep[11]=u;sweep[12]=n;sweep[13]=mkf3(body->pose.jawGap,body->pose.jawLen,history->strokeGap);
        sweep[14]=body->edgeA[0];sweep[15]=body->edgeB[0];sweep[16]=body->edgeA[1];sweep[17]=body->edgeB[1];
    }else{
        // Contact acts on the finite edge capsule. Fracture must start at its
        // leading surface, not lag behind it by a radius at the centre line.
        // Keep one direction over a stroke so a pause cannot sweep a fictitious
        // ribbon across the blade's own diameter.
        float3 direction=len3(history->entryAxis)>.5f?history->entryAxis:body->guideAxis;
        direction=direction-body->pose.jawAxis*dot3(direction,body->pose.jawAxis);
        float3 offset=contactUnit(direction,mkf3(0,-1,0))*body->pose.shaftR;
        sweep[0]=body->previous.shaftA+offset;sweep[1]=body->previous.shaftB+offset;
        sweep[2]=body->pose.shaftA+offset;sweep[3]=body->pose.shaftB+offset;
    }}
static XpbdToolState g_bladeCommand={},g_bladeInitial={};
static void invalidateBladeGraphs(){
    if(g_unifiedGraph)cudaGraphExecDestroy(g_unifiedGraph);g_unifiedGraph=nullptr;if(g_unifiedRecoveryGraph)cudaGraphExecDestroy(g_unifiedRecoveryGraph);g_unifiedRecoveryGraph=nullptr;
    if(g_unifiedContactGraph)cudaGraphExecDestroy(g_unifiedContactGraph);g_unifiedContactGraph=nullptr;
}
static void freeBlade(){
    cudaFree(d_bladeSweep);d_bladeSweep=nullptr;
    cudaFree(d_bladeState);d_bladeState=nullptr;cudaFree(d_bladeBody);d_bladeBody=nullptr;
    cudaFree(d_bladeContacts);d_bladeContacts=nullptr;cudaFree(d_bladeIds);d_bladeIds=nullptr;
    cudaFree(d_bladeBackCache);d_bladeBackCache=nullptr;cudaFree(d_bladeBackCacheSlots);d_bladeBackCacheSlots=nullptr;
    cudaFree(d_bladeCount);d_bladeCount=nullptr;cudaFree(d_bladeDegree);d_bladeDegree=nullptr;
    cudaFree(d_bladePhase);d_bladePhase=nullptr;cudaFree(d_bladeResult);d_bladeResult=nullptr;
    cudaFree(d_scissorsLoads);d_scissorsLoads=nullptr;
    cudaFree(d_bladeHistory);d_bladeHistory=nullptr;g_bladeCapacity=0;g_blade={};g_scissors={};g_scissorsMode=false;
    g_knifeBody=false;g_knifeBoxes[0]={};g_knifeBoxes[1]={};
    cudaFree(d_bladeManifold);d_bladeManifold=nullptr;cudaFree(d_bladeManifoldIds);d_bladeManifoldIds=nullptr;
    cudaFree(d_bladeManifoldCount);d_bladeManifoldCount=nullptr;cudaFree(d_bladeOwners);d_bladeOwners=nullptr;
    cudaFree(d_bladeQuadrature);d_bladeQuadrature=nullptr;cudaFree(d_bladeEdgeCount);d_bladeEdgeCount=nullptr;
}
// Setup-only boxes in the knife frame (edge axis, up, width). Neither cuts.
extern "C" int xpbd_set_knife_geometry(int enabled,const float* bounds){
    if(g_blade.enabled || g_scissorsMode)return XPBD_ERR_BAD_ARG;
    if(enabled){
        if(!bounds)return XPBD_ERR_BAD_ARG;
        for(int j=0;j<24;j++)if(!std::isfinite(bounds[j]))return XPBD_ERR_BAD_ARG;
        XpbdScissorsBackBox boxes[2];std::memcpy(boxes,bounds,sizeof(boxes));
        for(const auto& box:boxes)if(box.halfSize.x<=0 || box.halfSize.y<=0 || box.halfSize.z<=0 ||
            fabsf(len3(box.axisX)-1)>.001f || fabsf(len3(box.axisY)-1)>.001f ||
            fabsf(dot3(box.axisX,box.axisY))>.001f)return XPBD_ERR_BAD_ARG;
        std::memcpy(g_knifeBoxes,boxes,sizeof(boxes));
    }
    g_knifeBody=enabled!=0;g_bladeCapacity=0;invalidateBladeGraphs();return 0;
}
__global__ void k_ConfigureKnifeBody(CoupledTool* body,int enabled,XpbdScissorsBackBox blade,XpbdScissorsBackBox handle){
    body->knifeBody=enabled;body->scissors=0;body->backContact=enabled;
    body->backs[0]=blade;body->backs[1]=handle;
}
extern "C" int xpbd_set_blade(const XpbdBladeState* blade){
    if(!blade)return XPBD_ERR_BAD_ARG;
    if(g_scissorsMode && g_blade.enabled)return XPBD_ERR_BAD_ARG;
    g_scissorsMode=false;
    if(blade->enabled){
        const float input[]={blade->a.x,blade->a.y,blade->a.z,blade->b.x,blade->b.y,blade->b.z,blade->radius,blade->startForce,blade->cutForce,blade->driveStiffness,blade->mass,blade->friction};
        for(float v:input)if(!std::isfinite(v))return XPBD_ERR_BAD_ARG;
        if(blade->friction<0)return XPBD_ERR_BAD_ARG;
    }
    if(blade->enabled && (!(blade->radius>0) || !(blade->startForce>blade->cutForce) || !(blade->cutForce>0) || !(blade->driveStiffness>0) || !(blade->mass>0)))return XPBD_ERR_BAD_ARG;
    if(g_blade.startForce!=blade->startForce)invalidateBladeGraphs();
    if(g_blade.enabled!=blade->enabled){invalidateBladeGraphs();if(d_bladeBody)cudaMemset(d_bladeBody,0,sizeof(CoupledTool));if(d_bladePhase)cudaMemset(d_bladePhase,0,sizeof(int));if(d_bladeHistory)cudaMemset(d_bladeHistory,0,sizeof(BladeHistory));if(d_bladeResult)cudaMemset(d_bladeResult,0,6*sizeof(float));}
    XpbdToolState t={};t.shaftA=blade->a;t.shaftB=blade->b;t.shaftR=blade->radius;
    t.jawHinge=t.jawMid=(blade->a+blade->b)*.5f;t.jawAxis=contactUnit(blade->b-blade->a,mkf3(1,0,0));
    t.jawNormal=contactUnit(mkf3(0,1,0)-t.jawAxis*t.jawAxis.y,mkf3(1,0,0));
    if(!g_blade.enabled)g_bladeInitial=t;
    g_bladeCommand=t;g_blade=*blade;return 0;
}
extern "C" const int* xpbd_blade_phase_device(){return g_blade.enabled?d_bladePhase:nullptr;}
extern "C" int xpbd_get_blade(float* values){
    if(!values || !d_bladeBody)return XPBD_ERR_BAD_ARG;
    CoupledTool body;int phase;float reaction[6];
    cudaError_t e=cudaMemcpy(&body,d_bladeBody,sizeof(body),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(&phase,d_bladePhase,sizeof(int),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(reaction,d_bladeResult,6*sizeof(float),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return -(int)e;
    values[0]=body.pose.jawHinge.x;values[1]=body.pose.jawHinge.y;values[2]=body.pose.jawHinge.z;
    for(int k=0;k<4;k++)values[3+k]=reaction[k];values[7]=(float)phase;values[8]=reaction[4];values[9]=reaction[5];return 0;
}
static cudaError_t allocateBlade(){
    cudaError_t e=cudaSuccess;
    if(!d_bladeSweep)e=cudaMalloc(&d_bladeSweep,18*sizeof(float3));
    if(e==cudaSuccess&&!d_bladeState)e=cudaMalloc(&d_bladeState,sizeof(UnifiedState));
    if(e==cudaSuccess&&!d_bladeBody){e=cudaMalloc(&d_bladeBody,sizeof(CoupledTool));if(e==cudaSuccess)e=cudaMemset(d_bladeBody,0,sizeof(CoupledTool));}
    if(e==cudaSuccess&&!d_bladeCount)e=cudaMalloc(&d_bladeCount,sizeof(int));
    if(e==cudaSuccess&&!d_bladePhase){e=cudaMalloc(&d_bladePhase,sizeof(int));if(e==cudaSuccess)e=cudaMemset(d_bladePhase,0,sizeof(int));}
    if(e==cudaSuccess&&!d_bladeResult)e=cudaMalloc(&d_bladeResult,6*sizeof(float));
    if(e==cudaSuccess&&!d_scissorsLoads){e=cudaMalloc(&d_scissorsLoads,7*sizeof(float));if(e==cudaSuccess)e=cudaMemset(d_scissorsLoads,0,7*sizeof(float));}
    if(e==cudaSuccess&&!d_bladeHistory){e=cudaMalloc(&d_bladeHistory,sizeof(BladeHistory));if(e==cudaSuccess)e=cudaMemset(d_bladeHistory,0,sizeof(BladeHistory));}
    if(e==cudaSuccess&&!d_bladeDegree)e=cudaMalloc(&d_bladeDegree,g_N*sizeof(int));
    if(e==cudaSuccess&&!d_bladeBackCache){
        e=cudaMalloc(&d_bladeBackCache,CONTACT_CACHE_SIZE*sizeof(SurfaceContact));
        if(e==cudaSuccess)e=cudaMalloc(&d_bladeBackCacheSlots,CONTACT_CACHE_SIZE*sizeof(int));
        if(e==cudaSuccess)e=cudaMemset(d_bladeBackCacheSlots,0xff,CONTACT_CACHE_SIZE*sizeof(int));
    }
    if(e==cudaSuccess&&!d_bladeManifoldCount)e=cudaMalloc(&d_bladeManifoldCount,sizeof(int));
    if(e==cudaSuccess&&!d_bladeEdgeCount)e=cudaMalloc(&d_bladeEdgeCount,sizeof(int));
    if(e==cudaSuccess&&g_bladeCapacity<g_surfaceCapacity){
        invalidateBladeGraphs();cudaFree(d_bladeContacts);d_bladeContacts=nullptr;cudaFree(d_bladeIds);d_bladeIds=nullptr;g_bladeCapacity=0;
        e=cudaMalloc(&d_bladeContacts,BLADE_KIND_SLOTS*g_surfaceCapacity*sizeof(SurfaceContact));
        if(e==cudaSuccess)e=cudaMemset(d_bladeContacts,0,BLADE_KIND_SLOTS*g_surfaceCapacity*sizeof(SurfaceContact));
        if(e==cudaSuccess)e=cudaMalloc(&d_bladeIds,BLADE_KIND_SLOTS*g_surfaceCapacity*sizeof(int));
        cudaFree(d_bladeManifold);cudaFree(d_bladeManifoldIds);cudaFree(d_bladeOwners);cudaFree(d_bladeQuadrature);
        d_bladeManifold=nullptr;d_bladeManifoldIds=d_bladeOwners=nullptr;d_bladeQuadrature=nullptr;
        // One witness per capsule (edges, wrist) and up to 12 face witnesses per back box.
        size_t capacity=(size_t)(g_scissorsMode && g_scissors.backContact?26+(scissorsWrist()?1:0):g_knifeBody?25:3)*g_surfaceCapacity;
        if(e==cudaSuccess)e=cudaMalloc(&d_bladeManifold,capacity*sizeof(SurfaceContact));
        if(e==cudaSuccess)e=cudaMalloc(&d_bladeManifoldIds,capacity*sizeof(int));
        if(e==cudaSuccess)e=cudaMalloc(&d_bladeOwners,capacity*sizeof(int));
        if(e==cudaSuccess)e=cudaMalloc(&d_bladeQuadrature,capacity*sizeof(float));
        if(e==cudaSuccess)g_bladeCapacity=g_surfaceCapacity;
    }
    return e;
}
extern "C" int xpbd_set_scissors_wrist(float front,float length,float radius){
    // The contact slots and manifold capacity follow the shape count: set before enabling the scissors.
    if(g_blade.enabled)return XPBD_ERR_BAD_ARG;
    if(!std::isfinite(front) || !std::isfinite(length) || !std::isfinite(radius) || radius<0 || (radius>0 && !(length+front>0)))
        return XPBD_ERR_BAD_ARG;
    g_scissorsWrist[0]=front;g_scissorsWrist[1]=length;g_scissorsWrist[2]=radius;g_bladeCapacity=0;invalidateBladeGraphs();return 0;
}
__global__ void k_ConfigureScissors(CoupledTool* body,XpbdScissorsState state,float radius,float wristFront,float wristLength,float wristRadius){
    body->scissors=1;body->scissorsCutting=state.cutting;body->knifeBody=0;
    body->wristFront=wristFront;body->wristLength=wristLength;body->wristRadius=wristRadius;
    body->edgeA[0]=state.leftA;body->edgeB[0]=state.leftB;
    body->edgeA[1]=state.rightA;body->edgeB[1]=state.rightB;
    body->scissorsRadius=radius;
    body->maxGap=state.maxGap;
    body->backContact=state.backContact;body->backs[0]=state.leftBack;body->backs[1]=state.rightBack;
}
extern "C" int xpbd_set_scissors(const XpbdScissorsState* state){
    if(!state)return XPBD_ERR_BAD_ARG;
    if(!state->enabled){
        if(g_scissorsMode){invalidateBladeGraphs();g_blade.enabled=0;g_scissors.enabled=0;
            if(d_scissorsLoads)cudaMemset(d_scissorsLoads,0,7*sizeof(float));
            if(d_bladeBody)cudaMemset(d_bladeBody,0,sizeof(CoupledTool));
            if(d_bladePhase)cudaMemset(d_bladePhase,0,sizeof(int));
            if(d_bladeHistory)cudaMemset(d_bladeHistory,0,sizeof(BladeHistory));}
        return 0;
    }
    if(g_blade.enabled && !g_scissorsMode)return XPBD_ERR_BAD_ARG;
    const float* data=reinterpret_cast<const float*>(state);
    for(int j=0;j<32;j++)if(!std::isfinite(data[j]))return XPBD_ERR_BAD_ARG;
    if(!(state->length>0) || state->gap<0 || !(state->maxGap>0) || state->gap>state->maxGap || state->maxGap>1.7321f*state->length ||
       !(state->startForce>state->cutForce) || !(state->cutForce>0) || !(state->mass>0) || !(state->jawMass>0) ||
       !(state->driveStiffness>0) || !(state->jawStiffness>0) || !(state->jawSpeed>0) || state->friction<0 || !(state->squeeze>0) ||
       len3(state->axis)<1e-6f || len3(state->normal)<1e-6f || fabsf(dot3(contactUnit(state->axis,{}),contactUnit(state->normal,{})))>.001f ||
       len3(state->leftB-state->leftA)<1e-6f || len3(state->rightB-state->rightA)<1e-6f)return XPBD_ERR_BAD_ARG;
    if(g_blade.enabled && (memcmp(&state->leftA,&g_scissors.leftA,4*sizeof(float3))!=0 || state->length!=g_scissors.length || state->maxGap!=g_scissors.maxGap))return XPBD_ERR_BAD_ARG;
    if(state->backContact){
        for(const auto& box:{state->leftBack,state->rightBack}){
            const float* values=reinterpret_cast<const float*>(&box);for(int j=0;j<12;j++)if(!std::isfinite(values[j]))return XPBD_ERR_BAD_ARG;
            if(box.halfSize.x<=0 || box.halfSize.y<=0 || box.halfSize.z<=0 || fabsf(len3(box.axisX)-1)>.001f
                || fabsf(len3(box.axisY)-1)>.001f || fabsf(dot3(box.axisX,box.axisY))>.001f)return XPBD_ERR_BAD_ARG;
        }
    }
    if(g_blade.enabled && (state->backContact!=g_scissors.backContact || (state->backContact &&
        (memcmp(&state->leftBack,&g_scissors.leftBack,sizeof(state->leftBack)) || memcmp(&state->rightBack,&g_scissors.rightBack,sizeof(state->rightBack))))))return XPBD_ERR_BAD_ARG;
    if(!g_blade.enabled || g_blade.startForce!=state->startForce || g_scissors.cutting!=state->cutting)invalidateBladeGraphs();
    if(!g_blade.enabled && state->backContact!=g_scissors.backContact)g_bladeCapacity=0;
    XpbdToolState t={};t.jawHinge=t.jawMid=t.shaftA=t.shaftB=state->hinge;
    t.jawAxis=contactUnit(state->axis,mkf3(0,0,1));t.jawNormal=contactUnit(state->normal,mkf3(0,1,0));t.jawLen=state->length;t.jawGap=state->gap;
    // Native contact skin dictates the capsule radius, independent of Unity's marker radius.
    float radius=fmaxf(g_contactRadius,.005f*state->length);
    t.shaftR=radius; // coarse trust-region bound; scissors do not collide on kind 0
    if(!g_blade.enabled){g_bladeInitial=t;if(d_bladeBody)cudaMemset(d_bladeBody,0,sizeof(CoupledTool));
        if(d_scissorsLoads)cudaMemset(d_scissorsLoads,0,7*sizeof(float));
        if(d_bladePhase)cudaMemset(d_bladePhase,0,sizeof(int));if(d_bladeHistory)cudaMemset(d_bladeHistory,0,sizeof(BladeHistory));}
    g_bladeCommand=t;g_scissors=*state;g_scissorsMode=true;
    g_blade={state->hinge,state->hinge,radius,state->startForce,state->cutForce,state->driveStiffness,state->mass,state->friction,1};
    return 0;
}
extern "C" int xpbd_get_scissors(XpbdScissorsResult* result){
    if(!result || !g_scissorsMode || !g_blade.enabled || !d_bladeBody)return XPBD_ERR_BAD_ARG;
    CoupledTool body={};float force[6]={};int phase=0,count=0;
    cudaError_t e=cudaMemcpy(&body,d_bladeBody,sizeof(body),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(force,d_bladeResult,sizeof(force),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(&phase,d_bladePhase,sizeof(int),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(&count,d_bladeCount,sizeof(int),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return -(int)e;
    *result={};result->hinge=body.pose.jawHinge;result->axis=body.pose.jawAxis;result->normal=body.pose.jawNormal;
    ContactShape left=bodyContactShape(&body,body.pose,1,0,0),right=bodyContactShape(&body,body.pose,2,0,0);
    result->leftA=left.a;result->leftB=left.b;result->rightA=right.a;result->rightB=right.b;
    result->gap=body.pose.jawGap;result->radius=body.scissorsRadius;result->mass=g_scissors.mass;result->jawMass=g_scissors.jawMass;
    result->pressure=force[3];result->force=mkf3(force[0],force[1],force[2]);result->phase=phase;result->contactCount=count;return 0;
}
extern "C" int xpbd_get_scissors_loads(float* values){
    if(!values || !g_scissorsMode || !g_blade.enabled || !d_scissorsLoads)return XPBD_ERR_BAD_ARG;
    return -(int)cudaMemcpy(values,d_scissorsLoads,7*sizeof(float),cudaMemcpyDeviceToHost);
}

// Include the cutter in every coarse direction while leaving its own pose
// fixed. The grasper's joint-transport direction must not move a second tool.
__global__ void k_BladeCoarse(const UnifiedState* state,const UnifiedState* blade,CoarseBlock* blocks,bool bound){
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
__global__ void k_BladeReaction(UnifiedState* state,int* phase,float startForce,float* result,BladeHistory* history,bool advanceHistory=true,bool exportForce=false,float* loads=nullptr,float squeeze=0){
    const UnifiedState& s=*state;float3 total=mkf3(0,0,0);float pressure=0,approaching=0,leftLoad=0,rightLoad=0;
    // A scissor blade's compressive load is the tissue force resisting its
    // closing motion, on its edge and on its body alike (kinds 1 and 3, 2 and
    // 4): a blade sliding in a slit pushes the tissue ahead with its body. The
    // slit walls squeezing it across the shear plane compress nothing between
    // the blades and add nothing.
    float3 close1=mkf3(0,0,0),close2=close1;
    if(s.tool->scissors){
        const XpbdToolState& t=s.tool->pose;
        float3 u=contactUnit(t.jawAxis,mkf3(0,0,1)),n=contactUnit(t.jawNormal,mkf3(0,1,0));
        float sn=fminf(.95f,fmaxf(0.f,t.jawGap)/(2.f*fmaxf(t.jawLen,1e-6f))),cs=sqrtf(1-sn*sn);
        close1=u*sn-n*cs;close2=u*sn+n*cs;
    }
    for(int k=threadIdx.x;k<*s.contactCount;k+=32){
        const SurfaceContact& c=s.contacts[s.contactIds[k]];float3 f;Symmetric3 H={};unifiedContact(s,c,f,H);
        float3 normal;unifiedGap(s,c,unifiedPoint(c,s.pos),normal);
        float load=fmaxf(0.f,-dot3(f,normal));
        total=total+f;if(c.kind<3)pressure+=load;
        if(c.kind==1||c.kind==3)leftLoad+=fmaxf(0.f,-dot3(f,close1));
        if(c.kind==2||c.kind==4)rightLoad+=fmaxf(0.f,-dot3(f,close2));
        if(c.kind<3 && dot3(s.tool->guideAxis,normal)>1e-4f)approaching+=load;
    }
    float3 low=mkf3(1e30f,1e30f,1e30f),high=mkf3(-1e30f,-1e30f,-1e30f);
    if(advanceHistory)for(int i=threadIdx.x;i<s.n;i+=32){
        if(s.active&&!s.active[i])continue;float3 p=s.pos[i];
        low=mkf3(fminf(low.x,p.x),fminf(low.y,p.y),fminf(low.z,p.z));
        high=mkf3(fmaxf(high.x,p.x),fmaxf(high.y,p.y),fmaxf(high.z,p.z));
    }
    for(int d=16;d;d>>=1){
        low.x=fminf(low.x,__shfl_down_sync(0xffffffff,low.x,d));low.y=fminf(low.y,__shfl_down_sync(0xffffffff,low.y,d));low.z=fminf(low.z,__shfl_down_sync(0xffffffff,low.z,d));
        high.x=fmaxf(high.x,__shfl_down_sync(0xffffffff,high.x,d));high.y=fmaxf(high.y,__shfl_down_sync(0xffffffff,high.y,d));high.z=fmaxf(high.z,__shfl_down_sync(0xffffffff,high.z,d));
    }
    for(int d=16;d;d>>=1){total.x+=__shfl_down_sync(0xffffffff,total.x,d);total.y+=__shfl_down_sync(0xffffffff,total.y,d);total.z+=__shfl_down_sync(0xffffffff,total.z,d);pressure+=__shfl_down_sync(0xffffffff,pressure,d);leftLoad+=__shfl_down_sync(0xffffffff,leftLoad,d);rightLoad+=__shfl_down_sync(0xffffffff,rightLoad,d);approaching+=__shfl_down_sync(0xffffffff,approaching,d);}
    if(threadIdx.x)return;
    float enter=0,leave=1;float3 a=s.tool->pose.shaftA,v=s.tool->pose.shaftB-a;bool inside=true;
    for(int j=0;j<3;j++){
        float lo=compF3(low,j)-s.tool->pose.shaftR,hi=compF3(high,j)+s.tool->pose.shaftR,x=compF3(a,j),d=compF3(v,j);
        if(fabsf(d)<1e-9f){if(x<lo||x>hi)inside=false;}
        else{float p=(lo-x)/d,q=(hi-x)/d;enter=fmaxf(enter,fminf(p,q));leave=fminf(leave,fmaxf(p,q));}
    }
    inside=inside&&leave>=enter;

    if(advanceHistory || exportForce){
        result[0]=total.x;result[1]=total.y;result[2]=total.z;result[3]=pressure;
        result[4]=fmaxf(result[4],pressure);
    }
    // Commit a threshold event only from the accepted substep's reaction.
    if(!advanceHistory)return;
    if(s.tool->scissors){
        bool closing=s.tool->driveGap<s.tool->previous.jawGap-1e-6f;
        bool opening=s.tool->driveGap>s.tool->previous.jawGap+1e-6f;
        if(!s.tool->scissorsCutting || opening)history->strokeGap=s.tool->pose.jawGap;
        else history->strokeGap=fmaxf(history->strokeGap,s.tool->previous.jawGap);
        if(!s.tool->scissorsCutting || opening){
            *phase=BladeClearStroke;history->fractured=history->hadContact=history->squeezing=0;history->freeTravel=0;
        }
        // Closing blades push the tissue ahead of their edges and lose contact
        // for single substeps. The stroke lasts until the blades open or cutting
        // stops, so its pending crossings survive a lapse, and a fractured stroke
        // keeps cutting through it: re-testing the fracture law there leaves the
        // tissue swept after the last lapse uncut inside a closed cut. The
        // blades have left the tissue once their tips close by more than the
        // squeeze travel without edge contact. The fractured stroke then exits
        // for one substep and presses again, so new tissue must be squeezed by
        // both blades before it severs. A closing stroke presses from its
        // start: blades re-entering an earlier cut sweep material the cut has
        // displaced from its snapshot position before they touch any tissue.
        else if(!(s.cutContactCount?*s.cutContactCount:*s.contactCount)){
            if(history->fractured){
                history->freeTravel+=fmaxf(0.f,s.tool->previous.jawGap-s.tool->pose.jawGap);
                if(history->freeTravel>squeeze){*phase=BladeCutExit;history->fractured=history->squeezing=0;}
            }
            else if(*phase==BladeCutExit || *phase==BladeClearStroke)*phase=BladePressing;
        }
        else{
            history->hadContact=1;history->freeTravel=0;
            // CutExit commits the single sweep that left tissue. Renewed contact
            // is a new pressing gesture and must reach the fracture law again.
            if(*phase==BladeClearStroke || *phase==BladeWithdrawing || *phase==BladeCutExit)*phase=BladePressing;
            if(!history->fractured)result[5]=fmaxf(result[5],pressure);
            // Opposite contact normals can cancel force, so each blade's
            // compressive load is tested on its own. Both blades must squeeze
            // the tissue with at least half the initiation force: one loaded
            // blade and one grazing blade is pressing, not a shear. The squeeze
            // must also deform the tissue: the tips close by the squeeze travel
            // after both blades first load it, or a stiff patch reaching the
            // force at first touch would sever undeformed.
            if(!history->squeezing && leftLoad>0 && rightLoad>0){history->squeezing=1;history->squeezeGap=s.tool->previous.jawGap;}
            bool squeezed=history->squeezing && history->squeezeGap-s.tool->pose.jawGap>=squeeze;
            float required=.5f*.999f*startForce;
            if(!history->fractured && closing && squeezed && leftLoad>=required && rightLoad>=required){
                *phase=BladeCutting;history->fractured=1;
                if(loads){loads[2]=leftLoad;loads[3]=rightLoad;loads[4]=pressure;loads[5]+=1.f;}
            }
        }
        if(loads){loads[0]=leftLoad;loads[1]=rightLoad;loads[6]=history->squeezing;}
        return;
    }
    if(*phase==3 || *phase==4)*phase=0;
    // Keep the stroke direction across its free-space exit: reversing through
    // the fresh incision is withdrawal, not a new pressure-triggered stroke.
    // Releasing input in free space arms a genuinely new gesture.
    // A recoil can temporarily remove all contacts while the blade is still
    // inside the organ. A pause there must not erase preload/material history.
    // A solved substep can reach its target and have zero guide length while
    // the user is still withdrawing. Only a pause OUTSIDE tissue rearms the
    // gesture; otherwise the next substep could cut upward through the wound.
    int edgeContacts=s.cutContactCount?*s.cutContactCount:*s.contactCount;
    if(s.tool->guideLength<=1e-6f && !inside && !edgeContacts){
        *phase=4;history->entryAxis=mkf3(0,0,0);history->fractured=history->hadContact=0;history->freeTravel=0;return;
    }
    if(!history->fractured)result[5]=fmaxf(result[5],pressure);
    if(edgeContacts){
        history->freeTravel=0;history->hadContact=1;
        if(len3(history->entryAxis)<.5f && s.tool->guideLength>1e-6f)history->entryAxis=contactUnit(s.tool->guideAxis,total*(-1.f));
        if(*phase==3 || *phase==4)*phase=0;
    }else history->freeTravel+=len3(s.tool->pose.jawHinge-s.tool->previous.jawHinge);
    if(history->hadContact && !inside && !edgeContacts && history->freeTravel>2*s.tool->pose.shaftR){
        *phase=history->fractured?3:4;history->fractured=0;history->hadContact=0;return;
    }
    float direction=dot3(s.tool->guideAxis,history->entryAxis);
    // A single blade has one sharpened leading edge. Reversing through the
    // wound is withdrawal even if its blunt side touches a moving cut wall.
    // Do not reverse entryAxis and turn that contact into a second incision.
    if(*phase==BladeWithdrawing && inside)return;
    if(direction<-.1f){*phase=2;return;}
    if(*phase==2 && direction>.1f)*phase=history->fractured?1:0;
    if(!history->fractured && *phase!=2 && s.tool->guideLength>1e-6f && pressure>=.999f*startForce){
        *phase=1;history->fractured=1;
        history->entryAxis=contactUnit(s.tool->guideAxis,total*(-1.f));
    }

}
__global__ void k_CountCutContacts(const SurfaceContact* contacts,const int* ids,const int* count,int* edges){
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*count;k+=blockDim.x*gridDim.x)
        if(contacts[ids[k]].kind<3)atomicAdd(edges,1);
}
static void findBlade(cudaStream_t stream=0,bool stationary=false){
    cudaMemsetAsync(d_bladeCount,0,sizeof(int),stream);cudaMemsetAsync(d_bladeDegree,0,g_N*sizeof(int),stream);
    for(int index=0;index<bladeShapeCount();index++){
    int kind=bladeShapeKind(index);
    if(g_surfaceCount>0)k_FindSurface<<<Gr(g_surfaceCount),64,0,stream>>>(g_surfaceCount,kind,
        contactShape(g_bladeInitial,0,0,0),contactShape(g_bladeCommand,0,0,0),kind>=3?g_contactRadius:0,
        d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,d_bladeContacts,d_bladeDegree,d_bladeIds,d_bladeCount,d_surfaceCurrent,d_surfacePrevious,d_bladeBody,0,0,stationary,kind>=3?d_bladeBackCacheSlots:nullptr,kind>=3?d_bladeBackCache:nullptr,kind<3?d_bladePhase:nullptr,BLADE_KIND_SLOTS,kind>=3);
    }
    cudaMemsetAsync(d_bladeBackCacheSlots,0xff,CONTACT_CACHE_SIZE*sizeof(int),stream);
    if(g_surfaceCount>0)k_CacheBladeBackContacts<<<Gr(BLADE_KIND_SLOTS*g_surfaceCount),64,0,stream>>>(d_bladeContacts,d_bladeIds,d_bladeCount,d_bladeBackCacheSlots,d_bladeBackCache);
    cudaMemsetAsync(d_bladeManifoldCount,0,sizeof(int),stream);cudaMemsetAsync(d_bladeDegree,0,g_N*sizeof(int),stream);
    if(g_surfaceCount>0)k_UnifiedManifold<<<Gr(BLADE_KIND_SLOTS*g_surfaceCount),64,0,stream>>>(d_bladeContacts,d_bladeIds,d_bladeCount,d_surfaceVertices,
        d_pos,d_prevPos,d_bladeBody,0,0,g_contactRadius,d_bladeManifold,d_bladeManifoldIds,d_bladeManifoldCount,d_bladeOwners,d_bladeQuadrature,d_bladeDegree,BLADE_KIND_SLOTS);
    cudaMemsetAsync(d_bladeEdgeCount,0,sizeof(int),stream);
    k_CountCutContacts<<<128,64,0,stream>>>(d_bladeManifold,d_bladeManifoldIds,d_bladeManifoldCount,d_bladeEdgeCount);
}
