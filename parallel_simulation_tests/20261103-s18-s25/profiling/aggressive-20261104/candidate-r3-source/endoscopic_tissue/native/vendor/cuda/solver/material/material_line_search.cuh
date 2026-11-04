// Exact incremental-potential line search for the coupled material step.
// Inertia, elasticity, objective strain viscosity, ligament springs and live
// cut-rim edges retain their V18 energies. Check the cubic determinant along
// the ENTIRE segment, so a trial cannot invert and then recover at its endpoint.
// Contact/anchor witnesses are fixed in this material-only correction, so
// their energies stay constant. V18 vertex/tool sweeps solve those DOFs.
// energy[0] estimates the energy at the start of the search, the scale of the tolerance in k_MaterialLineApply.
// energy[k] for k>0 is the energy change of the step materialLineAlpha(k) (unifiedTetEnergyChange).
struct MaterialLineSearch { double energy[5]; int invalid; };
// In floor recovery (materialFloorRecovery) the material step keeps every tet at or above this fraction of the volume
// floor, while the vertex sweeps bound compression at the floor itself. The margin lets a tet that sits on the floor
// take the small compressions that the floor curvature of the stencil leaves in a direction (materialFloorCurvature):
// without one, any of them stops the step of the whole component (k_RestrictMaterialDirection).
static constexpr float MaterialFloorBand=.95f;
// Substeps a piece stays in floor recovery after the floor stopped its material step (k_MaterialFloorRecovery).
static constexpr int MaterialFloorRecoverySubsteps=6;
__device__ inline float materialLineAlpha(int k){return k==0?0.f:k==1?1.f:k==2?.5f:k==3?.25f:.125f;}
__device__ inline float3 materialLinePoint(const UnifiedState& s,const float3* start,int i,float a){
    return s.pos[i]+start[i]*a;
}
__device__ inline void materialLineAdd(MaterialLineSearch* out,double* e){
    unsigned mask=__activemask();auto key=(unsigned long long)out;
    if(__all_sync(mask,key==__shfl_sync(mask,key,__ffs(mask)-1))){
        auto group=cooperative_groups::coalesced_threads();
        for(int k=0;k<5;k++)e[k]=cooperative_groups::reduce(group,e[k],cooperative_groups::plus<double>());
        if(group.thread_rank())return;
    }
    for(int k=0;k<5;k++)atomicAdd(&out->energy[k],e[k]);
}
__global__ void k_MaterialLineClear(int n,MaterialLineSearch* search){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)search[i]={};
}
__global__ void k_MaterialLineNodes(const UnifiedState* state,const float3* start,MaterialLineSearch* search){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n||(s.active&&!s.active[i]))return;
    // The inertial, support and damping terms are quadratic in the node's step: g.step+|step|^2*curvature/2. Each step
    // runs to the trial position as k_MaterialLineApply writes it (k_MaterialLineTets).
    float3 p=s.pos[i],d=p-s.target[i],gradient=mkf3(0,0,0);float energy=0,curvature=0;
    if(s.invMass[i]>0){float inertia=1.f/(s.invMass[i]*s.h*s.h);energy+=.5f*inertia*dot3(d,d);gradient=d*inertia;curvature=inertia;}
    if(s.attach){float stiffness=s.attach[i],viscous=unifiedSupportDamping(s,i);float3 r=p-s.attachRest[i],q=p-s.previous[i];
        energy+=.5f*(stiffness*dot3(r,r)+viscous*dot3(q,q));gradient=gradient+r*stiffness+q*viscous;curvature+=stiffness+viscous;}
    double values[5]={energy};
    for(int k=1;k<5;k++){float3 step=materialLinePoint(s,start,i,materialLineAlpha(k))-p;values[k]=dot3(gradient,step)+.5f*curvature*dot3(step,step);}
    if(s.neighbors&&s.edgeSupport)for(int j=1;j<6;j+=2){int nb=s.neighbors[6*i+j];
        if(nb<0||(s.active&&!s.active[nb])||s.edgeSupport[6*i+j])continue;
        float3 e=p-s.pos[nb];float l=len3(e),stretch=l-len3(s.restNeighbors[6*i+j]);
        float K=.25f*(unifiedEdgeStiffness(s,i,j/2)+unifiedEdgeStiffness(s,nb,j/2));values[0]+=K*stretch*stretch;
        for(int k=1;k<5;k++){float a=materialLineAlpha(k);
            float3 move=(materialLinePoint(s,start,i,a)-p)-(materialLinePoint(s,start,nb,a)-s.pos[nb]);
            float growth=unifiedLengthChange(e,move,l);values[k]+=K*growth*(2.f*stretch+growth);}
    }
    materialLineAdd(search+s.component[i],values);
}
__device__ inline float materialLineJ(float a,float j0,float j1,float j2,float j3){return ((j3*a+j2)*a+j1)*a+j0;}
__device__ inline bool materialLineFeasible(float a,float floor,float j0,float j1,float j2,float j3){
    if(!(materialLineJ(a,j0,j1,j2,j3)>=floor))return false;
    // Check stationary points of the cubic determinant, not just its endpoint.
    if(fabsf(j3)<1e-12f){
        if(fabsf(j2)>1e-12f){float x=-j1/(2*j2);if(x>0&&x<a&&materialLineJ(x,j0,j1,j2,j3)<floor)return false;}
    }else{
        float disc=j2*j2-3*j3*j1;
        if(disc>=0){float d=sqrtf(disc);float x=(-j2-d)/(3*j3),y=(-j2+d)/(3*j3);
            if(x>0&&x<a&&materialLineJ(x,j0,j1,j2,j3)<floor)return false;
            if(y>0&&y<a&&materialLineJ(y,j0,j1,j2,j3)<floor)return false;}
    }
    return true;
}
// Project a blocked free-material search into the determinant tangent cone.
// This changes only the trial direction; the exact energy/path test remains the
// authority for position updates. Contact and anchor witnesses stay fixed.
__global__ void k_ProjectRecoveryDirection(const UnifiedState* state,const Symmetric3* H,const float3* direction,float3* correction,float* count){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=s.nt||!s.live[t])return;
    int id[4];unifiedTetIds(s,t,id);int flags=s.graspComponents[s.component[id[0]]];if(!(flags&8)||(flags&7))return;
    float3 a=s.pos[id[0]],u=s.pos[id[1]]-a,v=s.pos[id[2]]-a,w=s.pos[id[3]]-a;
    float det=unifiedKuhn(s,t).metric[6],J=dot3(u,cross3(v,w))*det;if(J>s.minimum+.001f)return;
    float3 g[4],h[4];g[1]=cross3(v,w)*det;g[2]=cross3(w,u)*det;g[3]=cross3(u,v)*det;g[0]=(g[1]+g[2]+g[3])*(-1.f);
    float dj=0,denom=0;for(int j=0;j<4;j++){dj+=dot3(g[j],direction[id[j]]);h[j]=materialCgFixed(s,id[j])?mkf3(0,0,0):H[id[j]].solve(g[j]);denom+=dot3(g[j],h[j]);}
    float violation=1e-5f-.9f*fmaxf(0.f,J-s.minimum)-dj;if(violation<=0||denom<=1e-20f)return;
    for(int j=0;j<4;j++)if(!materialCgFixed(s,id[j])){add3(correction+id[j],h[j]*(violation/denom));atomicAdd(count+id[j],1.f);}
}
__global__ void k_ApplyRecoveryDirection(int n,float3* direction,float3* correction,float* count,bool clearNext=false){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(count[i]>0)direction[i]=direction[i]+correction[i]*(1.f/count[i]);
    if(clearNext){correction[i]=mkf3(0,0,0);count[i]=0;}
}
// Restrict a failed material search locally. Contact/anchor sweeps still solve
// these vertices; they are not pinned in the physical model. Use the admissible
// fraction of the direction, rather than toggling whole vertices between full
// motion and zero as a tet approaches its volume floor. Recheck the entire
// nonlinear path after neighboring restrictions have been combined.
// A free recovering component, and one in floor recovery, scales as a whole instead, within the margin of
// MaterialFloorBand for the latter. Node by node, a tet held on the floor stops each node it shares, the nodes around
// them strain the tets in between, and once the line search rejects every trial the component's step stays zero: a
// fold pinched onto the floor never opens.
// The restriction reuses the coarse weights, so it clears the active nodes alone: an inactive node keeps weight 0.
__global__ void k_MaterialRestrictionClear(const UnifiedState* state,float* scale){
    int i=unifiedActiveNode(*state,blockIdx.x*blockDim.x+threadIdx.x);if(i>=0)scale[i]=1.f;
}
__global__ void k_RestrictMaterialDirection(const UnifiedState* state,const float3* direction,float* blocked,const int* previousChanged=nullptr,int* changed=nullptr){
    if(previousChanged && !*previousChanged)return;
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=s.nt||!s.live[t])return;
    int id[4];unifiedTetIds(s,t,id);float3 p[4],v[4];for(int j=0;j<4;j++){p[j]=s.pos[id[j]];v[j]=direction[id[j]];}
    float3 e1=p[1]-p[0],e2=p[2]-p[0],e3=p[3]-p[0],d1=v[1]-v[0],d2=v[2]-v[0],d3=v[3]-v[0];
    if(dot3(d1,d1)+dot3(d2,d2)+dot3(d3,d3)==0)return; // Rigid translation preserves J exactly.
    TetVolumePath J=unifiedVolumePath(e1,e2,e3,d1,d2,d3,unifiedKuhn(s,t).metric[6]);
    float j0=J.j0,j1=J.j1,j2=J.j2,j3=J.j3;
    int root=s.component[id[0]];bool whole=materialFreeRecovery(s,root)||materialFloorRecovery(s,root);
    float floor=fminf((materialFloorRecovery(s,root)?MaterialFloorBand:1.f)*s.minimum,j0);
    if(!materialLineFeasible(1.f,floor,j0,j1,j2,j3)){
        float lo=0,hi=1;
        for(int k=0;k<(whole?24:12);k++){
            float mid=.5f*(lo+hi);
            if(materialLineFeasible(mid,floor,j0,j1,j2,j3))lo=mid;else hi=mid;
        }
        if(whole)atomicMin((unsigned*)(blocked+root),__float_as_uint(lo));
        else for(int j=0;j<4;j++)atomicMin((unsigned*)(blocked+id[j]),__float_as_uint(lo));
        // Tets against a jaw or a blade sit on the floor by design: only a tet of free tissue that cuts the direction as
        // the PCG left it, before the cuts of earlier passes spread, makes a lock.
        bool freeTissue=!previousChanged;for(int j=0;j<4;j++)freeTissue=freeTissue && !materialCgFixed(s,id[j]);
        if(freeTissue)atomicOr(s.floorRecovery+root,MaterialFloorRestricted);
        if(changed)atomicExch(changed,1);
    }
}
__global__ void k_ApplyMaterialRestriction(int n,float3* direction,const float* blocked,const int* changed=nullptr,const UnifiedState* state=nullptr){
    if(changed && !*changed)return;
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    int root=state?state->component[i]:i;
    float a=blocked[state && (materialFreeRecovery(*state,root)||materialFloorRecovery(*state,root))?root:i];
    if(a<1)direction[i]=a>0?direction[i]*a:mkf3(0,0,0);
}
__global__ void k_MaterialLineTets(const UnifiedState* state,const float3* start,MaterialLineSearch* search){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=s.nt||!s.live[t])return;int id[4];unifiedTetIds(s,t,id);int root=s.component[id[0]];

    float3 p[4],v[4];for(int j=0;j<4;j++){p[j]=s.pos[id[j]];v[j]=start[id[j]];}
    TetVolumePath J=unifiedVolumePath(p[1]-p[0],p[2]-p[0],p[3]-p[0],v[1]-v[0],v[2]-v[0],v[3]-v[0],unifiedKuhn(s,t).metric[6]);
    double values[5]={};
    for(int k=1;k<5;k++){
        // The change to the trial positions as k_MaterialLineApply writes them, rounded to float, rather than to p+a*v:
        // their difference from p is exact in float, and the accepted step must lower the energy of the stored state.
        float a=materialLineAlpha(k),energy;float3 step[4];
        for(int j=0;j<4;j++)step[j]=(p[j]+v[j]*a)-p[j];
        values[k]=unifiedTetEnergyChange(s,t,p,step,&energy);values[0]=energy;
        float floor=fminf((materialFloorRecovery(s,root)?MaterialFloorBand:1.f)*s.minimum,J.j0);
        if(!materialLineFeasible(a,floor,J.j0,J.j1,J.j2,J.j3))atomicOr(&search[root].invalid,1<<k);
    }
    materialLineAdd(search+root,values);
}
__global__ void k_MaterialLineApply(const UnifiedState* state,const float3* start,const MaterialLineSearch* search){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n||materialCgFixed(s,i))return;
    const auto& q=search[s.component[i]];int best=0;double lowest=0,tolerance=1e-10*fmax(1.,fabs(q.energy[0]));
    for(int k=1;k<5;k++)if(!(q.invalid&(1<<k))&&isfinite(q.energy[k])&&q.energy[k]<lowest-tolerance){best=k;lowest=q.energy[k];}
    // A rejected/nonfinite direction must leave the accepted position intact:
    // multiplying an invalid direction by zero would still propagate NaNs.
    if(best>0)s.pos[i]=materialLinePoint(s,start,i,materialLineAlpha(best));
}
// Puts a held or touched component with a tet at the floor into floor recovery for MaterialFloorRecoverySubsteps when the
// volume floor cut its direction at a tet of free tissue (MaterialFloorRestricted) and its line search then found no
// step, and counts the recovery of every other component down. A component at rest takes no step either, but nothing
// cuts its direction. Only a piece cut free of every pinned node qualifies, and none that the cutting tool touches: the
// grasp and cut of attached tissue keep the ordinary step they are tuned with.
__global__ void k_MaterialFloorRecovery(const UnifiedState* state,const MaterialLineSearch* search){
    const UnifiedState& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0 || s.component[i]!=i)return;
    const auto& q=search[i];double tolerance=1e-10*fmax(1.,fabs(q.energy[0]));bool stepped=false;
    for(int k=1;k<5;k++)stepped|=!(q.invalid&(1<<k))&&isfinite(q.energy[k])&&q.energy[k]<-tolerance;
    int flags=s.graspComponents[i],left=s.floorRecovery[i]&~MaterialFloorRestricted;
    bool locked=!stepped && (s.floorRecovery[i]&MaterialFloorRestricted) && (flags&7) && (flags&8) && !(flags&MaterialCutterContact)
        && !(s.componentSupport[i]&1);
    s.floorRecovery[i]=locked?MaterialFloorRecoverySubsteps:max(0,left-1);
}

// Bound the *trial* strain before tangent projection; the accepted update still
// has to decrease the unchanged incremental potential and preserve every J.
__global__ void k_RecoveryTrust(const UnifiedState* state,const float3* d,float* scale){
 const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=s.nt||!s.live[t])return;
 int id[4];unifiedTetIds(s,t,id);int root=s.component[id[0]],flags=s.graspComponents[root];if(!(flags&8)||(flags&7))return;
 float a=1;for(int j=0;j<4;j++)for(int k=j+1;k<4;k++){
  float change=len3(d[id[j]]-d[id[k]]),length=len3(s.pos[id[j]]-s.pos[id[k]]);
  if(change>0)a=fminf(a,.1f*length/change);
 }
 atomicMin((unsigned*)(scale+root),__float_as_uint(a));
}
