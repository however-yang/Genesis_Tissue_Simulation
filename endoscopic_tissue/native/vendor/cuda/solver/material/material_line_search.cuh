// Exact incremental-potential line search for the coupled material step.
// Inertia, elasticity, objective strain viscosity, ligament springs and live
// cut-rim edges retain their V18 energies. Check the cubic determinant along
// the ENTIRE segment, so a trial cannot invert and then recover at its endpoint.
// Contact/anchor witnesses are fixed in this material-only correction, so
// their energies stay constant. V18 vertex/tool sweeps solve those DOFs.
struct MaterialLineSearch { double energy[5]; int invalid; };
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
    double energy[5]={};
    for(int k=0;k<5;k++){
        float a=materialLineAlpha(k);float3 p=materialLinePoint(s,start,i,a),d=p-s.target[i];
        double e=s.invMass[i]>0?.5*unifiedDot(d,d)/(s.invMass[i]*s.h*s.h):0;
        if(s.attach){d=p-s.attachRest[i];e+=.5*s.attach[i]*unifiedDot(d,d);
            d=p-s.previous[i];e+=.5*unifiedSupportDamping(s,i)*unifiedDot(d,d);}
        if(s.neighbors&&s.edgeSupport)for(int j=1;j<6;j+=2){int nb=s.neighbors[6*i+j];
            if(nb<0||(s.active&&!s.active[nb])||s.edgeSupport[6*i+j])continue;
            double stretch=len3(p-materialLinePoint(s,start,nb,a))-len3(s.restNeighbors[6*i+j]);
            e+=.25*(unifiedEdgeStiffness(s,i,j/2)+unifiedEdgeStiffness(s,nb,j/2))*stretch*stretch;
        }
        energy[k]=e;
    }
    materialLineAdd(search+s.component[i],energy);
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
    const int* id=s.ids+4*t;int flags=s.graspComponents[s.component[id[0]]];if(!(flags&8)||(flags&7))return;
    float3 a=s.pos[id[0]],u=s.pos[id[1]]-a,v=s.pos[id[2]]-a,w=s.pos[id[3]]-a;
    float det=s.restMetric[7*t+6],J=dot3(u,cross3(v,w))*det;if(J>s.minimum+.001f)return;
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
__global__ void k_MaterialRestrictionClear(int n,float* scale){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)scale[i]=1.f;
}
__global__ void k_RestrictMaterialDirection(const UnifiedState* state,const float3* direction,float* blocked,const int* previousChanged=nullptr,int* changed=nullptr){
    if(previousChanged && !*previousChanged)return;
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=s.nt||!s.live[t])return;
    const int* id=s.ids+4*t;float3 p[4],v[4];for(int j=0;j<4;j++){p[j]=s.pos[id[j]];v[j]=direction[id[j]];}
    float3 e1=p[1]-p[0],e2=p[2]-p[0],e3=p[3]-p[0],d1=v[1]-v[0],d2=v[2]-v[0],d3=v[3]-v[0];
    if(dot3(d1,d1)+dot3(d2,d2)+dot3(d3,d3)==0)return; // Rigid translation preserves J exactly.
    float det=s.restMetric[7*t+6],j0=dot3(e1,cross3(e2,e3))*det;
    float j1=(dot3(d1,cross3(e2,e3))+dot3(e1,cross3(d2,e3))+dot3(e1,cross3(e2,d3)))*det;
    float j2=(dot3(d1,cross3(d2,e3))+dot3(d1,cross3(e2,d3))+dot3(e1,cross3(d2,d3)))*det,j3=dot3(d1,cross3(d2,d3))*det;
    if(!materialLineFeasible(1.f,fminf(s.minimum,j0),j0,j1,j2,j3)){
        float lo=0,hi=1,floor=fminf(s.minimum,j0);
        int iterationFlags=s.graspComponents[s.component[id[0]]];
        int iterations=(iterationFlags&8)&&!(iterationFlags&7)?24:12;
        for(int k=0;k<iterations;k++){
            float mid=.5f*(lo+hi);
            if(materialLineFeasible(mid,floor,j0,j1,j2,j3))lo=mid;else hi=mid;
        }
        // A free recovering component must retain its coupled direction.
        int root=s.component[id[0]],flags=s.graspComponents[root];
        if((flags&8)&&!(flags&7))atomicMin((unsigned*)(blocked+root),__float_as_uint(lo));
        else for(int j=0;j<4;j++)atomicMin((unsigned*)(blocked+id[j]),__float_as_uint(lo));
        if(changed)atomicExch(changed,1);
    }
}
__global__ void k_ApplyMaterialRestriction(int n,float3* direction,const float* blocked,const int* changed=nullptr,const UnifiedState* state=nullptr){
    if(changed && !*changed)return;
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    int root=state?state->component[i]:i;int flags=state?state->graspComponents[root]:0;
    float a=blocked[(flags&8)&&!(flags&7)?root:i];
    if(a<1)direction[i]=a>0?direction[i]*a:mkf3(0,0,0);
}
__global__ void k_MaterialLineTets(const UnifiedState* state,const float3* start,MaterialLineSearch* search){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=s.nt||!s.live[t])return;const int* id=s.ids+4*t;int root=s.component[id[0]];

    float3 p[4],v[4];for(int j=0;j<4;j++){p[j]=s.pos[id[j]];v[j]=start[id[j]];}
    float3 e1=p[1]-p[0],e2=p[2]-p[0],e3=p[3]-p[0];
    float3 d1=v[1]-v[0],d2=v[2]-v[0],d3=v[3]-v[0];float det=s.restMetric[7*t+6];
    float j0=dot3(e1,cross3(e2,e3))*det;
    float j1=(dot3(d1,cross3(e2,e3))+dot3(e1,cross3(d2,e3))+dot3(e1,cross3(e2,d3)))*det;
    float j2=(dot3(d1,cross3(d2,e3))+dot3(d1,cross3(e2,d3))+dot3(e1,cross3(d2,d3)))*det;
    float j3=dot3(d1,cross3(d2,d3))*det;double energy[5]={};
    for(int k=0;k<5;k++){
        float a=materialLineAlpha(k);float3 q[4];for(int j=0;j<4;j++)q[j]=p[j]+v[j]*a;
        energy[k]=unifiedTetEnergy(s,t,q);
        if(k&&!materialLineFeasible(a,fminf(s.minimum,j0),j0,j1,j2,j3))atomicOr(&search[root].invalid,1<<k);
    }
    materialLineAdd(search+root,energy);
}
__global__ void k_MaterialLineApply(const UnifiedState* state,const float3* start,const MaterialLineSearch* search){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n||materialCgFixed(s,i))return;
    const auto& q=search[s.component[i]];int best=0;
    for(int k=1;k<5;k++)if(!(q.invalid&(1<<k))&&isfinite(q.energy[k])&&q.energy[k]<q.energy[best]-1e-10*fmax(1.,fabs(q.energy[0])))best=k;
    // A rejected/nonfinite direction must leave the accepted position intact:
    // multiplying an invalid direction by zero would still propagate NaNs.
    if(best>0)s.pos[i]=materialLinePoint(s,start,i,materialLineAlpha(best));
}

// Bound the *trial* strain before tangent projection; the accepted update still
// has to decrease the unchanged incremental potential and preserve every J.
__global__ void k_RecoveryTrust(const UnifiedState* state,const float3* d,float* scale){
 const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=s.nt||!s.live[t])return;
 const int* id=s.ids+4*t;int root=s.component[id[0]],flags=s.graspComponents[root];if(!(flags&8)||(flags&7))return;
 float a=1;for(int j=0;j<4;j++)for(int k=j+1;k<4;k++){
  float change=len3(d[id[j]]-d[id[k]]),length=len3(s.pos[id[j]]-s.pos[id[k]]);
  if(change>0)a=fminf(a,.1f*length/change);
 }
 atomicMin((unsigned*)(scale+root),__float_as_uint(a));
}
