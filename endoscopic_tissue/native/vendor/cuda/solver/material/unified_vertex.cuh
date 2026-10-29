#include "../../cutting/pressure_cut_state.cuh"
// Experimental primal vertex-block solver. Original implementation following
// Chen et al., Vertex Block Descent, SIGGRAPH 2024, sections 3 and 4.
// A fixed active contact plane is a unilateral quadratic energy. Material,
// contact and persistent Gaussian attachments enter the SAME local Hessian.
// This is not IPC: the determinant line bound protects tetrahedra, not CCD.
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
struct SafeCoarseCluster {float weight,gripWeight;int count,grips,pinned;};
struct UnifiedState {
    int n,nx,ny,nt,grips,coarseMode; bool coupledReady, recoveryOnly;
    UnifiedState* secondary;const int *cutPhase,*cutContactCount;float cutResistance;
    const int* componentSupport;
    int* component;int* graspComponents; // component flags: 1=anchor, 2=contact, 4=unheld jaw patch, 8=active-volume recovery
    int* safeCluster;SafeCoarseCluster* safeAggregate;int* safeJoined; // safeJoined: some tet joined a cluster (k_JoinSafeCoarse)
    const int *offsets,*incident,*ids,*live,*contactIds,*contactCount,*degree,*gripDegree,*neighbors,*edgeSupport,*active;
    const float3 *b0,*b1,*b2,*previous,*target,*attachRest,*restNeighbors;
    const float3* previousInverse;const float* restMetric;
    const float *mu,*bulk,*invMass,*attach,*contactQuadrature;
    float3* pos;float* coarseWeight;float* interaction;
    SurfaceContact* contacts; GripPoint* points; CoupledTool* tool;
    float h,minimum,damping,hardening,contactK,gripK,minGap,muS,muD,edgeK,skin,thickness;
    float3 toolTarget;float gapTarget;
    float3 patchOffset;float patchCoreRadius,patchFalloff;
};
// The material accelerator holds existing contact/anchor witnesses fixed.
// The full V18 vertex/tool sweeps still solve those DOFs; this is only a
// restricted descent space for the extra material correction.
__device__ inline bool materialFreeRecovery(const UnifiedState& s,int root){
    return s.graspComponents && (s.graspComponents[root]&8) && !(s.graspComponents[root]&7);
}
// The material accelerator holds existing contact/anchor witnesses fixed.
// The full V18 vertex/tool sweeps still solve those DOFs; this is only a
// restricted descent space for the extra material correction.
__device__ inline bool materialCgFixed(const UnifiedState& s,int i){
    return (s.recoveryOnly && !materialFreeRecovery(s,s.component[i])) || s.invMass[i]<=0 || (s.degree[i]>0 || (s.secondary && s.secondary->degree[i]>0)) || (s.grips>0 && s.gripDegree[i]>0);
}
__device__ inline float unifiedEdgeStiffness(const UnifiedState& s,int i,int axis){
    float volume=1;float length=0;
    for(int a=0;a<3;a++){
        float l=len3(s.restNeighbors[6*i+2*a]),r=len3(s.restNeighbors[6*i+2*a+1]);
        float spacing=l>0 && r>0?.5f*(l+r):fmaxf(l,r);volume*=spacing;if(a==axis)length=spacing;
    }
    // E*A/L, not E itself: otherwise refining a cut-rim edge makes it
    // artificially stiffer than the surrounding three-dimensional tissue.
    return s.edgeK*volume/fmaxf(length*length,1e-12f);
}
struct CoarseBlock { float3 gradient,step;float xx,yy,zz,xy,xz,yz,scale;int external;float3 center,pivot;float mass;int pins; };
// Ligaments are Kelvin-Voigt supports, using the existing tissue relaxation
// time. Internal strain damping cannot dissipate an almost-rigid organ swing;
// a purely elastic world support otherwise keeps re-impacting a stationary jaw.
// Only the original ligament nodes participate; free cut fragments get no drag.
__device__ inline float unifiedSupportDamping(const UnifiedState& s,int i){
    return s.attach && s.attach[i]>0 && s.damping>0 && s.h>0?s.attach[i]*s.damping/s.h:0;
}
// Previous F is fixed throughout a physical substep (including surface repair).
// Cache its inverse and the immutable rest lengths rather than evaluating them
// once for every incident vertex, color, sweep and reconstruction pass.
__global__ void k_CacheUnifiedTetData(const UnifiedState* state,float3* inverse,float* metric,bool rest){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=s.nt)return;
    float3 b0=s.b0[t],b1=s.b1[t],b2=s.b2[t];
    if(rest){
        float det=dot3(b0,cross3(b1,b2));metric[7*t+6]=det;
        float3 x[4]={mkf3(0,0,0),cross3(b1,b2)*(1.f/det),cross3(b2,b0)*(1.f/det),cross3(b0,b1)*(1.f/det)};
        int e=0;for(int j=0;j<4;j++)for(int k=j+1;k<4;k++)metric[7*t+e++]=len3(x[j]-x[k]);
    }
    if(!s.live[t])return;const int* id=s.ids+4*t;
    float3 a=s.previous[id[0]],u=s.previous[id[1]]-a,v=s.previous[id[2]]-a,w=s.previous[id[3]]-a;
    float3 f0=u*b0.x+v*b1.x+w*b2.x,f1=u*b0.y+v*b1.y+w*b2.y,f2=u*b0.z+v*b1.z+w*b2.z;
    float d=1.f/dot3(f0,cross3(f1,f2));
    inverse[3*t]=cross3(f1,f2)*d;inverse[3*t+1]=cross3(f2,f0)*d;inverse[3*t+2]=cross3(f0,f1)*d;
}
__device__ inline float unifiedRestLength(const UnifiedState& s,int t,int j,int k){
    int a=min(j,k),b=max(j,k);return s.restMetric[7*t+a*(7-a)/2+b-a-1];
}
// Aggregate repeated component addresses inside each active warp before global
// atomics. The sums and physical objective are unchanged; cut islands stay separate.
template<int N> __device__ inline bool unifiedReduce(const void* address,float (&v)[N]){
    unsigned mask=__activemask();unsigned long long key=(unsigned long long)address;
    unsigned long long first=__shfl_sync(mask,key,__ffs(mask)-1);
    if(__all_sync(mask,key==first)){
        auto group=cooperative_groups::coalesced_threads();
        for(int j=0;j<N;j++)v[j]=cooperative_groups::reduce(group,v[j],cooperative_groups::plus<float>());
        return group.thread_rank()==0;
    }
    return true;
}
__device__ inline void unifiedEnergyChangeAdd(CoarseBlock* b,float change){
    float values[1]={change};if(unifiedReduce(b,values))atomicAdd(&b->yy,values[0]);
}
struct Symmetric3 {
    float xx,yy,zz,xy,xz,yz;
    __device__ void diagonal(float s){xx+=s;yy+=s;zz+=s;}
    __device__ void outer(float3 v,float s){xx+=s*v.x*v.x;yy+=s*v.y*v.y;zz+=s*v.z*v.z;xy+=s*v.x*v.y;xz+=s*v.x*v.z;yz+=s*v.y*v.z;}
    __device__ float3 solve(float3 r)const{
        float a=yy*zz-yz*yz,b=xz*yz-xy*zz,c=xy*yz-xz*yy;
        float d=xx*zz-xz*xz,e=xy*xz-xx*yz,f=xx*yy-xy*xy;
        float det=xx*a+xy*b+xz*c;
        if(!(det>1e-25f))return mkf3(0,0,0);
        return mkf3(a*r.x+b*r.y+c*r.z,b*r.x+d*r.y+e*r.z,c*r.x+e*r.y+f*r.z)*(1.f/det);
    }
};
__device__ inline void unifiedStrainDamping(const UnifiedState& s,int t,
    float3 e1,float3 e2,float3 e3,float3 q,float3& gradient,Symmetric3& H){
    if(s.damping<=0)return;
    const float3 b0=s.b0[t],b1=s.b1[t],b2=s.b2[t];
    float3 f0=e1*b0.x+e2*b1.x+e3*b2.x,f1=e1*b0.y+e2*b1.y+e3*b2.y,f2=e1*b0.z+e2*b1.z+e3*b2.z;
    float3 r0=s.previousInverse[3*t],r1=s.previousInverse[3*t+1],r2=s.previousInverse[3*t+2];
    float3 g0=f0*r0.x+f1*r1.x+f2*r2.x,g1=f0*r0.y+f1*r1.y+f2*r2.y,g2=f0*r0.z+f1*r1.z+f2*r2.z;
    q=r0*q.x+r1*q.y+r2*q.z;f0=g0;f1=g1;f2=g2;
    float3 fq=f0*q.x+f1*q.y+f2*q.z;
    float3 dq=mkf3(dot3(f0,fq),dot3(f1,fq),dot3(f2,fq))-q;
    // Incremental spatial Green strain: mu*V*tau/(8*h) ||G'G-I||^2,
    // G=F*inverse(F_previous). A finite rigid rotation has exactly zero
    // dissipation. The previous configuration is fixed during this substep.
    // The Gauss-Newton Hessian is positive semidefinite in both solve spaces.
    float d=.5f*s.mu[t]*s.damping/s.h;
    gradient=gradient+(f0*dq.x+f1*dq.y+f2*dq.z)*d;
    H.outer(fq,d);float a=d*dot3(q,q);H.outer(f0,a);H.outer(f1,a);H.outer(f2,a);
}
// C2 compression barrier: normal deformation (J >= 2*minimum) is unchanged.
// Unlike a hard feasible-step clip, this contributes a restoring pressure before
// a tet reaches the determinant floor. Energy, gradient and curvature must be
// used together in every solve space. The existing path bound remains active.
struct CompressionBarrier { double energy;float gradient,curvature; };
__device__ inline CompressionBarrier unifiedCompressionBarrier(double J,float minimum,float stiffness){
    CompressionBarrier b={};if(minimum<=0 || J>=2*minimum)return b;
    double activation=2.*minimum,cutoff=activation*1e-4,j=fmax(J,cutoff),z=j,d=j-activation;
    double l=log(z/activation),k=stiffness;
    b.energy=-k*d*d*l;
    b.gradient=float(-k*(2*d*l+d*d/z));
    b.curvature=float(-k*(2*l+4*d/z-d*d/(z*z)));
    // The singularity is at physical zero volume, NOT at the artificial
    // positive guard: a pole at the guard fights the contact solve. Extend
    // through zero for diagnostic/legacy states; path guarding is unchanged.
    double offset=J-j;
    b.energy+=double(b.gradient)*offset+.5*double(b.curvature)*offset*offset;
    b.gradient+=b.curvature*float(offset);
    return b;
}
__device__ inline double unifiedDot(float3 a,float3 b){return double(a.x)*b.x+double(a.y)*b.y+double(a.z)*b.z;}
// The volume ratio J of a tet along the segment p+a*v, as the cubic j0+a*(j1+a*(j2+a*j3)) of the step length a. The
// edges e and their changes d run from the tet's first node, and det is the determinant of its inverse rest shape.
struct TetVolumePath { float j0,j1,j2,j3; };
__device__ inline TetVolumePath unifiedVolumePath(float3 e1,float3 e2,float3 e3,float3 d1,float3 d2,float3 d3,float det){
    return {dot3(e1,cross3(e2,e3))*det,(dot3(d1,cross3(e2,e3))+dot3(e1,cross3(d2,e3))+dot3(e1,cross3(e2,d3)))*det,
        (dot3(d1,cross3(d2,e3))+dot3(d1,cross3(e2,d3))+dot3(e1,cross3(d2,d3)))*det,dot3(d1,cross3(d2,d3))*det};
}
// len(e+d)-len(e) for len(e)=l, without subtracting two nearly equal lengths.
__device__ inline float unifiedLengthChange(float3 e,float3 d,float l){
    float sum=len3(e+d)+l;return sum>0?(2.f*dot3(e,d)+dot3(d,d))/sum:0.f;
}
// The sixth power of the tensile strain l/l0-1 of an edge, from l^2-l0^2 and l0^2: exact where l is close to l0, and 0
// for a compressed edge.
__device__ inline float unifiedTensionStrain6(float growth,float rest2){
    if(!(growth>0))return 0.f;
    float strain=growth/(rest2*(sqrtf(1.f+growth/rest2)+1.f)),s2=strain*strain;return s2*s2*s2;
}
// Energy change E(p+v)-E(p) of tet t, with the tet energy of the solve: stable Neo-Hookean elasticity, the compression
// barrier, objective strain damping and tension-only edge hardening (the terms of k_UnifiedVertex). Each term is
// expanded in powers of v around p, so no two nearly equal energies are subtracted: the change keeps its own relative
// accuracy in single precision, where the energies at p and p+v share all but their last digits. *energy, when given,
// receives an estimate of E(p) for scale tolerances.
__device__ inline float unifiedTetEnergyChange(const UnifiedState& s,int t,const float3 (&p)[4],const float3 (&v)[4],
    float* energy=nullptr){
    float3 e1=p[1]-p[0],e2=p[2]-p[0],e3=p[3]-p[0],d1=v[1]-v[0],d2=v[2]-v[0],d3=v[3]-v[0];
    float3 b0=s.b0[t],b1=s.b1[t],b2=s.b2[t];
    float3 f[3]={e1*b0.x+e2*b1.x+e3*b2.x,e1*b0.y+e2*b1.y+e3*b2.y,e1*b0.z+e2*b1.z+e3*b2.z};
    float3 df[3]={d1*b0.x+d2*b1.x+d3*b2.x,d1*b0.y+d2*b1.y+d3*b2.y,d1*b0.z+d2*b1.z+d3*b2.z};
    TetVolumePath J=unifiedVolumePath(e1,e2,e3,d1,d2,d3,s.restMetric[7*t+6]);
    float mu=s.mu[t],lambda=s.bulk[t]+mu/3.f,stretch=J.j0-1.f,dJ=J.j1+J.j2+J.j3;
    float change=mu*(dot3(f[0],df[0])+dot3(f[1],df[1])+dot3(f[2],df[2])+.5f*(dot3(df[0],df[0])+dot3(df[1],df[1])+dot3(df[2],df[2])))
        -mu*dJ+.5f*lambda*dJ*(2.f*stretch+dJ);
    float total=.5f*mu*(dot3(f[0],f[0])+dot3(f[1],f[1])+dot3(f[2],f[2])-3.f)-mu*stretch+.5f*lambda*stretch*stretch;
    // The barrier acts only below twice the volume floor, where its logarithm keeps double precision.
    if(s.minimum>0 && fminf(J.j0,J.j0+dJ)<2*s.minimum){
        double start=unifiedCompressionBarrier(J.j0,s.minimum,lambda).energy;total+=float(start);
        change+=float(unifiedCompressionBarrier(J.j0+(double(J.j1)+J.j2+J.j3),s.minimum,lambda).energy-start);
    }
    if(s.damping>0){
        // The incremental Green strain S=G'G-I of G=F*inverse(F_previous) changes from S0 to S0+S1+S2.
        float3 r0=s.previousInverse[3*t],r1=s.previousInverse[3*t+1],r2=s.previousInverse[3*t+2];
        float3 g[3]={f[0]*r0.x+f[1]*r1.x+f[2]*r2.x,f[0]*r0.y+f[1]*r1.y+f[2]*r2.y,f[0]*r0.z+f[1]*r1.z+f[2]*r2.z};
        float3 dg[3]={df[0]*r0.x+df[1]*r1.x+df[2]*r2.x,df[0]*r0.y+df[1]*r1.y+df[2]*r2.y,df[0]*r0.z+df[1]*r1.z+df[2]*r2.z};
        float s00=0,s01=0,s02=0,s11=0,s12=0,s22=0;
        #pragma unroll
        for(int j=0;j<3;j++)
            #pragma unroll
            for(int k=0;k<3;k++){
                float a0=dot3(g[j],g[k])-(j==k?1.f:0.f),a1=dot3(g[j],dg[k])+dot3(dg[j],g[k]),a2=dot3(dg[j],dg[k]);
                s00+=a0*a0;s01+=a0*a1;s02+=a0*a2;s11+=a1*a1;s12+=a1*a2;s22+=a2*a2;
            }
        float c=mu*s.damping/(8*s.h);total+=c*s00;change+=c*(2.f*s01+s11+2.f*s02+2.f*s12+s22);
    }
    if(s.hardening>0){
        float h=.5f*mu*s.hardening;
        #pragma unroll
        for(int j=0;j<4;j++)
            #pragma unroll
            for(int k=j+1;k<4;k++){
                float3 x=p[j]-p[k],y=v[j]-v[k];float l0=unifiedRestLength(s,t,j,k),rest2=l0*l0;
                float growth=dot3(x,x)-rest2,start=h*unifiedTensionStrain6(growth,rest2);total+=start;
                change+=h*unifiedTensionStrain6(growth+2.f*dot3(x,y)+dot3(y,y),rest2)-start;
            }
    }
    if(energy)*energy=total;
    return change;
}
// Pick the lower-energy feasible initial guess independently for each connected
// piece. Ballistic/rigid motion keeps its inertial prediction; a high-frequency
// elastic predictor may not inject unresolved strain energy into a fixed-budget
// solve. This changes only the starting iterate, never the inertial target.
// Each component's yy sums the energy change from the previous configuration to
// the inertial guess (unifiedTetEnergyChange): the guess stands unless it raises it.
__global__ void k_PredictorClear(int n,CoarseBlock* b){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){b[i].yy=0;b[i].external=0;}}
__global__ void k_PredictorNodeEnergy(const UnifiedState* state,CoarseBlock* b){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n||s.invMass[i]<=0)return;
    float3 p=s.previous[i],v=s.pos[i]-p,d=p-s.target[i];
    float change=(dot3(d,v)+.5f*dot3(v,v))/(s.invMass[i]*s.h*s.h);
    if(s.attach){d=p-s.attachRest[i];change+=s.attach[i]*(dot3(d,v)+.5f*dot3(v,v))+.5f*unifiedSupportDamping(s,i)*dot3(v,v);}
    if(s.neighbors&&s.edgeSupport)for(int a=1;a<6;a+=2){int j=s.neighbors[6*i+a];
        if(j<0||(s.active&&!s.active[j])||s.edgeSupport[6*i+a])continue;
        float3 e=p-s.previous[j];float l=len3(e),stretch=l-len3(s.restNeighbors[6*i+a]);
        float growth=unifiedLengthChange(e,v-(s.pos[j]-s.previous[j]),l);
        change+=.25f*(unifiedEdgeStiffness(s,i,a/2)+unifiedEdgeStiffness(s,j,a/2))*growth*(2.f*stretch+growth);}
    unifiedEnergyChangeAdd(b+s.component[i],change);
}
__global__ void k_PredictorTetEnergy(const UnifiedState* state,CoarseBlock* b){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=s.nt||!s.live[t])return;
    const int* id=s.ids+4*t;float3 p[4],q[4],v[4];
    for(int j=0;j<4;j++){p[j]=s.previous[id[j]];q[j]=s.pos[id[j]];v[j]=q[j]-p[j];}
    float det=s.restMetric[7*t+6];
    float oldJ=dot3(p[1]-p[0],cross3(p[2]-p[0],p[3]-p[0]))*det;
    // The finite stable material energy alone does not exclude inversion.
    // A ballistic guess must be feasible before any local/material solve starts.
    float nextJ=dot3(q[1]-q[0],cross3(q[2]-q[0],q[3]-q[0]))*det;
    if(!(nextJ>=fminf(s.minimum,oldJ)))atomicOr(&b[s.component[id[0]]].external,1);
    unifiedEnergyChangeAdd(b+s.component[id[0]],unifiedTetEnergyChange(s,t,p,v));
}
__global__ void k_SelectPredictor(const UnifiedState* state,const CoarseBlock* b){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    const CoarseBlock& e=b[s.component[i]];
    // Contact is included in the trial energy. A local contact must not reset
    // the inertial prediction of the entire connected organ unconditionally.
    if(e.external || !isfinite(e.yy)||e.yy>0)s.pos[i]=s.previous[i];
}
__device__ inline float3 unifiedPoint(const SurfaceContact& c,const float3* pos){
    float3 x=c.offset;for(int k=0;k<c.count;k++)x=x+pos[c.id[k]]*c.weight[k];return x;
}
__device__ inline float3 unifiedShift(const SurfaceContact& c,const CoupledTool* body){
    return body->pose.jawHinge-c.toolOrigin+c.gapGradient*(body->pose.jawGap-c.toolGap);
}
// Tangential slip of a contact witness from its anchor on the tool. A jaw
// contact of the friction grasp keeps its anchor fixed in the jaw across
// substeps (its carried spring elongation, see k_UnifiedStick); every other
// anchor is the witness at the start of the substep.
__device__ inline float3 unifiedSlip(const UnifiedState& s,const SurfaceContact& c,float3 x,float3 n){
    const ContactShape& r=c.referenceShape;
    float3 slip=x-c.previous-c.plane.motion-unifiedShift(c,s.tool)+r.u*c.stick.x+r.n*c.stick.y+r.w*c.stick.z;
    return slip-n*dot3(slip,n);
}
__device__ inline float unifiedGap(const UnifiedState& s,const SurfaceContact& c,float3 x,float3& n){
    n=c.plane.normal;
    if(s.componentSupport && !s.componentSupport[s.component[c.id[0]]] && !(s.graspComponents[s.component[c.id[0]]]&1)){
        // Material components without live tetrahedra use the particle/edge
        // branch and swept tool contact, not volumetric fracture traction.
        return 1.f;
    }
    if(s.cutPhase && c.kind<3 && bladeYields(*s.cutPhase)){
        // A cut can leave a dangling rim joined by edges but by no volume
        // element. Its global component still contains the supported organ.
        // Do not apply continuum fracture traction to that empty stencil:
        // its tiny mass has no corresponding volume Hessian to resist it.
        bool volume=false;
        for(int j=0;j<c.count && !volume;j++)
            for(int k=s.offsets[c.id[j]];k<s.offsets[c.id[j]+1];k++)
                if(s.live[s.incident[k]]){volume=true;break;}
        if(!volume)return 1.f;
    }
    float C=dot3(x-c.plane.target-unifiedShift(c,s.tool),n);
    if(c.kind==0 || (s.tool->scissors && c.kind<3)){
        float3 radial,surface;float d=shapeDistance(bodyContactShape(s.tool,s.tool->pose,c.kind,s.thickness,s.minGap),x,radial,surface);
        // Update a cylindrical normal as its material witness moves. Keep the
        // swept entrance side after a deep crossing rather than flipping it.
        if(s.cutPhase ? bladeYields(*s.cutPhase) : dot3(radial,n)>.5f){n=radial;C=d-s.skin;}
        else if(s.cutPhase){
            // CCD may report a hit at the beginning of a tangential exit.
            // Its incident plane must not exert pressure after the CURRENT
            // finite blade has left that material patch.
            ContactShape shape=bodyContactShape(s.tool,s.tool->pose,c.kind,s.thickness,s.minGap);float3 nn,ss;
            float lateral=shapeDistance(shape,x-n*(C+s.skin+shape.radius),nn,ss);
            if(lateral>fmaxf(1e-6f,1e-3f*shape.radius))C=fmaxf(C,lateral);
        }
    }
    return C;
}
// Blade traction is integrated over one contact patch. A finer reconstructed
// surface must not multiply the total normal stiffness or fracture strength.
__device__ inline float unifiedContactStiffness(const UnifiedState& s,const SurfaceContact& c){
    if(s.cutPhase && c.kind<3){
        float count=s.cutContactCount?*s.cutContactCount:*s.contactCount;
        return s.contactK/fmaxf(1.f,count);
    }
    if(s.cutPhase && c.kind>=3)return 200000.f*(s.contactQuadrature?s.contactQuadrature[&c-s.contacts]:1.f);
    return s.contactK*(s.contactQuadrature?s.contactQuadrature[&c-s.contacts]:1.f);
}
// Unilateral Kelvin-Voigt normal dissipation in the SAME contact objective.
// Clip the complete viscoelastic pressure at zero: release cannot pull
// tissue along, and distant/noncontacting material has no extra damping.
__device__ inline float unifiedWitnessInverseMass(const UnifiedState& s,const SurfaceContact& c){
    float inverseMass=0;for(int j=0;j<c.count;j++)inverseMass+=c.weight[j]*c.weight[j]*s.invMass[c.id[j]];return inverseMass;
}
__device__ inline float unifiedImpactStiffness(const UnifiedState& s,const SurfaceContact& c,float inverseMass){
    if(s.cutPhase && c.kind<3)return 0; // Preserve the accepted pressure/fracture blade law.
    // The effective mass belongs to this embedded material witness. Counting
    // manifold samples as extra inverse mass makes dissipation jump whenever
    // clipping adds/removes a polygon vertex. Use the effective, area-weighted
    // stiffness INSIDE the square root of the critical damping coefficient.
    if(inverseMass<=0||s.h<=0)return 0;
    float quadrature=s.contactQuadrature?s.contactQuadrature[&c-s.contacts]:1.f;
    float stiffness=s.cutPhase && c.kind>=3?200000.f:s.contactK;
    return 2.f*sqrtf(quadrature*stiffness/inverseMass)/s.h;
}
__device__ inline float unifiedImpactStiffness(const UnifiedState& s,const SurfaceContact& c){
    return unifiedImpactStiffness(s,c,unifiedWitnessInverseMass(s,c));
}
__device__ inline float unifiedPreviousDepth(const SurfaceContact& c,float3 normal){
    float oldGap=dot3(c.previous-c.plane.target+c.plane.motion,normal);
    return fmaxf(0.f,-oldGap);
}
// The contact law at witness x of a stencil of the given inverse mass
// (unifiedWitnessInverseMass).
__device__ inline void unifiedContactAt(const UnifiedState& s,const SurfaceContact& c,float3 x,float inverseMass,float3& force,Symmetric3& hessian){
    float3 n=c.plane.normal;
    float K=unifiedContactStiffness(s,c);
    float C=unifiedGap(s,c,x,n);force=mkf3(0,0,0);
    // Complementarity includes the tangent block: zero normal pressure means
    // zero friction stiffness, even when the relative tangential speed is zero.
    if(C>=0)return;
    // The footprint is clipped when the manifold is assembled/refreshed.
    // Keep that active set fixed during a descent sweep: dropping individual
    // boundary quadrature points mid-sweep makes the objective discontinuous.
    float pressure=-K*C,frictionK=K;
    if(s.cutPhase && c.kind<3 && bladeYields(*s.cutPhase)){float cap=s.cutResistance*unifiedContactStiffness(s,c)/s.contactK;if(pressure>cap){pressure=cap;K=0;}}
    float impactK=unifiedImpactStiffness(s,c,inverseMass);
    if(impactK>0){
        pressure+=impactK*(-C-unifiedPreviousDepth(c,n));
        if(pressure<=0)return; // Never turn unloading dissipation into adhesion.
        K+=impactK;
    }
    force=n*(-pressure);hessian.outer(n,K);
    float3 slip=unifiedSlip(s,c,x,n);
    float length=len3(slip),kt=frictionK;
    // Regularized Coulomb traction: exact static spring inside the cone, dynamic
    // slip outside. Never applies friction without compressive normal pressure.
    float limit=s.muS*pressure;
    if(kt*length>limit){
        float mu=s.muD;
        if(!s.cutPhase || c.kind>=3){
            // Meet the static spring continuously, then approach dynamic
            // friction smoothly. A direct muS -> muD switch drops tangential
            // force in one iteration and can repeatedly release/re-stick.
            float ratio=limit/fmaxf(kt*length,1e-12f);
            mu+=(s.muS-s.muD)*ratio*ratio;
        }
        kt=mu*pressure/fmaxf(length,1e-12f);
    }
    force=force+slip*kt;
    hessian.diagonal(kt);hessian.outer(n,-kt);
}
__device__ inline void unifiedContact(const UnifiedState& s,const SurfaceContact& c,float3& force,Symmetric3& hessian){
    unifiedContactAt(s,c,unifiedPoint(c,s.pos),unifiedWitnessInverseMass(s,c),force,hessian);
}
__device__ inline float3 unifiedGripPoint(const GripPoint& p,const float3* x){
    float3 r=p.offset;for(int j=0;j<p.count;j++)r=r+x[p.id[j]]*p.weight[j];return r;
}
__global__ void k_PredictorInteractionEnergy(const UnifiedState* state,CoarseBlock* b){
    const UnifiedState& s=*state;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<max(*s.contactCount,s.grips);k+=blockDim.x*gridDim.x){
        if(k<*s.contactCount){
            const SurfaceContact& c=s.contacts[s.contactIds[k]];float3 shift=unifiedShift(c,s.tool);
            float K=unifiedContactStiffness(s,c),energy[2]={};
            for(int j=0;j<2;j++){
                float3 x=unifiedPoint(c,j?s.pos:s.previous),normal;float C=fminf(0.f,unifiedGap(s,c,x,normal));
                float D=unifiedImpactStiffness(s,c),effectiveK=K+D;float compression=fmaxf(0.f,-C-D*unifiedPreviousDepth(c,normal)/fmaxf(effectiveK,1e-20f));energy[j]=.5f*effectiveK*compression*compression;
                if(s.cutPhase && c.kind<3 && bladeYields(*s.cutPhase)){float cap=s.cutResistance*K/s.contactK;float d=-C,y=cap/K;if(d>y)energy[j]=cap*(d-.5f*y);}
            }
            atomicAdd(&b[s.component[c.id[0]]].yy,energy[1]-energy[0]);
        }
        if(k<s.grips && s.points[k].valid){
            const GripPoint& a=s.points[k];float3 target=gripTarget(a,s.tool->pose,s.minGap);
            float3 old=unifiedGripPoint(a,s.previous)-target,trial=unifiedGripPoint(a,s.pos)-target;
            float stiffness=s.gripK/a.complianceScale;
            atomicAdd(&b[s.component[a.id[0]]].yy,.5f*stiffness*dot3(trial-old,trial+old));
        }
    }
}
// Lattice color of a vertex in the eight-color sweeps (parities of x, y, z).
__device__ inline int unifiedColor(const UnifiedState& s,int i){
    return ((i%s.nx)&1)|((((i/s.nx)%s.ny)&1)<<1)|(((i/(s.nx*s.ny))&1)<<2);
}
// Only the vertices of the sweep's color read their interactions, so only
// theirs are cleared and assembled before it.
__global__ void k_ClearUnifiedInteractions(const UnifiedState* state,int color){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n || unifiedColor(s,i)!=color)return;
    if((s.degree[i]>0 || (s.secondary && s.secondary->degree[i]>0)) || (s.grips>0 && s.gripDegree[i]>0))for(int j=0;j<9;j++)s.interaction[9*i+j]=0;
}
__device__ inline void scatterInteraction(float* out,float3 f,const Symmetric3& H,float w){
    atomicAdd(out,f.x*w);atomicAdd(out+1,f.y*w);atomicAdd(out+2,f.z*w);
    atomicAdd(out+3,H.xx*w);atomicAdd(out+4,H.yy*w);atomicAdd(out+5,H.zz*w);
    atomicAdd(out+6,H.xy*w);atomicAdd(out+7,H.xz*w);atomicAdd(out+8,H.yz*w);
}
// Sum v over the 32 lanes of a warp; every lane receives the sums.
template<int N> __device__ inline void unifiedWarpSum(float (&v)[N]){
    for(int j=0;j<N;j++){
        for(int offset=16;offset;offset/=2)v[j]+=__shfl_down_sync(0xffffffff,v[j],offset);
        v[j]=__shfl_sync(0xffffffff,v[j],0);
    }
}
// One warp per contact or grip point: each lane loads one stencil node, the
// lanes sum the witness, then scatter to their node if it has the swept color.
// One thread per contact walked the whole stencil through uncoalesced loads.
// Every lane evaluates the same contact law, which needs no broadcast.
constexpr int ASSEMBLY_BLOCKS=512;
__global__ void k_AssembleUnifiedInteractions(const UnifiedState* state,int color){
    const UnifiedState& s=*state;int lane=threadIdx.x%32,warps=gridDim.x*blockDim.x/32;
    for(int k=(blockIdx.x*blockDim.x+threadIdx.x)/32;k<max(*s.contactCount,s.grips);k+=warps){
        if(k<*s.contactCount){
            const SurfaceContact& c=s.contacts[s.contactIds[k]];bool node=lane<c.count;
            int id=node?c.id[lane]:0;float w=node?c.weight[lane]:0.f;float3 p=node?s.pos[id]*w:mkf3(0,0,0);
            float v[4]={p.x,p.y,p.z,node?w*w*s.invMass[id]:0.f};unifiedWarpSum(v);
            float3 f;Symmetric3 H={};unifiedContactAt(s,c,c.offset+mkf3(v[0],v[1],v[2]),v[3],f,H);
            if(node && unifiedColor(s,id)==color)scatterInteraction(s.interaction+9*id,f,H,w);
        }
        if(k<s.grips && s.points[k].valid){
            const GripPoint& a=s.points[k];bool node=lane<a.count;
            int id=node?a.id[lane]:0;float w=node?a.weight[lane]:0.f;float3 p=node?s.pos[id]*w:mkf3(0,0,0);
            float v[3]={p.x,p.y,p.z};unifiedWarpSum(v);
            float stiffness=s.gripK/a.complianceScale;Symmetric3 H={};H.diagonal(stiffness);
            float3 f=(a.offset+mkf3(v[0],v[1],v[2])-gripTarget(a,s.tool->pose,s.minGap))*stiffness;
            if(node && unifiedColor(s,id)==color)scatterInteraction(s.interaction+9*id,f,H,w);
        }
    }
}
// Warm-start from the inertial prediction, preserving unforced rigid motion.
// A cold start at previous positions plus a fixed small solve budget behaves
// like mesh-dependent viscosity. Accept each colored predictor increment up to
// the same affine determinant bound used by the material solve; contact planes
// were discovered over the full previous-to-predicted swept trajectory.
// One warp per vertex, lanes over its incident tets (see k_UnifiedVertex).
__global__ void k_UnifiedPredictor(const UnifiedState* state,int color){
    const UnifiedState& s=*state;
    int thread=blockIdx.x*blockDim.x+threadIdx.x,lane=thread%32,site=thread/32,cx=(s.nx+1)/2,cy=(s.ny+1)/2,nz=s.n/(s.nx*s.ny);
    int x=2*(site%cx)+(color&1),y=2*((site/cx)%cy)+((color>>1)&1),z=2*(site/(cx*cy))+((color>>2)&1);
    if(x>=s.nx || y>=s.ny || z>=nz)return;
    int i=x+s.nx*(y+s.ny*z);if(s.invMass[i]<=0)return;
    // Keep only the actual material contact/anchor witnesses at their accepted
    // configuration. Unconstrained tissue retains its inertial warm start.
    if((s.degree[i]>0 || (s.secondary && s.secondary->degree[i]>0)) || (s.grips>0 && s.gripDegree[i]>0))return;
    float3 p=s.pos[i],step=s.target[i]-p;float scale=1;
    for(int k=s.offsets[i]+lane;k<s.offsets[i+1];k+=32){
        int t=s.incident[k];if(!s.live[t])continue;const int* ids=s.ids+4*t;
        float3 a=s.pos[ids[0]],u=s.pos[ids[1]]-a,v=s.pos[ids[2]]-a,w=s.pos[ids[3]]-a;
        float det=s.restMetric[7*t+6],J=dot3(u,cross3(v,w))*det;
        float3 g1=cross3(v,w),g2=cross3(w,u),g3=cross3(u,v);
        float3 g=i==ids[0]?(g1+g2+g3)*(-1.f):i==ids[1]?g1:i==ids[2]?g2:g3;
        float dj=dot3(g,step)*det;
        if(dj<0)scale=fminf(scale,fmaxf(0.f,.9f*(J-s.minimum)/(-dj)));
    }
    for(int offset=16;offset;offset/=2)scale=fminf(scale,__shfl_xor_sync(0xffffffff,scale,offset));
    if(lane==0 && isfinite(step.x+step.y+step.z))s.pos[i]=p+step*scale;
}
// Sum one vertex's gradient and Hessian over the lanes of its warp. Every
// lane receives lane 0's total, so all lanes solve the same local step.
__device__ inline void unifiedWarpSum(float3& gradient,Symmetric3& H){
    float v[9]={gradient.x,gradient.y,gradient.z,H.xx,H.yy,H.zz,H.xy,H.xz,H.yz};unifiedWarpSum(v);
    gradient=mkf3(v[0],v[1],v[2]);H={v[3],v[4],v[5],v[6],v[7],v[8]};
}
// A colored sweep gives each vertex one warp whose lanes split its incident
// tets and edges: with one thread per vertex, a color's few thousand threads
// leave most of the GPU idle while each walks its tets in turn. The residual
// pass over all vertices keeps one thread per vertex.
__global__ void k_UnifiedVertex(const UnifiedState* state,int color,int localOnly,float3* residual=nullptr,Symmetric3* diagonal=nullptr){
    const UnifiedState& s=*state;
    int lanes=color<0?1:32,thread=blockIdx.x*blockDim.x+threadIdx.x,lane=thread%lanes,site=thread/lanes;
    int cx=(s.nx+1)/2,cy=(s.ny+1)/2,nz=s.n/(s.nx*s.ny);
    int x=2*(site%cx)+(color&1),y=2*((site/cx)%cy)+((color>>1)&1),z=2*(site/(cx*cy))+((color>>2)&1);
    if(color>=0 && (x>=s.nx || y>=s.ny || z>=nz))return;
    int i=color<0?site:x+s.nx*(y+s.ny*z);if(i>=s.n)return;
    if(residual){residual[i]=mkf3(0,0,0);diagonal[i]={1,1,1,0,0,0};}
    float inv=s.invMass[i];if(inv<=0 || (residual && materialCgFixed(s,i)))return;
    if(!residual && localOnly!=1 && s.coupledReady && !materialCgFixed(s,i))return;
    if(localOnly && s.coarseWeight[i]<=0 && (s.degree[i]==0 && !(s.secondary && s.secondary->degree[i]>0)) && !(s.grips>0 && s.gripDegree[i]>0))return;
    float3 p=s.pos[i],gradient=mkf3(0,0,0);Symmetric3 H={};
    if(lane==0){
        float inertia=1.f/(inv*s.h*s.h);H.diagonal(inertia);gradient=(p-s.target[i])*inertia;
        if(s.attach && s.attach[i]>0){float viscous=unifiedSupportDamping(s,i);H.diagonal(s.attach[i]+viscous);
            gradient=gradient+(p-s.attachRest[i])*s.attach[i]+(p-s.previous[i])*viscous;}
    }
    // Keep surviving cut-rim lattice edges that no longer have a complete tet.
    // Neighbor slots also exist outside the active tissue. Like k_RefreshLiveness,
    // require BOTH endpoints active: an inactive grid corner is not a support.
    // Do not use invMass here: an active pinned endpoint is a real support.
    if(s.neighbors && s.edgeSupport)for(int a=lane;a<6;a+=lanes){
        int j=s.neighbors[6*i+a];if(j<0 || (s.active && !s.active[j]))continue;
        int owner=(a&1)?6*i+a:6*j+(a^1);if(s.edgeSupport[owner])continue;
        float3 e=p-s.pos[j];float l=len3(e),l0=len3(s.restNeighbors[6*i+a]);
        if(l>1e-10f){float3 n=e*(1.f/l);float strain=l-l0,K=.5f*(unifiedEdgeStiffness(s,i,a/2)+unifiedEdgeStiffness(s,j,a/2));
            gradient=gradient+n*(K*strain);H.outer(n,K);H.diagonal(K*fmaxf(0.f,strain/l));}
    }
    for(int k=s.offsets[i]+lane;k<s.offsets[i+1];k+=lanes){
        int t=s.incident[k];if(!s.live[t])continue;const int* ids=s.ids+4*t;
        float3 a=s.pos[ids[0]],e1=s.pos[ids[1]]-a,e2=s.pos[ids[2]]-a,e3=s.pos[ids[3]]-a;
        float3 b[3]={s.b0[t],s.b1[t],s.b2[t]};
        float3 q=i==ids[0]?(b[0]+b[1]+b[2])*(-1.f):i==ids[1]?b[0]:i==ids[2]?b[1]:b[2];
        float3 Fq=e1*dot3(b[0],q)+e2*dot3(b[1],q)+e3*dot3(b[2],q);
        float det=s.restMetric[7*t+6];float J=dot3(e1,cross3(e2,e3))*det;
        float3 g1=cross3(e2,e3)*det,g2=cross3(e3,e1)*det,g3=cross3(e1,e2)*det;
        float3 g=i==ids[0]?(g1+g2+g3)*(-1.f):i==ids[1]?g1:i==ids[2]?g2:g3;
        // Stable Neo-Hookean without logarithm. lambdaStable = kappa + mu/3
        // preserves the requested small-strain Lame moduli and zero rest force.
        float mu=s.mu[t],lambda=s.bulk[t]+mu/3.f,pressure=lambda*(J-1.f)-mu;
        CompressionBarrier barrier=unifiedCompressionBarrier(J,s.minimum,lambda);
        gradient=gradient+Fq*mu+g*(pressure+barrier.gradient);
        H.diagonal(mu*dot3(q,q));H.outer(g,lambda+barrier.curvature);
        unifiedStrainDamping(s,t,e1,e2,e3,q,gradient,H);
        // The existing finite tensile reinforcement, expressed as an energy
        // gradient instead of a separate projection that could undo contact.
        int me=i==ids[0]?0:i==ids[1]?1:i==ids[2]?2:3;
        for(int j=0;j<4;j++)if(j!=me && s.hardening>0){
            float3 e=p-s.pos[ids[j]];float l=len3(e),l0=unifiedRestLength(s,t,me,j),strain=fmaxf(0.f,l/l0-1.f);
            if(strain>0 && l>1e-12f){float3 dir=e*(1.f/l);float s2=strain*strain,s4=s2*s2;float f=3.f*mu*s.hardening*s4*strain/l0;
                float h=15.f*mu*s.hardening*s4/(l0*l0);
                gradient=gradient+dir*f;H.diagonal(f/l);H.outer(dir,h-f/l);}
        }
    }
    if(lane==0 && ((s.degree[i]>0 || (s.secondary && s.secondary->degree[i]>0)) || (s.grips>0 && s.gripDegree[i]>0))){
        // Assemble each embedded contact/anchor once per color, rather than
        // recomputing its 24-node stencil separately at every supporting node.
        const float* f=s.interaction+9*i;gradient=gradient+mkf3(f[0],f[1],f[2]);
        H.xx+=f[3];H.yy+=f[4];H.zz+=f[5];H.xy+=f[6];H.xz+=f[7];H.yz+=f[8];
    }
    if(lanes>1)unifiedWarpSum(gradient,H);
    if(residual){residual[i]=gradient*(-1.f);diagonal[i]=H;return;}
    float3 step=H.solve(gradient)*(-1.f);float scale=1;
    // J is affine in this vertex with all other vertices fixed. Eight lattice
    // colors never share a tetrahedron, so this bound survives parallel writes.
    for(int k=s.offsets[i]+lane;k<s.offsets[i+1];k+=lanes){
        int t=s.incident[k];if(!s.live[t])continue;const int* ids=s.ids+4*t;
        float3 a=s.pos[ids[0]],u=s.pos[ids[1]]-a,v=s.pos[ids[2]]-a,w=s.pos[ids[3]]-a;
        float det=s.restMetric[7*t+6],J=dot3(u,cross3(v,w))*det;
        float3 g1=cross3(v,w),g2=cross3(w,u),g3=cross3(u,v);
        float3 g=i==ids[0]?(g1+g2+g3)*(-1.f):i==ids[1]?g1:i==ids[2]?g2:g3;
        float dj=dot3(g,step)*det;
        if(dj<0)scale=fminf(scale,fmaxf(0.f,.9f*(J-s.minimum)/(-dj)));
    }
    for(int offset=16;lanes>1 && offset;offset/=2)scale=fminf(scale,__shfl_xor_sync(0xffffffff,scale,offset));
    // Same-color vertices share neither a lattice tetrahedron nor an axial
    // cut-rim edge. Embedded interactions were assembled before this kernel,
    // so each vertex can commit directly without a second whole-grid pass.
    if(lane==0 && isfinite(step.x+step.y+step.z))s.pos[i]=p+step*scale;
}
__global__ void k_UnifiedTool(UnifiedState* state){
    UnifiedState& s=*state;CoupledTool& b=*s.tool;
    float A[4][4]={},r[4]={};float inertia=1.f/(b.inverseMass*s.h*s.h),gapInertia=b.gapInverseMass>0?1.f/(b.gapInverseMass*s.h*s.h):1e20f;
    if(threadIdx.x==0){
        for(int j=0;j<3;j++){A[j][j]=inertia;r[j]=(compF3(b.pose.jawHinge,j)-compF3(s.toolTarget,j))*inertia;}
        A[3][3]=gapInertia;r[3]=(b.pose.jawGap-s.gapTarget)*gapInertia;
    }
    for(int k=threadIdx.x;k<*s.contactCount;k+=blockDim.x){
        SurfaceContact& c=s.contacts[s.contactIds[k]];float3 force;Symmetric3 H={};unifiedContact(s,c,force,H);
        float3 hg=mkf3(H.xx*c.gapGradient.x+H.xy*c.gapGradient.y+H.xz*c.gapGradient.z,H.xy*c.gapGradient.x+H.yy*c.gapGradient.y+H.yz*c.gapGradient.z,H.xz*c.gapGradient.x+H.yz*c.gapGradient.y+H.zz*c.gapGradient.z);
        float mat[3][3]={{H.xx,H.xy,H.xz},{H.xy,H.yy,H.yz},{H.xz,H.yz,H.zz}};
        for(int i=0;i<3;i++){r[i]-=compF3(force,i);for(int j=0;j<3;j++)A[i][j]+=mat[i][j];A[i][3]+=compF3(hg,i);A[3][i]+=compF3(hg,i);}
        r[3]-=dot3(force,c.gapGradient);A[3][3]+=dot3(c.gapGradient,hg);
    }
    for(int k=threadIdx.x;k<s.grips;k+=blockDim.x)if(s.points[k].valid){
        const GripPoint& a=s.points[k];float stiffness=s.gripK/a.complianceScale;
        float3 target=gripTarget(a,b.pose,s.minGap),error=unifiedGripPoint(a,s.pos)-target;
        XpbdToolState plus=b.pose;float eps=1e-4f;plus.jawGap+=eps;
        float3 g=(gripTarget(a,plus,s.minGap)-target)*(1.f/eps);
        for(int i=0;i<3;i++){r[i]-=stiffness*compF3(error,i);A[i][i]+=stiffness;A[i][3]+=stiffness*compF3(g,i);A[3][i]+=stiffness*compF3(g,i);}
        r[3]-=stiffness*dot3(error,g);A[3][3]+=stiffness*dot3(g,g);
    }
    // One warp assembles the same four-DOF objective. A single lane walking
    // every polygon witness and grip stencil serialized every local sweep.
    for(int d=16;d;d>>=1){
        for(int i=0;i<4;i++){
            r[i]+=__shfl_down_sync(0xffffffffu,r[i],d);
            for(int j=0;j<4;j++)A[i][j]+=__shfl_down_sync(0xffffffffu,A[i][j],d);
        }
    }
    if(threadIdx.x!=0)return;
    if(b.guided){
        // Keyboard instrument is supported by the operator. Solve only motion
        // along the requested travel, plus aperture; lateral reaction is borne
        // by the guide, not converted to an unsolicited sideways displacement.
        float3 axis=b.guideAxis;float aa=0,ab=0,rr=0;
        for(int i=0;i<3;i++){rr+=compF3(axis,i)*r[i];ab+=compF3(axis,i)*A[i][3];for(int j=0;j<3;j++)aa+=compF3(axis,i)*A[i][j]*compF3(axis,j);}
        float dd=0,dg=0,den=aa*A[3][3]-ab*ab;
        if(b.guideLength>0 && den>1e-20f){dd=(-rr*A[3][3]+ab*r[3])/den;dg=(-aa*r[3]+ab*rr)/den;}
        else if(A[3][3]>0)dg=-r[3]/A[3][3];
        float current=dot3(b.pose.jawHinge-b.guideOrigin,axis);
        float accepted=fminf(b.guideLength,fmaxf(0.f,current+dd));
        b.pose=shiftTool(b.pose,b.guideOrigin+axis*accepted-b.pose.jawHinge);
        b.pose.jawGap=fminf(b.gapHi,fmaxf(b.gapLo,b.pose.jawGap+dg));return;
    }
    // SPD Cholesky solve; four physical DOFs, with no gap/translation latch.
    float L[4][4]={},y[4]={},dx[4]={};
    for(int i=0;i<4;i++)for(int j=0;j<=i;j++){float v=A[i][j];for(int k=0;k<j;k++)v-=L[i][k]*L[j][k];L[i][j]=i==j?sqrtf(fmaxf(v,1e-12f)):v/L[j][j];}
    for(int i=0;i<4;i++){float v=-r[i];for(int j=0;j<i;j++)v-=L[i][j]*y[j];y[i]=v/L[i][i];}
    for(int i=3;i>=0;i--){float v=y[i];for(int j=i+1;j<4;j++)v-=L[j][i]*dx[j];dx[i]=v/L[i][i];}
    b.pose=shiftTool(b.pose,mkf3(dx[0],dx[1],dx[2]));b.pose.jawGap=fminf(b.maxGap,fmaxf(0.f,b.pose.jawGap+dx[3]));
}
__global__ void k_UnifiedTargets(UnifiedState* state,bool preserveDriveWork=true){
    state->toolTarget=preserveDriveWork?state->tool->drivePrediction:state->tool->pose.jawHinge;
    state->gapTarget=state->tool->pose.jawGap;
}
__global__ void k_UnifiedRebind(UnifiedState* s,SurfaceContact* contacts,const int* ids,GripPoint* points,int grips,const int* count,const float* quadrature){
    s->contactCount=count;s->contactQuadrature=quadrature;s->contacts=contacts;s->contactIds=ids;s->points=points;s->grips=grips;
}
__global__ void k_UnifiedReaction(const UnifiedState* state){
    const UnifiedState& s=*state;for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<max(*s.contactCount,s.grips);k+=blockDim.x*gridDim.x){
    if(k<*s.contactCount){SurfaceContact& c=s.contacts[s.contactIds[k]];float3 force;Symmetric3 H={};unifiedContact(s,c,force,H);
        c.lambdaN=fmaxf(0.f,-dot3(force,c.plane.normal))*s.h*s.h;c.lambdaT=(force+c.plane.normal*(c.lambdaN/(s.h*s.h)))*(-s.h*s.h);}
    if(k<s.grips){GripPoint& a=s.points[k];a.lambda=a.valid?(unifiedGripPoint(a,s.pos)-gripTarget(a,s.tool->pose,s.minGap))*(-s.gripK*s.h*s.h/a.complianceScale):mkf3(0,0,0);}}
}

// Optional read-only diagnostic of the accepted grasper contact footprint.
// Mark material nodes, not reconstructed triangle indices: the latter change
// order and count on reconstruction. Retained CCD witnesses with zero pressure
// do not count as loaded tissue. Bits 1/2/4 identify shaft/upper/lower jaw.
__global__ void k_ReadPressureFootprint(const UnifiedState* state,int* flags){
    const auto& s=*state;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*s.contactCount;k+=blockDim.x*gridDim.x){
        const auto& c=s.contacts[s.contactIds[k]];float3 normal;
        if(unifiedGap(s,c,unifiedPoint(c,s.pos),normal)>=0)continue;
        float3 force;Symmetric3 H={};unifiedContact(s,c,force,H);
        if(-dot3(force,normal)<=0)continue;
        for(int j=0;j<c.count;j++)if(c.weight[j]>1e-6f)atomicOr(flags+c.id[j],1<<c.kind);
    }
}

// A smooth three-dimensional subspace transports the tool and its local tissue
// together. It minimizes the same objective; it adds no attachment or force.
// This accelerates the low-frequency mode that vertex-only relaxation handles
// poorly when very stiff contact couples a light tool to many fine-grid nodes.
__global__ void k_UnifiedPatchBasis(UnifiedState* state){
    UnifiedState& s=*state;float3 center=mkf3(0,0,0);float sum=0;
    for(int k=threadIdx.x;k<s.grips;k+=32)if(s.points[k].valid){
        atomicExch(s.graspComponents+s.component[s.points[k].id[0]],1);
        float w=1.f/s.points[k].complianceScale;center=center+gripTarget(s.points[k],s.tool->pose,s.minGap)*w;sum+=w;
    }
    // Unheld jaw pressure uses the actual contact footprint, not a large ball
    // centered on the tool. Blade and shaft contacts keep their existing basis.
    if(s.grips==0)for(int k=threadIdx.x;k<*s.contactCount;k+=32){
        int slot=s.contactIds[k];const auto& c=s.contacts[slot];if(c.kind==0)continue;
        float w=s.contactQuadrature?s.contactQuadrature[slot]:1.f;
        center=center+unifiedPoint(c,s.pos)*w;sum+=w;
        atomicOr(s.graspComponents+s.component[c.id[0]],4);
    }
    for(int d=16;d;d>>=1){center.x+=__shfl_down_sync(0xffffffff,center.x,d);center.y+=__shfl_down_sync(0xffffffff,center.y,d);center.z+=__shfl_down_sync(0xffffffff,center.z,d);sum+=__shfl_down_sync(0xffffffff,sum,d);}
    sum=__shfl_sync(0xffffffff,sum,0);
    center=mkf3(__shfl_sync(0xffffffff,center.x,0),__shfl_sync(0xffffffff,center.y,0),__shfl_sync(0xffffffff,center.z,0));
    center=sum>0?center*(1.f/sum):s.tool->pose.jawHinge+s.tool->pose.jawAxis*(.5f*s.tool->pose.jawLen);
    float radius=(sum>0?.5f:1.5f)*s.tool->pose.jawLen;
    for(int k=threadIdx.x;k<s.grips;k+=32)if(s.points[k].valid){
        const GripPoint& a=s.points[k];for(int j=0;j<a.count;j++)if(a.weight[j]>1e-6f)radius=fmaxf(radius,len3(s.pos[a.id[j]]-center));
    }
    if(s.grips==0 && sum>0)for(int k=threadIdx.x;k<*s.contactCount;k+=32){
        const auto& c=s.contacts[s.contactIds[k]];if(c.kind!=0)radius=fmaxf(radius,len3(unifiedPoint(c,s.pos)-center));
    }
    for(int d=16;d;d>>=1)radius=fmaxf(radius,__shfl_down_sync(0xffffffff,radius,d));
    if(threadIdx.x==0){s.patchOffset=center-s.tool->pose.jawHinge;s.patchCoreRadius=radius;s.patchFalloff=(sum>0?2.f:1.5f)*s.tool->pose.jawLen;}
}
__global__ void k_CoarseWeights(UnifiedState* state,int mode=0){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    if(i==0)state->coarseMode=mode;
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
__device__ inline void coarseAccumulate(CoarseBlock* b,float3 g,const Symmetric3& H){
    float v[9]={g.x,g.y,g.z,H.xx,H.yy,H.zz,H.xy,H.xz,H.yz};
    // Most interior warps belong to one component. Reduce their terms before
    // atomics, avoiding hundreds of thousands of updates to one root address.
    unsigned mask=__activemask();
    unsigned long long address=(unsigned long long)b;
    unsigned long long first=__shfl_sync(mask,address,__ffs(mask)-1);
    if(__all_sync(mask,address==first)){
        if(mask==0xffffffffu){
            for(int d=16;d;d>>=1)for(int j=0;j<9;j++)v[j]+=__shfl_down_sync(mask,v[j],d);
            if((threadIdx.x&31)!=0)return;
        }else{
            // Surface/cut tetrahedra leave sparse active warps. Compact those
            // lanes before reducing, instead of serializing nine atomics per
            // surviving lane at the same component root.
            auto group=cooperative_groups::coalesced_threads();
            for(int j=0;j<9;j++)v[j]=cooperative_groups::reduce(group,v[j],cooperative_groups::plus<float>());
            if(group.thread_rank()!=0)return;
        }
    }
    atomicAdd(&b->gradient.x,v[0]);atomicAdd(&b->gradient.y,v[1]);atomicAdd(&b->gradient.z,v[2]);
    atomicAdd(&b->xx,v[3]);atomicAdd(&b->yy,v[4]);atomicAdd(&b->zz,v[5]);
    atomicAdd(&b->xy,v[6]);atomicAdd(&b->xz,v[7]);atomicAdd(&b->yz,v[8]);
}
// Connected-component coarse modes accelerate long-range force transmission.
// Each severed piece has its own reduction/step, so a cut cannot introduce an
// artificial global spring between unrelated fragments.
__device__ inline int unifiedRoot(const int* parent,int i){while(parent[i]!=i)i=parent[i];return i;}
__global__ void k_InitComponents(int n,int* parent){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)parent[i]=i;}
__global__ void k_JoinComponents(int n,const int* neighbors,const int* active,int* parent){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n || (active && !active[i]))return;
    for(int a=1;a<6;a+=2){int j=neighbors[6*i+a];if(j<0 || (active && !active[j]))continue;
        while(true){int x=unifiedRoot(parent,i),y=unifiedRoot(parent,j);if(x==y)break;int hi=max(x,y),lo=min(x,y);if(atomicCAS(parent+hi,hi,lo)==hi)break;}}
}
__global__ void k_FlattenComponents(int n,int* parent){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)parent[i]=unifiedRoot(parent,i);}
// A nearly active volume bound must not reduce an entire graded direction to
// zero. Give that small connected set of tetrahedra a common scalar weight:
// the coarse update translates it exactly, including after contact releases.
// The ordinary vertex solver
// still resolves its material strain. No material or volume bound is changed.
__global__ void k_InitSafeCoarse(const UnifiedState* state){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i==0)*s.safeJoined=0;
    if(i>=s.n)return;s.safeCluster[i]=i;s.safeAggregate[i]={};
}
__global__ void k_MarkCoarseContact(const UnifiedState* state){
    const UnifiedState& s=*state;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*s.contactCount;k+=blockDim.x*gridDim.x)
        atomicOr(s.graspComponents+s.component[s.contacts[s.contactIds[k]].id[0]],2);
}
// Preserve feasible collective elastic descent after contact releases.
// Otherwise a collapsed local tet clips the PCG direction while the coarse
// correction is disabled solely because contact vanished. Flag 8 is numerical
// recovery only: it creates no anchor, tool motion, or reduced CG budget.
__global__ void k_MarkVolumeRecovery(const UnifiedState* state){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=s.nt||!s.live[t]||s.minimum<=0)return;const int* id=s.ids+4*t;
    float3 a=s.pos[id[0]];float J=dot3(s.pos[id[1]]-a,cross3(s.pos[id[2]]-a,s.pos[id[3]]-a))*s.restMetric[7*t+6];
    if(J<=s.minimum+.001f)atomicOr(s.graspComponents+s.component[id[0]],8);
}
__global__ void k_JoinSafeCoarse(const UnifiedState* state){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=s.nt || !s.live[t] || s.minimum<=0)return;
    const int* id=s.ids+4*t;
    float3 a=s.pos[id[0]];
    float J=dot3(s.pos[id[1]]-a,cross3(s.pos[id[2]]-a,s.pos[id[3]]-a))*s.restMetric[7*t+6];
    if(J>s.minimum+fmaxf(1e-5f,1e-3f*s.minimum))return;
    *s.safeJoined=1;
    for(int k=1;k<4;k++)while(true){
        int x=unifiedRoot(s.safeCluster,id[0]),y=unifiedRoot(s.safeCluster,id[k]);
        if(x==y)break;int hi=max(x,y),lo=min(x,y);if(atomicCAS(s.safeCluster+hi,hi,lo)==hi)break;
    }
}
// Without a join every cluster is a single node, whose finished weight is its own coarse weight: k_CoarseWeights has
// already set it to 0 on a pinned node, so flattening, accumulating and finishing change nothing then.
__global__ void k_FlattenSafeCoarse(const UnifiedState* state){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<s.n && *s.safeJoined)s.safeCluster[i]=unifiedRoot(s.safeCluster,i);
}
__global__ void k_AccumulateSafeCoarse(const UnifiedState* state){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n || !*s.safeJoined)return;
    SafeCoarseCluster* c=s.safeAggregate+s.safeCluster[i];
    atomicAdd(&c->weight,s.coarseWeight[i]);atomicAdd(&c->count,1);
    if(s.invMass[i]<=0)atomicExch(&c->pinned,1);
    if(s.grips>0 && s.gripDegree[i]>0){atomicAdd(&c->gripWeight,s.coarseWeight[i]);atomicAdd(&c->grips,1);}
}
__global__ void k_FinishSafeCoarse(const UnifiedState* state){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n || !*s.safeJoined)return;
    const SafeCoarseCluster& c=s.safeAggregate[s.safeCluster[i]];
    s.coarseWeight[i]=c.pinned?0.f:c.grips?c.gripWeight/c.grips:c.weight/fmaxf(1.f,float(c.count));
}
// A per-component mode accumulates into the block of each active component root, and every reader reaches a block
// through a component root (k_CoarseApply). Only those blocks are cleared and solved: the other blocks of the
// lattice-sized array are never read.
__device__ inline bool coarseRoot(const UnifiedState& s,int i){return i<s.n && s.component[i]==i && (!s.active || s.active[i]);}
__global__ void k_CoarseBegin(const UnifiedState* state,CoarseBlock* b){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(s.coarseMode){if(coarseRoot(s,i)){b[i]={};b[i].scale=1;}return;}
    if(i)return;float inertia=1.f/(s.tool->inverseMass*s.h*s.h);
    *b={};b->gradient=(s.tool->pose.jawHinge-s.toolTarget)*inertia;b->xx=b->yy=b->zz=inertia;b->scale=1;
}
__global__ void k_CoarseNodes(const UnifiedState* state,CoarseBlock* block){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n || (s.active && !s.active[i]))return;
    if(s.coarseMode)block+=s.component[i];
    if(s.coarseMode==6 && (s.invMass[i]<=0 || (s.attach && s.attach[i]>0)))atomicExch(&block->external,1);
    float w=s.coarseWeight[i];
    float inertia=w!=0?1.f/(s.invMass[i]*s.h*s.h):0;Symmetric3 H={};H.diagonal(inertia*w*w);
    float3 g=(s.pos[i]-s.target[i])*(inertia*w);
    if(s.attach){float viscous=unifiedSupportDamping(s,i);
        g=g+((s.pos[i]-s.attachRest[i])*s.attach[i]+(s.pos[i]-s.previous[i])*viscous)*w;H.diagonal((s.attach[i]+viscous)*w*w);}
    if(s.neighbors && s.edgeSupport)for(int a=1;a<6;a+=2){
        int j=s.neighbors[6*i+a];if(j<0 || (s.active && !s.active[j]) || s.edgeSupport[6*i+a])continue;
        float dw=w-s.coarseWeight[j];float3 e=s.pos[i]-s.pos[j];float l=len3(e),l0=len3(s.restNeighbors[6*i+a]);
        if(l>1e-10f){float3 n=e*(1.f/l);float strain=l-l0,K=.5f*(unifiedEdgeStiffness(s,i,a/2)+unifiedEdgeStiffness(s,j,a/2));
            g=g+n*(K*strain*dw);H.outer(n,K*dw*dw);H.diagonal(K*fmaxf(0.f,strain/l)*dw*dw);}
    }
    if(H.xx+H.yy+H.zz>0)coarseAccumulate(block,g,H);
}
// One reduction per participating warp/component, including partial warps.
// A global volume line bound otherwise serializes one atomic at every tet.
__device__ inline void coarseBound(CoarseBlock* block,float scale){
    unsigned mask=__activemask();int first=__ffs(mask)-1;
    unsigned long long address=(unsigned long long)block;
    unsigned long long root=__shfl_sync(mask,address,first);
    unsigned value=__float_as_uint(scale);
    if(__all_sync(mask,address==root)){
        value=__reduce_min_sync(mask,value);
        if((threadIdx.x&31)!=first)return;
    }
    atomicMin((unsigned*)&block->scale,value);
}
__global__ void k_CoarseTets(const UnifiedState* state,CoarseBlock* block,bool bound){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=s.nt || !s.live[t])return;
    const int* id=s.ids+4*t;if(s.coarseMode)block+=s.component[id[0]];float w[4];for(int j=0;j<4;j++)w[j]=s.coarseWeight[id[j]];
    float3 q=s.b0[t]*(w[1]-w[0])+s.b1[t]*(w[2]-w[0])+s.b2[t]*(w[3]-w[0]);
    if(dot3(q,q)<1e-20f)return;
    float3 a=s.pos[id[0]],e1=s.pos[id[1]]-a,e2=s.pos[id[2]]-a,e3=s.pos[id[3]]-a;
    float det=s.restMetric[7*t+6],J=dot3(e1,cross3(e2,e3))*det;
    float3 gJ=(cross3(e2,e3)*(w[1]-w[0])+cross3(e3,e1)*(w[2]-w[0])+cross3(e1,e2)*(w[3]-w[0]))*det;
    if(bound){float dj=dot3(gJ,block->step);if(dj<0)coarseBound(block,fminf(1.f,fmaxf(0.f,.9f*(J-s.minimum)/(-dj))));return;}
    float3 Fq=e1*dot3(s.b0[t],q)+e2*dot3(s.b1[t],q)+e3*dot3(s.b2[t],q);
    float mu=s.mu[t],lambda=s.bulk[t]+mu/3.f;
    CompressionBarrier barrier=unifiedCompressionBarrier(J,s.minimum,lambda);
    float3 g=Fq*mu+gJ*(lambda*(J-1)-mu+barrier.gradient);Symmetric3 H={};H.diagonal(mu*dot3(q,q));H.outer(gJ,lambda+barrier.curvature);
    unifiedStrainDamping(s,t,e1,e2,e3,q,g,H);
    for(int j=0;j<4;j++)for(int k=j+1;k<4;k++){
        float dw=w[j]-w[k];if(dw==0)continue;float3 e=s.pos[id[j]]-s.pos[id[k]];
        float l=len3(e),l0=unifiedRestLength(s,t,j,k),strain=fmaxf(0.f,l/l0-1.f);
        if(strain>0 && l>1e-12f){float s2=strain*strain,s4=s2*s2;float f=3*mu*s.hardening*s4*strain/l0,h=15*mu*s.hardening*s4/(l0*l0);float3 n=e*(1.f/l);
            g=g+n*(f*dw);H.diagonal(f/l*dw*dw);H.outer(n,(h-f/l)*dw*dw);}
    }
    coarseAccumulate(block,g,H);
}
__global__ void k_CoarseContactGrip(const UnifiedState* state,CoarseBlock* block){
    const UnifiedState& s=*state;for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<max(*s.contactCount,s.grips);k+=blockDim.x*gridDim.x){
        if(k<*s.contactCount){const SurfaceContact& c=s.contacts[s.contactIds[k]];float w=s.coarseMode?0:-1;
            if(s.coarseMode==6)atomicExch(&(block+s.component[c.id[0]])->external,1);
            for(int j=0;j<c.count;j++)w+=c.weight[j]*s.coarseWeight[c.id[j]];
            float3 f;Symmetric3 H={};unifiedContact(s,c,f,H);H.xx*=w*w;H.yy*=w*w;H.zz*=w*w;H.xy*=w*w;H.xz*=w*w;H.yz*=w*w;
            coarseAccumulate(s.coarseMode?block+s.component[c.id[0]]:block,f*w,H);}
        if(k<s.grips && s.points[k].valid){const GripPoint& a=s.points[k];float w=s.coarseMode?0:-1;
            if(s.coarseMode==6)atomicExch(&(block+s.component[a.id[0]])->external,1);
            for(int j=0;j<a.count;j++)w+=a.weight[j]*s.coarseWeight[a.id[j]];
            float stiffness=s.gripK/a.complianceScale;Symmetric3 H={};H.diagonal(stiffness*w*w);
            coarseAccumulate(s.coarseMode?block+s.component[a.id[0]]:block,(unifiedGripPoint(a,s.pos)-gripTarget(a,s.tool->pose,s.minGap))*(stiffness*w),H);}
    }
}
__global__ void k_CoarseSolve(const UnifiedState* state,CoarseBlock* block){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(state->coarseMode){if(!coarseRoot(*state,i))return;block+=i;}else if(i)return;
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
__global__ void k_CoarseContactBound(const UnifiedState* state,CoarseBlock* blocks){
    const UnifiedState& s=*state;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*s.contactCount;k+=blockDim.x*gridDim.x){
        const SurfaceContact& c=s.contacts[s.contactIds[k]];
        CoarseBlock* b=s.coarseMode?blocks+s.component[c.id[0]]:blocks;
        float w=s.coarseMode?0.f:-1.f;
        for(int j=0;j<c.count;j++)w+=c.weight[j]*s.coarseWeight[c.id[j]];
        float change=dot3(b->step,c.plane.normal)*w;
        if(change>=0)continue;
        ContactShape shape=bodyContactShape(s.tool,s.tool->pose,c.kind,s.thickness,s.minGap);
        float feature=shape.capsule?shape.radius:fminf(shape.half.x,fminf(shape.half.y,shape.half.z));
        // Bound the relative closing increment by the collider's thin feature,
        // not by an absolute penalty gap. An absolute bound can freeze every
        // global mode at a finite-compliance equilibrium. This trust region
        // permits pressure/traction to continue but prevents a coarse update
        // from crossing a blade before its manifold can be refreshed.
        coarseBound(b,fminf(1.f,.5f*feature/(-change)));
    }
}
__global__ void k_CoarseApply(const UnifiedState* state,const CoarseBlock* block){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    if(s.coarseMode)block+=s.component[i];
    if(s.coarseMode==6 && block->external)return;
    float3 d=block->step*block->scale;
    if(!isfinite(d.x+d.y+d.z))return;
    // An inactive node is its own component root without a cleared block (coarseRoot). Its block accumulates nothing,
    // so a per-component mode never moves it.
    if(!s.coarseMode || !s.active || s.active[i])s.pos[i]=s.pos[i]+d*s.coarseWeight[i];
    if(i==0 && !s.coarseMode)s.tool->pose=shiftTool(s.tool->pose,d);
}
// Exact rigid rotations are null modes of the isotropic material and objective
// viscosity. A positive vertex Hessian cannot express this cancellation across
// vertices: a fixed local iteration budget otherwise absorbs gravitational
// torque. Minimize the SAME inertial/support objective in a component rotation
// subspace. This adds no force or anchor and leaves every tet determinant and
// internal strain unchanged. Contact/grasp keep their coupled deformable solve,
// except the exact axial symmetry of a frictionless capsule on a free piece.
__device__ inline float3 unifiedRotate(float3 r,float3 omega){
    float a=len3(omega),sn,cs;sincosf(a,&sn,&cs);
    float b=a>1e-4f?sn/a:1-a*a/6;
    float c=a>1e-4f?(1-cs)/(a*a):.5f-a*a/24;
    return r+cross3(omega,r)*b+cross3(omega,cross3(omega,r))*c;
}
__global__ void k_RigidCenters(const UnifiedState* state,CoarseBlock* blocks){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n || (s.active&&!s.active[i]))return;CoarseBlock* b=blocks+s.component[i];
    if(s.coupledReady && !(s.graspComponents[s.component[i]]&7))return;
    // A persistent anchor already has a dedicated local, jointly transported
    // patch. Prioritize that graded deformation over a second organ-wide rigid
    // mode while held. Vertex/affine DOFs remain free; this adds no constraint.
    if(s.grips>0 && s.gripDegree[i]>0)atomicOr(&b->external,1);
    if((s.degree[i]>0 || (s.secondary && s.secondary->degree[i]>0)))atomicOr(&b->external,2);
    if(s.secondary && s.secondary->degree[i]>0)atomicOr(&b->external,3);
    if(s.attach && s.attach[i]>0)atomicOr(&b->external,4);
    if(s.invMass[i]<=0){add3(&b->pivot,s.pos[i]);atomicAdd(&b->pins,1);return;}
    float m=1.f/s.invMass[i];float values[4]={s.pos[i].x*m,s.pos[i].y*m,s.pos[i].z*m,m};
    if(unifiedReduce(b,values)){add3(&b->center,mkf3(values[0],values[1],values[2]));atomicAdd(&b->mass,values[3]);}
}
__global__ void k_RigidCenterFinish(const UnifiedState* state,CoarseBlock* b){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(!coarseRoot(*state,i))return;
    if(state->coupledReady && !(state->graspComponents[i]&7)){b[i].external=1;return;}
    b[i].center=b[i].pins==1?b[i].pivot:b[i].center*(1.f/fmaxf(b[i].mass,1e-20f));
    b[i].scale=0;
    // Supported/contacted bodies keep the deformable coupled solve. Only an
    // unsupported piece with exclusively frictionless capsule contacts has the
    // additional exact rotational symmetry used below.
    bool axial=b[i].external==2 && b[i].pins==0 && state->muS==0 &&
        len3(state->tool->pose.shaftB-state->tool->pose.shaftA)>1e-8f;
    b[i].external=(b[i].external&3)!=0;
    if(axial){
        float3 a=state->tool->pose.shaftA,axis=contactUnit(state->tool->pose.shaftB-a,mkf3(1,0,0));
        b[i].center=a+axis*dot3(b[i].center-a,axis);b[i].pins=-1;b[i].external=0;
    }
}
__global__ void k_RigidContactKinds(const UnifiedState* state,CoarseBlock* blocks){
    const UnifiedState& s=*state;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*s.contactCount;k+=blockDim.x*gridDim.x){
        const SurfaceContact& c=s.contacts[s.contactIds[k]];
        if(c.kind!=0)atomicOr(&blocks[s.component[c.id[0]]].external,8);
    }
}
__global__ void k_RigidNodes(const UnifiedState* state,CoarseBlock* blocks){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n || s.invMass[i]<=0 || (s.active&&!s.active[i]))return;
    CoarseBlock* b=blocks+s.component[i];if(b->external||b->pins>1)return;
    float inertia=1.f/(s.invMass[i]*s.h*s.h),stiffness=s.attach?s.attach[i]:0;
    float3 r=s.pos[i]-b->center,force=(s.pos[i]-s.target[i])*inertia;
    if(b->pins==-1){
        float3 axis=contactUnit(s.tool->pose.shaftB-s.tool->pose.shaftA,mkf3(1,0,0)),t=s.target[i]-b->center;
        float values[2]={(dot3(r,t)-dot3(r,axis)*dot3(t,axis))*inertia,dot3(axis,cross3(r,t))*inertia};
        if(unifiedReduce(b,values)){atomicAdd(&b->xx,values[0]);atomicAdd(&b->xy,values[1]);}return;
    }
    if(stiffness>0){float viscous=unifiedSupportDamping(s,i);
        force=force+(s.pos[i]-s.attachRest[i])*stiffness+(s.pos[i]-s.previous[i])*viscous;stiffness+=viscous;}
    Symmetric3 H={};H.diagonal((inertia+stiffness)*dot3(r,r));H.outer(r,-inertia-stiffness);
    coarseAccumulate(b,cross3(r,force),H);
}
__global__ void k_RigidEnergy(const UnifiedState* state,CoarseBlock* blocks){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n||s.invMass[i]<=0||(s.active&&!s.active[i]))return;
    CoarseBlock* b=blocks+s.component[i];if(b->external||b->pins>1)return;
    float3 r=s.pos[i]-b->center,d=unifiedRotate(r,b->step)-r;
    float inertia=1.f/(s.invMass[i]*s.h*s.h),stiffness=s.attach?s.attach[i]:0;
    float3 g=(s.pos[i]-s.target[i])*inertia;
    if(stiffness>0){float viscous=unifiedSupportDamping(s,i);g=g+(s.pos[i]-s.attachRest[i])*stiffness+(s.pos[i]-s.previous[i])*viscous;stiffness+=viscous;}
    // Difference form avoids subtracting two large, nearly equal energies.
    float values[1]={dot3(g,d)+.5f*(inertia+stiffness)*dot3(d,d)};
    if(unifiedReduce(b,values))atomicAdd(&b->scale,values[0]);
}
// A mathematical rigid transform preserves volume, but rounding its result
// back to world-space float positions does not. This matters for a cut piece
// after a long free fall, especially for thin compressed tetrahedra. Certify
// the actual stored coordinates, not the ideal rotation matrix.
__global__ void k_RigidFeasible(const UnifiedState* state,CoarseBlock* blocks){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=s.nt||!s.live[t])return;const int* id=s.ids+4*t;
    CoarseBlock* b=blocks+s.component[id[0]];
    if(b->external||b->pins>1||!(b->scale<=0))return;
    float3 p[4],q[4];for(int j=0;j<4;j++){
        p[j]=s.pos[id[j]];q[j]=s.invMass[id[j]]>0?b->center+unifiedRotate(p[j]-b->center,b->step):p[j];
    }
    float det=s.restMetric[7*t+6];
    float oldJ=dot3(p[1]-p[0],cross3(p[2]-p[0],p[3]-p[0]))*det;
    float nextJ=dot3(q[1]-q[0],cross3(q[2]-q[0],q[3]-q[0]))*det;
    if(!(nextJ>=fminf(s.minimum,oldJ)))atomicOr(&b->external,16);
}
__global__ void k_RigidApply(const UnifiedState* state,const CoarseBlock* blocks){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n||s.invMass[i]<=0||(s.active&&!s.active[i]))return;
    const CoarseBlock& b=blocks[s.component[i]];
    if(b.external||b.pins>1||!(b.scale<=0)||!isfinite(b.step.x+b.step.y+b.step.z))return;
    s.pos[i]=b.center+unifiedRotate(s.pos[i]-b.center,b.step);
}
// A contact polygon supplies all extremal material witnesses on the finite
// incident face. One centroid alone lets a triangle pivot through a thin jaw.
// At most 7 vertices arise from clipping a triangle by a rectangle; reserve 12
// including degeneracies, without a global active-contact truncation.
// The whole warp handles one contact: every lane clips the same polygon, then
// each builds one of its witnesses, which one thread would build in turn.
__device__ void unifiedManifoldContact(int slot,int lane,const SurfaceContact* source,
    const SurfaceVertex* vertices,const float3* pos,const float3* previous,const CoupledTool* body,
    float thickness,float minGap,float skin,SurfaceContact* out,int* ids,int* total,int* owners,float* quadrature,int* degree,int stride){
    const SurfaceContact& base=source[slot];if(!base.count)return;
    ContactShape shape=bodyContactShape(body,body->pose,base.kind,thickness,minGap);
    if(shape.capsule){
        if(lane)return;
        int index=atomicAdd(total,1);out[index]=base;ids[index]=index;owners[index]=slot;quadrature[index]=1;
        for(int j=0;j<base.count;j++)atomicAdd(degree+base.id[j],1);return;}
    const SurfaceVertex* v=vertices+3*(slot/stride);float3 axes[3]={shape.u,shape.n,shape.w};
    float3 normal=contactUnit(shape.u*dot3(base.plane.normal,base.referenceShape.u)+shape.n*dot3(base.plane.normal,base.referenceShape.n)+shape.w*dot3(base.plane.normal,base.referenceShape.w),base.plane.normal);int incidentAxis=0;
    for(int a=1;a<3;a++)if(fabsf(dot3(normal,axes[a]))>fabsf(dot3(normal,axes[incidentAxis])))incidentAxis=a;
    normal=axes[incidentAxis]*(dot3(normal,axes[incidentAxis])>=0?1.f:-1.f);
    float3 p[12],b[12],np[12],nb[12];
    for(int j=0;j<3;j++){float3 r=surfacePoint(v[j],pos)-shape.center;p[j]=mkf3(dot3(r,axes[0]),dot3(r,axes[1]),dot3(r,axes[2]));b[j]=mkf3(j==0,j==1,j==2);}
    int n=3;
    for(int a=0;a<3 && n;a++)if(a!=incidentAxis)for(int side=-1;side<=1 && n;side+=2){
        int m=0;float bound=compF3(shape.half,a)+skin;
        float3 prev=p[n-1],pb=b[n-1];float pd=side*compF3(prev,a)-bound;
        for(int j=0;j<n;j++){
            float d=side*compF3(p[j],a)-bound;
            if((d<=0)!=(pd<=0)){float t=pd/(pd-d);np[m]=prev+(p[j]-prev)*t;nb[m++]=pb+(b[j]-pb)*t;}
            if(d<=0){np[m]=p[j];nb[m++]=b[j];}prev=p[j];pb=b[j];pd=d;
        }
        n=m;for(int j=0;j<n;j++){p[j]=np[j];b[j]=nb[j];}
    }
    // No finite footprint means no current contact. Keeping an old witness
    // here creates an infinite supporting plane that drags withdrawing tissue.
    if(n<3)return;
    // Include an interior witness: boundary vertices can leave the finite face
    // while the triangle interior still crosses it during a nonlinear update.
    float3 center=mkf3(0,0,0);for(int j=0;j<n;j++)center=center+b[j];center=center*(1.f/n);
    // Integrate coverage in the source triangle's barycentric domain. Its
    // doubled area is one, including for an edge-on world-space triangle.
    // Each fan triangle distributes its area to its two boundary witnesses
    // and the center. Vanishing slivers now lose force continuously; adding a
    // duplicate clipping vertex no longer changes all quadrature weights.
    float sampleWeight[12]={};float coverage=0;
    for(int j=0;j<n;j++){
        int next=(j+1)%n;float3 u=b[j]-center,v=b[next]-center;
        float area=fabsf(u.y*v.z-u.z*v.y);coverage+=area;
        sampleWeight[j]+=area/3.f;sampleWeight[next]+=area/3.f;sampleWeight[n]+=area/3.f;
    }
    if(!(coverage>0.f))return;
    // Integrate normal traction over the actual projected tool-face area.
    // A barycentric fraction alone gives EVERY triangle the full stiffness:
    // remeshing or near edge-on faces then change the force discontinuously.
    // p is the clipped polygon, so use the original embedded triangle here.
    float3 tri0=surfacePoint(v[0],pos),tri1=surfacePoint(v[1],pos),tri2=surfacePoint(v[2],pos);
    float projectedArea=.5f*fabsf(dot3(cross3(tri1-tri0,tri2-tri0),normal));
    float featureLength=2.f*fminf(shape.half.x,fminf(shape.half.y,shape.half.z));
    float faceArea=featureLength*featureLength;
    float normalization=projectedArea/fmaxf(faceArea,1e-12f)/fmaxf(1.f,coverage);
    b[n]=center;n++;
    int start=0;if(lane==0)start=atomicAdd(total,n);
    start=__shfl_sync(0xffffffff,start,0);
    for(int sample=lane;sample<n;sample+=32){
        SurfaceContact c=base;c.count=0;c.offset=c.previous=mkf3(0,0,0);c.lambdaN=0;c.lambdaT=mkf3(0,0,0);
        c.plane.normal=normal;c.referenceShape=shape;c.toolOrigin=body->pose.jawHinge;c.toolGap=body->pose.jawGap;
        for(int a=0;a<3;a++){
            float f=compF3(b[sample],a);c.offset=c.offset+v[a].offset*f;c.previous=c.previous+surfacePoint(v[a],previous)*f;
            for(int j=0;j<v[a].count;j++){
                float w=f*v[a].weight[j];if(w<=1e-8f)continue;int id=v[a].id[j],index=0;
                while(index<c.count && c.id[index]!=id)index++;
                if(index==c.count){c.id[index]=id;c.weight[index]=0;c.count++;}c.weight[index]+=w;
            }
        }
        float3 x=unifiedPoint(c,pos);float support=fabsf(dot3(normal,shape.u))*shape.half.x+fabsf(dot3(normal,shape.n))*shape.half.y+fabsf(dot3(normal,shape.w))*shape.half.z;
        float3 surface=x+normal*(support-dot3(x-shape.center,normal));c.plane.target=surface+normal*skin;
        c.plane.motion=surface-witnessAt(shape,bodyContactShape(body,body->previous,c.kind,thickness,minGap),surface);
        c.gapGradient=jawGapGradient(body->pose,c.kind,thickness,minGap,surface,body);
        int index=start+sample;out[index]=c;ids[index]=index;owners[index]=slot;quadrature[index]=sampleWeight[sample]*normalization;
        for(int j=0;j<c.count;j++)atomicAdd(degree+c.id[j],1);
    }
}
// Launch blocks of 64 threads for k_UnifiedManifold. Its warps stride over the
// device-side active count, so the launch is independent of the surface size.
constexpr int MANIFOLD_BLOCKS=1024;
__global__ void k_UnifiedManifold(const SurfaceContact* source,const int* active,const int* count,
    const SurfaceVertex* vertices,const float3* pos,const float3* previous,const CoupledTool* body,
    float thickness,float minGap,float skin,SurfaceContact* out,int* ids,int* total,int* owners,float* quadrature,int* degree,int stride=3){
    int warps=gridDim.x*blockDim.x/32;
    for(int k=(blockIdx.x*blockDim.x+threadIdx.x)/32;k<*count;k+=warps)
        unifiedManifoldContact(active[k],threadIdx.x%32,source,vertices,pos,previous,body,thickness,minGap,skin,out,ids,total,owners,quadrature,degree,stride);
}
__global__ void k_UnifiedAggregate(const UnifiedState* state,const int* owners,SurfaceContact* base,bool stick){
    const UnifiedState& s=*state;
    // The allocation supports many possible witnesses, while a tool touches
    // only a small active set. Do not launch millions of idle threads for it.
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*s.contactCount;k+=blockDim.x*gridDim.x){
        const SurfaceContact& c=s.contacts[k];atomicAdd(&base[owners[k]].lambdaN,c.lambdaN);add3(&base[owners[k]].lambdaT,c.lambdaT);
        if(stick && c.lambdaN>0){
            float K=unifiedContactStiffness(s,c);
            atomicAdd(&base[owners[k]].stickStiffness,K);add3(&base[owners[k]].stickSpring,unifiedSlip(s,c,unifiedPoint(c,s.pos),c.plane.normal)*K);
        }
    }
}
// Persistent stick of the grasper jaws, after k_UnifiedAggregate. Measured from
// the substep start, a held load would stretch the tangential spring anew at
// every substep and the bite would creep along the pads under any load below
// the friction limit. A jaw contact sticks while the spring force of its loaded
// samples stays inside the static cone of their summed pressure: it keeps that
// elongation, so its anchor stays fixed in the jaw. Beyond the cone it slides:
// the anchor moves along the slip until the spring force equals the friction of
// the muS -> muD transition of unifiedContact. Deciding per contact keeps its
// low-pressure rim samples, which slide inside unifiedContact, from dragging
// the anchor of the loaded patch along. A contact without compressive pressure
// drops its anchor and so starts a new one at its next touch. Only the friction
// grasp (contact model 1) runs this: the anchor-assisted grasp ties its bite to
// the jaws with anchors, and its jaw contacts keep the substep anchor.
__global__ void k_UnifiedStick(const UnifiedState* state,const int* active,const int* count,SurfaceContact* contacts,float thickness,float minGap){
    const UnifiedState& s=*state;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*count;k+=blockDim.x*gridDim.x){
        SurfaceContact& c=contacts[active[k]];
        ContactShape shape=bodyContactShape(s.tool,s.tool->pose,c.kind,thickness,minGap);
        float3 e=mkf3(0,0,0);
        // Capsules (the shaft, scissor edges) keep the anchor at the substep start.
        if(c.stickStiffness>0 && c.lambdaN>0 && !shape.capsule){
            e=c.stickSpring*(1.f/c.stickStiffness);
            float spring=len3(c.stickSpring),limit=s.muS*c.lambdaN/(s.h*s.h);
            if(spring>limit){float ratio=limit/spring;e=e*((s.muD+(s.muS-s.muD)*ratio*ratio)*ratio/fmaxf(s.muS,1e-12f));}
        }
        c.stick=mkf3(dot3(e,shape.u),dot3(e,shape.n),dot3(e,shape.w));
    }
}
