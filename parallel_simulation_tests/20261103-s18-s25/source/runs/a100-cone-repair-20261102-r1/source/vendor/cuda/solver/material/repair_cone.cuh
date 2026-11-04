// Isolated R8 experiment: a feasible direction in the existing three-DOF
// repair basis, not a new force, projection of positions or relaxed bound.
#pragma once

namespace repair_cone {
struct Vector {double x,y,z;};
struct Matrix {double xx,yy,zz,xy,xz,yz;};
struct Plane {Vector normal;double lower;};
__device__ inline Vector add(Vector a,Vector b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
__device__ inline Vector mul(Vector a,double b){return {a.x*b,a.y*b,a.z*b};}
__device__ inline double dot(Vector a,Vector b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ inline Vector product(Matrix a,Vector b){return {
    a.xx*b.x+a.xy*b.y+a.xz*b.z,a.xy*b.x+a.yy*b.y+a.yz*b.z,a.xz*b.x+a.yz*b.y+a.zz*b.z};}
__device__ inline bool inverse(Matrix a,Matrix& b,double tolerance){
    double cxx=a.yy*a.zz-a.yz*a.yz,cyy=a.xx*a.zz-a.xz*a.xz,czz=a.xx*a.yy-a.xy*a.xy;
    double cxy=a.xz*a.yz-a.xy*a.zz,cxz=a.xy*a.yz-a.xz*a.yy,cyz=a.xy*a.xz-a.xx*a.yz;
    double determinant=a.xx*cxx+a.xy*cxy+a.xz*cxz;
    if(!(a.xx>0 && czz>0 && determinant>tolerance*a.xx*a.yy*a.zz) || !isfinite(determinant))return false;
    double scale=1./determinant;b={cxx*scale,cyy*scale,czz*scale,cxy*scale,cxz*scale,cyz*scale};return true;
}
__device__ inline double energy(Matrix h,Vector g,Vector x){return dot(g,x)+.5*dot(x,product(h,x));}
__device__ inline void consider(Matrix h,Vector g,const Plane* planes,int count,Vector candidate,Vector& best,double& cost){
    if(!isfinite(candidate.x+candidate.y+candidate.z))return;
    for(int p=0;p<count;p++)if(dot(planes[p].normal,candidate)<planes[p].lower-1e-9)return;
    double trial=energy(h,g,candidate);if(isfinite(trial) && trial<cost){best=candidate;cost=trial;}
}
// At most eight planes in three dimensions. Enumerate active sets of size
// zero to three; compare feasible candidates in the same positive quadratic.
// Final simulator bounds still run in their original precision afterwards.
__device__ inline bool solve(Matrix h,Vector g,const Plane* planes,int count,Vector& best){
    Matrix inv;if(!inverse(h,inv,1e-14))return false;
    Vector free=mul(product(inv,g),-1),directions[8];
    for(int i=0;i<count;i++)directions[i]=product(inv,planes[i].normal);
    best={0,0,0};double cost=0;consider(h,g,planes,count,free,best,cost);
    for(int i=0;i<count;i++){
        double a=dot(planes[i].normal,directions[i]);if(!(a>0))continue;
        double ri=planes[i].lower-dot(planes[i].normal,free);
        consider(h,g,planes,count,add(free,mul(directions[i],ri/a)),best,cost);
        for(int j=i+1;j<count;j++){
            double b=dot(planes[i].normal,directions[j]),c=dot(planes[j].normal,directions[j]);
            double rj=planes[j].lower-dot(planes[j].normal,free),det=a*c-b*b;
            if(det>1e-12*a*c){
                Vector candidate=add(add(free,mul(directions[i],(ri*c-rj*b)/det)),mul(directions[j],(rj*a-ri*b)/det));
                consider(h,g,planes,count,candidate,best,cost);
            }
            for(int k=j+1;k<count;k++){
                Matrix gram={a,c,dot(planes[k].normal,directions[k]),b,
                    dot(planes[i].normal,directions[k]),dot(planes[j].normal,directions[k])};
                Matrix gramInv;if(!inverse(gram,gramInv,1e-12))continue;
                Vector rhs={ri,rj,planes[k].lower-dot(planes[k].normal,free)},lambda=product(gramInv,rhs);
                Vector candidate=add(add(add(free,mul(directions[i],lambda.x)),mul(directions[j],lambda.y)),mul(directions[k],lambda.z));
                consider(h,g,planes,count,candidate,best,cost);
            }
        }
    }
    return true;
}
}

#ifndef REPAIR_CONE_ALGEBRA_ONLY
struct RepairCone {
    float3 original;float originalScale;
    unsigned long long worst;
    int count,enabled,searching,accepted;
    int tet[8];float4 planes[8];
};
__device__ inline repair_cone::Matrix repairMatrix(const CoarseBlock& b){return {b.xx,b.yy,b.zz,b.xy,b.xz,b.yz};}
__device__ inline repair_cone::Vector repairVector(float3 v){return {v.x,v.y,v.z};}
__device__ inline bool repairTetPlane(const UnifiedState& s,int t,float3& gradient,float& margin){
    if(t>=s.nt || !s.live[t])return false;
    int id[4];unifiedTetIds(s,t,id);float w[4];for(int j=0;j<4;j++)w[j]=s.coarseWeight[id[j]];
    const auto& kuhn=unifiedKuhn(s,t);
    float3 q=kuhn.b0*(w[1]-w[0])+kuhn.b1*(w[2]-w[0])+kuhn.b2*(w[3]-w[0]);
    if(dot3(q,q)<1e-20f)return false;
    float3 a=s.pos[id[0]],e1=s.pos[id[1]]-a,e2=s.pos[id[2]]-a,e3=s.pos[id[3]]-a;
    float determinant=kuhn.metric[6],J=dot3(e1,cross3(e2,e3))*determinant;
    gradient=(cross3(e2,e3)*(w[1]-w[0])+cross3(e3,e1)*(w[2]-w[0])+cross3(e1,e2)*(w[3]-w[0]))*determinant;
    margin=fmaxf(0.f,.9f*(J-s.minimum));return isfinite(margin+gradient.x+gradient.y+gradient.z);
}
__global__ void k_RepairConeBegin(const UnifiedState* state,CoarseBlock* blocks,RepairCone* cones){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);
    if(!coarseRoot(s,i))return;auto& c=cones[i];const auto& b=blocks[i];
    c={};c.original=b.step;c.originalScale=b.scale;c.worst=~0ULL;
    int flags=s.graspComponents[i];
    c.enabled=s.coarseMode==5 && s.grips>0 && (flags&1) && (flags&MaterialCutterContact) && b.scale<1.f;
    c.searching=c.enabled;
}
__global__ void k_RepairConeSelect(const UnifiedState* state,const CoarseBlock* blocks,RepairCone* cones){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=s.nt || !s.live[t])return;
    int ids[4];unifiedTetIds(s,t,ids);int root=s.component[ids[0]];auto& c=cones[root];
    if(!c.searching)return;float3 gradient;float margin;if(!repairTetPlane(s,t,gradient,margin))return;
    float derivative=dot3(gradient,blocks[root].step);if(!(derivative<0))return;
    float alpha=fminf(1.f,margin/(-derivative));if(!(alpha<1))return;
    // Positive float bit order, then tet ID, gives deterministic tie handling.
    unsigned long long key=((unsigned long long)__float_as_uint(alpha)<<32)|(unsigned)t;
    atomicMin(&c.worst,key);
}
__global__ void k_RepairConeSolve(const UnifiedState* state,CoarseBlock* blocks,RepairCone* cones){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);
    if(!coarseRoot(s,i))return;auto& c=cones[i];auto& b=blocks[i];if(!c.searching)return;
    if(c.worst==~0ULL || c.count>=8){c.searching=0;return;}
    int t=(int)(unsigned)c.worst;for(int j=0;j<c.count;j++)if(c.tet[j]==t){c.searching=0;return;}
    float3 gradient;float margin;
    if(!repairTetPlane(s,t,gradient,margin)){c.searching=0;return;}
    double length=sqrt(double(gradient.x)*gradient.x+double(gradient.y)*gradient.y+double(gradient.z)*gradient.z);
    if(!(length>0)){c.searching=0;return;}
    c.tet[c.count]=t;c.planes[c.count++]=make_float4(gradient.x/length,gradient.y/length,gradient.z/length,-margin/length);
    repair_cone::Plane planes[8];
    for(int j=0;j<c.count;j++){const auto& p=c.planes[j];planes[j]={{p.x,p.y,p.z},p.w};}
    repair_cone::Vector best;
    if(!repair_cone::solve(repairMatrix(b),repairVector(b.gradient),planes,c.count,best)){c.searching=0;return;}
    b.step=mkf3(best.x,best.y,best.z);c.worst=~0ULL;
}
__global__ void k_RepairConeRebound(const UnifiedState* state,CoarseBlock* blocks,const RepairCone* cones){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);
    if(coarseRoot(s,i) && cones[i].enabled)blocks[i].scale=1.f;
}
__global__ void k_RepairConeAccept(const UnifiedState* state,CoarseBlock* blocks,RepairCone* cones){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);
    if(!coarseRoot(s,i) || !cones[i].enabled)return;auto& c=cones[i];auto& b=blocks[i];
    // Use the final, fully bounded proposal, including both tools and fragments.
    // Keep the exact original float step AND scale on failure. This is only a
    // quadratic-model safeguard, not a claim of full nonlinear energy decrease.
    auto h=repairMatrix(b);auto g=repairVector(b.gradient);
    double before=repair_cone::energy(h,g,repairVector(c.original*c.originalScale));
    double after=repair_cone::energy(h,g,repairVector(b.step*b.scale));
    c.accepted=isfinite(after) && after<=before;
    if(!c.accepted){b.step=c.original;b.scale=c.originalScale;}
}
#endif
