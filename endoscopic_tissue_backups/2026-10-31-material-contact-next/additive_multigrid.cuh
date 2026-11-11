// Isolated additive multilevel preconditioner. The material operator, full
// displacement space, PCG budgets and physical line search are unchanged.
// B_l = .7 D_l^-1 + P_l B_(l+1) P_l^T; the last level uses the existing inverse.
// Keep the same diagonal block and cutoff. Form its cofactors in double to
// avoid cancellation at cut-rim blocks; round only the resulting direction.
__device__ inline float3 additiveFineSolve(const Symmetric3& h,float3 r){
    double xx=h.xx,yy=h.yy,zz=h.zz,xy=h.xy,xz=h.xz,yz=h.yz;
    double a=yy*zz-yz*yz,b=xz*yz-xy*zz,c=xy*yz-xz*yy;
    double d=xx*zz-xz*xz,e=xy*xz-xx*yz,f=xx*yy-xy*xy;
    double det=xx*a+xy*b+xz*c;
    if(!(det>1e-25))return mkf3(0,0,0);
    return mkf3(float((a*r.x+b*r.y+c*r.z)/det),float((b*r.x+d*r.y+e*r.z)/det),float((c*r.x+e*r.y+f*r.z)/det));
}
__global__ void k_AdditiveFineRestrict(const UnifiedState* state,const unsigned short* stencil,
    const float3* r,MultigridLevel L){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;
    int c=L.parent[i];if(!materialCgNode(stencil,i)||c<0)return;
    atomicAdd(&L.r[c].x,r[i].x);atomicAdd(&L.r[c].y,r[i].y);atomicAdd(&L.r[c].z,r[i].z);
}
__global__ void k_AdditiveRestrict(MultigridLevel L,MultigridLevel next){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=min(*L.count,L.capacity))return;
    int C=next.parent[c];if(C<0)return;
    atomicAdd(&next.r[C].x,L.r[c].x);atomicAdd(&next.r[C].y,L.r[c].y);atomicAdd(&next.r[C].z,L.r[c].z);
}
__global__ void k_AdditiveProlong(MultigridLevel L,MultigridLevel next){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=min(*L.count,L.capacity))return;
    float3 out=mgApply3(L.inverse+9*c,L.r[c])*MG_OMEGA;int C=next.parent[c];
    if(C>=0)out=out+next.post[C];L.post[c]=out;
}
__global__ void k_AdditiveFineFinish(const UnifiedState* state,const unsigned short* stencil,
    const Symmetric3* diagonal,const float3* r,MultigridLevel L,float3* z,MaterialCGSum* sums,int slot){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;
    if(!materialCgNode(stencil,i)){z[i]=mkf3(0,0,0);return;}
    float3 out=additiveFineSolve(diagonal[i],r[i])*MG_OMEGA;int c=L.parent[i];if(c>=0)out=out+L.post[c];
    z[i]=out;materialDotAdd(&sums[s.component[i]].next[slot],unifiedDot(r[i],out));
}
static void applyAdditiveMultigrid(cudaStream_t stream,const UnifiedState* state,int activeCount,
    const unsigned short* stencil,const Symmetric3* diagonal,const float3* r,float3* z,MaterialCGSum* sums,int slot,
    MultigridLevel* L,int top,const float* inverse){
    cudaMemsetAsync(L[1].r,0,(size_t)L[1].capacity*sizeof(float3),stream);
    k_AdditiveFineRestrict<<<(activeCount+63)/64,64,0,stream>>>(state,stencil,r,L[1]);
    for(int l=1;l<top;l++){
        cudaMemsetAsync(L[l+1].r,0,(size_t)L[l+1].capacity*sizeof(float3),stream);
        k_AdditiveRestrict<<<(L[l].capacity+63)/64,64,0,stream>>>(L[l],L[l+1]);
    }
    k_MgCoarsest<<<(3*MG_COARSEST+63)/64,64,0,stream>>>(L[top],inverse);
    for(int l=top-1;l>=1;l--)
        k_AdditiveProlong<<<(L[l].capacity+63)/64,64,0,stream>>>(L[l],L[l+1]);
    k_AdditiveFineFinish<<<(activeCount+63)/64,64,0,stream>>>(state,stencil,diagonal,r,L[1],z,sums,slot);
}
