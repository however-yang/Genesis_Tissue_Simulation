// A Kuhn-grid row has 15 blocks. Store only its diagonal and seven upper
// neighbors; lower blocks are their transposes. The symmetric positive strain
// curvature couples all four tet nodes and retains rigid motion null modes.
// This is a solver preconditioner/curvature approximation, not a new material
// energy. The exact V18 gradient and energy decide the physical motion.
__device__ inline int materialSlot(const UnifiedState& s,int i,int j){
    int dx=j%s.nx-i%s.nx,dy=(j/s.nx)%s.ny-(i/s.nx)%s.ny,dz=j/(s.nx*s.ny)-i/(s.nx*s.ny);
    int bits=abs(dx)+2*abs(dy)+4*abs(dz);return 7+(dx+dy+dz>=0?bits:-bits);
}
__global__ void k_MaterialStencilClear(const UnifiedState* state,float* matrix){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    if(materialCgFixed(s,i))return;
    for(int a=0;a<72;a++)matrix[a*s.n+i]=0;
    float inertia=1.f/(s.invMass[i]*s.h*s.h)+(s.attach?s.attach[i]:0)+unifiedSupportDamping(s,i);
    for(int a=0;a<3;a++)matrix[(4*a)*s.n+i]=inertia;
    if(s.neighbors&&s.edgeSupport)for(int a=0;a<6;a++){
        int j=s.neighbors[6*i+a];if(j<0||(s.active&&!s.active[j]))continue;
        int owner=(a&1)?6*i+a:6*j+(a^1);if(s.edgeSupport[owner])continue;
        float3 e=s.pos[i]-s.pos[j];float l=len3(e),l0=len3(s.restNeighbors[6*i+a]);
        if(l>1e-10f){float3 n=e*(1.f/l);float K=.5f*(unifiedEdgeStiffness(s,i,a/2)+unifiedEdgeStiffness(s,j,a/2));
            int slot=abs(materialSlot(s,i,j)-7);float tangent=K*fmaxf(0.f,(l-l0)/l);
            for(int u=0;u<3;u++)for(int v=0;v<3;v++){
                float H=K*compF3(n,u)*compF3(n,v)+(u==v?tangent:0);
                matrix[(3*u+v)*s.n+i]+=H;if(j>i)matrix[(9*slot+3*u+v)*s.n+i]-=H;
            }
        }
    }
}
__global__ void k_MaterialStencilTets(const UnifiedState* state,float* matrix){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=s.nt||!s.live[t])return;
    const int* id=s.ids+4*t;
    float3 e[3]={s.pos[id[1]]-s.pos[id[0]],s.pos[id[2]]-s.pos[id[0]],s.pos[id[3]]-s.pos[id[0]]};
    float3 b[3]={s.b0[t],s.b1[t],s.b2[t]},q[4]={(b[0]+b[1]+b[2])*(-1.f),b[0],b[1],b[2]},F[3];
    for(int a=0;a<3;a++)F[a]=e[0]*compF3(b[0],a)+e[1]*compF3(b[1],a)+e[2]*compF3(b[2],a);
    float3 cof[3]={cross3(F[1],F[2]),cross3(F[2],F[0]),cross3(F[0],F[1])},fq[4],gJ[4],G[3],pq[4],gq[4];
    for(int j=0;j<4;j++){
        fq[j]=F[0]*q[j].x+F[1]*q[j].y+F[2]*q[j].z;
        gJ[j]=cof[0]*q[j].x+cof[1]*q[j].y+cof[2]*q[j].z;
    }
    if(s.damping>0){
        float3 r[3]={s.previousInverse[3*t],s.previousInverse[3*t+1],s.previousInverse[3*t+2]};
        for(int a=0;a<3;a++)G[a]=F[0]*compF3(r[0],a)+F[1]*compF3(r[1],a)+F[2]*compF3(r[2],a);
        for(int j=0;j<4;j++){
            pq[j]=r[0]*q[j].x+r[1]*q[j].y+r[2]*q[j].z;gq[j]=G[0]*pq[j].x+G[1]*pq[j].y+G[2]*pq[j].z;
        }
    }
    float barrierCurvature=unifiedCompressionBarrier(dot3(F[0],cof[0]),s.minimum,s.bulk[t]+s.mu[t]/3.f).curvature;
    float mu=s.mu[t],scale=.5f*mu*s.damping/s.h,fft[9]={},ggt[9]={};
    for(int u=0;u<3;u++)for(int v=0;v<3;v++)for(int a=0;a<3;a++){
        fft[3*u+v]+=compF3(F[a],u)*compF3(F[a],v);
        if(s.damping>0)ggt[3*u+v]+=compF3(G[a],u)*compF3(G[a],v);
    }
    for(int i=0;i<4;i++){
        if(materialCgFixed(s,id[i]))continue;
        for(int j=0;j<4;j++){
            int slot=abs(materialSlot(s,id[i],id[j])-7);float dot=dot3(q[i],q[j]),pdot=s.damping>0?dot3(pq[i],pq[j]):0;
            float3 n=mkf3(0,0,0);float tangent=0,axial=0;
            if(i!=j&&s.hardening>0){
                float3 edge=s.pos[id[i]]-s.pos[id[j]];float l=len3(edge),l0=unifiedRestLength(s,t,i,j),strain=fmaxf(0.f,l/l0-1.f);
                if(strain>0&&l>1e-12f){n=edge*(1.f/l);float s2=strain*strain,s4=s2*s2;
                    tangent=3.f*mu*s.hardening*s4*strain/(l0*l);axial=15.f*mu*s.hardening*s4/(l0*l0)-tangent;}
            }
            for(int u=0;u<3;u++)for(int v=0;v<3;v++){
                float H=mu*(dot*fft[3*u+v]+compF3(fq[j],u)*compF3(fq[i],v)-(2.f/3.f)*compF3(fq[i],u)*compF3(fq[j],v))
                    +(s.bulk[t]+barrierCurvature)*compF3(gJ[i],u)*compF3(gJ[j],v);
                if(s.damping>0)H+=scale*(pdot*ggt[3*u+v]+compF3(gq[j],u)*compF3(gq[i],v));
                float extra=axial*compF3(n,u)*compF3(n,v)+(u==v?tangent:0);
                if(id[j]>=id[i])atomicAdd(matrix+(9*slot+3*u+v)*s.n+id[i],H-extra);
                if(extra!=0)atomicAdd(matrix+(3*u+v)*s.n+id[i],extra);
            }
        }
    }
}
__global__ void k_MaterialStencilDiagonal(const UnifiedState* state,const float* matrix,Symmetric3* diagonal){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    if(materialCgFixed(s,i))return;
    const float* v=matrix+i;diagonal[i]={v[0],v[4*s.n],v[8*s.n],v[s.n],v[2*s.n],v[5*s.n]};
}


__global__ void k_MaterialStencilIndices(int n,int nx,int ny,int* indices){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    int x=i%nx,y=(i/nx)%ny,z=i/(nx*ny);
    for(int slot=0;slot<15;slot++){
        int sign=slot<7?-1:1,bits=abs(slot-7),dx=sign*(bits&1),dy=sign*((bits>>1)&1),dz=sign*((bits>>2)&1);
        indices[slot*n+i]=(x+dx<0||x+dx>=nx||y+dy<0||y+dy>=ny||z+dz<0||z+dz>=n/(nx*ny))?-1:i+dx+nx*(dy+ny*dz);
    }
}
__global__ void k_MaterialStencilProduct(const UnifiedState* __restrict__ state,const float3* __restrict__ vector,
    const float* __restrict__ matrix,const int* __restrict__ indices,float3* __restrict__ result,MaterialCGSum* sums,int iteration){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    result[i]=mkf3(0,0,0);if(indices[7*s.n+i]<0||(iteration>=MaterialLoadedIterations && (s.graspComponents[s.component[i]]&7)))return;
    float3 out=mkf3(0,0,0);
    #pragma unroll
    for(int slot=0;slot<15;slot++){
        int j=indices[slot*s.n+i];if(j<0)continue;
        float3 v=vector[j];const float* a=matrix+9*abs(slot-7)*s.n+(slot<7?j:i);
        if(slot<7){
            out.x+=a[0*s.n]*v.x+a[3*s.n]*v.y+a[6*s.n]*v.z;
            out.y+=a[1*s.n]*v.x+a[4*s.n]*v.y+a[7*s.n]*v.z;
            out.z+=a[2*s.n]*v.x+a[5*s.n]*v.y+a[8*s.n]*v.z;
        }else{
            out.x+=a[0*s.n]*v.x+a[1*s.n]*v.y+a[2*s.n]*v.z;
            out.y+=a[3*s.n]*v.x+a[4*s.n]*v.y+a[5*s.n]*v.z;
            out.z+=a[6*s.n]*v.x+a[7*s.n]*v.y+a[8*s.n]*v.z;
        }
    }
    result[i]=out;materialDotAdd(&sums[s.component[i]].denominator,unifiedDot(vector[i],out));
}

// The fixed witnesses and component labels are unchanged during one PCG
// solve. Filter the stencil once, instead of chasing their buffers for every
// neighbor during all 32/80 matrix-vector products. Rebuild after each manifold
// and topology update; this is never a persistent contact/fixation cache.
__global__ void k_MaterialStencilActiveIndices(const UnifiedState* state,int* indices){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    bool fixed=materialCgFixed(s,i);int x=i%s.nx,y=(i/s.nx)%s.ny,z=i/(s.nx*s.ny);
    for(int slot=0;slot<15;slot++){
        int sign=slot<7?-1:1,bits=abs(slot-7),dx=sign*(bits&1),dy=sign*((bits>>1)&1),dz=sign*((bits>>2)&1);
        int j=(fixed||x+dx<0||x+dx>=s.nx||y+dy<0||y+dy>=s.ny||z+dz<0||z+dz>=s.n/(s.nx*s.ny))?-1:i+dx+s.nx*(dy+s.ny*dz);
        if(j>=0 && (materialCgFixed(s,j)||s.component[i]!=s.component[j]))j=-1;
        indices[slot*s.n+i]=j;
    }
}
