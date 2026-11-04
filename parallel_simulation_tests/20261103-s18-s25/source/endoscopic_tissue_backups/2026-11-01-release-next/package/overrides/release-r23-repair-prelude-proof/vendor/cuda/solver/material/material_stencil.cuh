#include <cuda_fp16.h>
// A Kuhn-grid row has 15 blocks. Store only its diagonal and seven upper
// neighbors; lower blocks are their transposes. The symmetric positive strain
// curvature couples all four tet nodes and retains rigid motion null modes.
// This is a solver preconditioner/curvature approximation, not a new material
// energy. The exact V18 gradient and energy decide the physical motion.
__device__ inline int materialSlot(const UnifiedState& s,int i,int j){
    int dx=j%s.nx-i%s.nx,dy=(j/s.nx)%s.ny-(i/s.nx)%s.ny,dz=j/(s.nx*s.ny)-i/(s.nx*s.ny);
    int bits=abs(dx)+2*abs(dy)+4*abs(dz);return 7+(dx+dy+dz>=0?bits:-bits);
}
// Adds a 3x3 block into one slot of a stencil row. Every index is fixed at compile time, which keeps the row in
// registers: indexing it at run time would spill it to local memory.
template<int S> __device__ __forceinline__ void materialRowAddSlot(float (&row)[72],const float (&block)[9]){
    #pragma unroll
    for(int k=0;k<9;k++)row[9*S+k]+=block[k];
}
__device__ __forceinline__ void materialRowAdd(float (&row)[72],int slot,const float (&block)[9]){
    switch(slot){
        case 0:materialRowAddSlot<0>(row,block);break;case 1:materialRowAddSlot<1>(row,block);break;
        case 2:materialRowAddSlot<2>(row,block);break;case 3:materialRowAddSlot<3>(row,block);break;
        case 4:materialRowAddSlot<4>(row,block);break;case 5:materialRowAddSlot<5>(row,block);break;
        case 6:materialRowAddSlot<6>(row,block);break;default:materialRowAddSlot<7>(row,block);break;
    }
}
// Each free node assembles its own row in registers: its inertia and support, its surviving cut-rim edges, and the
// terms its incident tets contribute, so no two threads add into one entry. A tet's own terms are recomputed for each
// of its free nodes, which costs far less than scattering them into shared rows with atomic additions. The row's
// diagonal block becomes the block Jacobi preconditioner in single precision. Its eight blocks are stored in half
// precision, each scaled by its largest entry (blockScale), which halves the bytes every matrix-vector product reads.
// The scaled entries keep three significant digits of their own block, an error far below what the PCG's fixed
// iteration budget leaves unresolved, and the transposed lower blocks read the same values, so the stored matrix
// stays symmetric. A block below the diagonal is stored by the row of its other node, so only its diagonal part
// (tension hardening) is computed here.
// The same pass writes the PCG right-hand side, the residual: minus the gradient of the node's energy, with the terms
// and the summation order of k_UnifiedVertex. A fixed node gets a zero residual and an identity diagonal.
template<bool AllowFragments=true> __global__ void k_MaterialStencilAssemble(const UnifiedState* state,__half* matrix,float* blockScale,float3* residual,Symmetric3* diagonal){
    const auto& s=*state;int node=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(node<0)return;
    if(materialCgFixed<AllowFragments>(s,node)){residual[node]=mkf3(0,0,0);diagonal[node]={1,1,1,0,0,0};return;}
    float3 p=s.pos[node];float inertia=1.f/(s.invMass[node]*s.h*s.h),row[72]={};
    float3 gradient=(p-s.target[node])*inertia;
    if(s.attach && s.attach[node]>0)gradient=gradient+(p-s.attachRest[node])*s.attach[node]+(p-s.previous[node])*unifiedSupportDamping(s,node);
    row[0]=row[4]=row[8]=inertia+(s.attach?s.attach[node]:0)+unifiedSupportDamping(s,node);
    if(s.neighbors&&s.edgeSupport)for(int a=0;a<6;a++){
        int j=s.neighbors[6*node+a];if(j<0||(s.active&&!s.active[j]))continue;
        int owner=(a&1)?6*node+a:6*j+(a^1);if(s.edgeSupport[owner])continue;
        float3 e=s.pos[node]-s.pos[j];float l=len3(e),l0=len3(s.restNeighbors[6*node+a]);
        if(l>1e-10f){float3 n=e*(1.f/l);float K=.5f*(unifiedEdgeStiffness(s,node,a/2)+unifiedEdgeStiffness(s,j,a/2));
            gradient=gradient+n*(K*(l-l0));
            float tangent=K*fmaxf(0.f,(l-l0)/l),block[9];
            #pragma unroll
            for(int u=0;u<3;u++)
                #pragma unroll
                for(int v=0;v<3;v++){float H=K*compF3(n,u)*compF3(n,v)+(u==v?tangent:0);row[3*u+v]+=H;block[3*u+v]=-H;}
            if(j>node)materialRowAdd(row,abs(materialSlot(s,node,j)-7),block);
        }
    }
    for(int k=s.offsets[node];k<s.offsets[node+1];k++){
        int t=s.incident[k];if(!s.live[t])continue;
        int id[4];unifiedTetIds(s,t,id);int i=id[0]==node?0:id[1]==node?1:id[2]==node?2:3;
        float3 e[3]={s.pos[id[1]]-s.pos[id[0]],s.pos[id[2]]-s.pos[id[0]],s.pos[id[3]]-s.pos[id[0]]};
        float3 b[3]={unifiedKuhn(s,t).b0,unifiedKuhn(s,t).b1,unifiedKuhn(s,t).b2},q[4]={(b[0]+b[1]+b[2])*(-1.f),b[0],b[1],b[2]},F[3];
        // The row node's shape vector, selected without indexing at run time.
        float3 qi=q[0];
        #pragma unroll
        for(int a=1;a<4;a++)if(i==a)qi=q[a];
        float mu=unifiedKuhn(s,t).muV;
        // Its gradient: elasticity with the compression barrier, then strain damping, as k_UnifiedVertex evaluates them.
        // It goes first, before the curvature terms below hold their registers.
        {
            float det=unifiedKuhn(s,t).metric[6],J=dot3(e[0],cross3(e[1],e[2]))*det;
            float3 g1=cross3(e[1],e[2])*det,g2=cross3(e[2],e[0])*det,g3=cross3(e[0],e[1])*det;
            float3 g=i==0?(g1+g2+g3)*(-1.f):i==1?g1:i==2?g2:g3;
            float3 Fq=e[0]*dot3(b[0],qi)+e[1]*dot3(b[1],qi)+e[2]*dot3(b[2],qi);
            float lambda=unifiedKuhn(s,t).bulkV+mu/3.f,pressure=lambda*(J-1.f)-mu;
            gradient=gradient+Fq*mu+g*(pressure+unifiedCompressionBarrier(J,s.minimum,lambda).gradient);
            Symmetric3 unused={};unifiedStrainDamping(s,t,e[0],e[1],e[2],qi,gradient,unused);
        }
        #pragma unroll
        for(int a=0;a<3;a++)F[a]=e[0]*compF3(b[0],a)+e[1]*compF3(b[1],a)+e[2]*compF3(b[2],a);
        float3 cof[3]={cross3(F[1],F[2]),cross3(F[2],F[0]),cross3(F[0],F[1])},fq[4],gJ[4],G[3]={},pq[4]={},gq[4]={};
        #pragma unroll
        for(int j=0;j<4;j++){
            fq[j]=F[0]*q[j].x+F[1]*q[j].y+F[2]*q[j].z;
            gJ[j]=cof[0]*q[j].x+cof[1]*q[j].y+cof[2]*q[j].z;
        }
        if(s.damping>0){
            float3 r[3]={s.previousInverse[3*t],s.previousInverse[3*t+1],s.previousInverse[3*t+2]};
            #pragma unroll
            for(int a=0;a<3;a++)G[a]=F[0]*compF3(r[0],a)+F[1]*compF3(r[1],a)+F[2]*compF3(r[2],a);
            #pragma unroll
            for(int j=0;j<4;j++){
                pq[j]=r[0]*q[j].x+r[1]*q[j].y+r[2]*q[j].z;gq[j]=G[0]*pq[j].x+G[1]*pq[j].y+G[2]*pq[j].z;
            }
        }
        float tetJ=dot3(F[0],cof[0]),tetLambda=unifiedKuhn(s,t).bulkV+unifiedKuhn(s,t).muV/3.f;
        float barrierCurvature=unifiedCompressionBarrier(tetJ,s.minimum,tetLambda).curvature;
        if(materialFloorRecovery(s,s.component[node]))barrierCurvature+=materialFloorCurvature(tetJ,s.minimum,tetLambda);
        float scale=.5f*mu*s.damping/s.h,fft[9]={},ggt[9]={};
        #pragma unroll
        for(int u=0;u<3;u++)
            #pragma unroll
            for(int v=0;v<3;v++)
                #pragma unroll
                for(int a=0;a<3;a++){
                    fft[3*u+v]+=compF3(F[a],u)*compF3(F[a],v);
                    if(s.damping>0)ggt[3*u+v]+=compF3(G[a],u)*compF3(G[a],v);
                }
        // The row node's own curvature terms, selected without indexing at run time.
        float3 fqi=fq[0],gJi=gJ[0],pqi=pq[0],gqi=gq[0];
        #pragma unroll
        for(int a=1;a<4;a++)if(i==a){fqi=fq[a];gJi=gJ[a];pqi=pq[a];gqi=gq[a];}
        #pragma unroll
        for(int j=0;j<4;j++){
            int slot=abs(materialSlot(s,node,id[j])-7);float dot=dot3(qi,q[j]),pdot=s.damping>0?dot3(pqi,pq[j]):0;
            float3 n=mkf3(0,0,0);float tangent=0,axial=0;
            if(i!=j&&s.hardening>0){
                float3 edge=s.pos[node]-s.pos[id[j]];float l=len3(edge),l0=unifiedRestLength(s,t,i,j),strain=fmaxf(0.f,l/l0-1.f);
                if(strain>0&&l>1e-12f){n=edge*(1.f/l);float s2=strain*strain,s4=s2*s2;
                    gradient=gradient+n*(3.f*mu*s.hardening*s4*strain/l0);
                    tangent=3.f*mu*s.hardening*s4*strain/(l0*l);axial=15.f*mu*s.hardening*s4/(l0*l0)-tangent;}
            }
            if(id[j]>=node){
                float block[9];
                #pragma unroll
                for(int u=0;u<3;u++)
                    #pragma unroll
                    for(int v=0;v<3;v++){
                        float H=mu*(dot*fft[3*u+v]+compF3(fq[j],u)*compF3(fqi,v)-(2.f/3.f)*compF3(fqi,u)*compF3(fq[j],v))
                            +(unifiedKuhn(s,t).bulkV+barrierCurvature)*compF3(gJi,u)*compF3(gJ[j],v);
                        if(s.damping>0)H+=scale*(pdot*ggt[3*u+v]+compF3(gq[j],u)*compF3(gqi,v));
                        float extra=axial*compF3(n,u)*compF3(n,v)+(u==v?tangent:0);
                        block[3*u+v]=H-extra;row[3*u+v]+=extra;
                    }
                materialRowAdd(row,slot,block);
            }else{
                #pragma unroll
                for(int u=0;u<3;u++)
                    #pragma unroll
                    for(int v=0;v<3;v++)row[3*u+v]+=axial*compF3(n,u)*compF3(n,v)+(u==v?tangent:0);
            }
        }
    }
    residual[node]=gradient*(-1.f);
    diagonal[node]={row[0],row[4],row[8],row[1],row[2],row[5]};
    #pragma unroll
    for(int b=0;b<8;b++){
        float largest=0;
        #pragma unroll
        for(int k=0;k<9;k++)largest=fmaxf(largest,fabsf(row[9*b+k]));
        float inverse=largest>0?1.f/largest:0.f;blockScale[b*s.n+node]=largest;
        #pragma unroll
        for(int k=0;k<9;k++)matrix[(9*b+k)*s.n+node]=__float2half_rn(row[9*b+k]*inverse);
    }
}


// The lattice offset of the neighbor in slot 0..14 of a stencil row: slot 7 is the node itself, slots above 7 its seven
// upper neighbors and slots below 7 the lower ones, mirrored.
__device__ inline int materialSlotOffset(const UnifiedState& s,int slot){
    int sign=slot<7?-1:1,bits=abs(slot-7);return sign*((bits&1)+s.nx*(((bits>>1)&1)+s.ny*((bits>>2)&1)));
}
// The product A*p of this iteration's direction p, and its p'Ap. The direction is formed here, p=z+beta*p from the
// preconditioned residual z and the previous direction, for the node and for each neighbor it reads: a neighbor is a
// CG node of the same component, so it forms the same value. The node stores its own into direction for the advance.
__global__ void k_MaterialStencilProduct(const UnifiedState* __restrict__ state,const float3* __restrict__ previous,
    const float3* __restrict__ preconditioned,float3* __restrict__ direction,
    const __half* __restrict__ matrix,const float* __restrict__ blockScale,const unsigned short* __restrict__ stencil,
    float3* __restrict__ result,MaterialCGSum* sums,int iteration){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;
    unsigned mask=stencil[i];
    result[i]=mkf3(0,0,0);if(!(mask>>7&1)||(iteration>=MaterialLoadedIterations && (s.graspComponents[s.component[i]]&7)))return;
    // beta: the r'z of the previous iteration over its rho (materialRho), 0 at iteration 0.
    const auto& q=sums[s.component[i]];double rho=q.next[(iteration+1)%3];
    float beta=rho>1e-30?(float)(q.next[(iteration+2)%3]/rho):0;
    float3 own=preconditioned[i]+previous[i]*beta,out=mkf3(0,0,0);direction[i]=own;
    #pragma unroll
    for(int slot=0;slot<15;slot++){
        if(!(mask>>slot&1))continue;
        int j=i+materialSlotOffset(s,slot),block=abs(slot-7),owner=slot<7?j:i;
        float3 p=slot==7?own:preconditioned[j]+previous[j]*beta;
        float3 v=p*blockScale[block*s.n+owner];const __half* h=matrix+9*block*s.n+owner;float a[9];
        #pragma unroll
        for(int k=0;k<9;k++)a[k]=__half2float(h[k*s.n]);
        if(slot<7){
            out.x+=a[0]*v.x+a[3]*v.y+a[6]*v.z;
            out.y+=a[1]*v.x+a[4]*v.y+a[7]*v.z;
            out.z+=a[2]*v.x+a[5]*v.y+a[8]*v.z;
        }else{
            out.x+=a[0]*v.x+a[1]*v.y+a[2]*v.z;
            out.y+=a[3]*v.x+a[4]*v.y+a[5]*v.z;
            out.z+=a[6]*v.x+a[7]*v.y+a[8]*v.z;
        }
    }
    result[i]=out;materialDotAdd(&sums[s.component[i]].denominator[iteration%3],unifiedDot(own,out));
}

// The fixed witnesses and component labels are unchanged during one PCG
// solve. Filter the stencil once, instead of chasing their buffers for every
// neighbor during all 32/80 matrix-vector products. Rebuild after each manifold
// and topology update; this is never a persistent contact/fixation cache.
// Bit k of a node's mask marks slot k of its row as a neighbor the PCG solves in
// the same component. A fixed node keeps an empty mask, so bit 7, the node
// itself, marks the nodes the PCG solves (materialCgNode).
template<bool AllowFragments=true> __global__ void k_MaterialStencilActiveMask(const UnifiedState* state,unsigned short* stencil){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;
    unsigned mask=0;
    if(!materialCgFixed<AllowFragments>(s,i)){
        int x=i%s.nx,y=(i/s.nx)%s.ny,z=i/(s.nx*s.ny);
        for(int slot=0;slot<15;slot++){
            int sign=slot<7?-1:1,bits=abs(slot-7),dx=sign*(bits&1),dy=sign*((bits>>1)&1),dz=sign*((bits>>2)&1);
            if(x+dx<0||x+dx>=s.nx||y+dy<0||y+dy>=s.ny||z+dz<0||z+dz>=s.n/(s.nx*s.ny))continue;
            int j=i+dx+s.nx*(dy+s.ny*dz);
            if(!materialCgFixed<AllowFragments>(s,j) && s.component[i]==s.component[j])mask|=1u<<slot;
        }
    }
    stencil[i]=(unsigned short)mask;
}
