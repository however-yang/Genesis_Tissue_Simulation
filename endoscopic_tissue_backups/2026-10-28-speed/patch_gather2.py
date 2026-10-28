"""Keep the gathered stencil row and the tet's per-corner terms in registers: no index is chosen at run time."""
from pathlib import Path

stencil = Path(r"D:\Simulation\genesis-world\endoscopic_tissue\native\vendor\cuda\solver\material\material_stencil.cuh")
s = stencil.read_text(encoding="utf8")
start = s.index("// Each free node gathers the blocks of its own row from its incident tets")
end = s.index("__global__ void k_MaterialStencilDiagonal(")
gather = r'''// Adds a 3x3 block into one slot of a stencil row. Every index is fixed at compile time, which keeps the row in
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
// Each free node gathers the blocks of its own row from its incident tets, so no two threads add into one entry.
// A tet's own terms are recomputed for each of its free nodes, which costs far less than scattering them into shared
// rows with atomic additions.
__global__ void k_MaterialStencilGather(const UnifiedState* state,float* matrix){
    const auto& s=*state;int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=s.n||materialCgFixed(s,node))return;
    float row[72]={};
    for(int k=s.offsets[node];k<s.offsets[node+1];k++){
        int t=s.incident[k];if(!s.live[t])continue;
        const int* id=s.ids+4*t;int i=id[0]==node?0:id[1]==node?1:id[2]==node?2:3;
        float3 e[3]={s.pos[id[1]]-s.pos[id[0]],s.pos[id[2]]-s.pos[id[0]],s.pos[id[3]]-s.pos[id[0]]};
        float3 b[3]={s.b0[t],s.b1[t],s.b2[t]},q[4]={(b[0]+b[1]+b[2])*(-1.f),b[0],b[1],b[2]},F[3];
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
        float barrierCurvature=unifiedCompressionBarrier(dot3(F[0],cof[0]),s.minimum,s.bulk[t]+s.mu[t]/3.f).curvature;
        float mu=s.mu[t],scale=.5f*mu*s.damping/s.h,fft[9]={},ggt[9]={};
        #pragma unroll
        for(int u=0;u<3;u++)
            #pragma unroll
            for(int v=0;v<3;v++)
                #pragma unroll
                for(int a=0;a<3;a++){
                    fft[3*u+v]+=compF3(F[a],u)*compF3(F[a],v);
                    if(s.damping>0)ggt[3*u+v]+=compF3(G[a],u)*compF3(G[a],v);
                }
        // The row node's own terms, selected without indexing at run time.
        float3 qi=q[0],fqi=fq[0],gJi=gJ[0],pqi=pq[0],gqi=gq[0];
        #pragma unroll
        for(int a=1;a<4;a++)if(i==a){qi=q[a];fqi=fq[a];gJi=gJ[a];pqi=pq[a];gqi=gq[a];}
        #pragma unroll
        for(int j=0;j<4;j++){
            int slot=abs(materialSlot(s,node,id[j])-7);float dot=dot3(qi,q[j]),pdot=s.damping>0?dot3(pqi,pq[j]):0;
            float3 n=mkf3(0,0,0);float tangent=0,axial=0;
            if(i!=j&&s.hardening>0){
                float3 edge=s.pos[node]-s.pos[id[j]];float l=len3(edge),l0=unifiedRestLength(s,t,i,j),strain=fmaxf(0.f,l/l0-1.f);
                if(strain>0&&l>1e-12f){n=edge*(1.f/l);float s2=strain*strain,s4=s2*s2;
                    tangent=3.f*mu*s.hardening*s4*strain/(l0*l);axial=15.f*mu*s.hardening*s4/(l0*l0)-tangent;}
            }
            float block[9];
            #pragma unroll
            for(int u=0;u<3;u++)
                #pragma unroll
                for(int v=0;v<3;v++){
                    float H=mu*(dot*fft[3*u+v]+compF3(fq[j],u)*compF3(fqi,v)-(2.f/3.f)*compF3(fqi,u)*compF3(fq[j],v))
                        +(s.bulk[t]+barrierCurvature)*compF3(gJi,u)*compF3(gJ[j],v);
                    if(s.damping>0)H+=scale*(pdot*ggt[3*u+v]+compF3(gq[j],u)*compF3(gqi,v));
                    float extra=axial*compF3(n,u)*compF3(n,v)+(u==v?tangent:0);
                    block[3*u+v]=H-extra;row[3*u+v]+=extra;
                }
            if(id[j]>=node)materialRowAdd(row,slot,block);
        }
    }
    #pragma unroll
    for(int a=0;a<72;a++)matrix[a*s.n+node]+=row[a];
}
'''
s = s[:start] + gather + s[end:]
stencil.write_text(s, encoding="utf8")
print("register gather in place")
