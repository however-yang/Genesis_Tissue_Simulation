"""Replace the atomic per-tet scatter of the CG stencil by a per-node gather of the same terms."""
from pathlib import Path

root = Path(r"D:\Simulation\genesis-world\endoscopic_tissue\native\vendor\cuda")
stencil = root / "solver" / "material" / "material_stencil.cuh"
s = stencil.read_text(encoding="utf8")
start = s.index("__global__ void k_MaterialStencilTets(")
end = s.index("__global__ void k_MaterialStencilDiagonal(")
gather = r'''// Each free node gathers the blocks of its own row from its incident tets, so no two threads add into one entry.
// A tet's own terms are recomputed for each of its free nodes, which costs far less than scattering them into shared
// rows with atomic additions.
__global__ void k_MaterialStencilGather(const UnifiedState* state,float* matrix){
    const auto& s=*state;int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=s.n||materialCgFixed(s,node))return;
    float row[72]={};
    for(int k=s.offsets[node];k<s.offsets[node+1];k++){
        int t=s.incident[k];if(!s.live[t])continue;
        const int* id=s.ids+4*t;int i=0;while(id[i]!=node)i++;
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
        for(int j=0;j<4;j++){
            int slot=abs(materialSlot(s,node,id[j])-7);float dot=dot3(q[i],q[j]),pdot=s.damping>0?dot3(pq[i],pq[j]):0;
            float3 n=mkf3(0,0,0);float tangent=0,axial=0;
            if(i!=j&&s.hardening>0){
                float3 edge=s.pos[node]-s.pos[id[j]];float l=len3(edge),l0=unifiedRestLength(s,t,i,j),strain=fmaxf(0.f,l/l0-1.f);
                if(strain>0&&l>1e-12f){n=edge*(1.f/l);float s2=strain*strain,s4=s2*s2;
                    tangent=3.f*mu*s.hardening*s4*strain/(l0*l);axial=15.f*mu*s.hardening*s4/(l0*l0)-tangent;}
            }
            for(int u=0;u<3;u++)for(int v=0;v<3;v++){
                float H=mu*(dot*fft[3*u+v]+compF3(fq[j],u)*compF3(fq[i],v)-(2.f/3.f)*compF3(fq[i],u)*compF3(fq[j],v))
                    +(s.bulk[t]+barrierCurvature)*compF3(gJ[i],u)*compF3(gJ[j],v);
                if(s.damping>0)H+=scale*(pdot*ggt[3*u+v]+compF3(gq[j],u)*compF3(gq[i],v));
                float extra=axial*compF3(n,u)*compF3(n,v)+(u==v?tangent:0);
                if(id[j]>=node)row[9*slot+3*u+v]+=H-extra;
                if(extra!=0)row[3*u+v]+=extra;
            }
        }
    }
    for(int a=0;a<72;a++)matrix[a*s.n+node]+=row[a];
}
'''
s = s[:start] + gather + s[end:]
stencil.write_text(s, encoding="utf8")

solver = root / "xpbd" / "xpbd_solver.cu"
x = solver.read_text(encoding="utf8")
old = "                k_MaterialStencilTets<<<Gr(g_T),64,0,g_captureStream>>>(d_unified,d_cgMatrix);\n"
new = "                k_MaterialStencilGather<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgMatrix);\n"
assert x.count(old) == 1
x = x.replace(old, new)
x = x.replace('xpbd_section_mark("cg assembly: stencil tets");', 'xpbd_section_mark("cg assembly: stencil gather");')
solver.write_text(x, encoding="utf8")
print("gather in place")
