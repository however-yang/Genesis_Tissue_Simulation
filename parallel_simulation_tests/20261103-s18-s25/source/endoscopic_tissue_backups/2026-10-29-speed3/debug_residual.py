"""Add a debug comparison of the merged PCG residual against the original residual pass, for a debug build only.

Usage: python debug_residual.py apply|restore  (from endoscopic_tissue/native/vendor/cuda)

apply copies the two touched files to <scratch>/debug-residual-backup and inserts, after the stencil assembly, the
original residual computation of k_UnifiedVertex (one thread per vertex) into d_cgA, a comparison against d_cgR and a
one-thread printf of the largest difference, the largest reference entry and the count of differing nodes.
restore copies the files back.
"""
import shutil
import sys
from pathlib import Path

backup = Path(r"C:\WINDOWS\TEMP\claude\D--Simulation-genesis-world\6de6da1d-3b43-4fa3-91f1-fd2ef998be2d\scratchpad\debug-residual-backup")
files = ["solver/material/material_stencil.cuh", "xpbd/xpbd_solver.cu"]
if sys.argv[1] == "restore":
    for f in files:
        shutil.copyfile(backup / Path(f).name, f)
    print("restored")
    sys.exit(0)
backup.mkdir(parents=True, exist_ok=True)
for f in files:
    shutil.copyfile(f, backup / Path(f).name)

KERNELS = r'''
// DEBUG: the original residual pass of k_UnifiedVertex (one thread per vertex).
__global__ void k_DebugResidualReference(const UnifiedState* state,float3* residual){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    residual[i]=mkf3(0,0,0);
    float inv=s.invMass[i];if(inv<=0 || materialCgFixed(s,i))return;
    float3 p=s.pos[i],gradient=mkf3(0,0,0);Symmetric3 H={};
    {
        float inertia=1.f/(inv*s.h*s.h);H.diagonal(inertia);gradient=(p-s.target[i])*inertia;
        if(s.attach && s.attach[i]>0){float viscous=unifiedSupportDamping(s,i);H.diagonal(s.attach[i]+viscous);
            gradient=gradient+(p-s.attachRest[i])*s.attach[i]+(p-s.previous[i])*viscous;}
    }
    if(s.neighbors && s.edgeSupport)for(int a=0;a<6;a++){
        int j=s.neighbors[6*i+a];if(j<0 || (s.active && !s.active[j]))continue;
        int owner=(a&1)?6*i+a:6*j+(a^1);if(s.edgeSupport[owner])continue;
        float3 e=p-s.pos[j];float l=len3(e),l0=len3(s.restNeighbors[6*i+a]);
        if(l>1e-10f){float3 n=e*(1.f/l);float strain=l-l0,K=.5f*(unifiedEdgeStiffness(s,i,a/2)+unifiedEdgeStiffness(s,j,a/2));
            gradient=gradient+n*(K*strain);H.outer(n,K);H.diagonal(K*fmaxf(0.f,strain/l));}
    }
    for(int k=s.offsets[i];k<s.offsets[i+1];k++){
        int t=s.incident[k];if(!s.live[t])continue;const int* ids=s.ids+4*t;
        float3 a=s.pos[ids[0]],e1=s.pos[ids[1]]-a,e2=s.pos[ids[2]]-a,e3=s.pos[ids[3]]-a;
        float3 b[3]={s.b0[t],s.b1[t],s.b2[t]};
        float3 q=i==ids[0]?(b[0]+b[1]+b[2])*(-1.f):i==ids[1]?b[0]:i==ids[2]?b[1]:b[2];
        float3 Fq=e1*dot3(b[0],q)+e2*dot3(b[1],q)+e3*dot3(b[2],q);
        float det=s.restMetric[7*t+6];float J=dot3(e1,cross3(e2,e3))*det;
        float3 g1=cross3(e2,e3)*det,g2=cross3(e3,e1)*det,g3=cross3(e1,e2)*det;
        float3 g=i==ids[0]?(g1+g2+g3)*(-1.f):i==ids[1]?g1:i==ids[2]?g2:g3;
        float mu=s.mu[t],lambda=s.bulk[t]+mu/3.f,pressure=lambda*(J-1.f)-mu;
        CompressionBarrier barrier=unifiedCompressionBarrier(J,s.minimum,lambda);
        gradient=gradient+Fq*mu+g*(pressure+barrier.gradient);
        H.diagonal(mu*dot3(q,q));H.outer(g,lambda+barrier.curvature);
        unifiedStrainDamping(s,t,e1,e2,e3,q,gradient,H);
        int me=i==ids[0]?0:i==ids[1]?1:i==ids[2]?2:3;
        for(int j=0;j<4;j++)if(j!=me && s.hardening>0){
            float3 e=p-s.pos[ids[j]];float l=len3(e),l0=unifiedRestLength(s,t,me,j),strain=fmaxf(0.f,l/l0-1.f);
            if(strain>0 && l>1e-12f){float3 dir=e*(1.f/l);float s2=strain*strain,s4=s2*s2;float f=3.f*mu*s.hardening*s4*strain/l0;
                float h=15.f*mu*s.hardening*s4/(l0*l0);
                gradient=gradient+dir*f;H.diagonal(f/l);H.outer(dir,h-f/l);}
        }
    }
    residual[i]=gradient*(-1.f);
}
__global__ void k_DebugResidualCompare(int n,const float3* merged,const float3* reference,unsigned* out){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    float3 d=merged[i]-reference[i],r=reference[i];
    float diff=fmaxf(fabsf(d.x),fmaxf(fabsf(d.y),fabsf(d.z))),mag=fmaxf(fabsf(r.x),fmaxf(fabsf(r.y),fabsf(r.z)));
    float relative=diff/fmaxf(mag,1e-30f);
    if(diff>0){atomicAdd(out+2,1u);atomicMax(out,__float_as_uint(diff));atomicMax(out+3,__float_as_uint(relative));}
    atomicMax(out+1,__float_as_uint(mag));
}
__global__ void k_DebugResidualPrint(unsigned* out){
    printf("RESIDUAL maxdiff %.9g maxref %.9g differing %u maxrel %.9g\n",__uint_as_float(out[0]),__uint_as_float(out[1]),out[2],__uint_as_float(out[3]));
    out[0]=out[1]=out[2]=out[3]=0;
}
'''


def patch(path, old, new):
    text = open(path, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in text else "\n"
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    assert text.count(old) == 1, (path, old[:60], text.count(old))
    open(path, "w", encoding="utf-8", newline="").write(text.replace(old, new))


patch("solver/material/material_stencil.cuh", "// The lattice offset of the neighbor in slot 0..14 of a stencil row",
      KERNELS + "// The lattice offset of the neighbor in slot 0..14 of a stencil row")
patch("xpbd/xpbd_solver.cu",
      """                k_MaterialStencilAssemble<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgR,d_cgH);""",
      """                k_MaterialStencilAssemble<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgR,d_cgH);
                k_DebugResidualReference<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgA);
                k_DebugResidualCompare<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_cgR,d_cgA,d_debugOut);
                k_DebugResidualPrint<<<1,1,0,g_captureStream>>>(d_debugOut);""")
patch("xpbd/xpbd_solver.cu",
      """    if(e==cudaSuccess && !d_globalCoarse)e=cudaMalloc(&d_globalCoarse,g_N*sizeof(CoarseBlock));""",
      """    if(e==cudaSuccess && !d_globalCoarse)e=cudaMalloc(&d_globalCoarse,g_N*sizeof(CoarseBlock));
    if(e==cudaSuccess && !d_debugOut){e=cudaMalloc(&d_debugOut,16);if(e==cudaSuccess)e=cudaMemset(d_debugOut,0,16);}""")
patch("xpbd/xpbd_solver.cu", "static UnifiedState* d_unified=nullptr;",
      """static UnifiedState* d_unified=nullptr;
static unsigned* d_debugOut=nullptr;""")
print("applied")
