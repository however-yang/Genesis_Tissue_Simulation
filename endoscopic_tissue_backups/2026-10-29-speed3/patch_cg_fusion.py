"""Fuse the PCG direction update into the next matrix-vector product and rotate the sums through three slots.

Usage: python patch_cg_fusion.py  (from endoscopic_tissue/native/vendor/cuda)
"""


def patch(path, edits):
    text = open(path, encoding="utf-8", newline="").read()
    crlf = "\r\n" in text
    for old, new in edits:
        if crlf:
            old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        assert text.count(old) == 1, (path, old[:80], text.count(old))
        text = text.replace(old, new)
    open(path, "w", encoding="utf-8", newline="").write(text)
    print("ok", path)


patch("solver/material/coupled_material.cuh", [
("""struct MaterialCGSum { double rho,next,denominator; };""",
"""// The PCG sums of a component for three consecutive iterations, by iteration modulo 3: p'Ap (denominator) and r'z
// (next) of iteration k. The r'z of iteration k-1 is the rho of iteration k (materialRho). The direction update of
// iteration k therefore runs inside the matrix-vector product of iteration k+1 (k_MaterialStencilProduct), and the
// advance of iteration k clears the slot of iteration k+1, with no kernel of their own.
struct MaterialCGSum { double denominator[3],next[3]; };
// The rho of an iteration; k_MaterialCGBegin stores the initial one as the r'z of iteration -1.
__device__ inline double materialRho(const MaterialCGSum& q,int iteration){return q.next[(iteration+2)%3];}"""),
("""__global__ void k_MaterialCGClear(int n,MaterialCGSum* sums,bool initial){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    sums[i].next=sums[i].denominator=0;if(initial)sums[i].rho=0;
}
__global__ void k_MaterialCGBegin(const UnifiedState* state,const float3* r,const Symmetric3* diagonal,
    float3* direction,float3* step,MaterialCGSum* sums){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    step[i]=mkf3(0,0,0);direction[i]=mkf3(0,0,0);
    if(materialCgFixed(s,i))return;
    direction[i]=diagonal[i].solve(r[i]);
    materialDotAdd(&sums[s.component[i]].rho,unifiedDot(r[i],direction[i]));
}""",
"""__global__ void k_MaterialCGClear(int n,MaterialCGSum* sums){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)sums[i]={};
}
// The first preconditioned residual z, and a zero direction before it: iteration 0 then forms its direction z+0*0
// as every later one forms z+beta*p (k_MaterialStencilProduct).
__global__ void k_MaterialCGBegin(const UnifiedState* state,const float3* r,const Symmetric3* diagonal,
    float3* preconditioned,float3* direction,float3* step,MaterialCGSum* sums){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    step[i]=mkf3(0,0,0);direction[i]=mkf3(0,0,0);
    if(materialCgFixed(s,i))return;
    preconditioned[i]=diagonal[i].solve(r[i]);
    materialDotAdd(&sums[s.component[i]].next[2],unifiedDot(r[i],preconditioned[i]));
}"""),
("""    MaterialCGSum* sums,int iteration){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    if(!materialCgNode(stencil,i)||(iteration>=MaterialLoadedIterations && (s.graspComponents[s.component[i]]&7)))return;
    auto& q=sums[s.component[i]];float alpha=q.denominator>1e-30&&q.rho>0?(float)(q.rho/q.denominator):0;
    step[i]=step[i]+direction[i]*alpha;float3 r=residual[i]-product[i]*alpha;residual[i]=r;
    float3 z=diagonal[i].solve(r);preconditioned[i]=z;
    materialDotAdd(&q.next,unifiedDot(r,z));
}
__global__ void k_MaterialCGDirection(const UnifiedState* state,const unsigned short* stencil,const float3* preconditioned,
    float3* direction,const MaterialCGSum* sums,int iteration){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    if(!materialCgNode(stencil,i)||(iteration>=MaterialLoadedIterations && (s.graspComponents[s.component[i]]&7)))return;
    const auto& q=sums[s.component[i]];float beta=q.rho>1e-30?(float)(q.next/q.rho):0;
    direction[i]=preconditioned[i]+direction[i]*beta;
}
__global__ void k_MaterialCGFinish(const UnifiedState* state,MaterialCGSum* sums,int iteration){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n||s.component[i]!=i||(s.active&&!s.active[i])||(iteration>=MaterialLoadedIterations && (s.graspComponents[i]&7)))return;
    sums[i].rho=sums[i].next;sums[i].next=sums[i].denominator=0;
}""",
"""    MaterialCGSum* sums,int iteration){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    // The slot of the next iteration held the sums of the one before last, which this iteration's product read last.
    if(coarseRoot(s,i)){sums[i].denominator[(iteration+1)%3]=0;sums[i].next[(iteration+1)%3]=0;}
    if(!materialCgNode(stencil,i)||(iteration>=MaterialLoadedIterations && (s.graspComponents[s.component[i]]&7)))return;
    auto& q=sums[s.component[i]];double rho=materialRho(q,iteration),denominator=q.denominator[iteration%3];
    float alpha=denominator>1e-30&&rho>0?(float)(rho/denominator):0;
    step[i]=step[i]+direction[i]*alpha;float3 r=residual[i]-product[i]*alpha;residual[i]=r;
    float3 z=diagonal[i].solve(r);preconditioned[i]=z;
    materialDotAdd(&q.next[iteration%3],unifiedDot(r,z));
}"""),
])
patch("solver/material/material_stencil.cuh", [
("""__global__ void k_MaterialStencilProduct(const UnifiedState* __restrict__ state,const float3* __restrict__ vector,
    const __half* __restrict__ matrix,const float* __restrict__ blockScale,const unsigned short* __restrict__ stencil,
    float3* __restrict__ result,MaterialCGSum* sums,int iteration){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    unsigned mask=stencil[i];
    result[i]=mkf3(0,0,0);if(!(mask>>7&1)||(iteration>=MaterialLoadedIterations && (s.graspComponents[s.component[i]]&7)))return;
    float3 out=mkf3(0,0,0);
    #pragma unroll
    for(int slot=0;slot<15;slot++){
        if(!(mask>>slot&1))continue;
        int j=i+materialSlotOffset(s,slot),block=abs(slot-7),owner=slot<7?j:i;
        float3 v=vector[j]*blockScale[block*s.n+owner];const __half* h=matrix+9*block*s.n+owner;float a[9];""",
"""// The product A*p of this iteration's direction p, and its p'Ap. The direction is formed here, p=z+beta*p from the
// preconditioned residual z and the previous direction, for the node and for each neighbor it reads: a neighbor is a
// CG node of the same component, so it forms the same value. The node stores its own into direction for the advance.
__global__ void k_MaterialStencilProduct(const UnifiedState* __restrict__ state,const float3* __restrict__ previous,
    const float3* __restrict__ preconditioned,float3* __restrict__ direction,
    const __half* __restrict__ matrix,const float* __restrict__ blockScale,const unsigned short* __restrict__ stencil,
    float3* __restrict__ result,MaterialCGSum* sums,int iteration){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
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
        float3 v=p*blockScale[block*s.n+owner];const __half* h=matrix+9*block*s.n+owner;float a[9];"""),
("""    result[i]=out;materialDotAdd(&sums[s.component[i]].denominator,unifiedDot(vector[i],out));
}""",
"""    result[i]=out;materialDotAdd(&sums[s.component[i]].denominator[iteration%3],unifiedDot(own,out));
}"""),
])
patch("xpbd/xpbd_solver.cu", [
("""static float3 *d_cgR=nullptr,*d_cgP=nullptr,*d_cgD=nullptr,*d_cgA=nullptr,*d_cgZ=nullptr;""",
"""// d_cgP and d_cgQ hold the PCG direction of alternate iterations (k_MaterialStencilProduct).
static float3 *d_cgR=nullptr,*d_cgP=nullptr,*d_cgQ=nullptr,*d_cgD=nullptr,*d_cgA=nullptr,*d_cgZ=nullptr;"""),
("""    if(e==cudaSuccess && !d_cgP)e=cudaMalloc(&d_cgP,g_N*sizeof(float3));""",
"""    if(e==cudaSuccess && !d_cgP)e=cudaMalloc(&d_cgP,g_N*sizeof(float3));
    if(e==cudaSuccess && !d_cgQ)e=cudaMalloc(&d_cgQ,g_N*sizeof(float3));"""),
("""                k_MaterialCGClear<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_cgS,true);
                k_MaterialCGBegin<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgR,d_cgH,d_cgP,d_cgD,d_cgS);""",
"""                k_MaterialCGClear<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_cgS);
                k_MaterialCGBegin<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgR,d_cgH,d_cgZ,d_cgQ,d_cgD,d_cgS);"""),
("""                    k_MaterialStencilProduct<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgP,d_cgMatrix,d_cgScale,d_cgStencil,d_cgA,d_cgS,cg);
                    k_MaterialCGAdvance<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgStencil,d_cgR,d_cgH,d_cgP,d_cgA,d_cgD,d_cgZ,d_cgS,cg);
                    k_MaterialCGDirection<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgStencil,d_cgZ,d_cgP,d_cgS,cg);
                    k_MaterialCGFinish<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgS,cg);""",
"""                    float3* direction=cg%2?d_cgQ:d_cgP;const float3* previous=cg%2?d_cgP:d_cgQ;
                    k_MaterialStencilProduct<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,previous,d_cgZ,direction,d_cgMatrix,d_cgScale,d_cgStencil,d_cgA,d_cgS,cg);
                    k_MaterialCGAdvance<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgStencil,d_cgR,d_cgH,direction,d_cgA,d_cgD,d_cgZ,d_cgS,cg);"""),
("""cudaFree(d_cgP);d_cgP=nullptr;""",
"""cudaFree(d_cgP);d_cgP=nullptr;cudaFree(d_cgQ);d_cgQ=nullptr;"""),
])
