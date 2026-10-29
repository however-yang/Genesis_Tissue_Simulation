"""Wire the material multigrid preconditioner (material_multigrid.cuh) into the PCG and the solver lifecycle.

Usage: python patch_multigrid.py  (from endoscopic_tissue/native/vendor/cuda/xpbd)
"""
path = "xpbd_solver.cu"
text = open(path, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in text else "\n"


def rep(old, new, count=1):
    global text
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    assert text.count(old) == count, (old[:80], text.count(old))
    text = text.replace(old, new)


rep('#include "../solver/material/material_stencil.cuh"',
    '#include "../solver/material/material_stencil.cuh"\n#include "../solver/material/material_multigrid.cuh"')
rep("    if (!d_topologyRecords) XK(cudaMalloc(&d_topologyRecords, 6 * sizeof(unsigned)));\n"
    "    XK(cudaMemset(d_topologyRecords, 0xff, 6 * sizeof(unsigned)));",
    "    if (!d_topologyRecords) XK(cudaMalloc(&d_topologyRecords, 8 * sizeof(unsigned)));\n"
    "    XK(cudaMemset(d_topologyRecords, 0xff, 8 * sizeof(unsigned)));")
rep("    d_edgeSupportRevision = d_topologyRecords + 4;",
    "    d_edgeSupportRevision = d_topologyRecords + 4; d_multigridRevision = d_topologyRecords + 6;")
rep("// derived from those flags; xpbd_bind allocates them together.\nstatic unsigned* d_topologyRecords=nullptr;",
    "// derived from those flags, and the multigrid hierarchy; xpbd_bind allocates them together.\n"
    "static unsigned* d_topologyRecords=nullptr;\nstatic unsigned* d_multigridRevision=nullptr;")
rep("    cudaFree(d_topologyRecords); d_topologyRecords = d_componentRevision = d_livenessRevision = d_edgeSupportRevision = nullptr;",
    "    cudaFree(d_topologyRecords);\n"
    "    d_topologyRecords = d_componentRevision = d_livenessRevision = d_edgeSupportRevision = d_multigridRevision = nullptr;")

HOST = """
// ---- The multigrid preconditioner of the material PCG (material_multigrid.cuh).
// Levels 1..g_multigridLevels, each sized for MG_SLOTS aggregates per site: the coarsest is the first whose capacity the
// dense solve takes (MG_COARSEST).
static int g_multigridLevels=0;
static MultigridLevel g_multigrid[MG_MAX_LEVELS+1]={};
// The dense coarsest operator and its Cholesky factor, the inverse of the factor, and the inverse operator.
static float *d_multigridDense=nullptr,*d_multigridFactorInverse=nullptr,*d_multigridInverse=nullptr;
// The fine correction before the post smoothing.
static float3* d_multigridFine=nullptr;
static void freeMultigrid(){
    for(int l=1;l<=g_multigridLevels;l++){
        MultigridLevel& L=g_multigrid[l];
        for(void* p:{(void*)L.slots,(void*)L.index,(void*)L.count,(void*)L.parent,(void*)L.site,(void*)L.component,
                     (void*)L.neighbor,(void*)L.A,(void*)L.inverse,(void*)L.r,(void*)L.z,(void*)L.post})cudaFree(p);
        L={};
    }
    g_multigridLevels=0;
    cudaFree(d_multigridDense);cudaFree(d_multigridFactorInverse);cudaFree(d_multigridInverse);cudaFree(d_multigridFine);
    d_multigridDense=d_multigridFactorInverse=d_multigridInverse=nullptr;d_multigridFine=nullptr;
}
static cudaError_t allocateMultigrid(){
    if(g_multigridLevels)return cudaSuccess;
    int sx=g_dimsX+1,sy=g_dimsY+1,sz=g_dimsZ+1,children=g_N;cudaError_t e=cudaSuccess;
    for(int l=1;l<=MG_MAX_LEVELS && e==cudaSuccess;l++){
        sx=(sx+1)/2;sy=(sy+1)/2;sz=(sz+1)/2;
        MultigridLevel& L=g_multigrid[l];L.sx=sx;L.sy=sy;L.sz=sz;
        int sites=sx*sy*sz,capacity=MG_SLOTS*sites;L.capacity=capacity;g_multigridLevels=l;
        if(e==cudaSuccess)e=cudaMalloc(&L.slots,(size_t)capacity*sizeof(int));
        if(e==cudaSuccess)e=cudaMalloc(&L.index,(size_t)capacity*sizeof(int));
        if(e==cudaSuccess)e=cudaMalloc(&L.count,sizeof(int));
        if(e==cudaSuccess)e=cudaMalloc(&L.parent,(size_t)children*sizeof(int));
        if(e==cudaSuccess)e=cudaMalloc(&L.site,(size_t)capacity*sizeof(int));
        if(e==cudaSuccess)e=cudaMalloc(&L.component,(size_t)capacity*sizeof(int));
        if(e==cudaSuccess)e=cudaMalloc(&L.neighbor,(size_t)27*capacity*sizeof(int));
        if(e==cudaSuccess)e=cudaMalloc(&L.A,(size_t)27*9*capacity*sizeof(float));
        if(e==cudaSuccess)e=cudaMalloc(&L.inverse,(size_t)9*capacity*sizeof(float));
        if(e==cudaSuccess)e=cudaMalloc(&L.r,(size_t)capacity*sizeof(float3));
        if(e==cudaSuccess)e=cudaMalloc(&L.z,(size_t)capacity*sizeof(float3));
        if(e==cudaSuccess)e=cudaMalloc(&L.post,(size_t)capacity*sizeof(float3));
        children=capacity;
        if(capacity<=MG_COARSEST)break;
    }
    const size_t dense=(size_t)9*MG_COARSEST*MG_COARSEST*sizeof(float);
    if(e==cudaSuccess)e=cudaMalloc(&d_multigridDense,dense);
    if(e==cudaSuccess)e=cudaMalloc(&d_multigridFactorInverse,dense);
    if(e==cudaSuccess)e=cudaMalloc(&d_multigridInverse,dense);
    if(e==cudaSuccess)e=cudaMalloc(&d_multigridFine,(size_t)g_N*sizeof(float3));
    if(e==cudaSuccess && g_multigrid[g_multigridLevels].capacity>MG_COARSEST)e=cudaErrorInvalidValue;
    if(e!=cudaSuccess)freeMultigrid();
    return e;
}
// The aggregates of every level, made again once the topology revision moves (the component labels are fresh).
static void rebuildMultigrid(){
    k_TopologyStale<<<1,1>>>(cut_topology_revision_device(),d_multigridRevision);
    const unsigned* stale=d_multigridRevision+1;int children=g_N;
    for(int l=1;l<=g_multigridLevels;l++){
        MultigridLevel& L=g_multigrid[l];
        k_MgClear<<<Gr(L.capacity),64>>>(L,stale);
        if(l==1)k_MgClaimFine<<<Gr(g_N),64>>>(d_unified,L,stale);
        else k_MgClaim<<<Gr(g_multigrid[l-1].capacity),64>>>(g_multigrid[l-1],L,stale);
        k_MgNumber<<<Gr(L.capacity),64>>>(L,stale);
        k_MgParent<<<Gr(children),64>>>(L,children,stale);
        k_MgNeighbors<<<Gr(L.capacity),64>>>(L,stale);
        children=L.capacity;
    }
}
// The Galerkin operators of this solve's stencil and the dense inverse of the coarsest one.
static void assembleMultigrid(cudaStream_t stream){
    k_MgGalerkinFine<<<Gr(27*g_multigrid[1].capacity),64,0,stream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgStencil,g_multigrid[1]);
    for(int l=2;l<=g_multigridLevels;l++)k_MgGalerkin<<<Gr(27*g_multigrid[l].capacity),64,0,stream>>>(g_multigrid[l-1],g_multigrid[l]);
    const MultigridLevel& coarsest=g_multigrid[g_multigridLevels];
    cudaMemsetAsync(d_multigridDense,0,(size_t)9*MG_COARSEST*MG_COARSEST*sizeof(float),stream);
    k_MgDense<<<Gr(27*coarsest.capacity),64,0,stream>>>(coarsest,d_multigridDense);
    k_MgCholesky<<<1,256,0,stream>>>(coarsest,d_multigridDense);
    k_MgFactorInverse<<<Gr(3*MG_COARSEST),64,0,stream>>>(coarsest,d_multigridDense,d_multigridFactorInverse);
    k_MgCoarseInverse<<<Gr(9*MG_COARSEST*MG_COARSEST),64,0,stream>>>(coarsest,d_multigridFactorInverse,d_multigridInverse);
}
// z = M^-1 r by one V(1,1) cycle, adding r'z into the slot `slot` of each component's PCG sums.
static void applyMultigrid(cudaStream_t stream,const float3* r,float3* z,MaterialCGSum* sums,int slot){
    MultigridLevel* L=g_multigrid;const int top=g_multigridLevels;
    cudaMemsetAsync(L[1].r,0,(size_t)L[1].capacity*sizeof(float3),stream);
    k_MgFinePre<<<Gr(g_N),64,0,stream>>>(d_unified,d_cgStencil,d_cgH,r,d_multigridFine);
    k_MgFineRestrict<<<Gr(g_N),64,0,stream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgStencil,r,d_multigridFine,L[1]);
    for(int l=1;l<top;l++){
        cudaMemsetAsync(L[l+1].r,0,(size_t)L[l+1].capacity*sizeof(float3),stream);
        k_MgPre<<<Gr(L[l].capacity),64,0,stream>>>(L[l]);
        k_MgRestrict<<<Gr(L[l].capacity),64,0,stream>>>(L[l],L[l+1]);
    }
    k_MgCoarsest<<<Gr(3*MG_COARSEST),64,0,stream>>>(L[top],d_multigridInverse);
    for(int l=top-1;l>=1;l--){
        k_MgProlong<<<Gr(L[l].capacity),64,0,stream>>>(L[l],L[l+1]);
        k_MgPost<<<Gr(L[l].capacity),64,0,stream>>>(L[l]);
    }
    k_MgFineProlong<<<Gr(g_N),64,0,stream>>>(d_unified,d_cgStencil,L[1],d_multigridFine);
    k_MgFinePost<<<Gr(g_N),64,0,stream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgStencil,d_cgH,r,d_multigridFine,z,sums,slot);
}

"""
rep("// Cut edges can change BETWEEN substeps and during final reconstruction.\n",
    HOST + "// Cut edges can change BETWEEN substeps and during final reconstruction.\n")
rep("    if(e==cudaSuccess && !d_cgMatrix)e=cudaMalloc(&d_cgMatrix,(size_t)72*g_N*sizeof(__half));",
    "    if(e==cudaSuccess && !d_cgMatrix)e=cudaMalloc(&d_cgMatrix,(size_t)72*g_N*sizeof(__half));\n"
    "    if(e==cudaSuccess)e=allocateMultigrid();")
rep("    e=cudaMemcpyAsync(d_unified,&s,sizeof(s),cudaMemcpyHostToDevice);if(e!=cudaSuccess)return -(int)e;",
    "    e=cudaMemcpyAsync(d_unified,&s,sizeof(s),cudaMemcpyHostToDevice);if(e!=cudaSuccess)return -(int)e;\n"
    "    rebuildMultigrid();")
rep("""                k_MaterialCGClear<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_cgS);
                k_MaterialCGBegin<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgR,d_cgH,d_cgZ,d_cgQ,d_cgD,d_cgS);
                xpbd_section_mark("cg: assembly");
                for(int cg=0;cg<(nonlinear==0 && !loaded?MaterialFreeIterations:MaterialLoadedIterations);cg++){
                    float3* direction=cg%2?d_cgQ:d_cgP;const float3* previous=cg%2?d_cgP:d_cgQ;
                    k_MaterialStencilProduct<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,previous,d_cgZ,direction,d_cgMatrix,d_cgScale,d_cgStencil,d_cgA,d_cgS,cg);
                    k_MaterialCGAdvance<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgStencil,d_cgR,d_cgH,direction,d_cgA,d_cgD,d_cgZ,d_cgS,cg);
                }""",
    """                k_MaterialCGClear<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_cgS);
                assembleMultigrid(g_captureStream);
                k_MaterialCGBegin<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgQ,d_cgD);
                applyMultigrid(g_captureStream,d_cgR,d_cgZ,d_cgS,2);
                xpbd_section_mark("cg: assembly");
                const int iterations=nonlinear==0 && !loaded?MaterialFreeIterations:MaterialLoadedIterations;
                for(int cg=0;cg<iterations;cg++){
                    float3* direction=cg%2?d_cgQ:d_cgP;const float3* previous=cg%2?d_cgP:d_cgQ;
                    k_MaterialStencilProduct<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,previous,d_cgZ,direction,d_cgMatrix,d_cgScale,d_cgStencil,d_cgA,d_cgS,cg);
                    k_MaterialCGAdvance<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgStencil,d_cgR,direction,d_cgA,d_cgD,d_cgS,cg);
                    // The last direction is formed by no further product.
                    if(cg+1<iterations)applyMultigrid(g_captureStream,d_cgR,d_cgZ,d_cgS,cg%3);
                }""")
rep("    cudaFree(d_cgMatrix);d_cgMatrix=nullptr;cudaFree(d_cgScale);d_cgScale=nullptr;",
    "    cudaFree(d_cgMatrix);d_cgMatrix=nullptr;cudaFree(d_cgScale);d_cgScale=nullptr;freeMultigrid();")
open(path, "w", encoding="utf-8", newline="").write(text)
print("ok")
