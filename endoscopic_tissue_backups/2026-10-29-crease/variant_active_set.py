"""Variant for a test build (apply after debug_crease_dump.py, debug_material_dump.py and debug_push.py): an active-set
material step. After the volume-floor restriction of the PCG direction, every node of a grasped component whose
direction it cut below half is held fixed, and the PCG solves again from the same residual with those nodes fixed;
that direction is restricted and line searched. The second solve runs in every substep here.

Usage: python variant_active_set.py apply   (from endoscopic_tissue/native/vendor/cuda; undo with git checkout)
"""
import sys
from pathlib import Path


def patch(path, old, new):
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, old[:80])
    Path(path).write_text(text.replace(old, new), encoding="utf-8")


assert sys.argv[1] == "apply"
VERTEX = "solver/material/unified_vertex.cuh"
patch(VERTEX, "*degree,*gripDegree,*neighbors,*edgeSupport,*active;", "*degree,*gripDegree,*neighbors,*edgeSupport,*active,*blockedNodes;")
patch(VERTEX, "|| (s.grips>0 && s.gripDegree[i]>0);\n}\n__device__ inline float unifiedEdgeStiffness",
      "|| (s.grips>0 && s.gripDegree[i]>0) || (s.blockedNodes && s.blockedNodes[i]);\n}\n__device__ inline float unifiedEdgeStiffness")
SEARCH = "solver/material/material_line_search.cuh"
text = Path(SEARCH).read_text(encoding="utf-8")
Path(SEARCH).write_text(text + """
__global__ void k_ClearBlockedNodes(int n,int* blocked){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)blocked[i]=0;}
// Holds fixed for the second solve the nodes of grasped components whose restricted direction lost over half its length.
__global__ void k_MarkBlockedNodes(const UnifiedState* state,const float3* free,const float3* used,int* blocked){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;
    float f=len3(free[i]),u=len3(used[i]);
    blocked[i]=f>0 && u<.5f*f && (s.graspComponents[s.component[i]]&7)?1:0;
}
__global__ void k_RestoreResidual(const UnifiedState* state,const float3* saved,float3* residual){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;
    residual[i]=materialCgFixed(s,i)?mkf3(0,0,0):saved[i];
}
""", encoding="utf-8")
SOLVER = "xpbd/xpbd_solver.cu"
patch(SOLVER, "static MaterialLineSearch* d_cgSearch=nullptr;\n", """static MaterialLineSearch* d_cgSearch=nullptr;
static float3 *d_cgR0=nullptr,*d_cgFree=nullptr;static int* d_materialBlocked=nullptr;
""")
patch(SOLVER, "    if(e==cudaSuccess && !d_cgSearch)e=cudaMalloc(&d_cgSearch,g_N*sizeof(MaterialLineSearch));\n",
      """    if(e==cudaSuccess && !d_cgSearch)e=cudaMalloc(&d_cgSearch,g_N*sizeof(MaterialLineSearch));
    if(e==cudaSuccess && !d_cgR0){cudaMalloc(&d_cgR0,g_N*sizeof(float3));cudaMalloc(&d_cgFree,g_N*sizeof(float3));
        cudaMalloc(&d_materialBlocked,g_N*sizeof(int));cudaMemset(d_materialBlocked,0,g_N*sizeof(int));}
""")
patch(SOLVER, "    s.degree=d_surfaceDegree;s.gripDegree=d_gripDegree;", "    s.blockedNodes=d_materialBlocked;s.degree=d_surfaceDegree;s.gripDegree=d_gripDegree;")
patch(SOLVER, """                k_MaterialStencilAssemble<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgR,d_cgH);
""", """                k_ClearBlockedNodes<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_materialBlocked);
                k_MaterialStencilAssemble<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgR,d_cgH);
                cudaMemcpyAsync(d_cgR0,d_cgR,g_N*sizeof(float3),cudaMemcpyDeviceToDevice,g_captureStream);
""")
patch(SOLVER, """                cudaMemcpyAsync(d_debugFree,d_cgD,g_N*sizeof(float3),cudaMemcpyDeviceToDevice,g_captureStream);
""", """                cudaMemcpyAsync(d_debugFree,d_cgD,g_N*sizeof(float3),cudaMemcpyDeviceToDevice,g_captureStream);
                cudaMemcpyAsync(d_cgFree,d_cgD,g_N*sizeof(float3),cudaMemcpyDeviceToDevice,g_captureStream);
""")
patch(SOLVER, """                xpbd_section_mark("cg: feasible");
""", """                xpbd_section_mark("cg: feasible");
                // Active set: hold the cut nodes fixed and solve again from the same residual.
                k_MarkBlockedNodes<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgFree,d_cgD,d_materialBlocked);
                k_MaterialStencilActiveMask<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgStencil);
                k_RestoreResidual<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgR0,d_cgR);
                k_MaterialCGClear<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_cgS);
                assembleMultigrid(g_captureStream);
                k_MaterialCGBegin<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgQ,d_cgD);
                applyMultigrid(g_captureStream,d_cgR,d_cgZ,d_cgS,2);
                for(int cg=0;cg<iterations;cg++){
                    float3* direction=cg%2?d_cgQ:d_cgP;const float3* previous=cg%2?d_cgP:d_cgQ;
                    k_MaterialStencilProduct<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,previous,d_cgZ,direction,d_cgMatrix,d_cgScale,d_cgStencil,d_cgA,d_cgS,cg);
                    k_MaterialCGAdvance<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgStencil,d_cgR,direction,d_cgA,d_cgD,d_cgS,cg);
                    if(cg+1<iterations)applyMultigrid(g_captureStream,d_cgR,d_cgZ,d_cgS,cg%3);
                }
                for(int feasible=0;feasible<8;feasible++){
                    int* changed=&d_cgSearch[feasible&1].invalid;
                    const int* previous=feasible?&d_cgSearch[(feasible-1)&1].invalid:nullptr;
                    cudaMemsetAsync(changed,0,sizeof(int),g_captureStream);
                    k_MaterialRestrictionClear<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_coarseWeight);
                    k_RestrictMaterialDirection<<<Gr(g_T),64,0,g_captureStream>>>(d_unified,d_cgD,d_coarseWeight,previous,changed);
                    k_ApplyMaterialRestriction<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_cgD,d_coarseWeight,changed,d_unified);
                }
""")
print("applied")
