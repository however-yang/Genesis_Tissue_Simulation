"""Leave the PCG iterations that cannot change anything out of the substep graph when no component is free."""
from pathlib import Path

root = Path(r"D:\Simulation\genesis-world\endoscopic_tissue\native\vendor\cuda")


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


material = root / "solver" / "material" / "coupled_material.cuh"
m = material.read_text(encoding="utf8")
m = replace_once(m, "__global__ void k_MaterialRecoveryRequested(", """// Flags whether some movable node belongs to a free component: one neither held, touched nor anchored. Only those
// nodes take part in the PCG iterations beyond MaterialLoadedIterations.
__global__ void k_MaterialFreeNodes(const UnifiedState* state,int* free){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    bool freeNode=i<s.n && s.invMass[i]>0 && (!s.active||s.active[i]) && !(s.graspComponents[s.component[i]]&7);
    if(__any_sync(0xffffffff,freeNode) && (threadIdx.x&31)==0)atomicExch(free,1);
}
__global__ void k_MaterialRecoveryRequested(""")
material.write_text(m, encoding="utf8")

solver = root / "xpbd" / "xpbd_solver.cu"
s = solver.read_text(encoding="utf8")
s = replace_once(s, """static cudaGraphExec_t g_unifiedRecoveryGraph=nullptr;
static bool g_unifiedRecoveryGraphDirty=false;
""", """static cudaGraphExec_t g_unifiedRecoveryGraph=nullptr;
static bool g_unifiedRecoveryGraphDirty=false;
// The substep graph for substeps with no free component, see the substep query in unifiedSubstep.
static cudaGraphExec_t g_unifiedLoadedGraph=nullptr;
static bool g_unifiedLoadedGraphDirty=false;
static int* d_substepQuery=nullptr;
""")
s = replace_once(s, "        g_unifiedGraphDirty=true;g_unifiedRecoveryGraphDirty=true;\n",
                 "        g_unifiedGraphDirty=true;g_unifiedRecoveryGraphDirty=true;g_unifiedLoadedGraphDirty=true;\n")
s = replace_once(s, """    int* request=&d_cgSearch[0].invalid;int recoveryRequested=0;
    e=cudaMemsetAsync(request,0,sizeof(int));if(e!=cudaSuccess)return -(int)e;
    k_MaterialRecoveryRequested<<<Gr(g_N),64>>>(d_unified,request);
    e=cudaMemcpy(&recoveryRequested,request,sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    cudaGraphExec_t& selectedGraph=recoveryRequested?g_unifiedRecoveryGraph:g_unifiedGraph;
    bool& selectedGraphDirty=recoveryRequested?g_unifiedRecoveryGraphDirty:g_unifiedGraphDirty;
""", """    // The same query also flags a free component. Within a substep the component flags only gain bits, so when no
    // movable node belongs to a free component here, every PCG iteration beyond MaterialLoadedIterations returns at
    // once for every node, and the loaded graph leaves them out.
    if(!d_substepQuery)e=cudaMalloc(&d_substepQuery,2*sizeof(int));if(e!=cudaSuccess)return -(int)e;
    int query[2]={};
    e=cudaMemsetAsync(d_substepQuery,0,2*sizeof(int));if(e!=cudaSuccess)return -(int)e;
    k_MaterialRecoveryRequested<<<Gr(g_N),64>>>(d_unified,d_substepQuery);
    k_MaterialFreeNodes<<<Gr(g_N),64>>>(d_unified,d_substepQuery+1);
    e=cudaMemcpy(query,d_substepQuery,2*sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    const int recoveryRequested=query[0];const bool loaded=!query[1];
    cudaGraphExec_t& selectedGraph=recoveryRequested?g_unifiedRecoveryGraph:loaded?g_unifiedLoadedGraph:g_unifiedGraph;
    bool& selectedGraphDirty=recoveryRequested?g_unifiedRecoveryGraphDirty:loaded?g_unifiedLoadedGraphDirty:g_unifiedGraphDirty;
""")
s = replace_once(s, "                for(int cg=0;cg<(nonlinear==0?MaterialFreeIterations:MaterialLoadedIterations);cg++){\n",
                 "                for(int cg=0;cg<(nonlinear==0 && !loaded?MaterialFreeIterations:MaterialLoadedIterations);cg++){\n")
s = replace_once(s, "if(g_unifiedRecoveryGraph)cudaGraphExecDestroy(g_unifiedRecoveryGraph);g_unifiedRecoveryGraph=nullptr;",
                 "if(g_unifiedRecoveryGraph)cudaGraphExecDestroy(g_unifiedRecoveryGraph);g_unifiedRecoveryGraph=nullptr;"
                 "if(g_unifiedLoadedGraph)cudaGraphExecDestroy(g_unifiedLoadedGraph);g_unifiedLoadedGraph=nullptr;"
                 "cudaFree(d_substepQuery);d_substepQuery=nullptr;")
solver.write_text(s, encoding="utf8")
print("loaded graph in place")
