"""Launch the material PCG and multigrid fine-level kernels over the active nodes only (UnifiedState::activeNodes).

The active flags never change after the build, so xpbd_build_tets lists the active nodes once. The kernels assemble,
mask, multiply, advance and smooth exactly the nodes they did before: every inactive node returned at once, and no
reader takes an entry of an inactive node.

Usage: python patch_active_nodes.py  (from endoscopic_tissue/native/vendor/cuda)
"""


def load(path):
    text = open(path, encoding="utf-8", newline="").read()
    return text, ("\r\n" if "\r\n" in text else "\n")


def save(path, text):
    open(path, "w", encoding="utf-8", newline="").write(text)


def rep(text, nl, old, new, count=1):
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    assert text.count(old) == count, (old[:90], text.count(old))
    return text.replace(old, new)


LIST = "int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;"

path = "solver/material/unified_vertex.cuh"
text, nl = load(path)
text = rep(text, nl, "    const KuhnTet* kuhn;const int* cellBase;int cells;\n",
           "    const KuhnTet* kuhn;const int* cellBase;int cells;\n"
           "    // The active nodes in increasing order, which the material PCG and the multigrid fine level launch over.\n"
           "    const int* activeNodes;int activeCount;\n")
text = rep(text, nl, "__device__ inline void unifiedTetIds(const UnifiedState& s,int t,int (&id)[4]){",
           "// The node of thread k of a launch over the active nodes, or -1 past them.\n"
           "__device__ inline int unifiedActiveNode(const UnifiedState& s,int k){return k<s.activeCount?s.activeNodes[k]:-1;}\n"
           "__device__ inline void unifiedTetIds(const UnifiedState& s,int t,int (&id)[4]){")
save(path, text)

path = "solver/material/material_stencil.cuh"
text, nl = load(path)
text = rep(text, nl, "    const auto& s=*state;int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=s.n)return;",
           "    const auto& s=*state;int node=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(node<0)return;")
text = rep(text, nl, "    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;\n    unsigned mask=stencil[i];",
           "    const auto& s=*state;" + LIST + "\n    unsigned mask=stencil[i];")
text = rep(text, nl, "    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;\n    unsigned mask=0;",
           "    const auto& s=*state;" + LIST + "\n    unsigned mask=0;")
save(path, text)

path = "solver/material/coupled_material.cuh"
text, nl = load(path)
text = rep(text, nl, "    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;\n    step[i]=mkf3(0,0,0);",
           "    const auto& s=*state;" + LIST + "\n    step[i]=mkf3(0,0,0);")
text = rep(text, nl, "    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;\n    // The slot of the next iteration",
           "    const auto& s=*state;" + LIST + "\n    // The slot of the next iteration")
save(path, text)

path = "solver/material/material_multigrid.cuh"
text, nl = load(path)
text = rep(text, nl, "    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;", "    const auto& s=*state;" + LIST, count=4)
save(path, text)

path = "xpbd/xpbd_solver.cu"
text, nl = load(path)
text = rep(text, nl, "static int     g_tetCells = 0;           // T/6: the tets of each type\n",
           "static int     g_tetCells = 0;           // T/6: the tets of each type\n"
           "static int*    d_activeNodes = nullptr;  // [g_activeNodes] the active nodes in increasing order (UnifiedState::activeNodes)\n"
           "static int     g_activeNodes = 0;\n")
text = rep(text, nl, "    cudaFree(d_tetCellBase); d_tetCellBase = nullptr; cudaFree(d_kuhnTets); d_kuhnTets = nullptr;\n    if (T > 0) {",
           "    cudaFree(d_tetCellBase); d_tetCellBase = nullptr; cudaFree(d_kuhnTets); d_kuhnTets = nullptr;\n"
           "    // The active flags never change after this build: list the active nodes once.\n"
           "    std::vector<int> activeNodes;\n"
           "    for (int c = 0; c < cc; c++) if (hActive[c] != 0) activeNodes.push_back(c);\n"
           "    cudaFree(d_activeNodes); d_activeNodes = nullptr; g_activeNodes = (int)activeNodes.size();\n"
           "    if (g_activeNodes > 0) {\n"
           "        XK(cudaMalloc(&d_activeNodes, activeNodes.size() * sizeof(int)));\n"
           "        XK(cudaMemcpy(d_activeNodes, activeNodes.data(), activeNodes.size() * sizeof(int), cudaMemcpyHostToDevice));\n"
           "    }\n"
           "    if (T > 0) {")
text = rep(text, nl, "    cudaFree(d_tetCellBase); d_tetCellBase = nullptr; cudaFree(d_kuhnTets); d_kuhnTets = nullptr; g_tetCells = 0;",
           "    cudaFree(d_tetCellBase); d_tetCellBase = nullptr; cudaFree(d_kuhnTets); d_kuhnTets = nullptr; g_tetCells = 0;\n"
           "    cudaFree(d_activeNodes); d_activeNodes = nullptr; g_activeNodes = 0;")
text = rep(text, nl, "    s.kuhn=d_kuhnTets;s.cellBase=d_tetCellBase;s.cells=std::max(g_tetCells,1);",
           "    s.kuhn=d_kuhnTets;s.cellBase=d_tetCellBase;s.cells=std::max(g_tetCells,1);s.activeNodes=d_activeNodes;s.activeCount=g_activeNodes;")
# The nodes outside the list keep the zeros of the allocation, as every solve wrote them before.
text = rep(text, nl, "    if(e==cudaSuccess && !d_cgStencil)e=cudaMalloc(&d_cgStencil,g_N*sizeof(unsigned short));",
           "    // The PCG launches over the active nodes alone (UnifiedState::activeNodes): the other nodes keep the zero\n"
           "    // stencil row, residual, direction and step set here, which the kernels over all nodes read.\n"
           "    const bool clearCG=!d_cgStencil;\n"
           "    if(e==cudaSuccess && !d_cgStencil)e=cudaMalloc(&d_cgStencil,g_N*sizeof(unsigned short));")
text = rep(text, nl, "    if(e==cudaSuccess && !d_cgSearch)e=cudaMalloc(&d_cgSearch,g_N*sizeof(MaterialLineSearch));\n",
           "    if(e==cudaSuccess && !d_cgSearch)e=cudaMalloc(&d_cgSearch,g_N*sizeof(MaterialLineSearch));\n"
           "    if(e==cudaSuccess && clearCG){\n"
           "        e=cudaMemset(d_cgStencil,0,g_N*sizeof(unsigned short));\n"
           "        for(float3* v:{d_cgZ,d_cgR,d_cgP,d_cgQ,d_cgD,d_cgA})if(e==cudaSuccess)e=cudaMemset(v,0,g_N*sizeof(float3));\n"
           "        if(e==cudaSuccess)e=cudaMemset(d_cgH,0,g_N*sizeof(Symmetric3));\n"
           "    }\n")
text = rep(text, nl, "    if(e==cudaSuccess)e=cudaMalloc(&d_multigridFine,(size_t)g_N*sizeof(float3));",
           "    if(e==cudaSuccess)e=cudaMalloc(&d_multigridFine,(size_t)g_N*sizeof(float3));\n"
           "    if(e==cudaSuccess)e=cudaMemset(d_multigridFine,0,(size_t)g_N*sizeof(float3));")
launches = [
    ("k_MaterialStencilAssemble<<<Gr(g_N),64,0,g_captureStream>>>", "k_MaterialStencilAssemble<<<Gr(g_activeNodes),64,0,g_captureStream>>>"),
    ("k_MaterialStencilActiveMask<<<Gr(g_N),64,0,g_captureStream>>>", "k_MaterialStencilActiveMask<<<Gr(g_activeNodes),64,0,g_captureStream>>>"),
    ("k_MaterialCGBegin<<<Gr(g_N),64,0,g_captureStream>>>", "k_MaterialCGBegin<<<Gr(g_activeNodes),64,0,g_captureStream>>>"),
    ("k_MaterialStencilProduct<<<Gr(g_N),64,0,g_captureStream>>>", "k_MaterialStencilProduct<<<Gr(g_activeNodes),64,0,g_captureStream>>>"),
    ("k_MaterialCGAdvance<<<Gr(g_N),64,0,g_captureStream>>>", "k_MaterialCGAdvance<<<Gr(g_activeNodes),64,0,g_captureStream>>>"),
    ("k_MgFinePre<<<Gr(g_N),64,0,stream>>>", "k_MgFinePre<<<Gr(g_activeNodes),64,0,stream>>>"),
    ("k_MgFineRestrict<<<Gr(g_N),64,0,stream>>>", "k_MgFineRestrict<<<Gr(g_activeNodes),64,0,stream>>>"),
    ("k_MgFineProlong<<<Gr(g_N),64,0,stream>>>", "k_MgFineProlong<<<Gr(g_activeNodes),64,0,stream>>>"),
    ("k_MgFinePost<<<Gr(g_N),64,0,stream>>>", "k_MgFinePost<<<Gr(g_activeNodes),64,0,stream>>>"),
]
for old, new in launches:
    count = text.count(old)
    assert count >= 1, old
    text = text.replace(old, new)
    print(old.split("<<<")[0], count)
left = [k for k in ("k_MaterialStencilAssemble<<<Gr(g_N)", "k_MaterialStencilActiveMask<<<Gr(g_N)", "k_MaterialStencilProduct<<<Gr(g_N)",
                    "k_MaterialCGAdvance<<<Gr(g_N)", "k_MaterialCGBegin<<<Gr(g_N)") if k in text]
print("launches over all nodes left:", left)
save(path, text)
