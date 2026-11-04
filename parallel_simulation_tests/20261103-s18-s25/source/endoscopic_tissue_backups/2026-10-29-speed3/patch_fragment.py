"""Make the fragment-contact passes free when no fragment needs them, and give the volume repair colour-ordered data.

Usage: python patch_fragment.py  (from endoscopic_tissue/native/vendor/cuda)
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


patch("contact/fragment_contact.cuh", [
("""__global__ void k_ApplyFragmentContact(int n,float3* pos,const float3* delta,const int* degree){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n && degree[i])pos[i]=pos[i]+delta[i]*(.8f/degree[i]);
}""",
"""// Applies and clears the corrections of one pass, leaving delta and degree zero for the next one. Every node the
// passes move belongs to a detached component with a query (k_FragmentQueries): without one they are all zero.
__global__ void k_ApplyFragmentContact(int n,float3* pos,float3* delta,int* degree,const int* queryCount){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n || !*queryCount || !degree[i])return;
    pos[i]=pos[i]+delta[i]*(.8f/degree[i]);delta[i]=mkf3(0,0,0);degree[i]=0;
}"""),
("""__global__ void k_UnsupportedTools(int n,const int* support,const int* components,const int* grasp,
    const float* inverseMass,const float3* previous,float3* pos,const CoupledTool* grasper,const CoupledTool* blade,
    float thickness,float minGap){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n || inverseMass[i]<=0 || support[components[i]] || (grasp[components[i]]&1))return;""",
"""// Its nodes are nodes with a query (k_FragmentQueries), so none moves without one.
__global__ void k_UnsupportedTools(int n,const int* support,const int* components,const int* grasp,
    const float* inverseMass,const float3* previous,float3* pos,const CoupledTool* grasper,const CoupledTool* blade,
    float thickness,float minGap,const int* queryCount){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || !*queryCount || inverseMass[i]<=0 || support[components[i]] || (grasp[components[i]]&1))return;"""),
("""__global__ void k_FragmentEdges(int n,const int* neighbors,const float3* rest,const int* component,const int* support,
    const int* grasp,const float* inverseMass,const float3* pos,float young,float h,float* lambda,float3* delta,int* degree){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || inverseMass[i]<=0 || support[component[i]] || (grasp[component[i]]&1))return;""",
"""// Its nodes are nodes with a query (k_FragmentQueries), so none moves without one.
__global__ void k_FragmentEdges(int n,const int* neighbors,const float3* rest,const int* component,const int* support,
    const int* grasp,const float* inverseMass,const float3* pos,float young,float h,float* lambda,float3* delta,int* degree,
    const int* queryCount){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || !*queryCount || inverseMass[i]<=0 || support[component[i]] || (grasp[component[i]]&1))return;"""),
])
patch("xpbd/xpbd_solver.cu", [
("""        cudaMemsetAsync(d_surfaceDelta,0,g_N*sizeof(float3));cudaMemsetAsync(d_fragmentDegree,0,g_N*sizeof(int));
        if(!prepare)k_FragmentEdges<<<Gr(g_N),64>>>(g_N,g_liveNbrIdx,g_liveRestNbr,d_unifiedComponents,d_fragmentSupported,d_graspComponents,d_invMass,d_pos,g_p.youngE,g_hFrame,d_fragmentLambda,d_surfaceDelta,d_fragmentDegree);""",
"""        if(!prepare)k_FragmentEdges<<<Gr(g_N),64>>>(g_N,g_liveNbrIdx,g_liveRestNbr,d_unifiedComponents,d_fragmentSupported,d_graspComponents,d_invMass,d_pos,g_p.youngE,g_hFrame,d_fragmentLambda,d_surfaceDelta,d_fragmentDegree,d_fragmentQueryCount);"""),
("""        k_ApplyFragmentContact<<<Gr(g_N),64>>>(g_N,d_pos,d_surfaceDelta,d_fragmentDegree);
        k_UnsupportedTools<<<Gr(g_N),64>>>(g_N,d_fragmentSupported,d_unifiedComponents,d_graspComponents,d_invMass,previous,d_pos,d_coupledTool,g_blade.enabled?d_bladeBody:nullptr,g_jawThickness,g_gp.jawGapClosed);""",
"""        k_ApplyFragmentContact<<<Gr(g_N),64>>>(g_N,d_pos,d_surfaceDelta,d_fragmentDegree,d_fragmentQueryCount);
        k_UnsupportedTools<<<Gr(g_N),64>>>(g_N,d_fragmentSupported,d_unifiedComponents,d_graspComponents,d_invMass,previous,d_pos,d_coupledTool,g_blade.enabled?d_bladeBody:nullptr,g_jawThickness,g_gp.jawGapClosed,d_fragmentQueryCount);"""),
("""    const float3* previous=prepare?d_fragmentBefore:d_prevPos;
    beginFeasible();""",
"""    const float3* previous=prepare?d_fragmentBefore:d_prevPos;
    // The passes add into d_surfaceDelta and d_fragmentDegree, which k_ApplyFragmentContact leaves cleared.
    cudaMemsetAsync(d_surfaceDelta,0,g_N*sizeof(float3));cudaMemsetAsync(d_fragmentDegree,0,g_N*sizeof(int));
    beginFeasible();"""),
# colour-ordered repair data
("""// The tets of colour c, by index: entries g_tColorOff[c] to g_tColorOff[c]+g_tColorCnt[c]-1 (host copy and device).
static std::vector<int> g_tetColorOrder;
static int* d_tetColorOrder=nullptr;""",
"""// The tets of colour c, by index: entries g_tColorOff[c] to g_tColorOff[c]+g_tColorCnt[c]-1 (host copy and device).
static std::vector<int> g_tetColorOrder;
static int* d_tetColorOrder=nullptr;
// The node ids and rest determinant of the tet at each entry of d_tetColorOrder (k_ColorTetData).
static int4* d_colorTetIds=nullptr;
static float* d_colorTetDet=nullptr;"""),
("""__global__ void k_RepairMinimumVolume(int off,int count,const int* order,const int* ids,const int* active,
    const float3* b0,const float3* b1,const float3* b2,const float* invMass,float3* pos,float minimum,const int* unsafe){
    if(!*unsafe)return;
    int k=off+blockIdx.x*blockDim.x+threadIdx.x;if(k>=off+count)return;
    int t=order[k];if(!active[t])return;
    int id[4]={ids[4*t],ids[4*t+1],ids[4*t+2],ids[4*t+3]};
    float3 p[4]={pos[id[0]],pos[id[1]],pos[id[2]],pos[id[3]]};
    float3 u=p[1]-p[0],v=p[2]-p[0],w=p[3]-p[0];float detB=dot3(b0[t],cross3(b1[t],b2[t]));""",
"""// The colour-ordered tet data of k_RepairMinimumVolume, whose launch over one colour then reads them contiguously.
__global__ void k_ColorTetData(int count,const int* order,const int* ids,const float3* b0,const float3* b1,const float3* b2,
    int4* colorIds,float* colorDet){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;
    int t=order[k];colorIds[k]=make_int4(ids[4*t],ids[4*t+1],ids[4*t+2],ids[4*t+3]);
    colorDet[k]=dot3(b0[t],cross3(b1[t],b2[t]));
}
__global__ void k_RepairMinimumVolume(int off,int count,const int* order,const int4* colorIds,const float* colorDet,
    const int* active,const float* invMass,float3* pos,float minimum,const int* unsafe){
    if(!*unsafe)return;
    int k=off+blockIdx.x*blockDim.x+threadIdx.x;if(k>=off+count)return;
    int t=order[k];if(!active[t])return;
    int4 c=colorIds[k];int id[4]={c.x,c.y,c.z,c.w};
    float3 p[4]={pos[id[0]],pos[id[1]],pos[id[2]],pos[id[3]]};
    float3 u=p[1]-p[0],v=p[2]-p[0],w=p[3]-p[0];float detB=colorDet[k];"""),
("""k_RepairMinimumVolume<<<(g_tColorCnt[c]+63)/64,64,0,g_captureStream>>>(g_tColorOff[c],g_tColorCnt[c],d_tetColorOrder,d_tetIds,d_tetActive,d_tetBr0,d_tetBr1,d_tetBr2,d_invMass,d_pos,g_minTetVolume,d_needFeasible+g_N);""",
"""k_RepairMinimumVolume<<<(g_tColorCnt[c]+63)/64,64,0,g_captureStream>>>(g_tColorOff[c],g_tColorCnt[c],d_tetColorOrder,d_colorTetIds,d_colorTetDet,d_tetActive,d_invMass,d_pos,g_minTetVolume,d_needFeasible+g_N);"""),
("""        XK(cudaMalloc(&d_tetColorOrder,  (size_t)T   * sizeof(int)));""",
"""        XK(cudaMalloc(&d_tetColorOrder,  (size_t)T   * sizeof(int)));
        XK(cudaMalloc(&d_colorTetIds,    (size_t)T   * sizeof(int4)));
        XK(cudaMalloc(&d_colorTetDet,    (size_t)T   * sizeof(float)));"""),
("""        XK(cudaMemcpy(d_tetColorOrder, g_tetColorOrder.data(),(size_t)T*sizeof(int), cudaMemcpyHostToDevice));""",
"""        XK(cudaMemcpy(d_tetColorOrder, g_tetColorOrder.data(),(size_t)T*sizeof(int), cudaMemcpyHostToDevice));
        k_ColorTetData<<<Gr(T),64>>>(T,d_tetColorOrder,d_tetIds,d_tetBr0,d_tetBr1,d_tetBr2,d_colorTetIds,d_colorTetDet);
        XK(cudaGetLastError());"""),
("""    cudaFree(d_tetColorOrder); d_tetColorOrder = nullptr; g_tetColorOrder.clear();""",
"""    cudaFree(d_tetColorOrder); d_tetColorOrder = nullptr; g_tetColorOrder.clear();
    cudaFree(d_colorTetIds); d_colorTetIds = nullptr; cudaFree(d_colorTetDet); d_colorTetDet = nullptr;"""),
])
