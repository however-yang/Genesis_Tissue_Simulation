"""Skip the safe-coarse cluster passes when no tet sits at the volume floor: every cluster is then one node, and they
leave every coarse weight as it is."""
from pathlib import Path

root = Path(r"D:\Simulation\genesis-world\endoscopic_tissue\native\vendor\cuda")
solver = root / "xpbd" / "xpbd_solver.cu"
vertex = root / "solver" / "material" / "unified_vertex.cuh"


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:90]
    return text.replace(old, new)


s = solver.read_text(encoding="utf8")
s = replace_once(s, "static SafeCoarseCluster* d_safeAggregate=nullptr;\n",
                 "static SafeCoarseCluster* d_safeAggregate=nullptr;\nstatic int* d_safeJoined=nullptr;\n")
s = replace_once(s, "    if(e==cudaSuccess && !d_safeAggregate)e=cudaMalloc(&d_safeAggregate,g_N*sizeof(SafeCoarseCluster));\n",
                 "    if(e==cudaSuccess && !d_safeAggregate)e=cudaMalloc(&d_safeAggregate,g_N*sizeof(SafeCoarseCluster));\n"
                 "    if(e==cudaSuccess && !d_safeJoined)e=cudaMalloc(&d_safeJoined,sizeof(int));\n")
s = replace_once(s, "s.safeCluster=d_safeCluster;s.safeAggregate=d_safeAggregate;s.n=g_N;",
                 "s.safeCluster=d_safeCluster;s.safeAggregate=d_safeAggregate;s.safeJoined=d_safeJoined;s.n=g_N;")
s = replace_once(s, "cudaFree(d_safeAggregate);d_safeAggregate=nullptr;",
                 "cudaFree(d_safeAggregate);d_safeAggregate=nullptr;cudaFree(d_safeJoined);d_safeJoined=nullptr;")
solver.write_text(s, encoding="utf8")

v = vertex.read_text(encoding="utf8")
v = replace_once(v, "    int* safeCluster;SafeCoarseCluster* safeAggregate;\n",
                 "    int* safeCluster;SafeCoarseCluster* safeAggregate;int* safeJoined; // safeJoined: some tet joined a cluster (k_JoinSafeCoarse)\n")
v = replace_once(v, """    if(i>=s.n)return;s.safeCluster[i]=i;s.safeAggregate[i]={};
}""", """    if(i==0)*s.safeJoined=0;
    if(i>=s.n)return;s.safeCluster[i]=i;s.safeAggregate[i]={};
}""")
v = replace_once(v, """    if(J>s.minimum+fmaxf(1e-5f,1e-3f*s.minimum))return;
    for(int k=1;k<4;k++)while(true){""", """    if(J>s.minimum+fmaxf(1e-5f,1e-3f*s.minimum))return;
    *s.safeJoined=1;
    for(int k=1;k<4;k++)while(true){""")
# Without a join every cluster is one node: its finished weight is its own coarse weight, which k_CoarseWeights has
# already set to 0 on a pinned node, so the remaining passes change nothing.
v = replace_once(v, """__global__ void k_FlattenSafeCoarse(const UnifiedState* state){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<s.n)""", """// Without a join every cluster is a single node, whose finished weight is its own coarse weight: k_CoarseWeights has
// already set it to 0 on a pinned node, so flattening, accumulating and finishing change nothing then.
__global__ void k_FlattenSafeCoarse(const UnifiedState* state){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<s.n && *s.safeJoined)""")
v = replace_once(v, """__global__ void k_AccumulateSafeCoarse(const UnifiedState* state){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n)return;""", """__global__ void k_AccumulateSafeCoarse(const UnifiedState* state){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n || !*s.safeJoined)return;""")
v = replace_once(v, """__global__ void k_FinishSafeCoarse(const UnifiedState* state){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n)return;""", """__global__ void k_FinishSafeCoarse(const UnifiedState* state){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n || !*s.safeJoined)return;""")
vertex.write_text(v, encoding="utf8")
print("safe-coarse gate written")
