"""List the predictor's vertices per color with k_SweepSites instead of launching 32 threads per lattice site.

Usage: python patch_predictor_list.py  (from endoscopic_tissue/native/vendor/cuda)
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


patch("solver/material/unified_vertex.cuh", [
("""// Lists, per lattice color, the vertices one local sweep solves (k_UnifiedVertex): every movable vertex while the
// coupled material solve is off, otherwise the contact and anchor witnesses it holds fixed, and in a surface repair
// (localOnly) also the vertices the regional mode moves. Each color then launches warps for its listed vertices
// alone. It also clears the interactions of every witness: the assembly of a color writes only the vertices of that
// color, so clearing them all before the sweep equals clearing each color before its assembly. Color c lists its
// vertices from sites+c*capacity on and counts them in count[c], zeroed before.
__global__ void k_SweepSites(const UnifiedState* state,int localOnly,int capacity,int* sites,int* count){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    bool solved=false;int color=-1;
    if(i<s.n){
        bool witness=(s.degree[i]>0 || (s.secondary && s.secondary->degree[i]>0)) || (s.grips>0 && s.gripDegree[i]>0);
        if(witness)for(int j=0;j<9;j++)s.interaction[9*i+j]=0;
        solved=s.invMass[i]>0 && !(localOnly!=1 && s.coupledReady && !materialCgFixed(s,i)) && !(localOnly && s.coarseWeight[i]<=0 && !witness);
        if(solved)color=unifiedColor(s,i);
    }""",
"""// The vertex sets k_SweepSites lists: those of a local sweep, of a local sweep in a surface repair, and of the
// predictor.
enum SweepList{SweepLocal,SweepRepair,SweepPredictor};
// Lists, per lattice color, the vertices that a colored pass moves. A local sweep (k_UnifiedVertex) solves every
// movable vertex while the coupled material solve is off, otherwise the contact and anchor witnesses that solve holds
// fixed, and in a surface repair also the vertices the regional mode moves. The predictor (k_UnifiedPredictor) moves
// the movable vertices other than witnesses. Each color then launches warps for its listed vertices alone. For a sweep
// it also clears the interactions of every witness: the assembly of a color writes only the vertices of that color,
// so clearing them all before the sweep equals clearing each color before its assembly. Color c lists its vertices
// from sites+c*capacity on and counts them in count[c], zeroed before.
__global__ void k_SweepSites(const UnifiedState* state,SweepList list,int capacity,int* sites,int* count){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    bool solved=false;int color=-1;
    if(i<s.n){
        bool witness=(s.degree[i]>0 || (s.secondary && s.secondary->degree[i]>0)) || (s.grips>0 && s.gripDegree[i]>0);
        if(witness && list!=SweepPredictor)for(int j=0;j<9;j++)s.interaction[9*i+j]=0;
        solved=s.invMass[i]>0 && (list==SweepPredictor?!witness:
            list==SweepRepair?(!(s.coarseWeight[i]<=0) || witness):!(s.coupledReady && !materialCgFixed(s,i)));
        if(solved)color=unifiedColor(s,i);
    }"""),
("""// One warp per vertex, lanes over its incident tets (see k_UnifiedVertex).
__global__ void k_UnifiedPredictor(const UnifiedState* state,int color){
    const UnifiedState& s=*state;
    int thread=blockIdx.x*blockDim.x+threadIdx.x,lane=thread%32,site=thread/32,cx=(s.nx+1)/2,cy=(s.ny+1)/2,nz=s.n/(s.nx*s.ny);
    int x=2*(site%cx)+(color&1),y=2*((site/cx)%cy)+((color>>1)&1),z=2*(site/(cx*cy))+((color>>2)&1);
    if(x>=s.nx || y>=s.ny || z>=nz)return;
    int i=x+s.nx*(y+s.ny*z);if(s.invMass[i]<=0)return;
    // Keep only the actual material contact/anchor witnesses at their accepted
    // configuration. Unconstrained tissue retains its inertial warm start.
    if((s.degree[i]>0 || (s.secondary && s.secondary->degree[i]>0)) || (s.grips>0 && s.gripDegree[i]>0))return;
    float3 p=s.pos[i],step=s.target[i]-p;float scale=1;
    for(int k=s.offsets[i]+lane;k<s.offsets[i+1];k+=32){
        int t=s.incident[k];if(!s.live[t])continue;const int* ids=s.ids+4*t;
        float3 a=s.pos[ids[0]],u=s.pos[ids[1]]-a,v=s.pos[ids[2]]-a,w=s.pos[ids[3]]-a;
        float det=s.restMetric[7*t+6],J=dot3(u,cross3(v,w))*det;
        float3 g1=cross3(v,w),g2=cross3(w,u),g3=cross3(u,v);
        float3 g=i==ids[0]?(g1+g2+g3)*(-1.f):i==ids[1]?g1:i==ids[2]?g2:g3;
        float dj=dot3(g,step)*det;
        if(dj<0)scale=fminf(scale,fmaxf(0.f,.9f*(J-s.minimum)/(-dj)));
    }
    for(int offset=16;offset;offset/=2)scale=fminf(scale,__shfl_xor_sync(0xffffffff,scale,offset));
    if(lane==0 && isfinite(step.x+step.y+step.z))s.pos[i]=p+step*scale;
}""",
"""// One warp per vertex, lanes over its incident tets (see k_UnifiedVertex), for the vertices of one color listed by
// k_SweepSites. The contact and anchor witnesses keep their accepted configuration; unconstrained tissue takes its
// inertial warm start.
__global__ void k_UnifiedPredictor(const UnifiedState* state,const int* sites,const int* count){
    const UnifiedState& s=*state;int lane=threadIdx.x%32,warps=gridDim.x*blockDim.x/32;
    for(int site=(blockIdx.x*blockDim.x+threadIdx.x)/32;site<*count;site+=warps){
        int i=sites[site];float3 p=s.pos[i],step=s.target[i]-p;float scale=1;
        for(int k=s.offsets[i]+lane;k<s.offsets[i+1];k+=32){
            int t=s.incident[k];if(!s.live[t])continue;const int* ids=s.ids+4*t;
            float3 a=s.pos[ids[0]],u=s.pos[ids[1]]-a,v=s.pos[ids[2]]-a,w=s.pos[ids[3]]-a;
            float det=s.restMetric[7*t+6],J=dot3(u,cross3(v,w))*det;
            float3 g1=cross3(v,w),g2=cross3(w,u),g3=cross3(u,v);
            float3 g=i==ids[0]?(g1+g2+g3)*(-1.f):i==ids[1]?g1:i==ids[2]?g2:g3;
            float dj=dot3(g,step)*det;
            if(dj<0)scale=fminf(scale,fmaxf(0.f,.9f*(J-s.minimum)/(-dj)));
        }
        for(int offset=16;offset;offset/=2)scale=fminf(scale,__shfl_xor_sync(0xffffffff,scale,offset));
        if(lane==0 && isfinite(step.x+step.y+step.z))s.pos[i]=p+step*scale;
    }
}"""),
("""// Warps of one local-sweep launch (k_UnifiedVertex), which stride over the listed vertices of a color.
constexpr int SWEEP_BLOCKS=1024;
""", ""),
("""// Lattice color of a vertex in the eight-color sweeps (parities of x, y, z).""",
"""// Blocks of 64 threads of one colored launch over listed vertices (k_UnifiedVertex, k_UnifiedPredictor), whose warps
// stride over the vertices of a color.
constexpr int SWEEP_BLOCKS=1024;
// Lattice color of a vertex in the eight-color sweeps (parities of x, y, z)."""),
])
patch("xpbd/xpbd_solver.cu", [
("""    for(int color=0;color<8;color++)k_UnifiedPredictor<<<Gr(32*predictorSites),64>>>(d_unified,color);""",
"""    cudaMemsetAsync(d_sweepCount,0,8*sizeof(int));
    k_SweepSites<<<Gr(g_N),64>>>(d_unified,SweepPredictor,predictorSites,d_sweepSites,d_sweepCount);
    for(int color=0;color<8;color++)k_UnifiedPredictor<<<SWEEP_BLOCKS,64>>>(d_unified,d_sweepSites+color*predictorSites,d_sweepCount+color);"""),
("""            k_SweepSites<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,1,sites,d_sweepSites,d_sweepCount);""",
"""            k_SweepSites<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,SweepRepair,sites,d_sweepSites,d_sweepCount);"""),
("""            k_SweepSites<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,0,sites,d_sweepSites,d_sweepCount);""",
"""            k_SweepSites<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,SweepLocal,sites,d_sweepSites,d_sweepCount);"""),
])
