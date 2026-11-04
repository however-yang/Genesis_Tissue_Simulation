"""Fuse the per-domain kernels of every coarse solve: the node, tet and contact/grip terms into k_CoarseAccumulate, the
tet and contact step bounds into k_CoarseBound, and the node, tet and contact energies of each pressure-line pass into
k_PressureLine. The terms and their sums are unchanged; only the launches merge.

Usage: python patch_coarse_fusion.py  (from endoscopic_tissue/native/vendor/cuda)
"""
import re


def load(path):
    text = open(path, encoding="utf-8", newline="").read()
    return text, ("\r\n" if "\r\n" in text else "\n")


def save(path, text):
    open(path, "w", encoding="utf-8", newline="").write(text)


def rep(text, nl, old, new, count=1):
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    assert text.count(old) == count, (old[:90], text.count(old))
    return text.replace(old, new)


# ---- unified_vertex.cuh: node terms, contact/grip terms and contact bounds as device functions, and the fused kernels.
path = "solver/material/unified_vertex.cuh"
text, nl = load(path)
text = rep(text, nl, """__global__ void k_CoarseNodes(const UnifiedState* state,CoarseBlock* block){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;
    float3 g=mkf3(0,0,0);Symmetric3 H={};CoarseBlock* target=nullptr;
    if(i<s.n && (!s.active || s.active[i])){
        if(s.coarseMode)block+=s.component[i];""", """// The coarse terms of node i: its inertia, support and cut-rim edges in the mode, returned with the block of its
// component, or null when the node adds nothing.
__device__ inline CoarseBlock* coarseNodeTerms(const UnifiedState& s,int i,CoarseBlock* block,float3& g,Symmetric3& H){
    CoarseBlock* target=nullptr;
    if(i<s.n && (!s.active || s.active[i])){
        if(s.coarseMode)block+=s.component[i];""")
text = rep(text, nl, """        if(H.xx+H.yy+H.zz>0)target=block;
    }
    coarseAccumulateBlock(target,g,H);
}""", """        if(H.xx+H.yy+H.zz>0)target=block;
    }
    return target;
}""")
text = rep(text, nl, """__global__ void k_CoarseTets(const UnifiedState* state,CoarseBlock* block,bool bound){
    float3 g=mkf3(0,0,0);Symmetric3 H={};float scale=1;
    CoarseBlock* target=coarseTetTerms(*state,blockIdx.x*blockDim.x+threadIdx.x,block,bound,g,H,scale);
    // The bound adds one value per warp, which the warp-level coarseBound sums at little cost (blockAccumulate).
    if(!bound)coarseAccumulateBlock(target,g,H);else if(target)coarseBound(target,scale);
}
__global__ void k_CoarseContactGrip(const UnifiedState* state,CoarseBlock* block){
    const UnifiedState& s=*state;for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<max(*s.contactCount,s.grips);k+=blockDim.x*gridDim.x){""",
 """// The contact and grip terms of a coarse mode, the threads of the launch striding over them from `start`.
__device__ inline void coarseContactGripTerms(const UnifiedState& s,CoarseBlock* block,int start,int stride){
    for(int k=start;k<max(*s.contactCount,s.grips);k+=stride){""")
text = rep(text, nl, """            coarseAccumulate(s.coarseMode?block+s.component[a.id[0]]:block,(unifiedGripPoint(a,s.pos)-gripTarget(a,s.tool->pose,s.minGap))*(stiffness*w),H);}
    }
}""", """            coarseAccumulate(s.coarseMode?block+s.component[a.id[0]]:block,(unifiedGripPoint(a,s.pos)-gripTarget(a,s.tool->pose,s.minGap))*(stiffness*w),H);}
    }
}
// The node, tet, contact and grip terms of a coarse mode in one launch: the first n threads add the terms of a node,
// the next nt those of a tet, and all of them then stride over the contacts and grips.
__global__ void k_CoarseAccumulate(const UnifiedState* state,CoarseBlock* block){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    float3 g=mkf3(0,0,0);Symmetric3 H={};float scale=1;
    CoarseBlock* target=t<s.n?coarseNodeTerms(s,t,block,g,H):coarseTetTerms(s,t-s.n,block,false,g,H,scale);
    coarseAccumulateBlock(target,g,H);
    coarseContactGripTerms(s,block,t,gridDim.x*blockDim.x);
}""")
text = rep(text, nl, """__global__ void k_CoarseContactBound(const UnifiedState* state,CoarseBlock* blocks){
    const UnifiedState& s=*state;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*s.contactCount;k+=blockDim.x*gridDim.x){""",
 """// The step bounds the contacts set on a coarse mode, the threads of the launch striding over them from `start`.
__device__ inline void coarseContactBoundTerms(const UnifiedState& s,CoarseBlock* blocks,int start,int stride){
    for(int k=start;k<*s.contactCount;k+=stride){""")
text = rep(text, nl, """        coarseBound(b,fminf(1.f,.5f*feature/(-change)));
    }
}""", """        coarseBound(b,fminf(1.f,.5f*feature/(-change)));
    }
}
// The tet and contact step bounds of a coarse mode in one launch, each thread bounding by one tet and then striding over
// the contacts.
__global__ void k_CoarseBound(const UnifiedState* state,CoarseBlock* block){
    const UnifiedState& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    float3 g;Symmetric3 H;float scale=1;
    CoarseBlock* target=coarseTetTerms(s,t,block,true,g,H,scale);
    // The bound adds one value per warp, which the warp-level coarseBound sums at little cost (blockAccumulate).
    if(target)coarseBound(target,scale);
    coarseContactBoundTerms(s,block,t,gridDim.x*blockDim.x);
}""")
save(path, text)

# ---- pressure_patch_line.cuh: the three energy kernels as device functions and one fused kernel.
path = "solver/material/pressure_patch_line.cuh"
text, nl = load(path)
text = rep(text, nl, """__global__ void k_PressureLineNodes(const UnifiedState* state,CoarseBlock* blocks,int pass=-1){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;""",
 """__device__ void pressureLineNode(const UnifiedState& s,CoarseBlock* blocks,int i,int pass){""")
text = rep(text, nl, """__global__ void k_PressureLineTets(const UnifiedState* state,CoarseBlock* blocks,int pass=-1){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;""",
 """__device__ void pressureLineTet(const UnifiedState& s,CoarseBlock* blocks,int t,int pass){""")
text = rep(text, nl, """__global__ void k_PressureLineContacts(const UnifiedState* state,CoarseBlock* blocks,int pass=-1){
    const auto& s=*state;if(!pressureLineActive(s))return;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<max(*s.contactCount,s.grips);i+=blockDim.x*gridDim.x){""",
 """__device__ void pressureLineContacts(const UnifiedState& s,CoarseBlock* blocks,int pass,int start,int stride){
    if(!pressureLineActive(s))return;
    for(int i=start;i<max(*s.contactCount,s.grips);i+=stride){""")
text = rep(text, nl, """__global__ void k_PressureLineSelect(""", """// One pressure-line pass in one launch: the first n threads add the energy change of a node, the next nt that of a
// tet, and all of them then stride over the contacts and grips.
__global__ void k_PressureLine(const UnifiedState* state,CoarseBlock* blocks,int pass){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<s.n)pressureLineNode(s,blocks,t,pass);else pressureLineTet(s,blocks,t-s.n,pass);
    pressureLineContacts(s,blocks,pass,t,gridDim.x*blockDim.x);
}
__global__ void k_PressureLineSelect(""")
save(path, text)

# ---- xpbd_solver.cu: the launches.
path = "xpbd/xpbd_solver.cu"
text, nl = load(path)
S = r"(,0,(?:g_captureStream|stream))?"
accumulate = re.compile(r"k_CoarseNodes<<<GrAccumulate\(g_N\),ACCUMULATE_THREADS" + S + r">>>\(d_unified,(\w+)\);\s*"
                        r"k_CoarseTets<<<GrAccumulate\(g_T\),ACCUMULATE_THREADS(?:,0,(?:g_captureStream|stream))?>>>\(d_unified,\2,false\);\s*"
                        r"k_CoarseContactGrip<<<128,64(?:,0,(?:g_captureStream|stream))?>>>\(d_unified,\2\);")
text, a = accumulate.subn(lambda m: f"k_CoarseAccumulate<<<GrAccumulate(g_N+g_T),ACCUMULATE_THREADS{m.group(1) or ''}>>>(d_unified,{m.group(2)});", text)
bound = re.compile(r"k_CoarseTets<<<GrAccumulate\(g_T\),ACCUMULATE_THREADS" + S + r">>>\(d_unified,(\w+),true\);\s*"
                   r"k_CoarseContactBound<<<128,64(?:,0,(?:g_captureStream|stream))?>>>\(d_unified,\2\);")
text, b = bound.subn(lambda m: f"k_CoarseBound<<<GrAccumulate(g_T),ACCUMULATE_THREADS{m.group(1) or ''}>>>(d_unified,{m.group(2)});", text)
line = re.compile(r"k_PressureLineNodes<<<Gr\(g_N\),64" + S + r">>>\(d_unified,(\w+),pass\);\s*"
                  r"k_PressureLineTets<<<Gr\(g_T\),64(?:,0,(?:g_captureStream|stream))?>>>\(d_unified,\2,pass\);\s*"
                  r"k_PressureLineContacts<<<128,64(?:,0,(?:g_captureStream|stream))?>>>\(d_unified,\2,pass\);")
text, c = line.subn(lambda m: f"k_PressureLine<<<Gr(g_N+g_T),64{m.group(1) or ''}>>>(d_unified,{m.group(2)},pass);", text)
save(path, text)
left = [k for k in ("k_CoarseNodes<<<", "k_CoarseTets<<<", "k_CoarseContactGrip<<<", "k_CoarseContactBound<<<",
                    "k_PressureLineNodes<<<", "k_PressureLineTets<<<", "k_PressureLineContacts<<<") if k in text]
print("accumulate", a, "bound", b, "pressure line", c, "unfused launches left", left)
