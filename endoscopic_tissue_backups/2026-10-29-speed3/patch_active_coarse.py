"""Launch the per-node kernels of the coarse, regional, tool and rigid modes over the active nodes only.

Every one of them returned at once for an inactive node, or wrote it a zero weight. The coarse weight of an inactive
node now stays at the zero of its allocation: the restriction clear (which reuses the weights) writes the active nodes
alone too, and every other writer over all nodes writes zeros. A kernel whose thread 0 also updates the tool block or
the mode keeps that duty on launch thread 0.

Usage: python patch_active_coarse.py  (from endoscopic_tissue/native/vendor/cuda)
"""
import re


def load(path):
    text = open(path, encoding="utf-8", newline="").read()
    return text, ("\r\n" if "\r\n" in text else "\n")


def save(path, text):
    open(path, "w", encoding="utf-8", newline="").write(text)


def rep(text, nl, old, new, count=1):
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    assert text.count(old) == count, (old[:100], text.count(old))
    return text.replace(old, new)


THREAD = "blockIdx.x*blockDim.x+threadIdx.x"

path = "solver/material/unified_vertex.cuh"
text, nl = load(path)
text = rep(text, nl, "__device__ inline bool coarseRoot(const UnifiedState& s,int i){return i<s.n && ",
           "__device__ inline bool coarseRoot(const UnifiedState& s,int i){return i>=0 && i<s.n && ")
text = rep(text, nl, f"""__global__ void k_CoarseWeights(UnifiedState* state,int mode=0){{
    const UnifiedState& s=*state;int i={THREAD};if(i>=s.n)return;
    if(i==0)state->coarseMode=mode;
""", f"""__global__ void k_CoarseWeights(UnifiedState* state,int mode=0){{
    const UnifiedState& s=*state;int k={THREAD},i=unifiedActiveNode(s,k);
    if(k==0)state->coarseMode=mode;
    if(i<0)return;
""")
text = rep(text, nl, f"""    const UnifiedState& s=*state;int i={THREAD};
    if(i==0)*s.safeJoined=0;
    if(i>=s.n)return;s.safeCluster[i]=i;s.safeAggregate[i]={{}};""",
           f"""    const UnifiedState& s=*state;int k={THREAD},i=unifiedActiveNode(s,k);
    if(k==0)*s.safeJoined=0;
    if(i<0)return;s.safeCluster[i]=i;s.safeAggregate[i]={{}};""")
text = rep(text, nl, f"""    const UnifiedState& s=*state;int i={THREAD};
    if(i<s.n && *s.safeJoined)s.safeCluster[i]=unifiedRoot(s.safeCluster,i);""",
           f"""    const UnifiedState& s=*state;int i=unifiedActiveNode(s,{THREAD});
    if(i>=0 && *s.safeJoined)s.safeCluster[i]=unifiedRoot(s.safeCluster,i);""")
text = rep(text, nl, f"""    const UnifiedState& s=*state;int i={THREAD};
    if(i>=s.n || !*s.safeJoined)return;""", f"""    const UnifiedState& s=*state;int i=unifiedActiveNode(s,{THREAD});
    if(i<0 || !*s.safeJoined)return;""", count=2)
text = rep(text, nl, f"""    const UnifiedState& s=*state;int i={THREAD};
    if(s.coarseMode){{if(coarseRoot(s,i)){{b[i]={{}};b[i].scale=1;}}return;}}
    if(i)return;""", f"""    const UnifiedState& s=*state;int k={THREAD};
    if(s.coarseMode){{int i=unifiedActiveNode(s,k);if(coarseRoot(s,i)){{b[i]={{}};b[i].scale=1;}}return;}}
    if(k)return;""")
text = rep(text, nl, """// The node, tet, contact and grip terms of a coarse mode in one launch: the first n threads add the terms of a node,
// the next nt those of a tet, and all of them then stride over the contacts and grips.""",
           """// The node, tet, contact and grip terms of a coarse mode in one launch: the first activeCount threads add the terms of
// an active node, the next nt those of a tet, and all of them then stride over the contacts and grips.""")
text = rep(text, nl, "    CoarseBlock* target=t<s.n?coarseNodeTerms(s,t,block,g,H):coarseTetTerms(s,t-s.n,block,false,g,H,scale);",
           "    CoarseBlock* target=t<s.activeCount?coarseNodeTerms(s,s.activeNodes[t],block,g,H):\n"
           "        coarseTetTerms(s,t-s.activeCount,block,false,g,H,scale);")
text = rep(text, nl, f"""    int i={THREAD};if(state->coarseMode){{if(!coarseRoot(*state,i))return;block+=i;}}else if(i)return;""",
           f"""    int k={THREAD};
    if(state->coarseMode){{int i=unifiedActiveNode(*state,k);if(!coarseRoot(*state,i))return;block+=i;}}else if(k)return;""")
text = rep(text, nl, f"""    const UnifiedState& s=*state;int i={THREAD};if(i>=s.n)return;
    if(s.coarseMode)block+=s.component[i];
    if(s.coarseMode==6 && block->external)return;
    float3 d=block->step*block->scale;
    if(!isfinite(d.x+d.y+d.z))return;
    // An inactive node is its own component root without a cleared block (coarseRoot). Its block accumulates nothing,
    // so a per-component mode never moves it.
    if(!s.coarseMode || !s.active || s.active[i])s.pos[i]=s.pos[i]+d*s.coarseWeight[i];
    if(i==0 && !s.coarseMode)s.tool->pose=shiftTool(s.tool->pose,d);""",
           f"""    const UnifiedState& s=*state;int k={THREAD},i=unifiedActiveNode(s,k);
    if(s.coarseMode){{if(i<0)return;block+=s.component[i];if(s.coarseMode==6 && block->external)return;}}
    else if(i<0 && k)return;
    float3 d=block->step*block->scale;
    if(!isfinite(d.x+d.y+d.z))return;
    if(i>=0)s.pos[i]=s.pos[i]+d*s.coarseWeight[i];
    if(k==0 && !s.coarseMode)s.tool->pose=shiftTool(s.tool->pose,d);""")
text = rep(text, nl, f"""    const UnifiedState& s=*state;int i={THREAD};
    if(i>=s.n || (s.active&&!s.active[i]))return;CoarseBlock* b=blocks+s.component[i];""",
           f"""    const UnifiedState& s=*state;int i=unifiedActiveNode(s,{THREAD});
    if(i<0)return;CoarseBlock* b=blocks+s.component[i];""")
text = rep(text, nl, f"""    int i={THREAD};if(!coarseRoot(*state,i))return;
    if(state->coupledReady && !(state->graspComponents[i]&7)){{b[i].external=1;return;}}""",
           f"""    int i=unifiedActiveNode(*state,{THREAD});if(!coarseRoot(*state,i))return;
    if(state->coupledReady && !(state->graspComponents[i]&7)){{b[i].external=1;return;}}""")
text = rep(text, nl, f"""    const UnifiedState& s=*state;int i={THREAD};
    if(i>=s.n || s.invMass[i]<=0 || (s.active&&!s.active[i]))return;""",
           f"""    const UnifiedState& s=*state;int i=unifiedActiveNode(s,{THREAD});
    if(i<0 || s.invMass[i]<=0)return;""")
text = rep(text, nl, f"""    const UnifiedState& s=*state;int i={THREAD};
    if(i>=s.n||s.invMass[i]<=0||(s.active&&!s.active[i]))return;""",
           f"""    const UnifiedState& s=*state;int i=unifiedActiveNode(s,{THREAD});
    if(i<0||s.invMass[i]<=0)return;""", count=2)
save(path, text)

path = "solver/material/pressure_patch_line.cuh"
text, nl = load(path)
text = rep(text, nl, f"""    const auto& s=*state;int i={THREAD};
    if(coarseRoot(s,i) && pressureLineActive(s))blocks[i].gradient=mkf3(0,0,0);""",
           f"""    const auto& s=*state;int i=unifiedActiveNode(s,{THREAD});
    if(coarseRoot(s,i) && pressureLineActive(s))blocks[i].gradient=mkf3(0,0,0);""")
text = rep(text, nl, """// One pressure-line pass in one launch: the first n threads add the energy change of a node, the next nt that of a
// tet, and all of them then stride over the contacts and grips.""",
           """// One pressure-line pass in one launch: the first activeCount threads add the energy change of an active node, the
// next nt that of a tet, and all of them then stride over the contacts and grips.""")
text = rep(text, nl, "    if(t<s.n)pressureLineNode(s,blocks,t,pass);else pressureLineTet(s,blocks,t-s.n,pass);",
           "    if(t<s.activeCount)pressureLineNode(s,blocks,s.activeNodes[t],pass);else pressureLineTet(s,blocks,t-s.activeCount,pass);")
text = rep(text, nl, f"""    const auto& s=*state;int i={THREAD};if(!coarseRoot(s,i) || !pressureLineActive(s))return;""",
           f"""    const auto& s=*state;int i=unifiedActiveNode(s,{THREAD});if(!coarseRoot(s,i) || !pressureLineActive(s))return;""")
save(path, text)

path = "solver/material/material_line_search.cuh"
text, nl = load(path)
text = rep(text, nl, f"""__global__ void k_MaterialRestrictionClear(int n,float* scale){{
    int i={THREAD};if(i<n)scale[i]=1.f;
}}""", f"""// The restriction reuses the coarse weights, so it clears the active nodes alone: an inactive node keeps weight 0.
__global__ void k_MaterialRestrictionClear(const UnifiedState* state,float* scale){{
    int i=unifiedActiveNode(*state,{THREAD});if(i>=0)scale[i]=1.f;
}}""")
save(path, text)

path = "xpbd/xpbd_solver.cu"
text, nl = load(path)
text = rep(text, nl, "    if(e==cudaSuccess && !d_coarseWeight)e=cudaMalloc(&d_coarseWeight,(size_t)g_N*sizeof(float));",
           "    // The coarse kernels visit the active nodes alone (UnifiedState::activeNodes): every other node keeps weight 0.\n"
           "    if(e==cudaSuccess && !d_coarseWeight){\n"
           "        e=cudaMalloc(&d_coarseWeight,(size_t)g_N*sizeof(float));\n"
           "        if(e==cudaSuccess)e=cudaMemset(d_coarseWeight,0,(size_t)g_N*sizeof(float));\n"
           "    }")
text, count = re.subn(r"k_MaterialRestrictionClear<<<Gr\(g_N\),64,0,g_captureStream>>>\(g_N,d_coarseWeight\);",
                      "k_MaterialRestrictionClear<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_coarseWeight);", text)
assert count == 2, count
names = ["k_CoarseWeights", "k_InitSafeCoarse", "k_FlattenSafeCoarse", "k_AccumulateSafeCoarse", "k_FinishSafeCoarse",
         "k_CoarseBegin", "k_CoarseSolve", "k_PressureLineClear", "k_PressureLineSelect", "k_CoarseApply",
         "k_RigidCenters", "k_RigidCenterFinish", "k_RigidNodes", "k_RigidEnergy", "k_RigidApply"]
for name in names:
    text, count = re.subn(name + r"<<<Gr\(g_N\),", name + "<<<Gr(g_activeNodes),", text)
    print(name, count)
text, count = re.subn(r"k_CoarseAccumulate<<<GrAccumulate\(g_N\+g_T\),", "k_CoarseAccumulate<<<GrAccumulate(g_activeNodes+g_T),", text)
print("k_CoarseAccumulate", count)
text, count = re.subn(r"k_PressureLine<<<Gr\(g_N\+g_T\),", "k_PressureLine<<<Gr(g_activeNodes+g_T),", text)
print("k_PressureLine", count)
left = re.findall(r"(?:" + "|".join(names + ["k_CoarseAccumulate", "k_PressureLine"]) + r")<<<Gr\w*\(g_N[^)]*\)", text)
print("launches over all nodes left:", left)
save(path, text)
