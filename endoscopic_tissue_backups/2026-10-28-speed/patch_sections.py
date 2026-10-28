"""Add opt-in section timing to the native solver: marks between kernel groups, read back per section name."""
from pathlib import Path

root = Path(r"D:\Simulation\genesis-world\endoscopic_tissue\native\vendor")
solver = root / "cuda" / "xpbd" / "xpbd_solver.cu"
s = solver.read_text(encoding="utf8")


def insert_after(text, anchor, addition):
    # Anchors start at a line start: a deeper-indented copy elsewhere does not match.
    key = "\n" + anchor
    assert text.count(key) == 1, anchor
    return text.replace(key, key + addition)


def insert_before(text, anchor, addition):
    key = "\n" + anchor
    assert text.count(key) == 1, anchor
    return text.replace(key, "\n" + addition + anchor)


s = insert_after(s, "#include <cmath>\n", "#include <algorithm>\n#include <cstdio>\n#include <string>\n#include <utility>\n")
s = insert_after(s, """    for(int p=0;p<3;p++){ms[p]=0;for(int s=0;s<steps;s++){float t=0;cudaEventElapsedTime(&t,g_profileEvents[4*s+p],g_profileEvents[4*s+p+1]);ms[p]+=t;}}
    return 0;
}
""", """// Opt-in section timing for performance work. While enabled, the unified substep issues its captured kernel
// sequence directly on the default stream, and the time between consecutive marks accumulates under the name of the
// later mark. Disabled, a mark costs one flag test and every captured graph stays as it is.
static bool g_sectionProfile=false;
static std::vector<std::pair<const char*,cudaEvent_t>> g_sectionMarks;
static std::vector<cudaEvent_t> g_sectionPool;
static std::vector<std::pair<std::string,double>> g_sectionTotals;
extern "C" void xpbd_section_mark(const char* name){
    if(!g_sectionProfile)return;
    if(g_sectionPool.size()<=g_sectionMarks.size()){cudaEvent_t e;cudaEventCreate(&e);g_sectionPool.push_back(e);}
    cudaEvent_t e=g_sectionPool[g_sectionMarks.size()];cudaEventRecord(e,0);g_sectionMarks.push_back({name,e});
}
extern "C" void xpbd_section_flush(){
    if(!g_sectionProfile || g_sectionMarks.empty())return;
    cudaEventSynchronize(g_sectionMarks.back().second);
    for(size_t k=1;k<g_sectionMarks.size();k++){
        float t=0;cudaEventElapsedTime(&t,g_sectionMarks[k-1].second,g_sectionMarks[k].second);
        const char* name=g_sectionMarks[k].first;
        auto found=std::find_if(g_sectionTotals.begin(),g_sectionTotals.end(),[&](const std::pair<std::string,double>& entry){return entry.first==name;});
        if(found==g_sectionTotals.end())g_sectionTotals.push_back({name,t});else found->second+=t;
    }
    g_sectionMarks.clear();
}
extern "C" void xpbd_profile_sections(int enabled){g_sectionProfile=enabled!=0;g_sectionMarks.clear();g_sectionTotals.clear();}
// Writes "name=ms" lines into text and returns 0, or returns the capacity the text needs.
extern "C" int xpbd_profile_sections_read(char* text,int capacity){
    std::string out;
    for(const auto& entry:g_sectionTotals){char line[200];snprintf(line,sizeof line,"%s=%.4f\\n",entry.first.c_str(),entry.second);out+=line;}
    if(!text || capacity<=(int)out.size())return (int)out.size()+1;
    memcpy(text,out.c_str(),out.size()+1);return 0;
}
""")
s = insert_before(s, "    if(d_coupledTool && g_toolArmed && d_incidentOffsets && g_frictionContact && d_surfaceActiveCount) return unifiedSubstep(h,previousTool,stepTool,componentsFresh);",
                  "    xpbd_section_mark(\"substep prologue: liveness, fragment contact, predict\");\n")
s = insert_after(s, "    e=prepareUnifiedManifold();if(e!=cudaSuccess)return -(int)e;\n",
                 "    xpbd_section_mark(\"pre: tool, contact discovery, manifold\");\n")
s = s.replace("""    cudaGraphExec_t& selectedGraph=recoveryRequested?g_unifiedRecoveryGraph:g_unifiedGraph;
    bool& selectedGraphDirty=recoveryRequested?g_unifiedRecoveryGraphDirty:g_unifiedGraphDirty;
    if(!selectedGraph || selectedGraphDirty){
        if(!g_captureStream)e=cudaStreamCreate(&g_captureStream);if(e!=cudaSuccess)return -(int)e;
        e=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(e!=cudaSuccess)return -(int)e;
""", """    cudaGraphExec_t& selectedGraph=recoveryRequested?g_unifiedRecoveryGraph:g_unifiedGraph;
    bool& selectedGraphDirty=recoveryRequested?g_unifiedRecoveryGraphDirty:g_unifiedGraphDirty;
    xpbd_section_mark("pre: predictor, recovery query");
    // Section timing issues the sequence directly, kernel after kernel, instead of capturing it.
    const bool direct=g_sectionProfile;
    cudaStream_t captureStream=g_captureStream;
    if(direct || !selectedGraph || selectedGraphDirty){
        if(direct)g_captureStream=0;
        else{
            if(!g_captureStream)e=cudaStreamCreate(&g_captureStream);if(e!=cudaSuccess)return -(int)e;
            e=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(e!=cudaSuccess)return -(int)e;
        }
""")
s = insert_after(s, "            if(it==0)k_MarkVolumeRecovery<<<Gr(g_T),64,0,g_captureStream>>>(d_unified);\n",
                 "            xpbd_section_mark(\"loop: manifold, contact marks\");\n")
s = insert_before(s, "            // Solve unconstrained material DOFs with contact/anchor witnesses\n",
                  "            xpbd_section_mark(\"loop: coarse modes\");\n")
s = insert_after(s, "                k_MaterialCGBegin<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgR,d_cgH,d_cgP,d_cgD,d_cgS);\n",
                 "                xpbd_section_mark(\"cg: assembly\");\n")
s = insert_after(s, """                    k_MaterialCGFinish<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgS,cg);
                }
""", "                xpbd_section_mark(\"cg: iterations\");\n")
s = insert_before(s, "                for(int feasible=0;feasible<8;feasible++){\n", "                xpbd_section_mark(\"cg: trust, projection\");\n")
s = insert_before(s, "                k_MaterialLineClear<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_cgSearch);\n", "                xpbd_section_mark(\"cg: feasible\");\n")
s = insert_after(s, "                k_MaterialReady<<<1,1,0,g_captureStream>>>(d_unified,true);\n", "                xpbd_section_mark(\"cg: line search\");\n")
s = insert_before(s, "            // A tissue-only regional mode lets pressure move a finite patch\n",
                  "            xpbd_section_mark(\"loop: contact discovery\");\n")
s = insert_before(s, "            k_CoarseWeights<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,0);\n",
                  "            xpbd_section_mark(\"loop: regional mode\");\n")
s = insert_before(s, "            k_UnifiedTool<<<1,32,0,g_captureStream>>>(d_unified);\n"
                     "            if(g_blade.enabled)k_UnifiedTool<<<1,32,0,g_captureStream>>>(d_bladeState);\n",
                  "            xpbd_section_mark(\"loop: local sweeps\");\n")
s = insert_before(s, "            { // Retain rotational inertia in every material/contact sweep.\n", "            xpbd_section_mark(\"loop: tool, safe and global coarse\");\n")
s = insert_after(s, """                k_RigidApply<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
            }
""", "            xpbd_section_mark(\"loop: rigid modes\");\n")
old_end = """        cudaGraph_t graph=nullptr;e=cudaStreamEndCapture(g_captureStream,&graph);if(e!=cudaSuccess)return -(int)e;
        if(selectedGraph){
            cudaGraphExecUpdateResultInfo info={};e=cudaGraphExecUpdate(selectedGraph,graph,&info);
            if(e==cudaErrorGraphExecUpdateFailure || (e==cudaSuccess && info.result!=cudaGraphExecUpdateSuccess)){
                cudaGetLastError();cudaGraphExecDestroy(selectedGraph);selectedGraph=nullptr;e=cudaSuccess;
            }
        }
        if(e==cudaSuccess && !selectedGraph)e=cudaGraphInstantiate(&selectedGraph,graph,0);
        cudaGraphDestroy(graph);if(e!=cudaSuccess)return -(int)e;selectedGraphDirty=false;
    }
    e=cudaGraphLaunch(selectedGraph,0);if(e!=cudaSuccess)return -(int)e;
"""
new_end = """        if(direct)g_captureStream=captureStream;
        else{
            cudaGraph_t graph=nullptr;e=cudaStreamEndCapture(g_captureStream,&graph);if(e!=cudaSuccess)return -(int)e;
            if(selectedGraph){
                cudaGraphExecUpdateResultInfo info={};e=cudaGraphExecUpdate(selectedGraph,graph,&info);
                if(e==cudaErrorGraphExecUpdateFailure || (e==cudaSuccess && info.result!=cudaGraphExecUpdateSuccess)){
                    cudaGetLastError();cudaGraphExecDestroy(selectedGraph);selectedGraph=nullptr;e=cudaSuccess;
                }
            }
            if(e==cudaSuccess && !selectedGraph)e=cudaGraphInstantiate(&selectedGraph,graph,0);
            cudaGraphDestroy(graph);if(e!=cudaSuccess)return -(int)e;selectedGraphDirty=false;
        }
    }
    if(!direct){e=cudaGraphLaunch(selectedGraph,0);if(e!=cudaSuccess)return -(int)e;}
"""
assert s.count(old_end) == 1
s = s.replace(old_end, new_end)
s = insert_before(s, "    e=solveFragmentContact();if(e!=cudaSuccess)return -(int)e;\n    profileMark(g_substepIndex-1,1);profileMark(g_substepIndex-1,2);\n",
                  "    xpbd_section_mark(\"post: free-component nullspace\");\n")
s = insert_after(s, "    e=solveFragmentContact();if(e!=cudaSuccess)return -(int)e;\n    profileMark(g_substepIndex-1,1);profileMark(g_substepIndex-1,2);\n",
                 "    xpbd_section_mark(\"post: fragment contact\");\n")
s = insert_before(s, "    profileMark(g_substepIndex-1,3);return -(int)cudaGetLastError();\n}\n\nextern \"C\" int xpbd_substep(void)",
                  "    xpbd_section_mark(\"post: blade and contact reactions, velocity\");\n")
solver.write_text(s, encoding="utf8")

physics = root / "cuda" / "physics.cu"
p = physics.read_text(encoding="utf8")
p = insert_after(p, "#include \"xpbd/xpbd_solver.h\"  // XPBD GO-LIVE (design rev-B 9-(4)): xpbd_bind + begin_frame/substep\n",
                 "extern \"C\" void xpbd_section_mark(const char* name);\nextern \"C\" void xpbd_section_flush();\n")
p = insert_after(p, """    if (!g_phys_ready) return -1;
    if (!recon_corner_pos()) return -2;

    // ── XPBD small-steps body""", "")
p = insert_before(p, "    int rc = xpbd_begin_frame(dt);\n    if (rc != 0) return rc;\n    int N = (g_pdesc.maxSubsteps > 0) ? g_pdesc.maxSubsteps : 20;",
                  "    xpbd_section_mark(\"step start\");\n")
p = insert_before(p, "        rc = xpbd_substep();\n        if (rc != 0) return rc;\n",
                  "        xpbd_section_mark(\"begin frame, cut tick and rebuild between substeps\");\n")
p = insert_after(p, """    cudaError_t le = cudaGetLastError();
    if (le == cudaSuccess) le = cudaDeviceSynchronize();   // commit the frame (replaces the errMax sync)
""", "    xpbd_section_mark(\"begin frame, cut tick and rebuild between substeps\");\n    xpbd_section_flush();\n")
physics.write_text(p, encoding="utf8")

api = root / "src" / "plugin_api.cpp"
a = api.read_text(encoding="utf8")
a = insert_after(a, "LCS_API int LCS_ProfileRead(int steps,float* ms){return xpbd_profile_read(steps,ms);}\n",
                 "extern \"C\" void xpbd_profile_sections(int);\nextern \"C\" int xpbd_profile_sections_read(char*,int);\n"
                 "LCS_API void LCS_ProfileSections(int enabled){xpbd_profile_sections(enabled);}\n"
                 "LCS_API int LCS_ProfileSectionsRead(char* text,int capacity){return xpbd_profile_sections_read(text,capacity);}\n")
api.write_text(a, encoding="utf8")
print("patched")
