#ifdef _WIN32
#define LCS_COMBINED_EXPORT __declspec(dllexport)
#else
#define LCS_COMBINED_EXPORT __attribute__((visibility("default")))
#endif
#include <cub/device/device_radix_sort.cuh>
#define LCS_STATE_TRACK_ALLOCATIONS
#include "../training/state_archive.h"
// Unified tissue solver and legacy XPBD entry points; see README.md (system architecture).
// physics.cu binds the live shared buffers here. Domain headers keep their original include order.

#include "xpbd_solver.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <cub/device/device_radix_sort.cuh>

#include "../training/training_api.h"
static int g_trainingFlags=0;
static void trainingSubstep(float);
static void trainingCapture();
static void trainingShutdown();

// Opt-in CUDA timeline measurements. Disabled during normal simulation.
static bool g_profile=false;
static std::vector<cudaEvent_t> g_profileEvents;
static void profileMark(int step,int phase){
    if(!g_profile)return;
    int index=step*4+phase;
    while((int)g_profileEvents.size()<=index){cudaEvent_t e;cudaEventCreate(&e);g_profileEvents.push_back(e);}
    cudaEventRecord(g_profileEvents[index]);
}
extern "C" void xpbd_profile_enable(int enabled){g_profile=enabled!=0;}
extern "C" int xpbd_profile_read(int steps,float* ms){
    if(!ms || steps<=0 || (int)g_profileEvents.size()<steps*4)return -1;
    cudaEventSynchronize(g_profileEvents[steps*4-1]);
    for(int p=0;p<3;p++){ms[p]=0;for(int s=0;s<steps;s++){float t=0;cudaEventElapsedTime(&t,g_profileEvents[4*s+p],g_profileEvents[4*s+p+1]);ms[p]+=t;}}
    return 0;
}
// Opt-in section timing for performance work. While enabled, the unified substep issues its captured kernel
// sequence directly on the default stream, and the time between consecutive marks accumulates under the name of the
// later mark. Disabled, a mark costs one flag test and every captured graph stays as it is.
static bool g_sectionProfile=false;
static std::vector<std::pair<const char*,cudaEvent_t>> g_sectionMarks;
static std::vector<cudaEvent_t> g_sectionPool;
static std::vector<std::pair<std::string,double>> g_sectionTotals;
#ifdef ENDO_KIDNEY_STAGE_TRACE
static void kidneyTraceMark(const char* name);
#endif
extern "C" void xpbd_section_mark(const char* name){
    if(!g_sectionProfile)return;
#ifdef ENDO_KIDNEY_STAGE_TRACE
    kidneyTraceMark(name);
#endif
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
    for(const auto& entry:g_sectionTotals){char line[200];snprintf(line,sizeof line,"%s=%.4f\n",entry.first.c_str(),entry.second);out+=line;}
    if(!text || capacity<=(int)out.size())return (int)out.size()+1;
    memcpy(text,out.c_str(),out.size()+1);return 0;
}

// ── minimal float3 helpers (kept local so this module is independent of common.cuh) ─────────────
__host__ __device__ __forceinline__ float3 mkf3(float x, float y, float z) { float3 r; r.x=x; r.y=y; r.z=z; return r; }
__host__ __device__ __forceinline__ float3 operator+(float3 a, float3 b) { return mkf3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ __forceinline__ float3 operator-(float3 a, float3 b) { return mkf3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ __forceinline__ float3 operator*(float3 a, float s)   { return mkf3(a.x*s, a.y*s, a.z*s); }
__host__ __device__ __forceinline__ float  dot3(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ __forceinline__ float  len3(float3 a)           { return sqrtf(dot3(a,a)); }
__host__ __device__ __forceinline__ float3 cross3(float3 a, float3 b) { return mkf3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
__host__ __device__ __forceinline__ float  compF3(float3 v, int k) { return (k==0) ? v.x : ((k==1) ? v.y : v.z); }

#include "../contact/friction_contact.cuh"
#include "../grasp/tool_coupling.cuh"
#include "../contact/surface_contact.cuh"
#include "../contact/contact_groups.cuh"
#include "../contact/fragment_contact.cuh"
static FragmentVertex* d_fragmentVertices=nullptr;
static FragmentHit* d_fragmentHits=nullptr;
static int* d_fragmentClosedCanonical=nullptr;
static unsigned long long *d_fragmentKeys=nullptr,*d_fragmentSortedKeys=nullptr;
static int *d_fragmentOrder=nullptr,*d_fragmentSortedOrder=nullptr;
static void* d_fragmentSortTemp=nullptr;
static size_t g_fragmentSortBytes=0;static int g_fragmentSortCapacity=0;
static int* d_fragmentGroups=nullptr;
static float3* d_fragmentLineSums=nullptr;
static FragmentConstraint* d_fragmentConstraints=nullptr;
static int* d_fragmentQuerySlots=nullptr;
static int *d_fragmentUniqueQueries=nullptr,*d_fragmentUniqueCount=nullptr;
static int *d_fragmentDuplicateHeads=nullptr,*d_fragmentDuplicateAmbiguous=nullptr,*d_fragmentDuplicateNext=nullptr;
static cudaStream_t g_fragmentRefreshStream=nullptr;
static int *d_fragmentConstraintCount=nullptr,*d_fragmentConstraintDegree=nullptr;
static int g_fragmentConstraintCapacity=0;
static int* d_fragmentContacted=nullptr;
static FragmentTreeNode* d_fragmentTree=nullptr;
static int g_fragmentTreeCapacity=0,g_fragmentTreeBase=0;
static FragmentBounds* d_fragmentBounds=nullptr;
static int* d_fragmentRoots=nullptr;
static int* d_fragmentRootCount=nullptr;
static int* d_fragmentSupported=nullptr;
static int* d_fragmentUnpinnedVolume=nullptr;
static int* d_fragmentDegree=nullptr;
static float* d_fragmentLambda=nullptr;
static float3* d_fragmentBefore=nullptr;
static float3* d_fragmentStart=nullptr;
// Host-only diagnosis; no device synchronization or changed GPU work.
// Host-only graph lifecycle optimization. The physics state and every device
// kernel are unchanged. It is opt-in, fixed before initialization, and falls
// back to the original rebuild on CUDA's documented update rejection.
static int g_runtimeCoreMode=0;
static int g_runtimeCoarseBlocks=0;
static bool runtimeSolverInitialized();
extern "C" LCS_COMBINED_EXPORT int LCS_GetRuntimeCoreMode(){return g_runtimeCoreMode;}
static unsigned long long g_runtimeGraphStats[8]={};
extern "C" LCS_COMBINED_EXPORT int LCS_SetRuntimeCoreMode(int mode){
    if(mode&~(256|2048|8192|32768))return -10001;
    if(runtimeSolverInitialized() && mode!=g_runtimeCoreMode)return -10002;
    g_runtimeCoreMode=mode;g_runtimeCoarseBlocks=(mode&8192)?3:0;return 0;
}
extern "C" LCS_COMBINED_EXPORT int LCS_ReadRuntimeGraphStats(unsigned long long* out,int capacity){
    if(!out||capacity<8)return -1;
    for(int i=0;i<8;i++)out[i]=g_runtimeGraphStats[i];return 8;
}
static cudaError_t installRuntimeGraph(cudaGraphExec_t& executable,cudaGraph_t graph){
    if(executable && (g_runtimeCoreMode&256)){
        g_runtimeGraphStats[0]++;
        cudaGraphExecUpdateResultInfo info={};
        cudaError_t error=cudaGraphExecUpdate(executable,graph,&info);
        if(error==cudaSuccess && info.result==cudaGraphExecUpdateSuccess){g_runtimeGraphStats[1]++;return cudaSuccess;}
        if(error!=cudaSuccess && error!=cudaErrorGraphExecUpdateFailure)return error;
        g_runtimeGraphStats[2]++;g_runtimeGraphStats[5]=(unsigned long long)info.result;
        cudaGetLastError();
    }
    if(executable){
        cudaError_t error=cudaGraphExecDestroy(executable);if(error!=cudaSuccess)return error;
        executable=nullptr;g_runtimeGraphStats[4]++;
    }
    cudaError_t error=cudaGraphInstantiate(&executable,graph,0);
    if(error==cudaSuccess)g_runtimeGraphStats[3]++;
    return error;
}

static int g_interactionStats[16]={};
extern "C" LCS_COMBINED_EXPORT int LCS_ReadInteractionStats(int* out,int capacity,int reset){
    if(!out||capacity<16)return -1;for(int i=0;i<16;i++)out[i]=g_interactionStats[i];
    if(reset)for(int i=0;i<16;i++)g_interactionStats[i]=0;return 16;
}
static int *d_fragmentQueries=nullptr,*d_fragmentQueryCount=nullptr;
static int g_fragmentCapacity=0;
#include "../grasp/gaussian_grasp.cuh"
#include "../grasp/grip_observation.cuh"
#include "../solver/material/unified_vertex.cuh"
#include "../contact/manifold_validation.cuh"
#include "../contact/coarse_bound_cache.cuh"
#include "../contact/fragment_coarse.cuh"
#include "../solver/material/pressure_patch_line.cuh"
#include "../solver/material/coupled_material.cuh"
#include "../solver/material/material_stencil.cuh"
#include "../solver/material/material_multigrid.cuh"
static __half* d_cgMatrix=nullptr;
static float* d_cgScale=nullptr;
static unsigned short* d_cgStencil=nullptr;
#include "../solver/material/material_line_search.cuh"
// d_cgP and d_cgQ hold the PCG direction of alternate iterations (k_MaterialStencilProduct).
static float3 *d_cgR=nullptr,*d_cgP=nullptr,*d_cgQ=nullptr,*d_cgD=nullptr,*d_cgA=nullptr,*d_cgZ=nullptr;
static Symmetric3* d_cgH=nullptr;
static MaterialCGSum* d_cgS=nullptr;
static MaterialLineSearch* d_cgSearch=nullptr;
static UnifiedState* d_unified=nullptr;
static float3* d_unifiedTarget=nullptr;
static float3* d_unifiedPreviousInverse=nullptr;
static float* d_unifiedRestMetric=nullptr;
static int g_unifiedTetCapacity=0;
static int* d_unifiedComponents=nullptr;
// Records of data derived from the live topology alone: the topology revision it was derived at, then 1 when the next
// refresh must derive it again (k_TopologyStale). ~0u marks data never derived for the bound topology. One record
// each for the component labels, the liveness flags and inverse masses of k_RefreshLiveness, and the tet edge support
// derived from those flags, and the multigrid hierarchy; xpbd_bind allocates them together.
static unsigned* d_topologyRecords=nullptr;
static unsigned* d_multigridRevision=nullptr;
static unsigned* d_componentRevision=nullptr;
static unsigned* d_livenessRevision=nullptr;
static unsigned* d_edgeSupportRevision=nullptr;
// A host write to the liveness flags or inverse masses: derive them, and the edge support, again.
static void markLivenessStale(){if(d_livenessRevision)cudaMemset(d_livenessRevision,0xff,4*sizeof(unsigned));}
static int* d_graspComponents=nullptr;
static int* d_floorRecovery=nullptr;   // [N] floor recovery per component root (materialFloorRecovery)
static int* d_safeCluster=nullptr;
static SafeCoarseCluster* d_safeAggregate=nullptr;
static int* d_safeJoined=nullptr;

static float* d_coarseWeight=nullptr;
static InteractionSum* d_unifiedInteraction=nullptr;
// The vertices each local sweep solves, listed per lattice color by k_SweepSites, and their counts.
static int *d_sweepSites=nullptr,*d_sweepCount=nullptr;
static CoarseBlock* d_coarseBlock=nullptr;
static CoarseBlock* d_globalCoarse=nullptr;
static SurfaceContact* d_manifold=nullptr;
static int *d_manifoldIds=nullptr,*d_manifoldCount=nullptr,*d_manifoldOwners=nullptr;
static float* d_manifoldQuadrature=nullptr;
static int g_manifoldCapacity=0;
static cudaGraphExec_t g_unifiedGraph=nullptr;
static bool g_unifiedGraphDirty=false;
static int g_fragmentGraphQueries[3]={-1,-1,-1};
static cudaGraphExec_t g_unifiedRecoveryGraph=nullptr;
static bool g_unifiedRecoveryGraphDirty=false;
// The substep graph for substeps with no free component, see the substep query in unifiedSubstep.
static cudaGraphExec_t g_unifiedLoadedGraph=nullptr;
static bool g_unifiedLoadedGraphDirty=false;
static int* d_substepQuery=nullptr;
static cudaGraphExec_t g_unifiedContactGraph=nullptr;
static bool g_unifiedContactGraphDirty=false;

// ── device buffers (SoA) ────────────────────────────────────────────────────────────────────────
static float3* d_pos      = nullptr;   // [N] current positions
static float3* d_prevPos  = nullptr;   // [N] positions at substep start (for v = (x-x_prev)/h)
static float* d_attachStiffness = nullptr;
static float3* d_attachRest = nullptr;
static float3* d_attachLambda = nullptr;
static float3* d_vel      = nullptr;   // [N] velocities
static float*  d_invMass  = nullptr;   // [N] inverse mass (0 = pinned)
static int*    d_edgeI    = nullptr;   // [E] colour-ordered edge endpoint i
static int*    d_edgeJ    = nullptr;   // [E] colour-ordered edge endpoint j
static float*  d_restLen  = nullptr;   // [E] colour-ordered rest length
static float*  d_lambda   = nullptr;   // [E] colour-ordered per-edge XPBD multiplier (reset each substep)
static int*    d_edgeActive = nullptr; // [E] colour-ordered per-edge active flag (1=live, 0=severed).
                                       // STANDALONE mode: the notify API flips it. BOUND mode:
                                       // k_RefreshLiveness RE-DERIVES it every substep from the live
                                       // nbrIdx/active. The greedy COLOURING stays static either way.

static int     g_N = 0, g_E = 0;
static std::vector<int> g_colorOff, g_colorCnt;   // host: [numColors] flat-range per colour
static int     g_numColors = 0;
static bool    g_ready = false;
static bool runtimeSolverInitialized(){return g_ready;}

// M1 (design §9-(4)): the input-edge -> colour-slot PERMUTATION is PERSISTED, not discarded, so a
// runtime sever can be propagated to the colour-ordered d_edgeActive. The host inverse map serves the
// xpbd_set_*_active notify API (standalone self-test); in BOUND mode k_RefreshLiveness supersedes the
// notify path, deriving liveness from the LIVE shared nbrIdx/bendPairs every substep (d_bendSrcIdx then
// holds LIVE bendPairs indices, uploaded by xpbd_bind).
static int* d_edgeSrcIdx = nullptr;               // [E] colour-slot -> original input edge index
static std::vector<int> g_edgeSlotOfInput;        // host: input edge index -> colour slot

// M2 (design §9-(4)): frame-scope coefficients computed ONCE by xpbd_begin_frame so the caller
// (eventually physics_step) can own the N-loop and interleave cut_ribbon_tick between substeps.
static float g_hFrame = 0.f, g_alphaTildeF = 0.f, g_velScaleF = 1.f, g_betaGammaHSF = 0.f;
static float g_alphaBF = 0.f, g_betaGammaHBF = 0.f;
static int   g_itersF = 1;
static bool  g_doBendF = false, g_frameReady = false;
static bool  g_doStrainF = false;   // STEP 7: true iff tets exist AND youngE>0 (recomputed each begin_frame)

// STEP 2 — bending elements (paper Eq13-16 as XPBD angle constraints). Colour-ordered like the edges.
static int*    d_bendI = nullptr, *d_bendJ = nullptr, *d_bendK = nullptr;  // [B] apex + two arms
static float*  d_bendTheta0 = nullptr;   // [B] rest angle theta0
static int*    d_bendActive = nullptr;   // [B] active flag (1=live; a cut/tear flips via the notify API)
static float*  d_bendLambda = nullptr;   // [B] XPBD multiplier (reset per substep)
static int     g_B = 0;
static std::vector<int> g_bColorOff, g_bColorCnt;
static int     g_bNumColors = 0;
static int*    d_bendSrcIdx = nullptr;            // [B] colour-slot -> original input element index (M1)
static std::vector<int> g_bendSlotOfInput;        // host: input element index -> colour slot

// STEP 7 — per-tet isochoric/volumetric Neo-Hookean strain (design rev-B §4[Strain]/§9-(7)). Stored by Kuhn tet
// type, each type in cell order (see xpbd_build_tets); the colour-ordered passes reach their tets through
// d_tetColorOrder. BOUND MODE ONLY: built once by xpbd_build_tets from the live cornerPos/active/nbrIdx
// dims captured at xpbd_bind. d_tetActive is NOT written by a notify path — it is RE-DERIVED every
// substep by k_RefreshLiveness from the SAME live nbrIdx/active flags (a tet's 3 member axis-edge owner
// slots + its 4 corners), so it is naturally permanent (nbrIdx never un-severs).
static int*    d_tetIds     = nullptr;   // [4*T] corner ids (p0,p1,p2,p3)
static float3* d_tetBr0     = nullptr;   // [T] Dm^-1 row 0 (rest inverse matrix, baked at build time)
static float3* d_tetBr1     = nullptr;   // [T] Dm^-1 row 1
static float3* d_tetBr2     = nullptr;   // [T] Dm^-1 row 2
static float*  d_tetMuV     = nullptr;   // [T] mu   * RestVolume  (material x rest-volume, h-independent)
static float*  d_tetLambdaV = nullptr;   // [T] bulk modulus kappa * RestVolume (historical buffer name)
static float*  d_tetAlphaDev= nullptr;   // [T] 1/(h^2*mu*V)     — recomputed once per xpbd_begin_frame
static float*  d_tetAlphaHyd= nullptr;   // [T] 1/(h^2*kappa*V) — recomputed once per xpbd_begin_frame
static float*  d_tetLambdaDevL = nullptr; // [T] XPBD multiplier, deviatoric (reset each substep)
static float*  d_tetLambdaHydL = nullptr; // [T] XPBD multiplier, hydrostatic (reset each substep)
static int*    d_tetActive  = nullptr;   // [T] 1=live (re-derived every substep by k_RefreshLiveness)
static int*    d_tetEdgeOwnerSlot = nullptr; // [3*T] the tet's 3 member AXIS-edge owner slots (6p+s)
static int*    d_edgeTetSupport = nullptr; // [6*N] live material support for each axis-edge slot
static int*    d_tetCellBase = nullptr;  // [T/6] corner-0 node of the cell of the k-th tet of every type
static KuhnTet* d_kuhnTets   = nullptr;  // [6] rest data of each Kuhn type, copied from its first tet (k_KuhnTable)
static int     g_T = 0;
static int     g_tetCells = 0;           // T/6: the tets of each type
static int*    d_activeNodes = nullptr;  // [g_activeNodes] the active nodes in increasing order (UnifiedState::activeNodes)
static int     g_activeNodes = 0;
static std::vector<int> g_tColorOff, g_tColorCnt;
static int     g_tNumColors = 0;
// The tets of colour c, by index: entries g_tColorOff[c] to g_tColorOff[c]+g_tColorCnt[c]-1 (host copy and device).
static std::vector<int> g_tetColorOrder;
static int* d_tetColorOrder=nullptr;
// The node ids and rest determinant of the tet at each entry of d_tetColorOrder (k_ColorTetData).
static int4* d_colorTetIds=nullptr;
static float* d_colorTetDet=nullptr;
static int* d_localTets=nullptr;
static std::vector<int> g_localOff,g_localCount;
static int g_localPasses=0;
static float g_minTetVolume=0;
static int g_volumePasses=0;
static cudaGraphExec_t g_pressureGraph=nullptr;
static float g_tissueHardening=80.f,g_bulkDampingSeconds=.003125f;
static float* d_fiberLambda=nullptr; // six finite-compliance tensile energies per live tet
static int* d_captureInside=nullptr;
static bool g_captureWedge=false;
static float3* d_feasibleStart=nullptr;
static float3* d_feasibleTarget=nullptr;
static int* d_incidentOffsets=nullptr;
static int* d_incidentTets=nullptr;
static int* d_needFeasible=nullptr;

static bool    g_haveTets = false;
static cudaGraphExec_t g_tetGraph=nullptr;
static cudaStream_t g_captureStream=nullptr;
static cudaGraphExec_t g_feasibleGraphs[8]={};
static void destroyFeasibleGraphs(){for(auto& graph:g_feasibleGraphs){if(graph)cudaGraphExecDestroy(graph);graph=nullptr;}}
static float g_graphOmega=0,g_graphDamping=0,g_graphH=0;
static int g_graphTetCount=0;
static void destroyTetGraph(){if(g_tetGraph)cudaGraphExecDestroy(g_tetGraph);g_tetGraph=nullptr;}

// STEP 8 — contact + two-jaw grasp (design rev-B §4[Contact]/§5/§6/§9-(8)). Per-corner, BOUND MODE ONLY;
// allocated (zero-filled, functionally INERT) in xpbd_bind's tail, so these are always valid once bound —
// no separate "build" call is needed (unlike tets, grasp/contact need no extra topology beyond cornerCount).
static int*    d_grasped          = nullptr;  // [N] 1 = captured (core OR ring)
static int*    d_graspCore        = nullptr;  // [N] 1 = hard-pinned core (subset of d_grasped)
static float*  d_coreOffU         = nullptr;  // [N] captured corner's jaw-local anchor (axis/normal/width)
static float*  d_coreOffN         = nullptr;
static float*  d_coreOffW         = nullptr;
static float* d_coreComplianceScale=nullptr;
static float3* d_coreLambda       = nullptr;  // [N] core-pin XPBD multiplier (reset /substep, like d_slabLambda)
static float*  d_coreForceEMA     = nullptr;  // [N] SUSTAINED force estimate for the break check -- audit-round
                                               // fix, see k_SolveCorePin's header for why this must NOT be the
                                               // raw per-substep |coreLambda|/h^2 (persistent across substeps;
                                               // reset only at capture, like d_coreOffU/N/W)
static int*    d_candScratch = nullptr;   // [N] k_MarkGraspCandidates scratch — allocated ONCE at bind,
static float*  d_uScratch    = nullptr;   // reused every capture event (redesign 2026-07-07: avoids a
static float*  d_vScratch    = nullptr;   // cudaMalloc/cudaFree pair on every rising-edge capture, which
static float*  d_nScratch    = nullptr;   // could storm if grasp01 crosses threshold rapidly)
static float*  d_slabLambda       = nullptr;  // [N] slab XPBD multiplier (ring corners; reset /substep)
static float*  d_slabLambdaPrev   = nullptr;  // [N] PREVIOUS substep's converged slab lambda (Fn timing)
static float*  d_jawContactLambda = nullptr;  // [N] jaw-as-contact multiplier (uncaptured corners)
static float*  d_shaftContactLambda = nullptr; // [N] shaft-capsule contact multiplier (everyone)
static float*  d_coulombAnchorU   = nullptr;  // [N] ring corner's tangential stick anchor (axis/width)
static float*  d_coulombAnchorW   = nullptr;
static bool    g_grasping = false;            // capture/release hysteresis latch (design §5 item 1)
static bool    g_toolArmed = false;            // true once xpbd_set_tool has EVER been called (design
                                               // safety: an all-zero default XpbdToolState is NOT a safe
                                               // "definitely inert" sentinel by itself -- its degenerate
                                               // jaw box sits at the world origin, which can coincide
                                               // with real corner geometry there; gate K=2 and the
                                               // contact/grasp kernels on this instead)
static XpbdToolState  g_tool = {}, g_toolFrom = {};
static int g_substepIndex = 0;
static XpbdGraspParams g_gp = { 0.5f, 0.3f, 0.02f, 0.55f, 0.3f, 1.0e-7f, 0.02f,
                                /*ccdVelThresh*/0.5f, /*ccdMaxDisp*/0.045f, /*ccdOmega*/0.25f,
                                /*omegaGrasp*/0.25f, /*coreAlpha*/1.0e-7f, /*coreBreakForce*/150.f,
                                /*graspMaxDisp*/0.045f, /*nExemptRings*/1 };
static bool g_frictionContact=false;
static bool g_anchorAssist=false;
static bool g_embeddedGrip=false;
static CoupledTool* d_coupledTool=nullptr;
static int g_guidedInstrument=0;
static float g_toolMass=.02f,g_toolStiffness=300.f,g_toolDamping=5.f,g_toolMaxTravel=.1f;
static float g_jawMass=.005f,g_jawStiffness=0,g_jawDamping=2.f,g_jawSpeed=1.f;
static float g_openJawGap=0;
static float3* d_contactMobility=nullptr;
static float* d_gripError=nullptr;
static float* d_contactResidual=nullptr;
static GripPoint* d_gripPoints=nullptr;
static GripSolveStep* d_gripStep=nullptr;
static GripPatch* d_gripPatch=nullptr;
static float* d_gripPatchWeights=nullptr;
static cudaGraphExec_t g_gripGraph=nullptr;
static void destroyGripGraph(){if(g_gripGraph)cudaGraphExecDestroy(g_gripGraph);g_gripGraph=nullptr;}
static int g_gripCount=0,g_gripCapacity=0;
static int *d_gripCounter=nullptr,*d_gripLive=nullptr,*d_gripDegree=nullptr;
static float* d_gripMoments=nullptr;
static float g_gripSigma=.15f,g_gripCompliance=1.f/8000.f,g_gripDamping=.02f;
static float g_pressureH=0;

static float g_muStatic=.5f,g_muDynamic=.3f,g_jawThickness=.1f,g_contactRadius=.02f;
static ContactRecord* d_contacts=nullptr;
static float* d_contactNormalLambda=nullptr;
static float3* d_contactTangentLambda=nullptr;

static int g_surfaceCount=0,g_surfaceCapacity=0;
static float* d_toolClearance=nullptr;
static XpbdToolState g_previousAcceptedTool={};
static cudaGraphExec_t g_surfaceGraph=nullptr;
static cudaGraphExec_t g_velocityGraph=nullptr;
static cudaGraphExec_t g_repairGraph=nullptr;
static void destroyRepairGraph(){if(g_repairGraph)cudaGraphExecDestroy(g_repairGraph);g_repairGraph=nullptr;}
static float g_velocityH=0;
static int g_graphClosing=0;
static float g_graphAlpha=0,g_graphMuS=0,g_graphMuD=0;
static void destroySurfaceGraph(){if(g_surfaceGraph)cudaGraphExecDestroy(g_surfaceGraph);g_surfaceGraph=nullptr;if(g_velocityGraph)cudaGraphExecDestroy(g_velocityGraph);g_velocityGraph=nullptr;}
static const float3* g_surfaceRest=nullptr;
static const float4* g_surfaceMaterialFP=nullptr;
static const int *g_surfaceTags=nullptr,*g_surfaceCorners=nullptr,*g_surfaceConn=nullptr;
static const unsigned* g_surfaceMasks=nullptr;
static const unsigned* g_surfaceBoundaryMasks=nullptr;
static const float3* g_surfaceExpanded=nullptr;
static SurfaceVertex* d_surfaceVertices=nullptr;
static float4* d_jawLoad=nullptr;
static ContactGroupBounds* d_contactGroups=nullptr;
static unsigned char* d_contactGroupRejected=nullptr;
static float4* d_surfaceCurrent=nullptr;
static float3* d_surfacePrevious=nullptr;
static SurfaceContact* d_surfaceContacts=nullptr;
static SurfaceContact* d_contactCache=nullptr;
static int* d_contactCacheSlots=nullptr;
static int* d_surfaceDegree=nullptr;
static int *d_surfaceActive=nullptr,*d_surfaceActiveCount=nullptr;
static float3* d_surfaceDelta=nullptr;
static float4* d_contactReaction=nullptr;

static float3  g_jawW = { 1.f, 0.f, 0.f };    // derived width axis = normalize(cross(jawAxis,jawNormal))

static XpbdParams g_p = { /*numSubsteps*/20, /*iters*/1, /*omega*/1.0f, /*damping*/0.02f,
                          /*ks*/7.5e4f, /*gx*/0.f, /*gy*/-9.81f, /*gz*/0.f, /*kb*/2.0e4f,
                          /*alpha*/0.f, /*betaS*/0.f, /*betaB*/0.f,
                          /*youngE*/0.f, /*poisson*/0.45f };

// ── REBIND state (design rev-B §9-(4)). In BOUND mode d_pos/d_vel ALIAS the shared live buffers
// (recon_corner_pos() / physics_vel()) and are NOT owned by this module; liveness (edgeActive/
// bendActive) and invMass are DERIVED at the start of every substep by k_RefreshLiveness from the
// LIVE shared nbrIdx / bendPairs.alive / active / pinned / mass — the SAME flags physics.cu's force
// gates read — so a cut.cu sever (nbrIdx=-1, alive=0, active=0) is picked up the very next substep
// with ZERO cut.cu changes. The colouring stays STATIC (a dead edge just early-outs). ─────────────────
static bool  g_bound         = false;
static int*  d_edgeOwnerSlot = nullptr;   // [E] owner slot (6p + s, s=2*axis+1) per colour-ordered edge
static const float3* g_liveRestNbr=nullptr;
static const int*             g_liveNbrIdx      = nullptr;   // [6*N] live topology (cut writes -1)
static const XpbdBendPairGpu* g_liveBendPairs   = nullptr;   // live bend pairs (cut writes alive=0)
static const int*             g_liveActive      = nullptr;   // [N] live active flags (cut freezes debris)
static const int*             g_livePinned      = nullptr;   // [N] anchored BC
static const float*           g_liveMass        = nullptr;   // [N] invMass derived, NEVER baked
static const float3*          g_liveExtForce    = nullptr;   // [N] gravity force (predict: (ext+haptic)/mass)
static const float3*          g_liveHapticForce = nullptr;   // [N] SS2.3 felt reaction (STEP 5: into predict)
static int g_dimsX = 0, g_dimsY = 0, g_dimsZ = 0;            // carried for the step-7 tet builder

static inline int Gr(int n) { int g = (n + 63) / 64; return g < 1 ? 1 : g; }
// One full wave of 64-thread blocks of a kernel whose warps stride over the vertices k_SweepSites lists. A larger grid
// leaves a partly filled second wave of warps that each walk several vertices; a smaller one leaves SMs idle.
template<class Kernel> static int fullWave(Kernel kernel){
    int device=0,processors=0,blocks=0;cudaGetDevice(&device);
    cudaDeviceGetAttribute(&processors,cudaDevAttrMultiProcessorCount,device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,64,0);return std::max(1,processors*blocks);
}
static int g_sweepBlocks=0,g_predictorBlocks=0;
// Grid of the kernels that sum per-component terms through blockAccumulate (unified_vertex.cuh).
static inline int GrAccumulate(int n) { int g = (n + ACCUMULATE_THREADS - 1) / ACCUMULATE_THREADS; return g < 1 ? 1 : g; }

// ── kernels ───────────────────────────────────────────────────────────────────────────────────
// Predict: save prev, integrate the external body forces into velocity, advance position. Pinned
// (w=0) stay put.
// STANDALONE mode (extForce == nullptr): the acceleration is the uniform gravity g from XpbdParams.
// BOUND mode (design §9-(4)/(5)): the acceleration is the SS2.3 body-force sum divided by mass —
//   a = (extForce[p] + hapticForce[p]) * invMass[p]
// i.e. gravity (extForce) PLUS the felt haptic reaction (hapticForce, migration STEP 5). This is the
// SAME (extForce + hapticForce)/mass the retired RK45 slope formed in k_ComputeSlope (physics.cu); the
// internal spring force Fint moved to the constraint solve, and the Rayleigh alpha to the velScale
// post-pass. hapticForce is the LIVE d_hapticForce haptics.cu scatters into each frame (aliased at
// bind); nullptr in standalone / when haptics is off. invMass is the refresh-derived value, so a
// debris-frozen corner (invMass 0) is skipped identically.
__global__ void k_predict(int N, float h, float3 g, const float3* extForce, const float3* hapticForce,
                          const float* invMass, float3* pos, float3* prevPos, float3* vel)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;
    prevPos[p] = pos[p];
    if (invMass[p] > 0.f) {
        float3 a;
        if (extForce) {
            float3 F = extForce[p];
            if (hapticForce) F = F + hapticForce[p];   // STEP 5: gravity + felt reaction (SS2.3)
            a = F * invMass[p];
        } else {
            a = g;                                     // standalone self-test: uniform gravity
        }
        float3 v = vel[p] + a * h;          // v += h * a
        vel[p] = v;
        pos[p] = pos[p] + v * h;            // x += h * v
    }
}

// RefreshLiveness (design §9-(4), BOUND mode only) — run at the START of every substep, BEFORE predict,
// so predict and the sweeps see fresh liveness/mass. Derives, from the LIVE shared buffers:
//   edgeActive[e] = nbrIdx[edgeOwnerSlot[e]] >= 0  AND  active[i] && active[j]
//                   (k_SeverLinks writes BOTH half-slots, so the owner slot alone captures a sever;
//                    the active[] test is k_AccStructural's second gate (`if (active[n] == 0) continue;`) — an edge to a
//                    frozen debris corner exerts NO force there, so the constraint must drop too)
//   bendActive[b] = bendPairs[bendSrcIdx[b]].alive   (k_AccBending's only gate (`if (bp.alive == 0) return;`) — a
//                    frozen member is handled by its w=0, matching the RK45 force-discard semantics)
//   invMass[c]    = (pinned[c] || active[c]==0) ? 0 : 1/mass[c]   (NEVER baked: cut.cu:451 flips
//                    active[] at runtime when it freezes fully-severed debris)
// One launch over max(N, E, B); no atomics (disjoint writes).
// STEP 7 addendum: a 4th domain T re-derives d_tetActive every substep from the SAME live nbrIdx/active
// flags — a tet dies PERMANENTLY the substep any of its 3 member axis-edge owner slots is severed
// (tetEdgeOwnerSlot[3t+0..2], the 6p+s convention) or any of its 4 corners (tetIds[4t+0..3]) goes
// inactive. Zero cut.cu changes, exactly mirroring the edge/bend mechanism above.
__global__ void k_RefreshLiveness(int N, int E, int B, int T,
                                  const int* nbrIdx, const int* edgeOwnerSlot,
                                  const int* edgeI, const int* edgeJ,
                                  const XpbdBendPairGpu* bendPairs, const int* bendSrcIdx,
                                  const int* active, const int* pinned, const float* mass,
                                  const int* tetIds, const int* tetEdgeOwnerSlot,
                                  int* edgeActive, int* bendActive, float* invMass, int* tetActive,
                                  const unsigned* stale)
{
    // Every output is a function of the live topology and the fixed flags alone: a non-null `stale` flag skips the
    // refresh while the topology revision stays the same (k_TopologyStale, d_livenessRevision).
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (stale && !*stale) return;
    if (t < E)
        edgeActive[t] = (nbrIdx[edgeOwnerSlot[t]] >= 0 &&
                         active[edgeI[t]] != 0 && active[edgeJ[t]] != 0) ? 1 : 0;
    if (t < B)
        bendActive[t] = (bendPairs[bendSrcIdx[t]].alive != 0) ? 1 : 0;
    // design §5 "the w-rule": w_i = 0 iff pinned[i] OR active[i]==0 -- every constraint kernel
    // (edges/bending/tets/contact/slab/core-pin) reads invMass, so a single source of truth here
    // covers all of them at once (no per-kernel special-casing needed).
    // redesign 2026-07-07: a core-pinned corner is NO LONGER included in this zero-invMass rule (the
    // rule used to also zero invMass for graspCore[i]!=0, kinematically freezing it) -- it keeps its
    // NORMAL invMass and is pulled toward the jaw's target by the compliant k_SolveCorePin constraint
    // instead (so it can be overpowered and break, per the design's "compliant + breakable core-pin"
    // decision). Only pinned/inactive corners get invMass=0 now; graspCore is no longer read here at
    // all (the parameter was removed along with this rule).
    if (t < N)
        invMass[t] = (pinned[t] != 0 || active[t] == 0) ? 0.f : (1.f / mass[t]);
    if (t < T) {
        bool live = nbrIdx[tetEdgeOwnerSlot[3*t+0]] >= 0 &&
                    nbrIdx[tetEdgeOwnerSlot[3*t+1]] >= 0 &&
                    nbrIdx[tetEdgeOwnerSlot[3*t+2]] >= 0 &&
                    active[tetIds[4*t+0]] != 0 && active[tetIds[4*t+1]] != 0 &&
                    active[tetIds[4*t+2]] != 0 && active[tetIds[4*t+3]] != 0;
        tetActive[t] = live ? 1 : 0;
    }
}

// The edge support of the live tets, derived again with the liveness flags only (`stale`, k_RefreshLiveness).
__global__ void k_ClearTetEdgeSupport(int n, int* support, const unsigned* stale)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && *stale) support[i]=0;
}
__global__ void k_MarkTetEdgeSupport(int T, const int* live, const int* slots, int* support, const unsigned* stale)
{
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=T || !live[t] || !*stale) return;
    for(int k=0;k<3;k++) atomicExch(support+slots[3*t+k],1);
}

// Solve one COLOUR of structural distance constraints. Within a colour no two edges share a vertex,
// so each position is written by at most one edge -> true Gauss-Seidel, NO atomics.
// C = |xi-xj| - L0 ; a~ = (1/ks)/h^2 ; dLambda = (-C - a~*lambda)/(wi+wj+a~) ; xi += w*omega*dLambda*n.
__global__ void k_solve_edges(int colorOff, int colorCnt, float alphaTilde, float omega,
                              float betaGammaH, const float3* prevPos,
                              const int* edgeI, const int* edgeJ, const float* restLen,
                              const int* edgeActive, const float* invMass, float* lambda, float3* pos,
                              const int* materialSupport, const int* ownerSlot, float fallbackInvEh2)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= colorCnt) return;
    int e  = colorOff + t;
    if (edgeActive[e] == 0) return;               // severed/inactive (step-5 cut): drop the constraint; colouring stays static
    if (materialSupport) {
        // A partial cut can remove every incident tet while leaving this edge intact.
        // Preserve that residual connection with k=E*L (A=L^2), only where no tet supports it.
        // This is a grid-scale cut-rim approximation, never an extra spring across a severed edge.
        if (materialSupport[ownerSlot[e]]) return;
        alphaTilde=fallbackInvEh2/restLen[e];
    }
    int i  = edgeI[e], j = edgeJ[e];
    float wi = invMass[i], wj = invMass[j];
    float wsum = wi + wj;
    if (wsum <= 0.f) return;                      // both endpoints pinned
    float3 d = pos[i] - pos[j];
    float  L = len3(d);
    if (L < 1e-9f) return;                        // coincident: undefined gradient, skip
    float3 n = d * (1.f / L);
    float  C = L - restLen[e];
    // STEP 3 Kelvin-Voigt damping (Macklin XPBD 2016): gamma = a~ * beta * h (betaGammaH is a~*betaS*h,
    // precomputed on the host; beta = cs is the dashpot STIFFNESS, not a compliance). grad C . (x - x_prev)
    // is the relative displacement along the spring this substep; the damped constraint reproduces the
    // dashpot force f = cs*(relative velocity along the spring). betaGammaH=0 -> the undamped form exactly.
    float  gamma  = betaGammaH;
    float  gradDx = dot3(n, (pos[i] - prevPos[i]) - (pos[j] - prevPos[j]));
    float  dLambda = (-C - alphaTilde * lambda[e] - gamma * gradDx) / ((1.f + gamma) * wsum + alphaTilde);
    // SOR under-relaxation applied CONSISTENTLY to BOTH lambda and position, so the compliant fixed point
    // C = -a~*lambda (=> DL = mg/ks) is omega-INVARIANT (omega only changes convergence rate, not the
    // equilibrium stiffness). omega=1 is exact per-iteration; <1 trades speed for stability.
    float  applied = omega * dLambda;
    lambda[e] += applied;
    pos[i] = pos[i] + n * (wi * applied);
    pos[j] = pos[j] - n * (wj * applied);
}

// Solve one COLOUR of BENDING angle constraints — the paper's Eq13-16 (k_AccBending) as XPBD. C = theta -
// theta0, theta = angle at apex i between (xi-xj) and (xi-xk). The gradient directions are the paper's Eq14
// FORCE directions (nv - c*nu) / (nu - c*nv) — i.e. the angle gradient WITHOUT the 1/(sin*|arm|) factor.
// This reproduces the paper's SOFTER, geometry-scaled bending (effective force magnitude
// kb*(theta-theta0)*sin(theta)) so the XPBD rest shape MATCHES the RK45 golden (verified: a clamped
// cantilever's droop matches to <1% vs 50% for the clean 1/sin gradient, which is ~10x too stiff at these
// arm lengths and would over-resist the grasp tent). |g|=sin(theta) so it is also NON-singular. Within a
// colour no two bend elements share a vertex -> no atomics.
__global__ void k_solve_bending(int colorOff, int colorCnt, float alphaB, float omega,
                                float betaGammaH, const float3* prevPos,
                                const int* bI, const int* bJ, const int* bK, const float* theta0,
                                const int* bActive, const float* invMass, float* lambda, float3* pos)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= colorCnt) return;
    int e = colorOff + t;
    if (bActive[e] == 0) return;
    int i = bI[e], j = bJ[e], k = bK[e];
    float wi = invMass[i], wj = invMass[j], wk = invMass[k];
    if (wi + wj + wk <= 0.f) return;                 // all three pinned
    float3 u = pos[i] - pos[j]; float Lu = len3(u);
    float3 v = pos[i] - pos[k]; float Lv = len3(v);
    if (Lu < 1e-9f || Lv < 1e-9f) return;
    float3 nu = u * (1.f / Lu), nv = v * (1.f / Lv);
    float c = fmaxf(-1.f, fminf(1.f, dot3(nu, nv)));
    float s = sqrtf(fmaxf(1.f - c*c, 0.f));
    if (s < 1e-4f) return;                            // collinear singularity (matches k_AccBending EPS_SIN2)
    float theta = acosf(c);
    float C = theta - theta0[e];
    // Eq14 force-direction gradients (NO 1/(sin*|arm|) normalization) -> paper-faithful soft bending,
    // matches the RK45 golden. gj = (nv - c*nu) ; gk = (nu - c*nv) ; gi = -(gj+gk)  (sums to zero).
    float3 gj = (nv - nu*c);
    float3 gk = (nu - nv*c);
    float3 gi = (gj + gk) * (-1.f);
    float wg = wi*dot3(gi,gi) + wj*dot3(gj,gj) + wk*dot3(gk,gk);
    // STEP 3 bending Kelvin-Voigt damping (cb*theta_dot): gamma = a~_b*betaB*h (betaGammaH precomputed);
    // grad C . (x - x_prev) summed over the 3 particles = the angle rate this substep.
    float gamma  = betaGammaH;
    float gradDx = dot3(gi, pos[i]-prevPos[i]) + dot3(gj, pos[j]-prevPos[j]) + dot3(gk, pos[k]-prevPos[k]);
    float denom = (1.f + gamma) * wg + alphaB;
    if (denom < 1e-12f) return;
    float dLambda = (-C - alphaB * lambda[e] - gamma * gradDx) / denom;
    float applied = omega * dLambda;
    lambda[e] += applied;
    pos[i] = pos[i] + gi * (wi * applied);
    pos[j] = pos[j] + gj * (wj * applied);
    pos[k] = pos[k] + gk * (wk * applied);
}

// STEP 7 — recompute the PER-TET alpha (design §12 R-compliance: alpha~1/h^2, recomputed together with
// every other h-derived coefficient whenever h changes). muV/lambdaV are h-INDEPENDENT (baked once at
// build time); this kernel runs ONCE per xpbd_begin_frame (not per substep — h is fixed for the frame).
__global__ void k_ComputeTetAlpha(int T, float invH2, const float* muV, const float* lambdaV,
                                  float* alphaDev, float* alphaHyd)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    // muV/lambdaV > 0 guarded at build time (degenerate/zero-material tets are never emitted with
    // youngE<=0 handled by the caller skipping the tet dispatch entirely — see xpbd_substep).
    alphaDev[t] = invH2 / muV[t];
    alphaHyd[t] = invH2 / lambdaV[t];
}

// Finite tensile reinforcement on live tetrahedra. The separate determinant
// feasibility pass is only a numerical non-inversion safeguard.
// This is not the full IPC contact/elasticity solver.
__device__ void projectTensile(int e,const int* ids,const int* active,
    const float3* b0,const float3* b1,const float3* b2,const float* invMass,float3* pos,const float* alphaDev,float* fiberLambda,float strength){
    if(!active[e])return;
    int id[4]={ids[4*e],ids[4*e+1],ids[4*e+2],ids[4*e+3]};
    float3 p[4]={pos[id[0]],pos[id[1]],pos[id[2]],pos[id[3]]};
    float3 b[3]={b0[e],b1[e],b2[e]};
    // Nonlinear tensile energy per material direction: mu*V*strength/2*s^6,
    // s=max(length/restLength-1,0). Finite XPBD compliance, not a maximum
    // permitted edge length. There is no force below the rest length.
    if(strength>0){
    float detB=dot3(b[0],cross3(b[1],b[2]));
    float3 rest[4]={mkf3(0,0,0),cross3(b[1],b[2])*(1.f/detB),cross3(b[2],b[0])*(1.f/detB),cross3(b[0],b[1])*(1.f/detB)};
    int fiber=0;
    for(int i=0;i<4;i++)for(int k=i+1;k<4;k++,fiber++){
        float3 edge=p[k]-p[i];float length=len3(edge),restLength=len3(rest[k]-rest[i]);
        float strain=fmaxf(length/restLength-1.f,0.f),gradient=3.f*strain*strain/restLength;
        float wi=invMass[id[i]],wk=invMass[id[k]],alpha=alphaDev[e]/fmaxf(strength,1e-10f);
        float& lambda=fiberLambda[6*e+fiber];
        if(strength>0 && strain>0 && length>0){
            float dl=(-strain*strain*strain-alpha*lambda)/((wi+wk)*gradient*gradient+alpha);
            float3 g=edge*(gradient/length);lambda+=dl;p[i]=p[i]-g*(wi*dl);p[k]=p[k]+g*(wk*dl);
        }
    }
    }
    for(int k=0;k<4;k++)pos[id[k]]=p[k];
    // Volume response is exclusively the finite bulk energy below. In particular,
    // do not add a zero-compliance [minimum,1.05] projection on top of it.

}

__global__ void k_TensileReinforcement(int off,int count,const int* order,const int* ids,const int* active,
    const float3* b0,const float3* b1,const float3* b2,const float* mass,float3* pos,const float* alphaDev,float* fiberLambda,float strength){
    int k=off+blockIdx.x*blockDim.x+threadIdx.x;
    if(k<off+count)projectTensile(order[k],ids,active,b0,b1,b2,mass,pos,alphaDev,fiberLambda,strength);
}

// Eight parity colours: no two vertices of one colour belong to the same
// background tetrahedron. With the other three vertices fixed, signed volume
// is LINEAR in this vertex displacement. Restrict only the dangerous vertex,
// rather than freezing the entire organ because one local element is compressed.
// This filters the complete substep (including anchors/contact), and post-repair.
// Current cut liveness is checked for every incident tet.
__global__ void k_FeasibleVertex(int n,int nx,int ny,int color,const int* offsets,const int* incident,
    const int* ids,const int* active,const float3* b0,const float3* b1,const float3* b2,
    const float3* target,float3* pos,const int* needed,float minimum){
    // Dispatch only this colour's lattice sites (rather than seven idle threads
    // for every useful one). Same independent vertices, same update order.
    int lane=blockIdx.x*blockDim.x+threadIdx.x,cx=(nx+1)/2,cy=(ny+1)/2,nz=n/(nx*ny);
    int x=2*(lane%cx)+(color&1),y=2*((lane/cx)%cy)+((color>>1)&1),z=2*(lane/(cx*cy))+((color>>2)&1);
    if(x>=nx || y>=ny || z>=nz)return;
    int i=x+nx*(y+ny*z);if(!needed[i])return;
    float3 delta=target[i]-pos[i];float accepted=1.f;
    if(!isfinite(delta.x+delta.y+delta.z))return;
    for(int k=offsets[i];k<offsets[i+1];k++){
        int t=incident[k];if(!active[t])continue;
        const int* id=ids+4*t;float3 a=pos[id[0]],u=pos[id[1]]-a,v=pos[id[2]]-a,w=pos[id[3]]-a;
        float detB=dot3(b0[t],cross3(b1[t],b2[t]));
        float j=dot3(u,cross3(v,w))*detB;
        float3 grad[3]={cross3(v,w),cross3(w,u),cross3(u,v)};
        float3 g=i==id[0]?(grad[0]+grad[1]+grad[2])*(-1.f):i==id[1]?grad[0]:i==id[2]?grad[1]:grad[2];
        float dj=dot3(g,delta)*detB;
        if(dj<0){float floor=fminf(minimum,fmaxf(0.f,j));
            if(j+dj<floor)accepted=fminf(accepted,fmaxf(0.f,.95f*(j-floor)/(-dj)));}

    }
    pos[i]=pos[i]+delta*accepted;
}
// Signed tetrahedral volume is multi-affine in four independent vertex
// interpolation parameters. All 16 corners bound every partially applied
// displacement, including the eight-colour order. Only uncertified tets need
// adjacency work; ordinary material vertices take their candidate directly.
__global__ void k_NeedFeasible(int count,const int* ids,const int* active,const float3* b0,const float3* b1,const float3* b2,
    const float3* start,const float3* pos,int* needed,float minimum,const int* candidateUnsafe){
    if(!*candidateUnsafe)return;
    int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=count || !active[e])return;
    float3 x[4],v[4];for(int k=0;k<4;k++){int i=ids[4*e+k];x[k]=start[i];v[k]=pos[i]-x[k];}
    float detB=dot3(b0[e],cross3(b1[e],b2[e]));bool unsafe=false;
    float3 u=x[1]-x[0],vv=x[2]-x[0],w=x[3]-x[0];
    float r0=len3(v[0]),r1=r0+len3(v[1]),r2=r0+len3(v[2]),r3=r0+len3(v[3]);
    float bound=r1*len3(cross3(vv,w))+r2*len3(cross3(w,u))+r3*len3(cross3(u,vv))
        +r1*r2*len3(w)+r1*r3*len3(vv)+r2*r3*len3(u)+r1*r2*r3;
    if(dot3(u,cross3(vv,w))*detB-bound*fabsf(detB)>=minimum+1e-5f)return;
    for(int mask=0;mask<16;mask++){
        float3 p[4];for(int k=0;k<4;k++)p[k]=x[k]+v[k]*((mask>>k)&1);
        float j=dot3(p[1]-p[0],cross3(p[2]-p[0],p[3]-p[0]))*detB;
        if(!isfinite(j) || j<minimum+1e-5f){unsafe=true;break;}
    }
    if(unsafe)for(int k=0;k<4;k++)atomicExch(needed+ids[4*e+k],1);
}
__global__ void k_PrepareFeasible(int n,const int* needed,const float3* start,float3* target,float3* pos){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n && needed[i]){target[i]=pos[i];pos[i]=start[i];}
}
static void beginFeasible(){
    if(d_feasibleStart && g_volumePasses>0)cudaMemcpyAsync(d_feasibleStart,d_pos,(size_t)g_N*sizeof(float3),cudaMemcpyDeviceToDevice);
}
// A simultaneous translation can be feasible even when moving its four
// vertices one at a time is not. Repair only invalid candidate volumes first,
// then accept the complete valid configuration without a coordinate-wise path.
// The colour-ordered tet data of k_RepairMinimumVolume, whose launch over one colour then reads them contiguously.
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
    float3 u=p[1]-p[0],v=p[2]-p[0],w=p[3]-p[0];float detB=colorDet[k];
    float J=dot3(u,cross3(v,w))*detB;if(!isfinite(J) || J>=minimum+1e-4f)return;
    float3 g[4];g[1]=cross3(v,w)*detB;g[2]=cross3(w,u)*detB;g[3]=cross3(u,v)*detB;g[0]=(g[1]+g[2]+g[3])*(-1.f);
    float den=0;for(int k=0;k<4;k++)den+=invMass[id[k]]*dot3(g[k],g[k]);if(den<1e-20f)return;
    float dl=(minimum+2e-4f-J)/den;
    for(int k=0;k<4;k++)pos[id[k]]=p[k]+g[k]*(invMass[id[k]]*dl);
}
__global__ void k_CheckCandidateVolume(int count,const int* ids,const int* active,
    const float3* b0,const float3* b1,const float3* b2,const float3* pos,float minimum,int* unsafe){
    int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=count || !active[t])return;
    float3 p=pos[ids[4*t]],u=pos[ids[4*t+1]]-p,v=pos[ids[4*t+2]]-p,w=pos[ids[4*t+3]]-p;
    float J=dot3(u,cross3(v,w))*dot3(b0[t],cross3(b1[t],b2[t]));
    if(!isfinite(J) || J<minimum)atomicExch(unsafe,1);
}
static cudaError_t finishFeasible(){
    if(!d_feasibleStart || g_volumePasses<=0 || g_T<=0)return cudaSuccess;
    int phase=g_substepIndex&7;
    if(!g_feasibleGraphs[phase]){
        cudaError_t e=cudaSuccess;if(!g_captureStream)e=cudaStreamCreate(&g_captureStream);if(e!=cudaSuccess)return e;
        e=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(e!=cudaSuccess)return e;
        if(d_contactMobility)cudaMemcpyAsync(d_contactMobility,d_pos,(size_t)g_N*sizeof(float3),cudaMemcpyDeviceToDevice,g_captureStream);
        cudaMemsetAsync(d_needFeasible,0,(size_t)(g_N+1)*sizeof(int),g_captureStream);
        k_CheckCandidateVolume<<<(g_T+63)/64,64,0,g_captureStream>>>(g_T,d_tetIds,d_tetActive,d_tetBr0,d_tetBr1,d_tetBr2,d_pos,g_minTetVolume,d_needFeasible+g_N);
        for(int pass=0;pass<2;pass++)for(int c=0;c<g_tNumColors;c++)if(g_tColorCnt[c]>0)
            k_RepairMinimumVolume<<<(g_tColorCnt[c]+63)/64,64,0,g_captureStream>>>(g_tColorOff[c],g_tColorCnt[c],d_tetColorOrder,d_colorTetIds,d_colorTetDet,d_tetActive,d_invMass,d_pos,g_minTetVolume,d_needFeasible+g_N);
        cudaMemsetAsync(d_needFeasible+g_N,0,sizeof(int),g_captureStream);
        k_CheckCandidateVolume<<<(g_T+63)/64,64,0,g_captureStream>>>(g_T,d_tetIds,d_tetActive,d_tetBr0,d_tetBr1,d_tetBr2,d_pos,g_minTetVolume,d_needFeasible+g_N);
        k_NeedFeasible<<<(g_T+63)/64,64,0,g_captureStream>>>(g_T,d_tetIds,d_tetActive,d_tetBr0,d_tetBr1,d_tetBr2,d_feasibleStart,d_pos,d_needFeasible,g_minTetVolume,d_needFeasible+g_N);
        k_PrepareFeasible<<<(g_N+63)/64,64,0,g_captureStream>>>(g_N,d_needFeasible,d_feasibleStart,d_feasibleTarget,d_pos);
        // Same eight-colour order and live cut flags; only CPU submissions change.
        int colorSites=((g_dimsX+2)/2)*((g_dimsY+2)/2)*((g_dimsZ+2)/2);
        for(int k=0;k<8;k++)k_FeasibleVertex<<<(colorSites+63)/64,64,0,g_captureStream>>>(g_N,g_dimsX+1,g_dimsY+1,(k+phase)&7,
            d_incidentOffsets,d_incidentTets,d_tetIds,d_tetActive,d_tetBr0,d_tetBr1,d_tetBr2,d_feasibleTarget,d_pos,d_needFeasible,g_minTetVolume);
        cudaGraph_t graph=nullptr;e=cudaStreamEndCapture(g_captureStream,&graph);if(e!=cudaSuccess)return e;
        e=cudaGraphInstantiate(&g_feasibleGraphs[phase],graph,0);cudaGraphDestroy(graph);if(e!=cudaSuccess)return e;
    }
    return cudaGraphLaunch(g_feasibleGraphs[phase],0);
}

// Decoupled Neo-Hookean energy: mu/2*(J^(-2/3)*I1-3) plus a finite bulk term.
// Guarded production mode uses kappa/2*log(J)^2; legacy mode uses (J-1)^2.
// mu=E/(2*(1+nu)); kappa=E/(3*(1-2*nu)). Both constraints vanish at rest.
// Solve dev/hyd as a coupled 2x2 XPBD block; isochoric shear avoids opposing
// nonzero rest forces. J is regularized only during candidate evaluation;
// finishFeasible protects accepted live tetrahedra.
__global__ void k_solve_tet(int colorOff, int colorCnt, float omega, float dampingOverH, float bulkDampingOverH,
                            const float3* prevPos,
                            const int* tetIds, const float3* Br0, const float3* Br1, const float3* Br2,
                            const int* tetActive, const float* alphaDev, const float* alphaHyd,
                            const float* invMass, float* lambdaDev, float* lambdaHyd, float3* pos,float safeguarded,const int* subset=nullptr,const CoupledTool* localTool=nullptr)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= colorCnt) return;
    int e = subset ? subset[colorOff+t] : colorOff+t;
    if (tetActive[e] == 0) return; // live cut topology, also used by elastic and viscous terms
    int id[4] = {tetIds[4*e], tetIds[4*e+1], tetIds[4*e+2], tetIds[4*e+3]};
    float3 p[4] = {pos[id[0]], pos[id[1]], pos[id[2]], pos[id[3]]};
    if(localTool){
        float3 center=localTool->pose.jawHinge+localTool->pose.jawAxis*(.5f*localTool->pose.jawLen);
        if(len3((p[0]+p[1]+p[2]+p[3])*.25f-center)>3.f*localTool->pose.jawLen)return;
    }
    float3 edge1=p[1]-p[0], edge2=p[2]-p[0], edge3=p[3]-p[0];
    float3 br[3] = {Br0[e], Br1[e], Br2[e]};
    float3 F[3];
    for (int c=0;c<3;c++) F[c] = edge1*compF3(br[0],c) + edge2*compF3(br[1],c) + edge3*compF3(br[2],c);
    float I1=dot3(F[0],F[0])+dot3(F[1],F[1])+dot3(F[2],F[2]);
    float3 cof[3] = {cross3(F[1],F[2]),cross3(F[2],F[0]),cross3(F[0],F[1])};
    float ad=alphaDev[e], ah=alphaHyd[e];
    float J=dot3(F[0],cof[0]),positiveJ=fmaxf(J,.05f);
    float iso=powf(positiveJ,-2.f/3.f);
    float q=fmaxf(iso*I1-3.f,0.f);
    // Isochoric Neo-Hookean energy. Tensile reinforcement is solved with
    // finite compliance in the coloured volume/material sweep.
    float CD=sqrtf(fmaxf(q,1e-12f));
    float devScale=1.f/CD;
    float volumeGradient=J>.05f?iso*I1/(3.f*positiveJ):0.f;
    // Logarithmic bulk strain has the same small-strain kappa, but its pressure
    // grows before a cell collapses. Quadratic (J-1)^2 has finite energy at J=0
    // and left severe squeezes supported mostly by the numerical determinant cap.
    float CH=safeguarded>0?logf(positiveJ)+fminf(0.f,(J-.05f)/.05f):J-1.f;
    float3 gd[4], gh[4];
    for (int k=1;k<4;k++) {
        float3 b=br[k-1];
        gh[k]=cof[0]*b.x + cof[1]*b.y + cof[2]*b.z;
        gd[k]=((F[0]*b.x + F[1]*b.y + F[2]*b.z)*iso-gh[k]*volumeGradient)*devScale;
        if(safeguarded>0)gh[k]=gh[k]*(1.f/positiveJ);
    }
    gd[0]=(gd[1]+gd[2]+gd[3])*(-1.f);
    gh[0]=(gh[1]+gh[2]+gh[3])*(-1.f);
    float dd=0.f, dh=0.f, hh=0.f, vd=0.f, vh=0.f;
    for (int k=0;k<4;k++) {
        float w=invMass[id[k]];
        dd+=w*dot3(gd[k],gd[k]); dh+=w*dot3(gd[k],gh[k]); hh+=w*dot3(gh[k],gh[k]);
        float3 dx=p[k]-prevPos[id[k]];
        vd+=dot3(gd[k],dx); vh+=dot3(gh[k],dx);
    }
    // Kelvin damping only on shape change. Damping the pressure equation
    // delayed volume recovery by ~40x at production h. Divide the deviatoric
    // row by its viscous scale to retain a symmetric positive block.
    float scale=1.f+dampingOverH;
    float bulkScale=safeguarded>0?1.f+bulkDampingOverH:scale;
    float effectiveAd=ad/scale, effectiveAh=ah/bulkScale;
    float A=dd+effectiveAd,B=dh,gamma=B/A;
    float schur=effectiveAh+effectiveAd*gamma*gamma;
    for(int k=0;k<4;k++){
        float3 residual=gh[k]-gd[k]*gamma;
        schur+=invMass[id[k]]*dot3(residual,residual);
    }
    float rd=(-CD-ad*lambdaDev[e]-dampingOverH*vd)/scale;
    float rh=(-CH-ah*lambdaHyd[e]-(bulkScale-1.f)*vh)/bulkScale;
    if(!(A>0) || !(schur>0))return;
    float dlH=omega*(rh-gamma*rd)/schur,dlD=(omega*rd-B*dlH)/A;
    if (!isfinite(dlD) || !isfinite(dlH)) return;
    // Trust region on each nonlinear update, not on deformation/edge length.
    // Scale multipliers with the step so contact forces retain their units.
    if(safeguarded>0){
        float largest=0;
        for(int k=0;k<4;k++)largest=fmaxf(largest,len3((gd[k]*dlD+gh[k]*dlH)*invMass[id[k]]));
        float limit=.15f/fmaxf(len3(br[0]),fmaxf(len3(br[1]),len3(br[2])));
        float accepted=largest>limit?limit/largest:1.f;dlD*=accepted;dlH*=accepted;
    }
    lambdaDev[e]+=dlD; lambdaHyd[e]+=dlH;
    for (int k=0;k<4;k++) pos[id[k]]=p[k]+(gd[k]*dlD+gh[k]*dlH)*invMass[id[k]];


}

// Finite bulk elasticity: C=log(J), alpha=1/(kappa*V*h^2). Pressure and material
// share lambdaHyd, so these iterations converge to one constitutive response.
__global__ void k_SolveBulk(int off,int count,const int* order,const int* ids,const int* active,
    const float3* b0,const float3* b1,const float3* b2,const float* mass,
    const float3* previous,float3* pos,const float* alpha,float* lambda,float damping){
    int k=off+blockIdx.x*blockDim.x+threadIdx.x;if(k>=off+count)return;
    int e=order[k];if(!active[e])return;
    int id[4]={ids[4*e],ids[4*e+1],ids[4*e+2],ids[4*e+3]};
    float3 p[4]={pos[id[0]],pos[id[1]],pos[id[2]],pos[id[3]]},b[3]={b0[e],b1[e],b2[e]},f[3];
    for(int k=0;k<3;k++)f[k]=(p[1]-p[0])*compF3(b[0],k)+(p[2]-p[0])*compF3(b[1],k)+(p[3]-p[0])*compF3(b[2],k);
    float3 cof[3]={cross3(f[1],f[2]),cross3(f[2],f[0]),cross3(f[0],f[1])},g[4];
    float j=dot3(f[0],cof[0]);if(!isfinite(j))return;float positiveJ=fmaxf(j,.05f);
    float strain=logf(positiveJ)+fminf(0.f,(j-.05f)/.05f);
    for(int k=1;k<4;k++)g[k]=(cof[0]*b[k-1].x+cof[1]*b[k-1].y+cof[2]*b[k-1].z)*(1.f/positiveJ);
    g[0]=(g[1]+g[2]+g[3])*(-1.f);float den=0,rate=0;
    for(int k=0;k<4;k++){den+=mass[id[k]]*dot3(g[k],g[k]);rate+=dot3(g[k],p[k]-previous[id[k]]);}
    if(den<=0)return;
    float dl=.8f*(-strain-alpha[e]*lambda[e]-damping*rate)/((1.f+damping)*den+alpha[e]);
    if(!isfinite(dl))return;
    float largest=0;for(int k=0;k<4;k++)largest=fmaxf(largest,len3(g[k])*mass[id[k]]*fabsf(dl));
    float limit=.15f/fmaxf(len3(b[0]),fmaxf(len3(b[1]),len3(b[2])));
    if(largest>limit)dl*=limit/largest;
    lambda[e]+=dl;for(int k=0;k<4;k++)pos[id[k]]=p[k]+g[k]*(mass[id[k]]*dl);
}

__global__ void k_SolveAttachments(int n, float invH2, const float* stiffness,
    const float3* rest, const float* invMass, float3* lambda, float3* pos)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n || stiffness[i]<=0.f || invMass[i]<=0.f) return;
    float a=invH2/stiffness[i], w=invMass[i];
    float3 dl=((pos[i]-rest[i])+lambda[i]*a)*(-1.f/(w+a));
    lambda[i]=lambda[i]+dl;
    pos[i]=pos[i]+dl*w;
}


// Velocity update from the solved positions — RAW, no damping yet. Pinned stay at zero. Split from the
// old fused (raw*velScale) form so the STEP 7 perpendicular cs post-pass (design §4[Structural]) can run
// strictly BETWEEN the raw velocity write and the velScale (Rayleigh/legacy-damping) multiply.
__global__ void k_update_velocity(int N, float invH, const float* invMass,
                                  const float3* pos, const float3* prevPos, float3* vel)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;
    if (invMass[p] > 0.f)
        vel[p] = (pos[p] - prevPos[p]) * invH;
    else
        vel[p] = mkf3(0.f, 0.f, 0.f);
}

// STEP 7 — PERPENDICULAR velocity post-pass (design §4[Structural], the cs damping the along-constraint
// gamma term does NOT reach). Runs AFTER the raw velocity write, BEFORE velScale. Damps only the
// component of the relative velocity PERPENDICULAR to the (current, post-solve) edge direction — the
// along-edge component is already damped by k_solve_edges' gamma term; damping it again here would
// double-damp and break the step-3 cs-isolation golden. Dispatched over the EXISTING structural edge
// colours (one launch per colour, no atomics — within a colour no two edges share a vertex).
// Explicit one-step dashpot impulse per unit mass: vi += -h*wi*cs*vrelPerp (stability margin ~5.5e-3<<2
// at the paper's cs=0.92,m=1,h~1e-3 — design §4[Structural], safe by orders of magnitude).
__global__ void k_perp_damp_edges(int colorOff, int colorCnt, float cs, float h, const float3* pos,
                                  const int* edgeI, const int* edgeJ, const int* edgeActive,
                                  const float* invMass, float3* vel)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= colorCnt) return;
    int e = colorOff + t;
    if (edgeActive[e] == 0) return;
    int i = edgeI[e], j = edgeJ[e];
    float wi = invMass[i], wj = invMass[j];
    if (wi + wj <= 0.f) return;
    float3 d = pos[i] - pos[j];
    float L = len3(d);
    if (L < 1e-9f) return;
    float3 n = d * (1.f / L);
    float3 vrel = vel[i] - vel[j];
    float3 vrelPerp = vrel - n * dot3(vrel, n);
    float3 impulse = vrelPerp * (h * cs);
    if (wi > 0.f) vel[i] = vel[i] - impulse * wi;
    if (wj > 0.f) vel[j] = vel[j] + impulse * wj;
}

// Final post-solve damping multiply: velScale = Rayleigh/legacy-damping (design §12 R-compliance,
// ~h-power). Split out so k_perp_damp_edges (above) runs strictly before it, per design §4[Structural].
__global__ void k_apply_velscale(int N, float velScale, const float* invMass, float3* vel)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;
    vel[p] = (invMass[p] > 0.f) ? vel[p] * velScale : mkf3(0.f, 0.f, 0.f);
}

// ── STEP 8 kernels: contact + two-jaw grasp (design rev-B §4[Contact]/§5/§6) ───────────────────────

// CAPTURE candidate marking (design §5 item 1): one thread per corner. The jaw-region predicate is an
// oriented BOX in the jaw-local (axis,normal,w=axis x normal) frame: axial span [0,jawLen] from the
// hinge, radial |v|<=jawHalfWidth, and within the slab half-thickness + a capture-instant skin (widens
// the band so corners near the CLOSING jaws are included at the capture moment only -- design §5 item 1
// [audit rev-B]: the set is latched ONCE, corners entering later are deliberately NOT captured). Writes
// the candidate's jaw-local (u,v,n) for the host-side ellipse-core computation (a rare, once-per-capture
// event -- cheap to round-trip to host).
__global__ void k_MarkGraspCandidates(int N, float3 hinge, float3 axis, float3 normal, float3 w,
                                      float jawLen, float jawHalfWidth, float gapHalf, float captureSkin,
                                      const int* active, const int* pinned, const float3* pos,
                                      int* candidate, float* outU, float* outV, float* outN)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    candidate[c] = 0;
    if (active[c] == 0 || pinned[c] != 0) return;
    float3 rel = pos[c] - hinge;
    float u = dot3(rel, axis), v = dot3(rel, w), n = dot3(rel, normal);
    if (u < 0.f || u > jawLen) return;
    if (fabsf(v) > jawHalfWidth) return;
    if (fabsf(n) > gapHalf + captureSkin) return;
    candidate[c] = 1;
    outU[c] = u; outV[c] = v; outN[c] = n;
}

// RELEASE (design §5 item core lifecycle): damp the velocity of every currently-grasped corner x0.1
// (GripperTool.cs:600-601 pattern) so the spring-back starts clean, BEFORE the host clears d_grasped/
// d_graspCore (which would otherwise leave this kernel with nothing to identify as "was grasped").
__global__ void k_ReleaseDampVelocity(int N, const int* grasped, float3* vel)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    if (grasped[c] != 0) vel[c] = vel[c] * 0.1f;
}

// Vector XPBD attachment: retaining the direction of lambda avoids accumulating incompatible
// scalar impulses when the target crosses a node. Compliance is scaled by h^2 at dispatch.
__global__ void k_SolveCorePin(int N, float omegaGrasp, float coreAlpha, float graspMaxDisp,
                               float3 hinge, float3 axis, float3 normal, float3 w,
                               const float* offU, const float* offN, const float* offW,
                               const float* invMass, float3* pos, float3* coreLambda,
                               const int* graspCore,int tangentOnly=0,const float* complianceScale=nullptr)
{
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if (c>=N || !graspCore[c] || invMass[c]<=0.f) return;
    float wc=invMass[c];
    if(complianceScale)coreAlpha*=complianceScale[c];
    float3 target=hinge+axis*offU[c]+normal*offN[c]+w*offW[c];
    float3 applied=((pos[c]-target)+coreLambda[c]*coreAlpha)*(-omegaGrasp/(wc+coreAlpha));
    if(tangentOnly)applied=applied-normal*dot3(applied,normal);
    float corrLen=len3(applied)*wc;
    if (corrLen>graspMaxDisp) applied=applied*(graspMaxDisp/corrLen);
    coreLambda[c]=coreLambda[c]+applied;
    pos[c]=pos[c]+applied*wc;
}

// Low-pass the physical force with a time constant of one second, independent of substep count.
// A single tool reposition cannot masquerade as sustained overload. Actual tissue stress still
// controls breakage; this filter is not a cure for an unstable target trajectory.
__global__ void k_CoreBreakCheck(int N, float coreBreakForce, float h,
                                 const float3* coreLambda, float* coreForceEMA,
                                 int* graspCore, int* grasped)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    if (graspCore[c] == 0) return;
    const float kEMA = -expm1f(-h);   // see kernel header derivation above
    float FcoreRaw = len3(coreLambda[c]) / (h * h);
    coreForceEMA[c] += (FcoreRaw - coreForceEMA[c]) * kEMA;
    if (coreForceEMA[c] > coreBreakForce) {
        graspCore[c] = 0;
        grasped[c]   = 0;
        // NOTE: an explicit vel[c]*=0.1 "clean release" cushion (matching k_ReleaseDampVelocity) was
        // attempted here in an earlier revision and found to be dead code: this kernel runs BEFORE
        // xpbd_substep's own k_update_velocity call, which unconditionally overwrites vel from
        // (pos-prevPos)/h every substep -- any write to vel here is clobbered within the same substep.
        // k_ReleaseDampVelocity's damping (xpbd_set_tool, host-side, once per FRAME on manual release)
        // does NOT have this problem since it runs strictly between one substep's k_update_velocity and
        // the next's k_predict. Achieving the same cushion for an in-substep automatic break would
        // require rewriting d_prevPos (not d_vel) so the upcoming k_update_velocity computes the damped
        // value -- deferred as a minor, narrow polish item (Task 9 checklist), not attempted in this fix.
    }
}

// CCD ANTI-TUNNELLING GATE (redesign 2026-07-07, Option A — supersedes the old hard
// k_SolveJawShaftContact non-penetration constraint). The paper's Finger-Proxy (haptics.cu) is the
// ONLY tissue non-penetration authority now — this kernel exists SOLELY to stop a fast-moving corner
// from tunnelling clean through the thin shaft/jaw geometry between substeps, not to eject anything
// that is merely sitting inside the tool. Four mandatory rules (design spec's own words):
//   (a) fires ONLY when the corner's OWN velocity component toward the tool exceeds ccdVelThresh —
//       a statically-embedded corner (near-zero approach speed) gets ZERO correction;
//   (b) caps the applied per-substep correction at ccdMaxDisp regardless of nominal penetration depth;
//   (c) uses its OWN ccdOmega, independent of xp.omega (structural/tet) and omegaGrasp (core/slab);
//   (d) early-outs on wc<=0 (pinned/inactive) for BOTH branches, and on grasped[c]!=0 for the JAW-SLAB
//       branch ONLY (not the shaft-capsule branch) — "never fights the compliant core-pin or a pinned
//       neighbour" (design spec's own words) applies specifically to the jaw-slab box test, which
//       operates in the SAME jaw-local coordinate frame k_SolveSlabCoulomb/k_SolveCorePin already own
//       (a genuine double-solve risk); the shaft capsule is a completely separate world-space geometric
//       primitive neither grasp kernel ever tests against, so there is no double-solve risk there to
//       avoid. The OLD (retired) kernel protected EVERY corner against the shaft, grasped or not, and
//       neither k_SolveCorePin nor k_SolveSlabCoulomb has ANY shaft-capsule awareness — excluding
//       grasped corners from the shaft branch too would remove their shaft-tunnelling protection
//       entirely, a regression vs. the design this replaces, not a narrowing.
//
// Rule (b)'s cap MUST scale lambda by the SAME factor as the position correction, not cap position
// alone while accumulating an uncapped lambda. This codebase's own load-bearing invariant
// (k_solve_edges's comment: "omega applied CONSISTENTLY to both lambda and position, so the compliant
// fixed point C=-alpha~*lambda stays omega-INVARIANT") requires this; capping only the position
// silently breaks that identity and makes shaftLambda/jawLambda read HIGHER than what was actually
// applied whenever the cap engages (every other constraint kernel in this file — edges, bending, tet,
// and k_SolveSlabCoulomb's STICK/gap-closing branches — honors this identity without exception).
//
// Known limitations (deliberately NOT further hardened, see the redesign spec's discussion): this is a
// discrete per-substep position check, not a true swept/continuous (TOI) test — a corner whose motion
// carries it from clearly-outside on one side to clearly-outside on the other side within a single
// substep is invisible to this check. The shaft-capsule branch's version of this gap requires a corner
// to move more than 2*shaftR in one substep (~400-1800 world-units/second at real scene scale) — 100x+
// beyond anything this codebase's force sources (gravity, elastic/tet solve, damping, a core-pin
// break's transient force) can produce, and the tool geometry is frozen for an entire frame's substeps
// (only xpbd_set_tool, once per frame, changes it), so it is not reachable in practice. The jaw-slab
// branch's version has a lower threshold (tens to ~160 u/s) and is reachable via a violent tension
// spike right before a core-pin break — narrowed (not fully closed) by the grasped[c]==0 exclusion
// during the whole tension-buildup phase plus the break-path velocity damping in k_SolveCorePin below.
// Also: this kernel tests the PHYSICAL tool position (xpbd_set_tool's raw input), not haptics.cu's
// already-surface-clamped PROXY position — a deliberate, considered choice (the proxy's guarantees
// never constrain the underlying simulated mesh corners this kernel protects), not an oversight.
__global__ void k_SolveCCDContact(int N, float ccdOmega, float ccdVelThresh, float ccdMaxDisp, float contactAlpha,
                                  float3 shaftA, float3 shaftB, float shaftR,
                                  float3 hinge, float3 axis, float3 normal, float3 w,
                                  float jawLen, float jawHalfWidth, float gapHalf,
                                  const int* grasped, const float* invMass, const float3* vel, float3* pos,
                                  float* jawLambda, float* shaftLambda)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    float wc = invMass[c];
    if (wc <= 0.f) return;                                       // rule (d): pinned/inactive (both branches)

    // -- shaft capsule (closest-point-on-segment, reimplemented independently of haptics.cu). Runs for
    // EVERY corner including grasped ones (see kernel header) -- this is the ONLY shaft-tunnelling
    // protection in the whole solver; neither grasp kernel checks the shaft at all. --
    {
        float3 ab = shaftB - shaftA;
        float ab2 = dot3(ab, ab);
        float tt = ab2 > 1e-12f ? dot3(pos[c] - shaftA, ab) / ab2 : 0.f;
        tt = fmaxf(0.f, fminf(1.f, tt));
        float3 closest = shaftA + ab * tt;
        float3 dvec = pos[c] - closest;
        float dist = len3(dvec);
        float C = dist - shaftR;
        if (C < 0.f) {
            float3 n = dist > 1e-9f ? dvec * (1.f / dist) : mkf3(0.f, 1.f, 0.f);
            float vApproach = -dot3(vel[c], n);                  // rule (a): speed moving INTO the tool
            if (vApproach > ccdVelThresh) {
                float denom = wc + contactAlpha;
                float dLambda = (-C - contactAlpha * shaftLambda[c]) / denom;
                float applied = ccdOmega * dLambda;               // rule (c)
                float3 corr = n * (wc * applied);
                float corrLen = len3(corr);
                // scale BOTH applied (before lambda accumulation) and corr by the SAME factor when the
                // cap engages, instead of capping corr alone (rule b) while leaving shaftLambda uncapped.
                if (corrLen > ccdMaxDisp) {
                    float capScale = ccdMaxDisp / corrLen;
                    applied *= capScale;
                    corr = corr * capScale;
                }
                shaftLambda[c] += applied;
                pos[c] = pos[c] + corr;
            }
        }
    }

    // -- jaw slab as a contact BOX (uncaptured corners caught inside the closing jaws only) --
    if (grasped[c] == 0) {
        float3 rel = pos[c] - hinge;
        float u = dot3(rel, axis), v = dot3(rel, w), n = dot3(rel, normal);
        if (u >= 0.f && u <= jawLen && fabsf(v) <= jawHalfWidth) {
            float absN = fabsf(n);
            if (absN > gapHalf) {
                float Ctrue = gapHalf - absN;
                float sgn = (n >= 0.f) ? 1.f : -1.f;
                float3 nOut = normal * sgn;                       // outward direction for this branch
                float vApproach = -dot3(vel[c], nOut);            // rule (a)
                if (vApproach > ccdVelThresh) {
                    float denom = wc + contactAlpha;
                    float dLambda = (-Ctrue - contactAlpha * jawLambda[c]) / denom;
                    float applied = ccdOmega * dLambda;           // rule (c)
                    float corrMag = wc * applied;
                    // same consistent-scaling fix as the shaft branch above.
                    if (fabsf(corrMag) > ccdMaxDisp) {
                        float capScale = ccdMaxDisp / fabsf(corrMag);
                        applied *= capScale;
                        corrMag *= capScale;
                    }
                    jawLambda[c] += applied;
                    pos[c] = pos[c] - normal * (sgn * corrMag);
                }
            }
        }
    }
}

// GRASP (design §5 items 2-3): ring corners ONLY (grasped && !core) -- the slab compression band
// [-gapHalf,+gapHalf] (same two-sided box math as the jaw-contact case above, but LIVE for every
// captured ring corner rather than gated to "uncaptured") plus the Coulomb tangential stick/slip anchor,
// Fn = |slabLambdaPrev|/h^2 from the PREVIOUS substep (breaks the chicken-and-egg, design §5 item 3).
// The anchor is stored in the JAW-LOCAL (u,w) frame so a translating/rotating jaw carries it along
// (no false slip from rigid tool motion); only genuine RELATIVE tissue motion trips the cone.
__global__ void k_SolveSlabCoulomb(int N, float omega, float h, float contactAlpha, float mu, float graspMaxDisp,
                                   float3 hinge, float3 axis, float3 normal, float3 w,
                                   float jawLen, float jawHalfWidth, float gapHalf,
                                   const int* grasped, const int* graspCore, const float* invMass,
                                   const float* slabLambdaPrev, float* slabLambda,
                                   float* anchorU, float* anchorW, float3* pos)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    if (grasped[c] == 0 || graspCore[c] != 0) return;
    float wc = invMass[c];
    if (wc <= 0.f) return;                     // active[]==0 re-filter (cut severed it mid-grasp, §5)

    float3 rel = pos[c] - hinge;
    float u = dot3(rel, axis), v = dot3(rel, w), n = dot3(rel, normal);

    if (u >= 0.f && u <= jawLen && fabsf(v) <= jawHalfWidth) {
        float absN = fabsf(n);
        if (absN > gapHalf) {
            float Ctrue = gapHalf - absN;
            float sgn = (n >= 0.f) ? 1.f : -1.f;
            float denom = wc + contactAlpha;
            float dLambda = (-Ctrue - contactAlpha * slabLambda[c]) / denom;
            float applied = omega * dLambda;
            // redesign 2026-07-07: this normal/gap-closing branch is STRUCTURALLY IDENTICAL to
            // k_SolveCCDContact's jaw-slab branch (same Ctrue/sgn/dLambda/applied pattern) -- capped
            // the SAME way, scaling BOTH applied (before lambda accumulation) and the position
            // correction by the identical factor when the cap engages (see k_SolveCCDContact's header
            // for why capping position alone while leaving lambda uncapped is wrong).
            float corrMag = wc * applied;
            if (fabsf(corrMag) > graspMaxDisp) {
                float capScale = graspMaxDisp / fabsf(corrMag);
                applied *= capScale;
                corrMag *= capScale;
            }
            slabLambda[c] += applied;
            pos[c] = pos[c] - normal * (sgn * corrMag);
            rel = pos[c] - hinge; n = dot3(rel, normal);   // refresh n for the anchor reconstruction below
        }
    }

    float Fn = fabsf(slabLambdaPrev[c]) / (h * h);
    float3 anchorWorld = hinge + axis * anchorU[c] + w * anchorW[c] + normal * n;  // normal = slab's job
    float3 tangential = pos[c] - anchorWorld;                                     // already in-plane
    float dist = len3(tangential);
    // coneRadius: the position-space correction a friction force Fn sustains this substep, via the SAME
    // w*h^2 XPBD force<->position relationship the constraint solve itself uses (a documented,
    // dimensionally-grounded choice -- design §5 item 3 leaves the exact conversion to the implementer;
    // tune mu against the video's slip-out feel, §11 item 6).
    float coneRadius = mu * Fn * (h * h * wc);
    if (dist <= coneRadius + 1e-9f) {
        // redesign 2026-07-07: STICK is now a genuinely COMPLIANT pull, not a rigid full-snap — omega
        // (now omegaGrasp, passed in from the call site) under-relaxes this correction exactly like
        // every other XPBD constraint's dLambda*omega pattern, so the solve converges over the K
        // coupling passes instead of crawling a stiff instantaneous snap each substep. Capped at
        // graspMaxDisp, same Stability-Measure-D rationale as the branch above.
        float3 corr = tangential * omega;
        float corrLen = len3(corr);
        if (corrLen > graspMaxDisp) corr = corr * (graspMaxDisp / corrLen);
        pos[c] = pos[c] - corr;                              // STICK: compliant, capped in-plane attachment
    } else {
        float3 dir = tangential * (1.f / dist);
        pos[c] = pos[c] - dir * coneRadius;                  // SLIP: cap the correction at the cone
        float3 newAnchor = anchorWorld + dir * (dist - coneRadius);   // partial-advance (haptics.cu:588
        float3 newRel = newAnchor - hinge;                             // pattern, reimplemented) so the
        anchorU[c] = dot3(newRel, axis);                               // anchor keeps pulling at mu*Fn
        anchorW[c] = dot3(newRel, w);                                  // DURING the slide.
    }
}

// ── host API ────────────────────────────────────────────────────────────────────────────────────
#define XK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { xpbd_shutdown(); return -(int)_e; } } while (0)

extern "C" int xpbd_init(int particleCount, const float* pos3, const float* invMass,
                         int edgeCount, const int* edges, const float* restLen)
{
    // Validate args + run the colouring BEFORE tearing down any existing instance (fail-preserving:
    // a bad re-init must not destroy a working sim; only a DEVICE failure below can — see header).
    if (particleCount <= 0 || edgeCount < 0 || !pos3 || !invMass) return XPBD_ERR_BAD_ARG;
    if (edgeCount > 0 && (!edges || !restLen)) return XPBD_ERR_BAD_ARG;   // restLen REQUIRED with edges —
                                                   // a silent rest-0 default would collapse every spring (C2)

    // ── greedy edge colouring (host): no two edges in a colour share a vertex ─────────────────────
    // 64-bit used-colour bitmask per vertex; for a 6-neighbour lattice max degree 6 -> <=7 colours.
    std::vector<unsigned long long> vmask(particleCount, 0ull);
    std::vector<int> edgeColor(edgeCount, 0);
    int numColors = 0;
    for (int e = 0; e < edgeCount; e++) {
        int i = edges[2*e+0], j = edges[2*e+1];
        if (i < 0 || i >= particleCount || j < 0 || j >= particleCount) return XPBD_ERR_BAD_ARG;
        unsigned long long used = vmask[i] | vmask[j];
        int c = 0; while (c < 64 && (used & (1ull << c))) c++;
        if (c >= 64) return XPBD_ERR_COLOR_OVERFLOW; // colour budget exhausted (vertex degree >= 64): FAIL
                                                   // LOUDLY rather than aliasing to a used colour (which would
                                                   // race the no-atomics GS). Unreachable for a 6-neighbour
                                                   // lattice (deg<=6). Existing instance is PRESERVED.
        edgeColor[e] = c;
        vmask[i] |= (1ull << c); vmask[j] |= (1ull << c);
        if (c + 1 > numColors) numColors = c + 1;
    }
    if (g_ready) xpbd_shutdown();
    g_N = particleCount; g_E = edgeCount;
    XK(allocateQueryGroups(g_N));
    g_numColors = numColors;
    // bucket edges by colour -> a flat colour-ordered permutation with per-colour [off,cnt).
    g_colorCnt.assign(numColors, 0);
    for (int e = 0; e < g_E; e++) g_colorCnt[edgeColor[e]]++;
    g_colorOff.assign(numColors, 0);
    for (int c = 1; c < numColors; c++) g_colorOff[c] = g_colorOff[c-1] + g_colorCnt[c-1];
    std::vector<int>   hEdgeI(g_E), hEdgeJ(g_E);
    std::vector<float> hRest(g_E);
    std::vector<int>   hActive(g_E, 1);            // every edge live; a cut/tear flips entries via the notify API
    std::vector<int>   hSrc(g_E);                  // M1: colour-slot -> input index (persisted, not discarded)
    std::vector<int>   cursor = g_colorOff;
    g_edgeSlotOfInput.assign(g_E, -1);
    for (int e = 0; e < g_E; e++) {
        int c = edgeColor[e]; int dst = cursor[c]++;
        hEdgeI[dst] = edges[2*e+0]; hEdgeJ[dst] = edges[2*e+1]; hRest[dst] = restLen ? restLen[e] : 0.f;
        hSrc[dst] = e; g_edgeSlotOfInput[e] = dst;
    }

    // ── allocate + upload ─────────────────────────────────────────────────────────────────────────
    XK(cudaMalloc(&d_pos,     (size_t)g_N * sizeof(float3)));
    XK(cudaMalloc(&d_prevPos, (size_t)g_N * sizeof(float3)));
    XK(cudaMalloc(&d_vel,     (size_t)g_N * sizeof(float3)));
    XK(cudaMalloc(&d_invMass, (size_t)g_N * sizeof(float)));
    if (g_E > 0) {
        XK(cudaMalloc(&d_edgeI,     (size_t)g_E * sizeof(int)));
        XK(cudaMalloc(&d_edgeJ,     (size_t)g_E * sizeof(int)));
        XK(cudaMalloc(&d_restLen,   (size_t)g_E * sizeof(float)));
        XK(cudaMalloc(&d_lambda,    (size_t)g_E * sizeof(float)));
        XK(cudaMalloc(&d_edgeActive,(size_t)g_E * sizeof(int)));
        XK(cudaMalloc(&d_edgeSrcIdx,(size_t)g_E * sizeof(int)));
    }
    XK(cudaMemcpy(d_pos,     pos3,    (size_t)g_N * sizeof(float3), cudaMemcpyHostToDevice));
    XK(cudaMemcpy(d_invMass, invMass, (size_t)g_N * sizeof(float),  cudaMemcpyHostToDevice));
    XK(cudaMemset(d_vel,     0, (size_t)g_N * sizeof(float3)));
    XK(cudaMemcpy(d_prevPos, pos3,    (size_t)g_N * sizeof(float3), cudaMemcpyHostToDevice));
    if (g_E > 0) {
        XK(cudaMemcpy(d_edgeI,     hEdgeI.data(),  (size_t)g_E * sizeof(int),   cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_edgeJ,     hEdgeJ.data(),  (size_t)g_E * sizeof(int),   cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_restLen,   hRest.data(),   (size_t)g_E * sizeof(float), cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_edgeActive,hActive.data(), (size_t)g_E * sizeof(int),   cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_edgeSrcIdx,hSrc.data(),    (size_t)g_E * sizeof(int),   cudaMemcpyHostToDevice));
        XK(cudaMemset(d_lambda, 0, (size_t)g_E * sizeof(float)));
    }
    g_frameReady = false;                          // params/topology changed: caller must begin a new frame
    g_ready = true;
    return 0;
}

// ── M1 notify path (design §9-(4)): propagate a runtime sever / mass change through the PERSISTED
// input->colour-slot permutation into the colour-ordered device flags. At the rebind step this is
// superseded by k_RefreshLiveness (deriving liveness from the LIVE shared nbrIdx/bendPairs/active/mass
// every substep); the notify API remains for the standalone self-test + targeted host-side control. ────
extern "C" int xpbd_set_edge_active(int inputEdgeIndex, int active)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    if (inputEdgeIndex < 0 || inputEdgeIndex >= (int)g_edgeSlotOfInput.size()) return XPBD_ERR_BAD_ARG;
    int slot = g_edgeSlotOfInput[inputEdgeIndex];
    int v = active ? 1 : 0;
    int r = -(int)cudaMemcpy(d_edgeActive + slot, &v, sizeof(int), cudaMemcpyHostToDevice);
    markLivenessStale();
    return r;
}

extern "C" int xpbd_set_bend_active(int inputBendIndex, int active)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    if (inputBendIndex < 0 || inputBendIndex >= (int)g_bendSlotOfInput.size()) return XPBD_ERR_BAD_ARG;
    int slot = g_bendSlotOfInput[inputBendIndex];
    int v = active ? 1 : 0;
    int r = -(int)cudaMemcpy(d_bendActive + slot, &v, sizeof(int), cudaMemcpyHostToDevice);
    markLivenessStale();
    return r;
}

extern "C" int xpbd_set_particle_inv_mass(int particle, float invMass)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    if (particle < 0 || particle >= g_N) return XPBD_ERR_BAD_ARG;
    int r = -(int)cudaMemcpy(d_invMass + particle, &invMass, sizeof(float), cudaMemcpyHostToDevice);
    markLivenessStale();
    return r;
}

extern "C" void xpbd_set_params(const XpbdParams* p) { if (p) { g_p = *p; g_frameReady = false; } }
                                                       // params changed -> h-derived coefficients stale;
                                                       // force a fresh xpbd_begin_frame (R-compliance)
extern "C" int  xpbd_color_count() { return g_numColors; }
extern "C" int  xpbd_bend_color_count() { return g_bNumColors; }

// §9-(4) sever-propagation VERIFY diagnostics ("readback d_edgeActive").
extern "C" int xpbd_get_edge_count() { return g_ready ? g_E : XPBD_ERR_NOT_READY; }
extern "C" int xpbd_readback_edge_active(int* out)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    if (!out) return XPBD_ERR_BAD_ARG;
    if (g_E == 0) return 0;
    return -(int)cudaMemcpy(out, d_edgeActive, (size_t)g_E * sizeof(int), cudaMemcpyDeviceToHost);
}

extern "C" int xpbd_set_bending(int bendCount, const int* ijk, const float* theta0)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    // ONLY an explicit bendCount==0 clears the set; a NEGATIVE count (e.g. an underflowed size) is a
    // caller bug and must FAIL LOUD, not silently wipe a working bend set (fail-preserving contract).
    // theta0 is REQUIRED alongside ijk — a silent theta0=0 default would fold the rest shape (C2).
    if (bendCount < 0 || (bendCount > 0 && (!ijk || !theta0))) return XPBD_ERR_BAD_ARG;

    // greedy colouring over the THREE vertices of each element (share ANY vertex -> different colour),
    // run BEFORE the existing bend set is freed (fail-preserving: a colour-overflow / bad-index failure
    // leaves the OLD set intact and the module fully usable — see the header failure semantics).
    std::vector<int> col;
    int numColors = 0;
    if (bendCount > 0) {
        std::vector<unsigned long long> vmask(g_N, 0ull);
        col.assign(bendCount, 0);
        for (int e = 0; e < bendCount; e++) {
            int i = ijk[3*e+0], j = ijk[3*e+1], k = ijk[3*e+2];
            if (i < 0 || i >= g_N || j < 0 || j >= g_N || k < 0 || k >= g_N) return XPBD_ERR_BAD_ARG;
            unsigned long long used = vmask[i] | vmask[j] | vmask[k];
            int c = 0; while (c < 64 && (used & (1ull << c))) c++;
            if (c >= 64) return XPBD_ERR_COLOR_OVERFLOW;   // fail loudly; OLD bend set preserved
            col[e] = c; vmask[i] |= (1ull<<c); vmask[j] |= (1ull<<c); vmask[k] |= (1ull<<c);
            if (c+1 > numColors) numColors = c+1;
        }
    }

    // colouring succeeded (or clearing requested): free any prior bending set + install the new one.
    cudaFree(d_bendI); cudaFree(d_bendJ); cudaFree(d_bendK);
    cudaFree(d_bendTheta0); cudaFree(d_bendActive); cudaFree(d_bendLambda); cudaFree(d_bendSrcIdx);
    d_bendI=d_bendJ=d_bendK=d_bendActive=d_bendSrcIdx=nullptr; d_bendTheta0=d_bendLambda=nullptr;
    g_B = 0; g_bColorOff.clear(); g_bColorCnt.clear(); g_bNumColors = 0; g_bendSlotOfInput.clear();
    g_frameReady = false;                          // topology changed: caller must begin a new frame
    if (bendCount <= 0) return 0;
    g_B = bendCount;
    g_bNumColors = numColors;
    g_bColorCnt.assign(numColors, 0);
    for (int e = 0; e < g_B; e++) g_bColorCnt[col[e]]++;
    g_bColorOff.assign(numColors, 0);
    for (int cc = 1; cc < numColors; cc++) g_bColorOff[cc] = g_bColorOff[cc-1] + g_bColorCnt[cc-1];
    std::vector<int> hI(g_B), hJ(g_B), hK(g_B), hAct(g_B, 1), hSrcB(g_B); std::vector<float> hT(g_B);
    std::vector<int> cur = g_bColorOff;
    g_bendSlotOfInput.assign(g_B, -1);
    for (int e = 0; e < g_B; e++) {
        int dst = cur[col[e]]++;
        hI[dst]=ijk[3*e+0]; hJ[dst]=ijk[3*e+1]; hK[dst]=ijk[3*e+2]; hT[dst]=theta0?theta0[e]:0.f;
        hSrcB[dst] = e; g_bendSlotOfInput[e] = dst;
    }
    // On any alloc/upload failure: free the partial bend buffers + reset bend state before returning
    // (mirror step-1's XK-calls-shutdown pattern, so a failed set leaves NO half-allocated set that a
    // subsequent xpbd_step could dispatch k_solve_bending against with a null buffer — review minor).
    #define XB(call) do { cudaError_t _e=(call); if (_e!=cudaSuccess) { \
        cudaFree(d_bendI); cudaFree(d_bendJ); cudaFree(d_bendK); \
        cudaFree(d_bendTheta0); cudaFree(d_bendActive); cudaFree(d_bendLambda); cudaFree(d_bendSrcIdx); \
        d_bendI=d_bendJ=d_bendK=d_bendActive=d_bendSrcIdx=nullptr; d_bendTheta0=d_bendLambda=nullptr; \
        g_B=0; g_bNumColors=0; g_bColorOff.clear(); g_bColorCnt.clear(); g_bendSlotOfInput.clear(); \
        return -(int)_e; } } while(0)
    XB(cudaMalloc(&d_bendI,(size_t)g_B*sizeof(int)));      XB(cudaMalloc(&d_bendJ,(size_t)g_B*sizeof(int)));
    XB(cudaMalloc(&d_bendK,(size_t)g_B*sizeof(int)));      XB(cudaMalloc(&d_bendTheta0,(size_t)g_B*sizeof(float)));
    XB(cudaMalloc(&d_bendActive,(size_t)g_B*sizeof(int))); XB(cudaMalloc(&d_bendLambda,(size_t)g_B*sizeof(float)));
    XB(cudaMalloc(&d_bendSrcIdx,(size_t)g_B*sizeof(int)));
    XB(cudaMemcpy(d_bendI,hI.data(),(size_t)g_B*sizeof(int),cudaMemcpyHostToDevice));
    XB(cudaMemcpy(d_bendJ,hJ.data(),(size_t)g_B*sizeof(int),cudaMemcpyHostToDevice));
    XB(cudaMemcpy(d_bendK,hK.data(),(size_t)g_B*sizeof(int),cudaMemcpyHostToDevice));
    XB(cudaMemcpy(d_bendTheta0,hT.data(),(size_t)g_B*sizeof(float),cudaMemcpyHostToDevice));
    XB(cudaMemcpy(d_bendActive,hAct.data(),(size_t)g_B*sizeof(int),cudaMemcpyHostToDevice));
    XB(cudaMemcpy(d_bendSrcIdx,hSrcB.data(),(size_t)g_B*sizeof(int),cudaMemcpyHostToDevice));
    XB(cudaMemset(d_bendLambda,0,(size_t)g_B*sizeof(float)));
    markLivenessStale();
    #undef XB
    return 0;
}

// ── REBIND / GO-LIVE (design rev-B §9-(4)) ──────────────────────────────────────────────────────────
// Builds the module's topology FROM the live shared buffers (one-time device readback), reusing the
// audited xpbd_init/xpbd_set_bending machinery (validation -> colouring -> permutation -> install) on
// host lists, then CONVERTS to bound mode: the private pos/vel are freed and replaced by ALIASES of the
// shared cornerPos/vel; the owner-slot and live-source maps are uploaded for k_RefreshLiveness.
// Fail-preserving up through the colouring (inherited from xpbd_init); a DEVICE failure during install/
// conversion leaves the module shut down (header semantics).
extern "C" int xpbd_bind(float3* cornerPos, float3* vel,
                         const int* nbrIdx, const float3* restNbr,
                         const void* bendPairsRaw, int bendCapacity,
                         const int* active, const int* pinned,
                         const float* mass,
                         const float3* extForce, const float3* hapticForce,
                         int dimsX, int dimsY, int dimsZ, int cornerCount)
{
    if (!cornerPos || !vel || !nbrIdx || !restNbr || !active || !pinned || !mass || !extForce ||
        !hapticForce || cornerCount <= 0) return XPBD_ERR_BAD_ARG;
    if (bendCapacity < 0 || (bendCapacity > 0 && !bendPairsRaw)) return XPBD_ERR_BAD_ARG;
    const int cc = cornerCount;

    // 1) One-time READBACK of the live topology + state (no module state touched yet — a readback
    //    failure preserves any existing instance).
    std::vector<int>             hNbr((size_t)6 * cc);
    std::vector<float3>          hRestNbr((size_t)6 * cc);
    std::vector<float3>          hPos(cc);
    std::vector<int>             hActive(cc), hPinned(cc);
    std::vector<float>           hMass(cc);
    std::vector<XpbdBendPairGpu> hBp(bendCapacity > 0 ? bendCapacity : 1);
    #define XR(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) return -(int)_e; } while (0)
    XR(cudaMemcpy(hNbr.data(),     nbrIdx,    (size_t)6 * cc * sizeof(int),    cudaMemcpyDeviceToHost));
    XR(cudaMemcpy(hRestNbr.data(), restNbr,   (size_t)6 * cc * sizeof(float3), cudaMemcpyDeviceToHost));
    XR(cudaMemcpy(hPos.data(),     cornerPos, (size_t)cc * sizeof(float3),     cudaMemcpyDeviceToHost));
    XR(cudaMemcpy(hActive.data(),  active,    (size_t)cc * sizeof(int),        cudaMemcpyDeviceToHost));
    XR(cudaMemcpy(hPinned.data(),  pinned,    (size_t)cc * sizeof(int),        cudaMemcpyDeviceToHost));
    XR(cudaMemcpy(hMass.data(),    mass,      (size_t)cc * sizeof(float),      cudaMemcpyDeviceToHost));
    if (bendCapacity > 0)
        XR(cudaMemcpy(hBp.data(), bendPairsRaw, (size_t)bendCapacity * sizeof(XpbdBendPairGpu),
                      cudaMemcpyDeviceToHost));
    #undef XR

    // 2) CANONICAL EDGE DEDUP (design §9-(4)): emit each lattice edge ONCE from its POSITIVE slot
    //    s = 2*axis+1 (slot order 0=-x,1=+x,2=-y,3=+y,4=-z,5=+z — cut.cu k_SeverLinks writes exactly
    //    these). Emitted regardless of active[] — liveness is PER-SUBSTEP (k_RefreshLiveness), the
    //    topology is static. ownerSlot[e] = 6p+s is what the refresh reads back through.
    std::vector<int>   hEdges;  hEdges.reserve((size_t)3 * cc * 2);
    std::vector<float> hRest;   hRest.reserve((size_t)3 * cc);
    std::vector<int>   hOwner;  hOwner.reserve((size_t)3 * cc);
    for (int p = 0; p < cc; p++)
        for (int axis = 0; axis < 3; axis++) {
            int s = 2 * axis + 1;
            int n = hNbr[(size_t)6 * p + s];
            if (n < 0) continue;
            if (n >= cc) return XPBD_ERR_BAD_ARG;
            float3 r = hRestNbr[(size_t)6 * p + s];
            hEdges.push_back(p); hEdges.push_back(n);
            hRest.push_back(sqrtf(r.x * r.x + r.y * r.y + r.z * r.z));   // L0 = length3(restNbr[6p+s])
            hOwner.push_back(6 * p + s);
        }
    const int E = (int)hRest.size();

    // 3) BEND ELEMENTS from the live pairs that are ALIVE at bind time (a sever is permanent — dead
    //    pairs can never return, so they are not emitted). srcIdx = index into the LIVE bendPairs array.
    std::vector<int>   hIjk;  hIjk.reserve((size_t)bendCapacity * 3);
    std::vector<float> hTh;   hTh.reserve(bendCapacity);
    std::vector<int>   hSrcLive; hSrcLive.reserve(bendCapacity);
    for (int b = 0; b < bendCapacity; b++) {
        const XpbdBendPairGpu& bp = hBp[b];
        if (bp.alive == 0) continue;
        if (bp.i < 0 || bp.i >= cc || bp.j < 0 || bp.j >= cc || bp.k < 0 || bp.k >= cc)
            return XPBD_ERR_BAD_ARG;
        hIjk.push_back(bp.i); hIjk.push_back(bp.j); hIjk.push_back(bp.k);
        hTh.push_back(bp.theta0);
        hSrcLive.push_back(b);
    }
    const int B = (int)hTh.size();

    // 4) INSTALL via the audited standalone path (validates + colours BEFORE tearing down any previous
    //    instance, then allocates + uploads the colour-ordered arrays + permutation maps). invMass here
    //    is only the bind-time seed — k_RefreshLiveness re-derives it every substep.
    std::vector<float> hInvMass(cc);
    for (int c = 0; c < cc; c++)
        hInvMass[c] = (hPinned[c] != 0 || hActive[c] == 0) ? 0.f : (1.f / hMass[c]);
    int rc = xpbd_init(cc, (const float*)hPos.data(), hInvMass.data(),
                       E, E > 0 ? hEdges.data() : nullptr, E > 0 ? hRest.data() : nullptr);
    if (rc != 0) return rc;
    rc = xpbd_set_bending(B, B > 0 ? hIjk.data() : nullptr, B > 0 ? hTh.data() : nullptr);
    if (rc != 0) { xpbd_shutdown(); return rc; }

    // 5) CONVERT to bound mode. Owner-slot map: device slot -> 6p+s (permuted through the colour map);
    //    bend source map: device slot -> LIVE bendPairs index (overwrites the identity srcIdx that
    //    xpbd_set_bending installed — in bound mode d_bendSrcIdx indexes the LIVE array).
    if (E > 0) {
        std::vector<int> devOwner(E);
        for (int e = 0; e < E; e++) devOwner[g_edgeSlotOfInput[e]] = hOwner[e];
        XK(cudaMalloc(&d_edgeOwnerSlot, (size_t)E * sizeof(int)));
        XK(cudaMemcpy(d_edgeOwnerSlot, devOwner.data(), (size_t)E * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (B > 0) {
        std::vector<int> devSrc(B);
        for (int b = 0; b < B; b++) devSrc[g_bendSlotOfInput[b]] = hSrcLive[b];
        XK(cudaMemcpy(d_bendSrcIdx, devSrc.data(), (size_t)B * sizeof(int), cudaMemcpyHostToDevice));
    }
    // Swap the private pos/vel for ALIASES of the shared live buffers. From g_bound=true on, shutdown
    // will not free them.
    cudaFree(d_pos); d_pos = cornerPos;
    cudaFree(d_vel); d_vel = vel;
    g_bound = true;
    g_liveRestNbr=restNbr; g_liveNbrIdx = nbrIdx; g_liveBendPairs = (const XpbdBendPairGpu*)bendPairsRaw;
    g_liveActive = active; g_livePinned = pinned; g_liveMass = mass;
    g_liveExtForce = extForce; g_liveHapticForce = hapticForce;
    // Data derived for an earlier binding describes another topology.
    if (!d_topologyRecords) XK(cudaMalloc(&d_topologyRecords, 8 * sizeof(unsigned)));
    XK(cudaMemset(d_topologyRecords, 0xff, 8 * sizeof(unsigned)));
    d_componentRevision = d_topologyRecords; d_livenessRevision = d_topologyRecords + 2;
    d_edgeSupportRevision = d_topologyRecords + 4; d_multigridRevision = d_topologyRecords + 6;
    g_dimsX = dimsX; g_dimsY = dimsY; g_dimsZ = dimsZ;
    // 6) BIND-TIME STATE (§9-(4)): d_prevPos = cornerPos (no first-frame velocity kick); the CURRENT
    //    shared vel contents are ADOPTED (a moving liver is not zeroed — d_vel now aliases it, untouched).
    XK(cudaMemcpy(d_prevPos, cornerPos, (size_t)cc * sizeof(float3), cudaMemcpyDeviceToDevice));

    // 7) STEP 8 grasp/contact state (design §5/§6): allocated here (not a separate call, unlike tets —
    //    grasp needs no extra topology beyond cornerCount) so it is ALWAYS valid once bound. Zero-filled:
    //    functionally INERT (d_grasped=0 everywhere, g_tool defaults to a zeroed XpbdToolState so the
    //    contact kernels' shaftR=0/jawGap=0 never produce a spurious violation) until physics.cu's caller
    //    starts calling xpbd_set_tool each frame.
    XK(cudaMalloc(&d_surfaceDegree,(size_t)cc*sizeof(int)));
    XK(cudaMalloc(&d_surfaceActiveCount,sizeof(int)));
    XK(cudaMalloc(&d_jawLoad,sizeof(float4)));
    XK(cudaMemset(d_jawLoad,0,sizeof(float4)));
    XK(cudaMalloc(&d_surfaceDelta,(size_t)cc*sizeof(float3)));
    XK(cudaMalloc(&d_contactReaction,sizeof(float4)));
    XK(cudaMemset(d_contactReaction,0,sizeof(float4)));
    XK(cudaMalloc(&d_contacts,(size_t)3*cc*sizeof(ContactRecord)));
    XK(cudaMalloc(&d_contactNormalLambda,(size_t)3*cc*sizeof(float)));
    XK(cudaMalloc(&d_contactTangentLambda,(size_t)3*cc*sizeof(float3)));
    XK(cudaMalloc(&d_grasped,            (size_t)cc * sizeof(int)));
    XK(cudaMalloc(&d_graspCore,          (size_t)cc * sizeof(int)));
    XK(cudaMalloc(&d_coreOffU,           (size_t)cc * sizeof(float)));
    XK(cudaMalloc(&d_coreOffN,           (size_t)cc * sizeof(float)));
    XK(cudaMalloc(&d_coreOffW,           (size_t)cc * sizeof(float)));
    XK(cudaMalloc(&d_coreComplianceScale,(size_t)cc*sizeof(float)));
    XK(cudaMalloc(&d_coreLambda,         (size_t)cc * sizeof(float3)));
    XK(cudaMalloc(&d_coreForceEMA,       (size_t)cc * sizeof(float)));
    XK(cudaMalloc(&d_candScratch,        (size_t)cc * sizeof(int)));
    XK(cudaMalloc(&d_uScratch,           (size_t)cc * sizeof(float)));
    XK(cudaMalloc(&d_vScratch,           (size_t)cc * sizeof(float)));
    XK(cudaMalloc(&d_nScratch,           (size_t)cc * sizeof(float)));
    XK(cudaMalloc(&d_slabLambda,         (size_t)cc * sizeof(float)));
    XK(cudaMalloc(&d_slabLambdaPrev,     (size_t)cc * sizeof(float)));
    XK(cudaMalloc(&d_jawContactLambda,   (size_t)cc * sizeof(float)));
    XK(cudaMalloc(&d_shaftContactLambda, (size_t)cc * sizeof(float)));
    XK(cudaMalloc(&d_coulombAnchorU,     (size_t)cc * sizeof(float)));
    XK(cudaMalloc(&d_coulombAnchorW,     (size_t)cc * sizeof(float)));
    XK(cudaMemset(d_grasped,          0, (size_t)cc * sizeof(int)));
    XK(cudaMemset(d_graspCore,        0, (size_t)cc * sizeof(int)));
    XK(cudaMemset(d_coreOffU,         0, (size_t)cc * sizeof(float)));
    XK(cudaMemset(d_coreOffN,         0, (size_t)cc * sizeof(float)));
    XK(cudaMemset(d_coreOffW,         0, (size_t)cc * sizeof(float)));
    XK(cudaMemset(d_coreLambda,       0, (size_t)cc * sizeof(float3)));
    XK(cudaMemset(d_coreForceEMA,     0, (size_t)cc * sizeof(float)));
    XK(cudaMemset(d_slabLambda,       0, (size_t)cc * sizeof(float)));
    XK(cudaMemset(d_slabLambdaPrev,   0, (size_t)cc * sizeof(float)));
    XK(cudaMemset(d_jawContactLambda, 0, (size_t)cc * sizeof(float)));
    XK(cudaMemset(d_shaftContactLambda, 0, (size_t)cc * sizeof(float)));
    XK(cudaMemset(d_coulombAnchorU,   0, (size_t)cc * sizeof(float)));
    XK(cudaMemset(d_coulombAnchorW,   0, (size_t)cc * sizeof(float)));
    g_tool = XpbdToolState{};
    g_grasping = false;
    g_toolArmed = false;

    g_frameReady = false;                          // caller must begin a fresh frame
    return 0;
}

// ── STEP 7: per-tet isochoric/volumetric Neo-Hookean strain BUILD (design rev-B §4[Strain]/§9-(7)) ───────────────
// Host-side generic 3x3 inverse (double precision — build-time only, off the hot path).
static bool Inv3x3d(const double m[3][3], double out[3][3])
{
    double det = m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
               - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
               + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
    if (fabs(det) < 1e-20) return false;
    double invDet = 1.0 / det;
    out[0][0] =  (m[1][1]*m[2][2]-m[1][2]*m[2][1]) * invDet;
    out[0][1] = -(m[0][1]*m[2][2]-m[0][2]*m[2][1]) * invDet;
    out[0][2] =  (m[0][1]*m[1][2]-m[0][2]*m[1][1]) * invDet;
    out[1][0] = -(m[1][0]*m[2][2]-m[1][2]*m[2][0]) * invDet;
    out[1][1] =  (m[0][0]*m[2][2]-m[0][2]*m[2][0]) * invDet;
    out[1][2] = -(m[0][0]*m[1][2]-m[0][2]*m[1][0]) * invDet;
    out[2][0] =  (m[1][0]*m[2][1]-m[1][1]*m[2][0]) * invDet;
    out[2][1] = -(m[0][0]*m[2][1]-m[0][1]*m[2][0]) * invDet;
    out[2][2] =  (m[0][0]*m[1][1]-m[0][1]*m[1][0]) * invDet;
    return true;
}

extern "C" int xpbd_build_tets()
{
    destroyFeasibleGraphs();
    destroyTetGraph();
    cudaFree(d_localTets);d_localTets=nullptr;g_localPasses=0;g_localOff.clear();g_localCount.clear();
    if (!g_ready || !g_bound) return XPBD_ERR_NOT_READY;   // bound-mode only (needs dims + live topology)
    if (g_dimsX <= 0 || g_dimsY <= 0 || g_dimsZ <= 0) return XPBD_ERR_BAD_ARG;
    const int cc = g_N;
    const int nx = g_dimsX + 1, ny = g_dimsY + 1;   // corner-grid extents (VOXEL dims + 1 per axis —
                                                    // matches C# GridConventions.CornerId(i,j,k,dims))
    auto cid = [&](int x, int y, int z) { return x + nx * (y + ny * z); };

    // One-time readback of the buffers needed at BUILD time. d_pos aliases the shared cornerPos and is
    // still the REST configuration (xpbd_build_tets must run before the first xpbd_begin_frame, per the
    // header contract) — so Dm/Dm^-1 baked here are the genuine rest-state inverse rest matrices.
    std::vector<float3> hPos(cc);
    std::vector<int>    hActive(cc);
    #define XT(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) return -(int)_e; } while (0)
    XT(cudaMemcpy(hPos.data(),    d_pos,        (size_t)cc * sizeof(float3), cudaMemcpyDeviceToHost));
    XT(cudaMemcpy(hActive.data(), g_liveActive, (size_t)cc * sizeof(int),    cudaMemcpyDeviceToHost));
    #undef XT

    // Kuhn/"diagonal" 6-tet split, main diagonal c0-c7 (local bit-order corner index c[i] = (i&1,
    // (i>>1)&1, (i>>2)&1); design §4[Strain]).
    static const int TET_TABLE[6][4] = { {0,1,3,7}, {0,3,2,7}, {0,2,6,7}, {0,6,4,7}, {0,4,5,7}, {0,5,1,7} };

    std::vector<int>    hTetIds;     hTetIds.reserve(4096);
    std::vector<float3> hBr0, hBr1, hBr2; hBr0.reserve(1024); hBr1.reserve(1024); hBr2.reserve(1024);
    std::vector<float>  hMuV, hLambdaV;    hMuV.reserve(1024); hLambdaV.reserve(1024);
    std::vector<int>    hOwnerSlot;  hOwnerSlot.reserve(3072);
    std::vector<int>    hType;       hType.reserve(1024);   // Kuhn tet type (TET_TABLE row) of each emitted tet

    // Shear/bulk moduli from E/nu: mu=E/(2*(1+nu)); kappa=E/(3*(1-2*nu)). youngE<=0
    // (strain OFF) still builds the tet mesh — so the CUT lifecycle can be exercised independent of the
    // material tune — with mu=lambda=1 placeholders that xpbd_begin_frame's g_doStrainF gate (youngE>0)
    // ensures are never dispatched as a real compliance.
    float nu = g_p.poisson, mu, lambda;
    if (g_p.youngE > 0.f && nu > -1.f && nu < 0.5f) {
        mu     = g_p.youngE / (2.f * (1.f + nu));
        lambda = g_p.youngE / (3.f*(1.f-2.f*nu)); // bulk modulus kappa
    } else {
        mu = lambda = 1.f;
    }
    if (!(mu > 0.f)) mu = 1.f;
    if (!(lambda > 0.f)) lambda = 1.f;

    // The rest lattice is regular up to float rounding, so all tets of one Kuhn type share one rest shape. It is
    // baked from the lattice spacing, identical for every tet of the type: the material kernels read it per type
    // (UnifiedState::kuhn), the other kernels from the per-tet arrays, and both see the same values.
    double spacing[3] = { ((double)hPos[cid(g_dimsX, 0, 0)].x - (double)hPos[0].x) / g_dimsX,
                          ((double)hPos[cid(0, g_dimsY, 0)].y - (double)hPos[0].y) / g_dimsY,
                          ((double)hPos[cid(0, 0, g_dimsZ)].z - (double)hPos[0].z) / g_dimsZ };
    double typeB[6][3][3], typeV[6];
    for (int ti = 0; ti < 6; ti++) {
        double Dm[3][3];
        for (int k = 1; k < 4; k++)
            for (int a = 0; a < 3; a++) Dm[a][k - 1] = ((TET_TABLE[ti][k] >> a) & 1) * spacing[a];
        if (!Inv3x3d(Dm, typeB[ti])) return XPBD_ERR_IRREGULAR_REST;
        typeV[ti] = fabs(Dm[0][0]*(Dm[1][1]*Dm[2][2]-Dm[1][2]*Dm[2][1]) - Dm[0][1]*(Dm[1][0]*Dm[2][2]-Dm[1][2]*Dm[2][0])
                         + Dm[0][2]*(Dm[1][0]*Dm[2][1]-Dm[1][1]*Dm[2][0])) / 6.0;
    }

    for (int z = 0; z < g_dimsZ; z++)
    for (int y = 0; y < g_dimsY; y++)
    for (int x = 0; x < g_dimsX; x++) {
        bool allActive = true;
        for (int li = 0; li < 8 && allActive; li++) {
            int dx = li & 1, dy = (li >> 1) & 1, dz = (li >> 2) & 1;
            if (hActive[cid(x + dx, y + dy, z + dz)] == 0) allActive = false;
        }
        if (!allActive) continue;   // emit only when all 8 cell corners are active (design §4[Strain])

        // A uniform Freudenthal split gives both cells the SAME triangulation
        // on every shared face. Alternating only the x bit on x+y+z parity
        // mismatches the y/z faces and is not a conforming tetrahedral mesh.
        const int flip = 0;
        for (int ti = 0; ti < 6; ti++) {
            int gidTet[4]; float3 restP[4];
            for (int k = 0; k < 4; k++) {
                int li2 = TET_TABLE[ti][k] ^ flip;
                int dx = li2 & 1, dy = (li2 >> 1) & 1, dz = (li2 >> 2) & 1;
                gidTet[k] = cid(x + dx, y + dy, z + dz);
                restP[k] = hPos[gidTet[k]];
            }
            double e1[3] = { restP[1].x-restP[0].x, restP[1].y-restP[0].y, restP[1].z-restP[0].z };
            double e2[3] = { restP[2].x-restP[0].x, restP[2].y-restP[0].y, restP[2].z-restP[0].z };
            double e3[3] = { restP[3].x-restP[0].x, restP[3].y-restP[0].y, restP[3].z-restP[0].z };
            double Dm[3][3] = { {e1[0],e2[0],e3[0]}, {e1[1],e2[1],e3[1]}, {e1[2],e2[2],e3[2]} };
            double B[3][3];
            if (!Inv3x3d(Dm, B)) continue;   // singular rest tet: skip
            double det = Dm[0][0]*(Dm[1][1]*Dm[2][2]-Dm[1][2]*Dm[2][1])
                       - Dm[0][1]*(Dm[1][0]*Dm[2][2]-Dm[1][2]*Dm[2][0])
                       + Dm[0][2]*(Dm[1][0]*Dm[2][1]-Dm[1][1]*Dm[2][0]);
            double V = fabs(det) / 6.0;
            if (V < 1e-12) continue;         // degenerate-tet guard (XPBDSolverGPU.cs:208-231 pattern)
            double scale = 0, deviation = 0;
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) {
                    scale = fmax(scale, fabs(typeB[ti][a][b]));
                    deviation = fmax(deviation, fabs(B[a][b] - typeB[ti][a][b]));
                }
            if (deviation > 1e-4 * scale) return XPBD_ERR_IRREGULAR_REST;
            memcpy(B, typeB[ti], sizeof(B));
            V = typeV[ti];

            // the tet's 3 member AXIS edges (popcount(local_a^local_b)==1). Axis identity is computed
            // from the ORIGINAL (pre-flip) local indices — XOR-ing both endpoints by the SAME flip mask
            // preserves which bit differs — then the SAME flip resolves the owning (lower-coordinate)
            // corner's global id/slot.
            int axisEdges = 0;
            for (int a = 0; a < 4 && axisEdges < 3; a++)
            for (int b = a + 1; b < 4 && axisEdges < 3; b++) {
                int lu = TET_TABLE[ti][a], lv = TET_TABLE[ti][b];
                int diff = lu ^ lv;
                if (diff != 1 && diff != 2 && diff != 4) continue;   // not a single-bit (axis) edge
                int axis = (diff == 1) ? 0 : (diff == 2 ? 1 : 2);
                int lu2 = lu ^ flip, lv2 = lv ^ flip;
                int lowerLocal = (((lu2 >> axis) & 1) == 0) ? lu2 : lv2;   // bit=0 side = lower along axis
                int dx = lowerLocal & 1, dy = (lowerLocal >> 1) & 1, dz = (lowerLocal >> 2) & 1;
                int lowerGid = cid(x + dx, y + dy, z + dz);
                hOwnerSlot.push_back(6 * lowerGid + (2 * axis + 1));
                axisEdges++;
            }
            if (axisEdges != 3) continue;   // invariant guard (never trips for the fixed Kuhn table)

            hTetIds.push_back(gidTet[0]); hTetIds.push_back(gidTet[1]);
            hTetIds.push_back(gidTet[2]); hTetIds.push_back(gidTet[3]);
            hBr0.push_back(mkf3((float)B[0][0], (float)B[0][1], (float)B[0][2]));
            hBr1.push_back(mkf3((float)B[1][0], (float)B[1][1], (float)B[1][2]));
            hBr2.push_back(mkf3((float)B[2][0], (float)B[2][1], (float)B[2][2]));
            hMuV.push_back(mu * (float)V);
            hLambdaV.push_back(lambda * (float)V);
            hType.push_back(ti);
        }
    }

    const int T = (int)hMuV.size();

    // greedy 4-vertex tet colouring (share ANY of the 4 corners -> different colour), same pattern as
    // the bending 3-vertex colouring.
    std::vector<unsigned long long> vmask(cc, 0ull);
    std::vector<int> tetColor(T, 0);
    int numColors = 0;
    for (int e = 0; e < T; e++) {
        int i0=hTetIds[4*e+0], i1=hTetIds[4*e+1], i2=hTetIds[4*e+2], i3=hTetIds[4*e+3];
        unsigned long long used = vmask[i0] | vmask[i1] | vmask[i2] | vmask[i3];
        int c = 0; while (c < 64 && (used & (1ull << c))) c++;
        if (c >= 64) return XPBD_ERR_COLOR_OVERFLOW;
        tetColor[e] = c;
        vmask[i0] |= (1ull<<c); vmask[i1] |= (1ull<<c); vmask[i2] |= (1ull<<c); vmask[i3] |= (1ull<<c);
        if (c + 1 > numColors) numColors = c + 1;
    }

    g_tColorCnt.assign(numColors, 0);
    for (int e = 0; e < T; e++) g_tColorCnt[tetColor[e]]++;
    g_tColorOff.assign(numColors, 0);
    for (int c = 1; c < numColors; c++) g_tColorOff[c] = g_tColorOff[c-1] + g_tColorCnt[c-1];
    // Device order: the six Kuhn tet types one after the other, each in cell order (x fastest). Consecutive tets are
    // then the same tet of neighbouring cells, so a thread per tet gathers neighbouring nodes, and threads walking the
    // incident tets of neighbouring nodes read neighbouring tets at every step (incident lists are in tet order).
    // Colour order scatters the 24 tets of a node over the whole array. The colour-ordered passes read their tets
    // through g_tetColorOrder.
    std::vector<int> place(T), typeCursor(7, 0);
    for (int e = 0; e < T; e++) typeCursor[hType[e] + 1]++;
    for (int k = 1; k < 7; k++) typeCursor[k] += typeCursor[k - 1];
    for (int e = 0; e < T; e++) place[e] = typeCursor[hType[e]]++;
    std::vector<int> cursor = g_tColorOff;
    g_tetColorOrder.assign(T, 0);
    for (int e = 0; e < T; e++) g_tetColorOrder[cursor[tetColor[e]]++] = place[e];
    std::vector<int>    dIds((size_t)4*T);
    std::vector<float3> dBr0(T), dBr1(T), dBr2(T);
    std::vector<float>  dMuV(T), dLambdaV(T);
    std::vector<int>    dOwner((size_t)3*T);
    for (int e = 0; e < T; e++) {
        int dst = place[e];
        dIds[4*dst+0]=hTetIds[4*e+0]; dIds[4*dst+1]=hTetIds[4*e+1];
        dIds[4*dst+2]=hTetIds[4*e+2]; dIds[4*dst+3]=hTetIds[4*e+3];
        dBr0[dst]=hBr0[e]; dBr1[dst]=hBr1[e]; dBr2[dst]=hBr2[e];
        dMuV[dst]=hMuV[e]; dLambdaV[dst]=hLambdaV[e];
        dOwner[3*dst+0]=hOwnerSlot[3*e+0]; dOwner[3*dst+1]=hOwnerSlot[3*e+1]; dOwner[3*dst+2]=hOwnerSlot[3*e+2];
    }
    // Every emitting cell gives one tet of each type, so the k-th tet of every type lies in the k-th cell, and corner 0
    // of each tet (TET_TABLE) is corner 0 of its cell. The material kernels derive the corner ids from that corner
    // (unifiedTetIds): check that the stored ids are exactly those.
    const int cells = T / 6;
    if (T != 6 * cells) return XPBD_ERR_IRREGULAR_REST;
    std::vector<int> cellBase(cells);
    for (int k = 0; k < cells; k++) cellBase[k] = dIds[4 * k];
    for (int ti = 0; ti < 6 && cells > 0; ti++)
        for (int k = 0; k < cells; k++)
            for (int j = 0; j < 4; j++)
                if (dIds[4 * (ti * cells + k) + j] - cellBase[k] != dIds[4 * ti * cells + j] - cellBase[0])
                    return XPBD_ERR_IRREGULAR_REST;

    cudaFree(d_tetCellBase); d_tetCellBase = nullptr; cudaFree(d_kuhnTets); d_kuhnTets = nullptr;
    // The active flags never change after this build: list the active nodes once.
    std::vector<int> activeNodes;
    for (int c = 0; c < cc; c++) if (hActive[c] != 0) activeNodes.push_back(c);
    cudaFree(d_activeNodes); d_activeNodes = nullptr; g_activeNodes = (int)activeNodes.size();
    if (g_activeNodes > 0) {
        XK(cudaMalloc(&d_activeNodes, activeNodes.size() * sizeof(int)));
        XK(cudaMemcpy(d_activeNodes, activeNodes.data(), activeNodes.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (T > 0) {
        XK(cudaMalloc(&d_tetCellBase,   (size_t)cells * sizeof(int)));
        XK(cudaMemcpy(d_tetCellBase,    cellBase.data(), (size_t)cells*sizeof(int), cudaMemcpyHostToDevice));
        XK(cudaMalloc(&d_kuhnTets,      6 * sizeof(KuhnTet)));
        XK(cudaMalloc(&d_tetIds,        (size_t)4*T * sizeof(int)));
        XK(cudaMalloc(&d_tetBr0,        (size_t)T   * sizeof(float3)));
        XK(cudaMalloc(&d_tetBr1,        (size_t)T   * sizeof(float3)));
        XK(cudaMalloc(&d_tetBr2,        (size_t)T   * sizeof(float3)));
        XK(cudaMalloc(&d_tetMuV,        (size_t)T   * sizeof(float)));
        XK(cudaMalloc(&d_tetLambdaV,    (size_t)T   * sizeof(float)));
        XK(cudaMalloc(&d_tetAlphaDev,   (size_t)T   * sizeof(float)));
        XK(cudaMalloc(&d_tetAlphaHyd,   (size_t)T   * sizeof(float)));
        XK(cudaMalloc(&d_tetLambdaDevL, (size_t)T   * sizeof(float)));
        XK(cudaMalloc(&d_tetLambdaHydL, (size_t)T   * sizeof(float)));
        XK(cudaMalloc(&d_tetActive,     (size_t)T   * sizeof(int)));
        XK(cudaMalloc(&d_fiberLambda,(size_t)6*T*sizeof(float)));
        XK(cudaMalloc(&d_tetEdgeOwnerSlot, (size_t)3*T * sizeof(int)));
        XK(cudaMalloc(&d_tetColorOrder,  (size_t)T   * sizeof(int)));
        XK(cudaMalloc(&d_colorTetIds,    (size_t)T   * sizeof(int4)));
        XK(cudaMalloc(&d_colorTetDet,    (size_t)T   * sizeof(float)));
        XK(cudaMalloc(&d_edgeTetSupport, (size_t)6*g_N * sizeof(int)));

        XK(cudaMemcpy(d_tetIds,        dIds.data(),    (size_t)4*T*sizeof(int),   cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_tetBr0,        dBr0.data(),    (size_t)T*sizeof(float3), cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_tetBr1,        dBr1.data(),    (size_t)T*sizeof(float3), cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_tetBr2,        dBr2.data(),    (size_t)T*sizeof(float3), cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_tetMuV,        dMuV.data(),    (size_t)T*sizeof(float),  cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_tetLambdaV,    dLambdaV.data(),(size_t)T*sizeof(float),  cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_tetEdgeOwnerSlot, dOwner.data(),(size_t)3*T*sizeof(int), cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_tetColorOrder, g_tetColorOrder.data(),(size_t)T*sizeof(int), cudaMemcpyHostToDevice));
        k_ColorTetData<<<Gr(T),64>>>(T,d_tetColorOrder,d_tetIds,d_tetBr0,d_tetBr1,d_tetBr2,d_colorTetIds,d_colorTetDet);
        XK(cudaGetLastError());
        XK(cudaMemset(d_tetAlphaDev,   0, (size_t)T*sizeof(float)));
        XK(cudaMemset(d_tetAlphaHyd,   0, (size_t)T*sizeof(float)));
        XK(cudaMemset(d_tetLambdaDevL, 0, (size_t)T*sizeof(float)));
        XK(cudaMemset(d_tetLambdaHydL, 0, (size_t)T*sizeof(float)));
        // Seed active=1: k_RefreshLiveness re-derives the true value before the FIRST predict of the
        // FIRST substep, so this seed is never read as a load-bearing liveness value.
        std::vector<int> ones(T, 1);
        XK(cudaMemcpy(d_tetActive, ones.data(), (size_t)T*sizeof(int), cudaMemcpyHostToDevice));
    }
    g_T = T; g_tetCells = cells; g_tNumColors = numColors; g_haveTets = true;
    markLivenessStale();   // the seeded tet flags and the new edge support array
    g_frameReady = false;   // topology changed: caller must begin a new frame
    return 0;
}

extern "C" int xpbd_tet_count()       { return g_T; }
extern "C" int xpbd_tet_color_count() { return g_tNumColors; }
extern "C" int xpbd_readback_tet_ids(int* out)
{
    if (g_T <= 0) return 0;
    if (!out) return XPBD_ERR_BAD_ARG;
    return -(int)cudaMemcpy(out, d_tetIds, (size_t)4 * g_T * sizeof(int), cudaMemcpyDeviceToHost);
}

// ── STEP 8: contact + two-jaw grasp host API (design rev-B §4[Contact]/§5/§6) ───────────────────────
extern "C" void xpbd_set_grasp_params(const XpbdGraspParams* p) { if (p) g_gp = *p; }
extern "C" int* xpbd_grasp_flags() { return g_bound ? d_grasped : nullptr; }
extern "C" int* xpbd_grasp_core_flags() { return g_bound ? d_graspCore : nullptr; }
extern "C" int xpbd_read_grasp_core(int count,int* flags){
    if(!g_bound || count!=g_N || !flags)return XPBD_ERR_BAD_ARG;
    cudaError_t e=cudaMemcpy(flags,d_graspCore,(size_t)g_N*sizeof(int),cudaMemcpyDeviceToHost);
    return e==cudaSuccess?0:-(int)e;
}
// audit-round addition: readback for k_CoreBreakCheck's smoothed break-detection signal (VERIFY/tests
// only) -- lets a test log the actual converged EMA value instead of relying solely on the pass/fail
// graspCore transition, so a future regression in settled break-force magnitude fails loudly.
extern "C" float* xpbd_grasp_core_force_ema() { return g_bound ? d_coreForceEMA : nullptr; }

extern "C" int xpbd_grasp_corner_count()
{
    if (!g_bound) return 0;
    std::vector<int> h(g_N);
    if (cudaMemcpy(h.data(), d_grasped, (size_t)g_N * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
    int n = 0; for (int v : h) if (v) n++;
    return n;
}
extern "C" int xpbd_grasp_core_count()
{
    if (!g_bound) return 0;
    std::vector<int> h(g_N);
    if (cudaMemcpy(h.data(), d_graspCore, (size_t)g_N * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
    int n = 0; for (int v : h) if (v) n++;
    return n;
}

// Uploads the per-frame tool/grasp state and runs the CAPTURE (rising edge of grasp01) / RELEASE
// (falling edge) state machine (design §5 item 1 hysteresis). See xpbd_solver.h for the full contract.
extern "C" int xpbd_set_tissue_guard(float minVolume,int passes,int count,const int* inside,int capture){
    if(!g_bound || g_hFrame>0 || !std::isfinite(minVolume) || minVolume<0 || minVolume>.95f || passes<0 || passes>4 || count!=g_N || !inside)return XPBD_ERR_BAD_ARG;
    destroyTetGraph();destroySurfaceGraph();
    destroyFeasibleGraphs();
    if(g_pressureGraph){cudaGraphExecDestroy(g_pressureGraph);g_pressureGraph=nullptr;}
    cudaFree(d_captureInside);d_captureInside=nullptr;
    cudaError_t e=cudaMalloc(&d_captureInside,(size_t)g_N*sizeof(int));if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(d_captureInside,inside,(size_t)g_N*sizeof(int),cudaMemcpyHostToDevice);
    if(e==cudaSuccess && !d_feasibleStart)e=cudaMalloc(&d_feasibleStart,(size_t)g_N*sizeof(float3));
    if(e==cudaSuccess && !d_feasibleTarget)e=cudaMalloc(&d_feasibleTarget,(size_t)g_N*sizeof(float3));
    if(e==cudaSuccess && !d_needFeasible)e=cudaMalloc(&d_needFeasible,(size_t)(g_N+1)*sizeof(int));
    if(e==cudaSuccess && !d_incidentOffsets){
        std::vector<int> ids(4*g_T),offsets(g_N+1,0),incident(4*g_T);
        e=cudaMemcpy(ids.data(),d_tetIds,ids.size()*sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
        for(int i:ids)offsets[i+1]++;
        for(int i=1;i<=g_N;i++)offsets[i]+=offsets[i-1];
        auto cursor=offsets;for(int t=0;t<g_T;t++)for(int k=0;k<4;k++)incident[cursor[ids[4*t+k]]++]=t;
        e=cudaMalloc(&d_incidentOffsets,offsets.size()*sizeof(int));if(e!=cudaSuccess)return -(int)e;
        e=cudaMalloc(&d_incidentTets,incident.size()*sizeof(int));if(e!=cudaSuccess)return -(int)e;
        e=cudaMemcpy(d_incidentOffsets,offsets.data(),offsets.size()*sizeof(int),cudaMemcpyHostToDevice);if(e!=cudaSuccess)return -(int)e;
        e=cudaMemcpy(d_incidentTets,incident.data(),incident.size()*sizeof(int),cudaMemcpyHostToDevice);
    }
    if(e==cudaSuccess){g_minTetVolume=minVolume;g_volumePasses=passes;g_captureWedge=capture!=0;}
    return -(int)e;
}
// Shared by separate and packed feedback; writes diagnostic output only.
#include "../grasp/runtime_grip_feedback.cuh"
static cudaError_t observeGrip(){
    if(!d_gripError)return cudaErrorInvalidValue;
    if(g_embeddedGrip && g_grasping && g_gripCount>0){
        if(g_runtimeCoreMode&2048)return observeRuntimeGrip();
        k_ReadGripObservation<<<1,1>>>(g_gripCount,d_gripPoints,g_surfaceMasks,
            g_liveActive,d_pos,d_coupledTool,g_tool,g_gp.jawGapClosed,d_gripError);
        return cudaGetLastError();
    }
    return cudaMemsetAsync(d_gripError,0,3*sizeof(float));
}
extern "C" int xpbd_get_grasp_lock(float* gap){
    if(!gap)return XPBD_ERR_BAD_ARG;*gap=g_tool.jawGap;
    if(g_embeddedGrip){
        if(!g_grasping || g_gripCount<=0)return 0;
        cudaError_t e=observeGrip();if(e!=cudaSuccess)return -(int)e;
        float live=0;e=cudaMemcpy(&live,d_gripError+1,sizeof(float),cudaMemcpyDeviceToHost);
        if(e!=cudaSuccess)return -(int)e;return live>0?1:0;
    }

    return g_anchorAssist && g_captureWedge && d_captureInside && g_grasping?1:0;
}
// Keep candidates strictly inside the finite wedge between the inner jaw faces.
__global__ void k_InsideJawWedge(int n,XpbdToolState tool,float minGap,const int* inside,
    int* candidate,const float* u,const float* v,const float* normal){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n || !candidate[i])return;
    float sn=fminf(.95f,tool.jawGap/(2*fmaxf(tool.jawLen,1e-6f))),cs=sqrtf(1-sn*sn);
    float aperture=fmaxf(0.f,u[i]*sn/cs+.5f*minGap/cs);
    if(!inside[i] || u[i]<.1f*tool.jawLen || u[i]>tool.jawLen*cs || fabsf(v[i])>tool.jawHalfWidth || fabsf(normal[i])>aperture)candidate[i]=0;
}
extern "C" int xpbd_set_contact_material(const float3* rest,const float4* featurePoints){
    if(!g_bound)return XPBD_ERR_NOT_READY;
    g_surfaceRest=rest;g_surfaceMaterialFP=featurePoints;return 0;
}
// Surface support fallback for a sub-cell bite: a genuine embedded material
// point can lie between the jaws even when no lattice corner lies there.
// Clip each current material triangle against the finite inner-jaw wedge.
// Vertex-only selection misses a tiny bite through the middle of a coarse triangle.
__global__ void k_EmbeddedBite(int count,const SurfaceVertex* vertices,const unsigned* masks,
    const float3* pos,XpbdToolState tool,float minGap,const int* active,const int* pinned,int* candidate){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    SurfaceVertex v[3]={vertices[3*i],vertices[3*i+1],vertices[3*i+2]};
    float3 axis=contactUnit(tool.jawAxis,mkf3(0,0,1)),normal=contactUnit(tool.jawNormal,mkf3(1,0,0));
    float3 width=contactUnit(cross3(axis,normal),mkf3(0,1,0));
    float3 polygon[12],next[12];
    for(int j=0;j<3;j++){
        if(!v[j].count || (masks && v[j].mask!=(masks[v[j].voxel]&4095u)))return;
        float3 r=surfacePoint(v[j],pos)-tool.jawHinge;
        polygon[j]=mkf3(dot3(r,axis),dot3(r,width),dot3(r,normal));
    }
    float sn=fminf(.95f,tool.jawGap/(2*fmaxf(tool.jawLen,1e-6f))),cs=sqrtf(1-sn*sn);
    float3 planes[6]={mkf3(-1,0,0),mkf3(1,0,0),mkf3(0,-1,0),mkf3(0,1,0),mkf3(-sn/cs,0,1),mkf3(-sn/cs,0,-1)};
    float bounds[6]={-.1f*tool.jawLen,tool.jawLen*cs,tool.jawHalfWidth,tool.jawHalfWidth,.5f*minGap/cs,.5f*minGap/cs};
    int n=3;
    for(int plane=0;plane<6 && n;plane++){
        int m=0;float3 a=polygon[n-1];float da=dot3(planes[plane],a)-bounds[plane];
        for(int j=0;j<n;j++){
            float3 b=polygon[j];float db=dot3(planes[plane],b)-bounds[plane];
            if((da<=0)!=(db<=0))next[m++]=a+(b-a)*(da/(da-db));
            if(db<=0)next[m++]=b;
            a=b;da=db;
        }
        n=m;for(int j=0;j<n;j++)polygon[j]=next[j];
    }
    if(n<3)return;
    float area=0;for(int j=1;j<n-1;j++)area+=len3(cross3(polygon[j]-polygon[0],polygon[j+1]-polygon[0]));
    if(area<1e-10f)return;
    for(int j=0;j<3;j++)for(int k=0;k<v[j].count;k++)if(v[j].weight[k]>.05f){
        int id=v[j].id[k];if(active[id] && !pinned[id])atomicExch(candidate+id,1);
    }
}
__global__ void k_BiteCoordinates(int n,const int* candidate,const float3* pos,float3 hinge,float3 axis,float3 normal,float3 width,float* u,float* v,float* nn){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n || !candidate[i])return;
    float3 r=pos[i]-hinge;u[i]=dot3(r,axis);v[i]=dot3(r,width);nn[i]=dot3(r,normal);
}
extern "C" int xpbd_set_contact_surface(int count,const int* tags,const float3* expanded,
    const int* corners,const unsigned* masks,const int* conn,const unsigned* boundaryMasks) {
    if(!g_bound)return XPBD_ERR_NOT_READY;
    if(count<0 || (count && (!tags||!expanded||!corners)))return XPBD_ERR_BAD_ARG;
    if(!d_contactCache){
        cudaError_t e=cudaMalloc(&d_contactCache,CONTACT_CACHE_SIZE*sizeof(SurfaceContact));
        if(e==cudaSuccess)e=cudaMalloc(&d_contactCacheSlots,CONTACT_CACHE_SIZE*sizeof(int));
        if(e!=cudaSuccess)return -(int)e;
    }
    cudaMemsetAsync(d_contactCacheSlots,0xff,CONTACT_CACHE_SIZE*sizeof(int));
    if(g_surfaceCount>0 && g_toolArmed && g_substepIndex>0)
        k_CacheContacts<<<Gr(3*g_surfaceCount),64>>>(d_surfaceContacts,d_surfaceActive,d_surfaceActiveCount,d_contactCacheSlots,d_contactCache);
    if(count>g_surfaceCapacity){
        cudaFree(d_contactGroups);cudaFree(d_contactGroupRejected);d_contactGroups=nullptr;d_contactGroupRejected=nullptr;
        cudaFree(d_surfaceCurrent);cudaFree(d_surfacePrevious);d_surfaceCurrent=nullptr;d_surfacePrevious=nullptr;
        cudaFree(d_surfaceVertices);cudaFree(d_surfaceContacts);cudaFree(d_surfaceActive);d_surfaceVertices=nullptr;d_surfaceContacts=nullptr;d_surfaceActive=nullptr;
        // Reserve growth for cut-created faces. Reallocating the surface,
        // quadrature manifold and blade buffers at every 1024 new triangles
        // forces device synchronization in the middle of a cutting gesture.
        g_surfaceCapacity=0;int capacity=((count+count/2+1023)/1024)*1024;
        destroySurfaceGraph();destroyRepairGraph();
        cudaError_t e=cudaMalloc(&d_surfaceVertices,(size_t)3*capacity*sizeof(SurfaceVertex));
        if(e==cudaSuccess)e=cudaMalloc(&d_surfaceContacts,(size_t)3*capacity*sizeof(SurfaceContact));
        if(e==cudaSuccess)e=cudaMemset(d_surfaceContacts,0,(size_t)3*capacity*sizeof(SurfaceContact));
        if(e==cudaSuccess)e=cudaMalloc(&d_surfaceActive,(size_t)3*capacity*sizeof(int));
        if(e==cudaSuccess)e=cudaMalloc(&d_surfaceCurrent,(size_t)3*capacity*sizeof(float4));
        if(e==cudaSuccess)e=cudaMalloc(&d_surfacePrevious,(size_t)3*capacity*sizeof(float3));
        if(e==cudaSuccess)e=cudaMalloc(&d_contactGroups,(size_t)contactGroupCount(capacity)*sizeof(ContactGroupBounds));
        if(e==cudaSuccess)e=cudaMalloc(&d_contactGroupRejected,(size_t)6*contactGroupCount(capacity));
        if(e!=cudaSuccess){g_surfaceCount=0;return -(int)e;}g_surfaceCapacity=capacity;
    }
    if(count!=g_surfaceCount){
        // Cutting changes launch bounds frequently. Retain the executable and
        // update its captured parameters instead of re-instantiating hundreds
        // of identical PCG nodes on every changed surface.
        g_unifiedGraphDirty=true;g_unifiedRecoveryGraphDirty=true;g_unifiedLoadedGraphDirty=true;
        g_unifiedContactGraphDirty=true;
    }
    g_surfaceCount=count;g_surfaceTags=tags;g_surfaceExpanded=expanded;g_surfaceCorners=corners;
    g_surfaceMasks=masks;g_surfaceConn=conn;g_surfaceBoundaryMasks=boundaryMasks;return 0;
}
extern "C" int xpbd_rebind_contact_surface(){
    if(g_surfaceCount>0)k_BindSurface<<<Gr(3*g_surfaceCount),64>>>(3*g_surfaceCount,g_surfaceTags,g_surfaceExpanded,
        g_surfaceCorners,g_surfaceMasks,g_surfaceConn,d_pos,d_surfaceVertices,g_surfaceRest,g_surfaceMaterialFP,g_surfaceBoundaryMasks);
    return -(int)cudaGetLastError();
}
extern "C" int xpbd_get_contact_reaction(float* force,int* contacts){
    if(!g_bound || !force || !contacts)return XPBD_ERR_BAD_ARG;
    float4 r={};cudaError_t e=cudaMemcpy(&r,d_contactReaction,sizeof(r),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return -(int)e;
    force[0]=r.x;force[1]=r.y;force[2]=r.z;
    *contacts=(int)ceilf(r.w/fmaxf(1.f,(float)g_p.numSubsteps));return 0;
}
extern "C" int xpbd_get_jaw_load(float* result){
    if(!g_bound || !result || !d_jawLoad)return XPBD_ERR_BAD_ARG;
    return -(int)cudaMemcpy(result,d_jawLoad,sizeof(float4),cudaMemcpyDeviceToHost);
}
static void queryBladeDepth(int triangles,float* output);
extern "C" int xpbd_surface_depth(int triangles,float* depth){
    if(!g_bound || !depth)return XPBD_ERR_BAD_ARG;
    // Shared serial-query scratch; no allocator/device-wide free in the frame loop.
    if(!d_toolClearance){cudaError_t e=cudaMalloc(&d_toolClearance,9*sizeof(float));if(e!=cudaSuccess)return -(int)e;}
    float* d=d_toolClearance+8;cudaError_t e=cudaMemsetAsync(d,0,sizeof(float));if(e!=cudaSuccess)return -(int)e;
    if(g_toolArmed && triangles>0)for(int kind=0;kind<3;kind++)
        k_SurfaceDepth<<<Gr(triangles),64>>>(triangles,contactShape(g_tool,kind,g_jawThickness,g_gp.jawGapClosed),g_surfaceExpanded,d);
    // The pressure-tool slot is independent of the grasper. Query its accepted
    // CUDA geometry too: scissors contribute both finite edge capsules, while
    // a single knife contributes its capsule. Commands are spring targets and
    // must never substitute for the actual pose/aperture after reconstruction.
    if(triangles>0)queryBladeDepth(triangles,d);
    e=cudaMemcpy(depth,d,sizeof(float),cudaMemcpyDeviceToHost);return -(int)e;
}
static XpbdToolState translatedTool(XpbdToolState tool,float3 delta){
    tool.shaftA=tool.shaftA+delta;tool.shaftB=tool.shaftB+delta;tool.jawHinge=tool.jawHinge+delta;tool.jawMid=tool.jawMid+delta;return tool;
}
static void queryToolDepth(int triangles,XpbdToolState tool,float* output){
    for(int kind=0;kind<3;kind++)k_SurfaceDepth<<<Gr(triangles),64>>>(triangles,contactShape(tool,kind,g_jawThickness,g_gp.jawGapClosed),g_surfaceExpanded,output);
}
// Correct only along the attempted translation, after the nonlinear DC rebuild.
// This prevents a locally embedded contact solve from accepting a tool pose that
// the actual rendered surface cannot accommodate. No lateral ghost/proxy sliding.
extern "C" int xpbd_resolve_tool_translation(int triangles,float* correction){
    if(!correction)return XPBD_ERR_BAD_ARG;correction[0]=correction[1]=correction[2]=0;
    if(!g_bound || !g_frictionContact || !g_toolArmed || triangles<=0)return 0;
    if(!d_toolClearance){cudaError_t e=cudaMalloc(&d_toolClearance,9*sizeof(float));if(e!=cudaSuccess)return -(int)e;}
    cudaMemsetAsync(d_toolClearance,0,9*sizeof(float));queryToolDepth(triangles,g_tool,d_toolClearance+8);
    float depth=0;cudaError_t e=cudaMemcpy(&depth,d_toolClearance+8,sizeof(float),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    float tolerance=fmaxf(1e-5f,.5f*g_contactRadius);if(depth<=tolerance)return 0;
    float3 reverse=g_previousAcceptedTool.jawHinge-g_tool.jawHinge;
    // A kinematic tool may reject part of a NEW inward command; residual
    // reconstruction error is not permission to eject a stationary instrument.
    // Never extrapolate beyond the last accepted pose or oppose withdrawal.
    float3 outward=contactUnit(g_tool.jawAxis,mkf3(0,0,1))*(-1.f);
    if(len3(reverse)<1e-6f || dot3(reverse,outward)<0)return 0;
    cudaMemsetAsync(d_toolClearance,0,sizeof(float));
    queryToolDepth(triangles,translatedTool(g_tool,reverse),d_toolClearance);
    float previousDepth=0;e=cudaMemcpy(&previousDepth,d_toolClearance,sizeof(float),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return -(int)e;
    if(previousDepth>=depth)return 0;
    float lo=0,hi=1;
    if(previousDepth<=tolerance)for(int it=0;it<4;it++){
        float mid=.5f*(lo+hi);cudaMemsetAsync(d_toolClearance,0,sizeof(float));
        queryToolDepth(triangles,translatedTool(g_tool,reverse*mid),d_toolClearance);
        e=cudaMemcpy(&depth,d_toolClearance,sizeof(float),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
        if(depth<=tolerance)hi=mid;else lo=mid;
    }
    float3 delta=reverse*hi;g_tool=translatedTool(g_tool,delta);g_toolFrom=g_tool;
    correction[0]=delta.x;correction[1]=delta.y;correction[2]=delta.z;return 1;
}

// Setup only: extra material sweeps use existing coloured elements and live cut flags.
// No extra masses, duplicate material or independent collision surface are introduced.
extern "C" int xpbd_set_local_refinement(float x,float y,float z,float radius,int passes){
    if(!g_bound || g_hFrame>0 || passes<0 || passes>3 || !(radius>0) || !std::isfinite(radius) || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))return XPBD_ERR_BAD_ARG;
    destroyTetGraph();destroySurfaceGraph();cudaFree(d_localTets);d_localTets=nullptr;
    g_localPasses=passes;g_localOff.assign(g_tNumColors,0);g_localCount.assign(g_tNumColors,0);
    if(!passes)return 0;
    std::vector<int> ids(4*g_T),selected;std::vector<float3> rest(g_N);
    cudaError_t e=cudaMemcpy(ids.data(),d_tetIds,ids.size()*sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(rest.data(),d_pos,rest.size()*sizeof(float3),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    float3 center=mkf3(x,y,z);
    for(int c=0;c<g_tNumColors;c++){
        g_localOff[c]=(int)selected.size();
        for(int slot=g_tColorOff[c];slot<g_tColorOff[c]+g_tColorCnt[c];slot++){
            int t=g_tetColorOrder[slot];
            bool touchesRegion=false;for(int k=0;k<4;k++)touchesRegion|=len3(rest[ids[4*t+k]]-center)<radius;
            if(touchesRegion)selected.push_back(t);
        }
        g_localCount[c]=(int)selected.size()-g_localOff[c];
    }
    if(selected.empty()){g_localPasses=0;return 0;}
    e=cudaMalloc(&d_localTets,selected.size()*sizeof(int));if(e!=cudaSuccess){g_localPasses=0;return -(int)e;}
    return -(int)cudaMemcpy(d_localTets,selected.data(),selected.size()*sizeof(int),cudaMemcpyHostToDevice);
}
__global__ void k_MaterialQuality(int count,const int* ids,const int* active,const float3* b0,const float3* b1,
    const float3* b2,const float3* pos,float* output,const float* volume=nullptr){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !active[i])return;
    float3 a=pos[ids[4*i]],e1=pos[ids[4*i+1]]-a,e2=pos[ids[4*i+2]]-a,e3=pos[ids[4*i+3]]-a;
    float3 f[3];for(int k=0;k<3;k++)f[k]=e1*compF3(b0[i],k)+e2*compF3(b1[i],k)+e3*compF3(b2[i],k);
    float j=dot3(f[0],cross3(f[1],f[2]));if(!isfinite(j)){atomicAdd(output+1,1.f);return;}
    unsigned* address=(unsigned*)output;unsigned old=*address,assumed;
    while(j<__uint_as_float(old)){assumed=old;old=atomicCAS(address,assumed,__float_as_uint(j));if(old==assumed)break;}
    if(j<=0)atomicAdd(output+1,1.f);
    if(volume){atomicAdd(output+2,j*volume[i]);atomicAdd(output+3,volume[i]);atomicMax((unsigned*)(output+4),__float_as_uint(fmaxf(0,j)));}
}
extern "C" int xpbd_get_material_quality(float* output){
    if(!g_bound || !output)return XPBD_ERR_BAD_ARG;
    if(!d_toolClearance){cudaError_t e=cudaMalloc(&d_toolClearance,9*sizeof(float));if(e!=cudaSuccess)return -(int)e;}
    float seed[2]={1.f,0.f};cudaError_t e=cudaMemcpy(d_toolClearance,seed,sizeof(seed),cudaMemcpyHostToDevice);if(e!=cudaSuccess)return -(int)e;
    if(g_T>0)k_MaterialQuality<<<Gr(g_T),64>>>(g_T,d_tetIds,d_tetActive,d_tetBr0,d_tetBr1,d_tetBr2,d_pos,d_toolClearance);
    return -(int)cudaMemcpy(output,d_toolClearance,sizeof(seed),cudaMemcpyDeviceToHost);
}

extern "C" int xpbd_get_material_stats(float* output){
    if(!g_bound || !output)return XPBD_ERR_BAD_ARG;
    if(!d_toolClearance){cudaError_t e=cudaMalloc(&d_toolClearance,9*sizeof(float));if(e!=cudaSuccess)return -(int)e;}
    float seed[5]={1,0,0,0,0};cudaError_t e=cudaMemcpy(d_toolClearance,seed,sizeof(seed),cudaMemcpyHostToDevice);if(e!=cudaSuccess)return -(int)e;
    if(g_T>0)k_MaterialQuality<<<Gr(g_T),64>>>(g_T,d_tetIds,d_tetActive,d_tetBr0,d_tetBr1,d_tetBr2,d_pos,d_toolClearance,d_tetMuV);
    e=cudaMemcpy(seed,d_toolClearance,sizeof(seed),cudaMemcpyDeviceToHost);
    output[0]=seed[0];output[1]=seed[1];output[2]=seed[3]>0?seed[2]/seed[3]:1;output[3]=seed[4];return -(int)e;
}

// Diagnostic only: rest-volume weighted local errors cannot cancel as a signed
// whole-organ mean can. Two 64-bin histograms: all live tets and the jaw vicinity.
__global__ void k_VolumeError(int n,const int* ids,const int* active,const float3* b0,const float3* b1,const float3* b2,
    const float3* p,const float* volume,float3 center,float radius,float* out){
    int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=n || !active[e])return;
    int a=ids[4*e],b=ids[4*e+1],c=ids[4*e+2],d=ids[4*e+3];
    float j=dot3(p[b]-p[a],cross3(p[c]-p[a],p[d]-p[a]))*dot3(b0[e],cross3(b1[e],b2[e]));
    float error=isfinite(j)?fabsf(j-1.f):1e6f,w=volume[e];
    bool local=len3((p[a]+p[b]+p[c]+p[d])*.25f-center)<radius;
    for(int region=0;region<(local?2:1);region++){
        float* v=out+68*region;atomicAdd(v,w);atomicAdd(v+1,w*error);atomicAdd(v+2,w*error*error);
        if(error>.2f)atomicAdd(v+3,w);
        int bin=min(63,(int)floorf(fminf(error,1.575f)/.025f));atomicAdd(v+4+bin,w);
    }
}
extern "C" int xpbd_set_tissue_material(float tensileStrength,float bulkDampingSeconds){
    if(!g_bound || g_hFrame>0 || !std::isfinite(tensileStrength) || tensileStrength<0 || tensileStrength>1000 ||
       !std::isfinite(bulkDampingSeconds) || bulkDampingSeconds<0 || bulkDampingSeconds>1)return XPBD_ERR_BAD_ARG;
    destroyFeasibleGraphs();
    if(g_pressureGraph){cudaGraphExecDestroy(g_pressureGraph);g_pressureGraph=nullptr;}
    g_tissueHardening=tensileStrength;g_bulkDampingSeconds=bulkDampingSeconds;destroyTetGraph();return 0;
}
extern "C" int xpbd_get_volume_error(float* output){
    if(!g_bound || !output)return XPBD_ERR_BAD_ARG;
    float *scratch=nullptr;cudaError_t err=cudaMalloc(&scratch,136*sizeof(float));if(err!=cudaSuccess)return -(int)err;
    err=cudaMemset(scratch,0,136*sizeof(float));
    if(err!=cudaSuccess){cudaFree(scratch);return -(int)err;}
    if(g_T>0)k_VolumeError<<<Gr(g_T),64>>>(g_T,d_tetIds,d_tetActive,d_tetBr0,d_tetBr1,d_tetBr2,d_pos,d_tetMuV,
        g_tool.jawHinge+g_tool.jawAxis*(g_tool.jawLen*.5f),g_toolArmed?2.f*g_tool.jawLen:0.f,scratch);
    float h[136];err=cudaMemcpy(h,scratch,sizeof(h),cudaMemcpyDeviceToHost);cudaFree(scratch);
    if(err!=cudaSuccess)return -(int)err;
    for(int r=0;r<2;r++){
        float* v=h+68*r;float* o=output+5*r;float w=v[0],sum=0;int b=0;
        if(w>0){for(;b<63;b++){sum+=v[4+b];if(sum>=.95f*w)break;}}
        o[0]=w>0?v[1]/w:0;o[1]=w>0?sqrtf(v[2]/w):0;o[2]=w>0?v[3]/w:0;
        o[3]=w>0?(b==63?INFINITY:(b+1)*.025f):0;o[4]=w;
    }
    return 0;
}

extern "C" int xpbd_set_jaw_geometry(const float* bounds)
{
    if(!g_bound || g_hFrame>0 || !bounds)return XPBD_ERR_BAD_ARG;
    for(int j=0;j<2;j++){
        for(int k=0;k<6;k++)if(!std::isfinite(bounds[6*j+k]))return XPBD_ERR_BAD_ARG;
        for(int k=0;k<3;k++)if(bounds[6*j+3+k]<=0)return XPBD_ERR_BAD_ARG;
    }
    for(int j=0;j<2;j++)h_jawMeshBounds[j]={mkf3(bounds[6*j],bounds[6*j+1],bounds[6*j+2]),mkf3(bounds[6*j+3],bounds[6*j+4],bounds[6*j+5]),1};
    return -(int)cudaMemcpyToSymbol(c_jawMeshBounds,h_jawMeshBounds,sizeof(h_jawMeshBounds));
}
extern "C" int xpbd_set_contact_model(int enabled,float muS,float muD,float thickness,float radius)
{
    if(!g_bound)return XPBD_ERR_NOT_READY;
    if(enabled<0 || enabled>2 || !std::isfinite(muS)||!std::isfinite(muD)||!std::isfinite(thickness)||!std::isfinite(radius)||
       muS<0 || muD<0 || muD>muS || thickness<=0 || radius<0)return XPBD_ERR_BAD_ARG;
    g_frictionContact=enabled!=0;g_anchorAssist=enabled==2;g_muStatic=muS;g_muDynamic=muD;g_jawThickness=thickness;g_contactRadius=radius;
    g_grasping=false;cudaMemset(d_grasped,0,(size_t)g_N*sizeof(int));cudaMemset(d_graspCore,0,(size_t)g_N*sizeof(int));
    memset(h_jawMeshBounds,0,sizeof(h_jawMeshBounds));
    cudaMemcpyToSymbol(c_jawMeshBounds,h_jawMeshBounds,sizeof(h_jawMeshBounds));
    return 0;
}

extern "C" int xpbd_set_grasp_patch(float sigma,float compliance,float damping){
    if(!g_bound || g_hFrame>0 || !std::isfinite(sigma+compliance+damping) || sigma<=0 || compliance<=0 || damping<0)return XPBD_ERR_BAD_ARG;
    cudaError_t e=cudaSuccess;
    if(!d_gripCounter)e=cudaMalloc(&d_gripCounter,sizeof(int));
    if(e==cudaSuccess && !d_gripStep)e=cudaMalloc(&d_gripStep,sizeof(GripSolveStep));
    if(e==cudaSuccess && !d_gripLive)e=cudaMalloc(&d_gripLive,sizeof(int));
    if(e==cudaSuccess && !d_gripMoments)e=cudaMalloc(&d_gripMoments,4*sizeof(float));
    if(e==cudaSuccess && !d_gripDegree)e=cudaMalloc(&d_gripDegree,(size_t)g_N*sizeof(int));
    if(e==cudaSuccess && !d_gripPatch)e=cudaMalloc(&d_gripPatch,sizeof(GripPatch));
    if(e==cudaSuccess && !d_gripPatchWeights)e=cudaMalloc(&d_gripPatchWeights,(size_t)3*g_N*sizeof(float));
    if(e!=cudaSuccess)return -(int)e;
    e=cudaMemset(d_gripLive,0,sizeof(int));if(e!=cudaSuccess)return -(int)e;
    g_gripSigma=sigma;g_gripCompliance=compliance;g_gripDamping=damping;g_embeddedGrip=true;return 0;
}
extern "C" int xpbd_set_instrument_control(int guided){g_guidedInstrument=guided?1:0;return 0;}
extern "C" int xpbd_set_tool_dynamics(float mass,float stiffness,float damping,float maxTravel){
    if(!g_bound || g_hFrame>0 || !std::isfinite(mass+stiffness+damping+maxTravel) ||
       mass<=0 || stiffness<=0 || damping<0 || maxTravel<=0)return XPBD_ERR_BAD_ARG;
    cudaError_t e=cudaSuccess;
    if(!d_coupledTool)e=cudaMalloc(&d_coupledTool,sizeof(CoupledTool));
    if(e==cudaSuccess && !d_contactMobility)e=cudaMalloc(&d_contactMobility,(size_t)g_N*sizeof(float3));
    if(e==cudaSuccess && !d_gripError)e=cudaMalloc(&d_gripError,3*sizeof(float));
    if(e==cudaSuccess && !d_contactResidual)e=cudaMalloc(&d_contactResidual,3*sizeof(float));
    if(e==cudaSuccess)e=cudaMemset(d_coupledTool,0,sizeof(CoupledTool));
    if(e!=cudaSuccess)return -(int)e;
    g_toolMass=mass;g_toolStiffness=stiffness;g_toolDamping=damping;g_toolMaxTravel=maxTravel;
    destroySurfaceGraph();destroyRepairGraph();destroyGripGraph();return 0;
}
static void refreshGrip();
extern "C" int xpbd_set_jaw_dynamics(float mass,float stiffness,float damping,float speed){
    if(!d_coupledTool || g_hFrame>0 || !std::isfinite(mass+stiffness+damping+speed) ||
       mass<=0 || stiffness<=0 || damping<0 || speed<=0)return XPBD_ERR_BAD_ARG;
    g_jawMass=mass;g_jawStiffness=stiffness;g_jawDamping=damping;g_jawSpeed=speed;return 0;
}
static int syncSolvedTool(){
    if(!d_coupledTool || !g_toolArmed)return 0;
    CoupledTool b={};cudaError_t e=cudaMemcpy(&b,d_coupledTool,sizeof(b),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return -(int)e;
    if(b.initialized)g_tool=g_toolFrom=b.pose;
    return 0;
}
extern "C" int xpbd_get_solved_tool(float* hinge,float* error,int* live){
    if(!d_coupledTool || !hinge || !error || !live)return XPBD_ERR_BAD_ARG;
    int rc=syncSolvedTool();if(rc)return rc;
    hinge[0]=g_tool.jawHinge.x;hinge[1]=g_tool.jawHinge.y;hinge[2]=g_tool.jawHinge.z;
    cudaError_t e=observeGrip();if(e!=cudaSuccess)return -(int)e;
    float values[2]={};e=cudaMemcpy(values,d_gripError,sizeof(values),cudaMemcpyDeviceToHost);
    *error=values[0];*live=(int)values[1];return -(int)e;
}
extern "C" int xpbd_get_solved_jaw(float* gap){
    if(!d_coupledTool || !gap)return XPBD_ERR_BAD_ARG;
    int rc=syncSolvedTool();if(rc)return rc;*gap=g_tool.jawGap;return 0;
}
extern "C" int xpbd_sync_solved_tool(){return syncSolvedTool();}
static void refreshGrip(){
    cudaMemsetAsync(d_gripDegree,0,(size_t)g_N*sizeof(int));
    cudaMemsetAsync(d_gripLive,0,sizeof(int));
    cudaMemsetAsync(d_grasped,0,(size_t)g_N*sizeof(int));cudaMemsetAsync(d_graspCore,0,(size_t)g_N*sizeof(int));
    if(g_gripCount>0)k_RefreshGrip<<<Gr(g_gripCount),64>>>(g_gripCount,d_gripPoints,g_surfaceMasks,g_liveActive,d_gripDegree,d_grasped,d_graspCore,d_gripLive);
    cudaMemsetAsync(d_gripPatchWeights,0,(size_t)3*g_N*sizeof(float));k_ClearPatch<<<1,1>>>(d_gripPatch);
    if(g_gripCount>0){
        k_BuildPatch<<<Gr(g_gripCount),64>>>(g_gripCount,d_gripPoints,d_gripPatchWeights,d_gripPatch);
        k_NormalizePatch<<<Gr(g_N),64>>>(g_N,d_gripPatchWeights,d_gripPatch);
        k_BuildPatchShape<<<Gr(g_gripCount),64>>>(g_gripCount,g_N,d_gripPoints,g_gripSigma,d_gripPatchWeights,d_gripPatch);
    }
}
// Physical commit boundary, independent of whether a client asks for feedback.
extern "C" int xpbd_commit_grip_topology(){
    if(g_bound && g_embeddedGrip && g_grasping)refreshGrip();
    return -(int)cudaGetLastError();
}
static int captureGrip(){
    if(g_surfaceCount<=0)return 0;
    destroyGripGraph();
    if(g_gripCapacity<g_surfaceCount){
        GripPoint* points=nullptr;cudaError_t e=cudaMalloc(&points,(size_t)g_surfaceCount*sizeof(GripPoint));if(e!=cudaSuccess)return -(int)e;
        cudaFree(d_gripPoints);d_gripPoints=points;g_gripCapacity=g_surfaceCount;
    }
    k_BindSurface<<<Gr(3*g_surfaceCount),64>>>(3*g_surfaceCount,g_surfaceTags,g_surfaceExpanded,g_surfaceCorners,g_surfaceMasks,g_surfaceConn,
        d_pos,d_surfaceVertices,g_surfaceRest,g_surfaceMaterialFP,g_surfaceBoundaryMasks);
    cudaMemsetAsync(d_gripCounter,0,sizeof(int));cudaMemsetAsync(d_gripMoments,0,4*sizeof(float));
    // Capture at the actual instrument, not the command spring endpoint.
    XpbdToolState captureTool=d_coupledTool?g_previousAcceptedTool:g_tool;
    k_CaptureGrip<<<Gr(g_surfaceCount),64>>>(g_surfaceCount,d_surfaceVertices,g_surfaceMasks,d_pos,g_liveActive,g_livePinned,captureTool,g_gp.jawGapClosed,d_gripPoints,d_gripCounter,d_gripMoments);
    cudaError_t e=cudaMemcpy(&g_gripCount,d_gripCounter,sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    if(g_gripCount<=0)return 0;
    k_GripWeights<<<Gr(g_gripCount),64>>>(g_gripCount,d_gripPoints,g_gripSigma,d_gripMoments);
    k_NormalizeGrip<<<Gr(g_gripCount),64>>>(g_gripCount,d_gripPoints,g_gripSigma,d_gripMoments);
    if(g_trainingFlags)trainingCapture();
    refreshGrip();g_grasping=true;if(!d_coupledTool)g_toolFrom=g_tool;
    return -(int)cudaGetLastError();
}
// Tool poses live in device memory; one graph is reused through the whole hold.
// This avoids hundreds of small CPU kernel submissions per rendered grasp frame.
static cudaError_t solveGrip(){
    if(!g_embeddedGrip || !g_grasping || g_gripCount<=0)return cudaSuccess;
    if(!g_gripGraph){
        cudaError_t e=cudaSuccess;if(!g_captureStream)e=cudaStreamCreate(&g_captureStream);if(e!=cudaSuccess)return e;
        e=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(e!=cudaSuccess)return e;
        if(d_coupledTool){
            k_BeginPatchEval<<<1,1,0,g_captureStream>>>(d_gripPatch);
            k_EvalPatch<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_gripPatchWeights,d_pos,d_prevPos,d_invMass,d_coupledTool,d_gripPatch);
            k_SolvePatch<<<1,1,0,g_captureStream>>>(d_gripPatch,d_gripStep,d_coupledTool,g_gripSigma);
            k_ApplyPatch<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_gripPatchWeights,d_invMass,d_pos,d_coupledTool,d_gripPatch);
        } else for(int i=0;i<4;i++){
            cudaMemsetAsync(d_surfaceDelta,0,(size_t)g_N*sizeof(float3),g_captureStream);
            k_SolveGrip<<<Gr(g_gripCount),64,0,g_captureStream>>>(g_gripCount,d_gripPoints,d_pos,d_prevPos,d_invMass,d_gripDegree,d_gripStep,d_surfaceDelta,d_gripLive,d_coupledTool);
            k_ApplySurface<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_pos,d_surfaceDelta);
            if(d_coupledTool)k_ApplyTool<<<1,1,0,g_captureStream>>>(d_coupledTool);
        }
        cudaGraph_t graph=nullptr;e=cudaStreamEndCapture(g_captureStream,&graph);if(e!=cudaSuccess)return e;
        e=cudaGraphInstantiate(&g_gripGraph,graph,0);cudaGraphDestroy(graph);if(e!=cudaSuccess)return e;
    }
    return cudaGraphLaunch(g_gripGraph,0);
}

extern "C" int xpbd_set_tool(const XpbdToolState* t)
{
    if (!g_bound) return XPBD_ERR_NOT_READY;
    if (!t) return XPBD_ERR_BAD_ARG;
    g_previousAcceptedTool=g_toolArmed?g_tool:*t;
    g_toolFrom = g_toolArmed ? g_tool : *t;
    g_tool = *t;
    g_toolArmed = true;
    const int cc = g_N;

    float3 axis = g_tool.jawAxis; float alen = len3(axis);
    axis = (alen > 1e-9f) ? axis * (1.f / alen) : mkf3(0.f, 0.f, 1.f);
    float3 normal = g_tool.jawNormal; float nlen = len3(normal);
    normal = (nlen > 1e-9f) ? normal * (1.f / nlen) : mkf3(0.f, 1.f, 0.f);
    g_jawW = cross3(axis, normal);
    float wlen = len3(g_jawW);
    g_jawW = (wlen > 1e-9f) ? g_jawW * (1.f / wlen) : mkf3(1.f, 0.f, 0.f);

    float effGap  = fmaxf(g_tool.jawGap, g_gp.jawGapClosed);
    float gapHalf = 0.5f * effGap;

    bool nowAbove = g_tool.grasp01 > g_gp.graspThreshold;
    bool nowBelow = g_tool.grasp01 < g_gp.releaseThreshold;
    if(nowBelow)g_openJawGap=fmaxf(g_openJawGap,g_tool.jawGap);

    if(g_frictionContact && !g_anchorAssist) {
        // No candidate latch, center pin, or artificial velocity damping on release.
        g_grasping=nowAbove || (g_grasping && !nowBelow);
        return 0;
    }

    if(g_embeddedGrip && g_anchorAssist){
        // Pressing G while maximally open is only an intent to close. Capturing
        // the entire open wedge here welds two separated tissue patches before
        // a bite exists and makes the anchor itself oppose every closing move.
        bool biteFormed=g_jawStiffness<=0 || g_openJawGap<=1e-6f ||
            g_previousAcceptedTool.jawGap<=.8f*g_openJawGap;
        if(!g_grasping && nowAbove && biteFormed)return captureGrip();
        if(g_grasping && nowBelow){
            k_ReleaseDampVelocity<<<Gr(cc),64>>>(cc,d_grasped,d_vel);
            g_gripCount=0;g_grasping=false;refreshGrip();
        }
        return -(int)cudaGetLastError();
    }

    if (!g_grasping && nowAbove)
    {
        // ── CAPTURE (rising edge, design §5 item 1): latch the set ONCE — corners entering the band
        // LATER are deliberately not captured (the [audit rev-B] semantics note in xpbd_solver.h). ──
        // redesign 2026-07-07: reuse pre-allocated persistent scratch buffers (d_candScratch/d_uScratch/
        // d_vScratch/d_nScratch, allocated once in xpbd_bind's tail) instead of malloc/free on every
        // capture event — avoids a cudaMalloc/cudaFree storm if grasp01 crosses the capture threshold
        // rapidly (e.g. threshold chatter, or a fast repeated grip gesture).
        k_MarkGraspCandidates<<<Gr(cc), 64>>>(cc, g_tool.jawHinge, axis, normal, g_jawW,
            g_tool.jawLen, g_tool.jawHalfWidth, gapHalf, g_gp.contactSkin,
            g_liveActive, g_livePinned, d_pos, d_candScratch, d_uScratch, d_vScratch, d_nScratch);
        if(g_anchorAssist && g_captureWedge && d_captureInside){
            k_InsideJawWedge<<<Gr(cc),64>>>(cc,g_tool,g_gp.jawGapClosed,d_captureInside,d_candScratch,d_uScratch,d_vScratch,d_nScratch);
            if(g_surfaceCount>0)k_EmbeddedBite<<<Gr(g_surfaceCount),64>>>(g_surfaceCount,d_surfaceVertices,g_surfaceMasks,d_pos,g_tool,g_gp.jawGapClosed,g_liveActive,g_livePinned,d_candScratch);
        } else if(g_anchorAssist){
            // Require actual pressure on a jaw, as well as proximity to its grasp volume.
            // Shaft-only contact or closing in empty space must never latch tissue.
            cudaMemsetAsync(d_contactNormalLambda,0,(size_t)cc*sizeof(float));
            if(g_surfaceCount>0)k_TouchedByJaws<<<Gr(3*g_surfaceCount),64>>>(3*g_surfaceCount,d_surfaceContacts,d_surfaceVertices,g_surfaceMasks,d_contactNormalLambda);
            k_KeepTouched<<<Gr(cc),64>>>(cc,d_candScratch,d_contactNormalLambda);
        }
        if(g_captureWedge)k_BiteCoordinates<<<Gr(cc),64>>>(cc,d_candScratch,d_pos,g_tool.jawHinge,axis,normal,g_jawW,d_uScratch,d_vScratch,d_nScratch);
        std::vector<int> hCand(cc); std::vector<float> hU(cc), hV(cc), hN(cc);
        cudaMemcpy(hCand.data(), d_candScratch, (size_t)cc * sizeof(int),   cudaMemcpyDeviceToHost);
        cudaMemcpy(hU.data(),    d_uScratch,    (size_t)cc * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hV.data(),    d_vScratch,    (size_t)cc * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hN.data(),    d_nScratch,    (size_t)cc * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<int> idxs;
        for (int c = 0; c < cc; c++) if (hCand[c]) idxs.push_back(c);
        if(g_anchorAssist && idxs.empty())return 0; // retry on contact while closing; no mid-air capture

        std::vector<int>   hGrasped(cc, 0), hCore(cc, 0);
        std::vector<float> hOffU(cc, 0.f), hOffN(cc, 0.f), hOffW(cc, 0.f);
        std::vector<float> hAnchU(cc, 0.f), hAnchW(cc, 0.f),hComplianceScale(cc,1.f),hWeight(cc,0.f);
        if (!idxs.empty())
        {
            // NORMALIZED-ELLIPSE core (design §5): normalized by the CAPTURED SET's own u/v extents (NOT
            // the fixed jaw geometry) — GripperTool.cs:507-532.
            float uMin = 1e30f, uMax = -1e30f, vMin = 1e30f, vMax = -1e30f;
            for (int c : idxs) { uMin = fminf(uMin, hU[c]); uMax = fmaxf(uMax, hU[c]);
                                 vMin = fminf(vMin, hV[c]); vMax = fmaxf(vMax, hV[c]); }
            float uMid = 0.5f * (uMin + uMax), uHalf = fmaxf(0.5f * (uMax - uMin), 1e-6f);
            float vMid = 0.5f * (vMin + vMax), vHalf = fmaxf(0.5f * (vMax - vMin), 1e-6f);
            float frac = g_gp.coreEllipseFrac;
            int bestIdx = -1; float bestD2 = 1e30f;
            for (int c : idxs)
            {
                hGrasped[c] = 1;
                float nu = (hU[c] - uMid) / uHalf, nv = (hV[c] - vMid) / vHalf;
                float d2 = nu * nu + nv * nv;
                if ((g_anchorAssist && g_captureWedge && d_captureInside) || d2 < frac * frac) hCore[c] = 1;
                hWeight[c]=.15f+.85f*powf(fmaxf(0.f,1.f-d2),2.f);
                if (d2 < bestD2) { bestD2 = d2; bestIdx = c; }
            }
            bool anyCore = false; for (int c : idxs) if (hCore[c]) { anyCore = true; break; }
            if (!anyCore && bestIdx >= 0) hCore[bestIdx] = 1;   // MANDATORY fallback (design §5): the
                                                                 // grasp must never be core-less.
            // Normalize total patch stiffness, not per-node stiffness. A smooth
            // edge taper transfers load into tissue without a hard welded rim.
            float totalWeight=0;for(int c:idxs)if(hCore[c])totalWeight+=hWeight[c];
            for(int c:idxs)if(hCore[c])hComplianceScale[c]=totalWeight/fmaxf(hWeight[c],1e-6f);
            for (int c : idxs)
            {
                hOffU[c] = hU[c]; hOffN[c] = hN[c]; hOffW[c] = hV[c];   // jaw-local anchor at capture time
                hAnchU[c] = hU[c]; hAnchW[c] = hV[c];                   // coulomb anchor starts at zero slip
            }
        }
        cudaMemcpy(d_coreComplianceScale,hComplianceScale.data(),(size_t)cc*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_grasped,        hGrasped.data(), (size_t)cc * sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_graspCore,      hCore.data(),     (size_t)cc * sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_coreOffU,       hOffU.data(),     (size_t)cc * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coreOffN,       hOffN.data(),     (size_t)cc * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coreOffW,       hOffW.data(),     (size_t)cc * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coulombAnchorU, hAnchU.data(),    (size_t)cc * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coulombAnchorW, hAnchW.data(),    (size_t)cc * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_slabLambda,     0, (size_t)cc * sizeof(float));
        cudaMemset(d_slabLambdaPrev, 0, (size_t)cc * sizeof(float));   // Fn seed 0 on capture (design §5 item 3)
        cudaMemset(d_coreForceEMA,   0, (size_t)cc * sizeof(float));   // fresh break-detection state per grasp
                                                                        // session (unlike d_coreLambda, this is
                                                                        // NOT substep-reset, so a stale EMA from
                                                                        // a PRIOR grasp must not leak into this one
        g_grasping = true;
        g_toolFrom = g_tool; // capture offsets belong to this pose
    }
    else if (g_grasping && nowBelow)
    {
        // ── RELEASE (falling edge): damp x0.1 THEN clear. No invMass save/restore bookkeeping is needed
        // AT ALL (redesign 2026-07-07): core corners keep their NORMAL invMass the whole time they're
        // captured (k_RefreshLiveness's w-rule no longer reads graspCore, see that kernel's comment) --
        // clearing d_graspCore/d_grasped here simply stops k_SolveCorePin/k_SolveSlabCoulomb from acting
        // on the corner going forward; there is no frozen invMass state to undo. ──
        k_ReleaseDampVelocity<<<Gr(cc), 64>>>(cc, d_grasped, d_vel);
        cudaMemset(d_grasped,   0, (size_t)cc * sizeof(int));
        cudaMemset(d_graspCore, 0, (size_t)cc * sizeof(int));
        g_grasping = false;
    }

    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

// ── M2 API split (design §9-(4)): xpbd_begin_frame computes the h-derived coefficients ONCE per outer
// frame (§12 R-compliance: ALL h-powers recomputed together — a~ ~1/h^2, gamma ~1/h, velScale ~h);
// xpbd_substep advances exactly ONE substep. The eventual live caller (physics_step) owns the N-loop and
// interleaves cut_ribbon_tick() between substeps; xpbd_step below stays as the monolithic wrapper for
// the standalone self-test. (xpbd_step IS begin_frame + N x substep by construction; selftest check N
// locks that loop SHAPE + that foreign work interleaved between substeps does not perturb the state —
// the refactor's physics fidelity is carried by checks A..M + the oracle, not by N alone.) ─────────────
extern "C" int xpbd_begin_frame(float dt)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    g_frameReady = false;                               // a FAILED begin must not leave the OLD frame armed
    int N = g_p.numSubsteps > 0 ? g_p.numSubsteps : 1;
    float h = dt / (float)N;
    if (!(h > 0.f) || !std::isfinite(h)) return XPBD_ERR_BAD_DT;  // rejects <=0, NaN (!(NaN>0)) AND +Inf
                                                        // (Inf would make gamma = 0*Inf = NaN downstream)
    float alphaTilde = g_p.ks > 0.f ? 1.f / (g_p.ks * h * h) : 0.f;          // a~ = (1/ks)/h^2
    // STEP 3 damping: alpha (Rayleigh) as the implicit per-substep velocity term v *= 1/(1+alpha*h);
    // cs/cb (Kelvin-Voigt) as the Macklin XPBD damped-constraint term gamma = a~*beta*h.
    g_velScaleF    = (1.f - g_p.damping) / (1.f + g_p.alpha * h);
    g_betaGammaHSF = alphaTilde * g_p.betaS * h;
    g_alphaTildeF  = alphaTilde;
    g_itersF       = g_p.iters > 0 ? g_p.iters : 1;
    g_doBendF      = (g_B > 0 && g_p.kb > 0.f);
    float alphaB   = g_doBendF ? 1.f / (g_p.kb * h * h) : 0.f;     // a~_bend = (1/kb)/h^2
    g_betaGammaHBF = alphaB * g_p.betaB * h;
    g_alphaBF      = alphaB;
    g_hFrame       = h;
    // STEP 7: recompute the PER-TET alpha (1/h^2-scaled, §12 R-compliance) once per frame — muV/lambdaV
    // are h-independent material x rest-volume bakes from xpbd_build_tets.
    g_doStrainF = (g_haveTets && g_T > 0 && g_p.youngE > 0.f);
    if (g_doStrainF)
        k_ComputeTetAlpha<<<Gr(g_T), 64>>>(g_T, 1.f / (h * h), d_tetMuV, d_tetLambdaV, d_tetAlphaDev, d_tetAlphaHyd);
    g_substepIndex = 0;
    if(d_contactResidual)cudaMemsetAsync(d_contactResidual,0,3*sizeof(float));
    if(g_bound){cudaMemsetAsync(d_contactReaction,0,sizeof(float4));cudaMemsetAsync(d_jawLoad,0,sizeof(float4));}
    if(g_frictionContact && g_surfaceCount>0)
        k_BindSurface<<<Gr(3*g_surfaceCount),64>>>(3*g_surfaceCount,g_surfaceTags,g_surfaceExpanded,
            g_surfaceCorners,g_surfaceMasks,g_surfaceConn,d_pos,d_surfaceVertices,g_surfaceRest,g_surfaceMaterialFP,g_surfaceBoundaryMasks);
    g_frameReady   = true;
    return 0;
}

// Cache the exact coloured Gauss-Seidel order in a CUDA graph. Live cut flags,
// positions and compliance buffers remain device inputs; no constraints are skipped.
static cudaError_t solveTetGraph(float damping){
    if(g_tetGraph && (g_graphOmega!=g_p.omega || g_graphDamping!=damping || g_graphTetCount!=g_T || g_graphH!=g_hFrame))destroyTetGraph();
    if(!g_tetGraph){
        cudaError_t e=cudaSuccess;
        if(!g_captureStream)e=cudaStreamCreate(&g_captureStream);
        if(e!=cudaSuccess)return e;
        e=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(e!=cudaSuccess)return e;
        for(int c=0;c<g_tNumColors;c++)if(g_tColorCnt[c]>0)
            k_solve_tet<<<Gr(g_tColorCnt[c]),64,0,g_captureStream>>>(g_tColorOff[c],g_tColorCnt[c],g_p.omega,damping,g_hFrame>0?g_bulkDampingSeconds/g_hFrame:0.f,d_prevPos,
                d_tetIds,d_tetBr0,d_tetBr1,d_tetBr2,d_tetActive,d_tetAlphaDev,d_tetAlphaHyd,d_invMass,d_tetLambdaDevL,d_tetLambdaHydL,d_pos,g_volumePasses>0?1.f:0.f,d_tetColorOrder);
        for(int pass=0;pass<g_localPasses;pass++)for(int c=0;c<g_tNumColors;c++)if(g_localCount[c]>0)
            k_solve_tet<<<Gr(g_localCount[c]),64,0,g_captureStream>>>(g_localOff[c],g_localCount[c],g_p.omega,damping,g_hFrame>0?g_bulkDampingSeconds/g_hFrame:0.f,d_prevPos,
                d_tetIds,d_tetBr0,d_tetBr1,d_tetBr2,d_tetActive,d_tetAlphaDev,d_tetAlphaHyd,d_invMass,d_tetLambdaDevL,d_tetLambdaHydL,d_pos,g_volumePasses>0?1.f:0.f,d_localTets);
        for(int pass=0;pass<g_volumePasses;pass++)for(int c=0;c<g_tNumColors;c++)if(g_tColorCnt[c]>0)
            k_TensileReinforcement<<<Gr(g_tColorCnt[c]),64,0,g_captureStream>>>(g_tColorOff[c],g_tColorCnt[c],d_tetColorOrder,
                d_tetIds,d_tetActive,d_tetBr0,d_tetBr1,d_tetBr2,d_invMass,d_pos,d_tetAlphaDev,d_fiberLambda,g_tissueHardening);
        cudaGraph_t graph=nullptr;e=cudaStreamEndCapture(g_captureStream,&graph);if(e!=cudaSuccess)return e;
        e=cudaGraphInstantiate(&g_tetGraph,graph,0);cudaGraphDestroy(graph);if(e!=cudaSuccess)return e;
        g_graphOmega=g_p.omega;g_graphDamping=damping;g_graphTetCount=g_T;g_graphH=g_hFrame;
    }
    return cudaGraphLaunch(g_tetGraph,0);
}

static cudaError_t solvePressureGraph(){
    if(g_pressureGraph && g_pressureH!=g_hFrame){cudaGraphExecDestroy(g_pressureGraph);g_pressureGraph=nullptr;}
    if(!g_pressureGraph){
        cudaError_t e=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(e!=cudaSuccess)return e;
        for(int c=0;c<g_tNumColors;c++)if(g_tColorCnt[c]>0){
            if(d_coupledTool && g_doStrainF)
                // Let shear redistribute contact loads into the nearby tissue,
                // instead of alternating contact with a bulk-only projection.
                k_solve_tet<<<Gr(g_tColorCnt[c]),64,0,g_captureStream>>>(g_tColorOff[c],g_tColorCnt[c],g_p.omega,
                    g_p.strainDamping/g_hFrame,g_bulkDampingSeconds/g_hFrame,d_prevPos,
                    d_tetIds,d_tetBr0,d_tetBr1,d_tetBr2,d_tetActive,d_tetAlphaDev,d_tetAlphaHyd,d_invMass,d_tetLambdaDevL,d_tetLambdaHydL,d_pos,1.f,d_tetColorOrder,d_coupledTool);
            else k_SolveBulk<<<Gr(g_tColorCnt[c]),64,0,g_captureStream>>>(g_tColorOff[c],g_tColorCnt[c],d_tetColorOrder,
                d_tetIds,d_tetActive,d_tetBr0,d_tetBr1,d_tetBr2,d_invMass,d_prevPos,d_pos,d_tetAlphaHyd,d_tetLambdaHydL,g_bulkDampingSeconds/fmaxf(g_hFrame,1e-8f));
        }
        cudaGraph_t graph=nullptr;e=cudaStreamEndCapture(g_captureStream,&graph);if(e!=cudaSuccess)return e;
        e=cudaGraphInstantiate(&g_pressureGraph,graph,0);cudaGraphDestroy(graph);if(e!=cudaSuccess)return e;g_pressureH=g_hFrame;
    }
    return cudaGraphLaunch(g_pressureGraph,0);
}

static cudaError_t solveSurfaceGraph(float alpha,int closing){
    if(g_surfaceGraph && (g_graphAlpha!=alpha || g_graphClosing!=closing || g_graphMuS!=g_muStatic || g_graphMuD!=g_muDynamic))destroySurfaceGraph();
    if(!g_surfaceGraph){
        cudaError_t e=cudaSuccess;if(!g_captureStream)e=cudaStreamCreate(&g_captureStream);if(e!=cudaSuccess)return e;
        e=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(e!=cudaSuccess)return e;
        if(d_coupledTool){for(int refresh=0;refresh<2;refresh++){
            k_RefreshActiveWitness<<<128,64,0,g_captureStream>>>(d_surfaceActive,d_surfaceActiveCount,d_surfaceContacts,d_surfaceVertices,d_pos,d_prevPos,d_coupledTool,g_jawThickness,g_gp.jawGapClosed,g_contactRadius);
            k_SolveContactIsland<<<1,32,0,g_captureStream>>>(4,alpha,g_muStatic,g_muDynamic,d_invMass,d_pos,d_surfaceContacts,d_surfaceActive,d_surfaceActiveCount,d_coupledTool);
        }}
        else for(int it=0;it<8;it++){
            cudaMemsetAsync(d_surfaceDelta,0,(size_t)g_N*sizeof(float3),g_captureStream);
            k_SolveSurface<<<128,64,0,g_captureStream>>>(3*g_surfaceCount,alpha,g_muStatic,g_muDynamic,
                d_invMass,d_pos,d_surfaceDegree,d_surfaceContacts,d_surfaceDelta,d_grasped,closing,d_surfaceActive,d_surfaceActiveCount,d_coupledTool);
            k_ApplySurface<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_pos,d_surfaceDelta);
            if(d_coupledTool)k_ApplyTool<<<1,1,0,g_captureStream>>>(d_coupledTool);
        }
        cudaGraph_t graph=nullptr;e=cudaStreamEndCapture(g_captureStream,&graph);if(e!=cudaSuccess)return e;
        e=cudaGraphInstantiate(&g_surfaceGraph,graph,0);cudaGraphDestroy(graph);if(e!=cudaSuccess)return e;
        g_graphAlpha=alpha;g_graphClosing=closing;g_graphMuS=g_muStatic;g_graphMuD=g_muDynamic;
    }
    return cudaGraphLaunch(g_surfaceGraph,0);
}

// Active contact IDs/count are GPU inputs; no graph recapture when only the
// reconstructed triangle count changes. Reallocation invalidates all pointers.
static cudaError_t repairSurfaceGraph(){
    if(!g_repairGraph){
        cudaError_t e=cudaSuccess;if(!g_captureStream)e=cudaStreamCreate(&g_captureStream);if(e!=cudaSuccess)return e;
        e=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(e!=cudaSuccess)return e;
        if(d_coupledTool){for(int refresh=0;refresh<2;refresh++){
            k_RefreshActiveWitness<<<128,64,0,g_captureStream>>>(d_surfaceActive,d_surfaceActiveCount,d_surfaceContacts,d_surfaceVertices,d_pos,d_prevPos,d_coupledTool,g_jawThickness,g_gp.jawGapClosed,g_contactRadius);
            k_SolveContactIsland<<<1,32,0,g_captureStream>>>(8,0,0,0,d_invMass,d_pos,d_surfaceContacts,d_surfaceActive,d_surfaceActiveCount,d_coupledTool);
        }}
        else for(int it=0;it<16;it++){
            cudaMemsetAsync(d_surfaceDelta,0,(size_t)g_N*sizeof(float3),g_captureStream);
            k_SolveSurface<<<128,64,0,g_captureStream>>>(3*g_surfaceCapacity,0,0,0,d_invMass,d_pos,d_surfaceDegree,
                d_surfaceContacts,d_surfaceDelta,d_grasped,0,d_surfaceActive,d_surfaceActiveCount,d_coupledTool);
            k_ApplySurface<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_pos,d_surfaceDelta);
            if(d_coupledTool)k_ApplyTool<<<1,1,0,g_captureStream>>>(d_coupledTool);
        }
        cudaGraph_t graph=nullptr;e=cudaStreamEndCapture(g_captureStream,&graph);if(e!=cudaSuccess)return e;
        e=cudaGraphInstantiate(&g_repairGraph,graph,0);cudaGraphDestroy(graph);if(e!=cudaSuccess)return e;
    }
    return cudaGraphLaunch(g_repairGraph,0);
}

static cudaError_t dampSurfaceGraph(float h){
    if(g_velocityGraph && (g_velocityH!=h)){cudaGraphExecDestroy(g_velocityGraph);g_velocityGraph=nullptr;}
    if(!g_velocityGraph){
        cudaError_t e=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(e!=cudaSuccess)return e;
        for(int it=0;it<2;it++){
            cudaMemsetAsync(d_surfaceDelta,0,(size_t)g_N*sizeof(float3),g_captureStream);
            if(d_coupledTool)k_ClearToolVelocityDelta<<<1,1,0,g_captureStream>>>(d_coupledTool);
            k_SurfaceVelocity<<<128,64,0,g_captureStream>>>(3*g_surfaceCount,1.f/h,d_invMass,d_surfaceDegree,d_surfaceContacts,d_vel,d_surfaceDelta,d_surfaceActive,d_surfaceActiveCount,d_coupledTool);
            k_ApplySurface<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_vel,d_surfaceDelta);
            if(d_coupledTool)k_ApplyToolVelocity<<<1,1,0,g_captureStream>>>(d_coupledTool);
        }
        cudaGraph_t graph=nullptr;e=cudaStreamEndCapture(g_captureStream,&graph);if(e!=cudaSuccess)return e;
        e=cudaGraphInstantiate(&g_velocityGraph,graph,0);cudaGraphDestroy(graph);if(e!=cudaSuccess)return e;
        g_velocityH=h;
    }
    return cudaGraphLaunch(g_velocityGraph,0);
}

// Remove residual overlap on the CURRENT nonlinear DC surface. This updates
// material nodes, not an independently displaced display mesh. No artificial
// tangential anchor or extra force is added by this geometric correction.
#include "../cutting/pressure_blade.cuh"
#include "tool_feedback.cuh"
#include "../zero_mode_gates.cuh"


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
        if(e==cudaSuccess)e=cudaMalloc(&L.r,(size_t)capacity*sizeof(MgResidual));
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
    if(e==cudaSuccess)e=cudaMemset(d_multigridFine,0,(size_t)g_N*sizeof(float3));
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
    auto blocks=[](int capacity){return (capacity+MG_GALERKIN_ROWS-1)/MG_GALERKIN_ROWS;};
    k_MgGalerkinFine<<<blocks(g_multigrid[1].capacity),8*MG_GALERKIN_ROWS,0,stream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgStencil,g_multigrid[1]);
    for(int l=2;l<=g_multigridLevels;l++)
        k_MgGalerkin<<<blocks(g_multigrid[l].capacity),8*MG_SLOTS*MG_GALERKIN_ROWS,0,stream>>>(g_multigrid[l-1],g_multigrid[l]);
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
    cudaMemsetAsync(L[1].r,0,(size_t)L[1].capacity*sizeof(MgResidual),stream);
    k_MgFinePre<<<Gr(g_activeNodes),64,0,stream>>>(d_unified,d_cgStencil,d_cgH,r,d_multigridFine);
    k_MgFineRestrict<<<Gr(g_activeNodes),64,0,stream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgStencil,r,d_multigridFine,L[1]);
    for(int l=1;l<top;l++){
        cudaMemsetAsync(L[l+1].r,0,(size_t)L[l+1].capacity*sizeof(MgResidual),stream);
        k_MgPre<<<Gr(L[l].capacity),64,0,stream>>>(L[l]);
        k_MgRestrict<<<Gr(L[l].capacity),64,0,stream>>>(L[l],L[l+1]);
    }
    k_MgCoarsest<<<Gr(3*MG_COARSEST),64,0,stream>>>(L[top],d_multigridInverse);
    for(int l=top-1;l>=1;l--){
        k_MgProlong<<<Gr(L[l].capacity),64,0,stream>>>(L[l],L[l+1]);
        k_MgPost<<<Gr(L[l].capacity),64,0,stream>>>(L[l]);
    }
    k_MgFineProlong<<<Gr(g_activeNodes),64,0,stream>>>(d_unified,d_cgStencil,L[1],d_multigridFine);
    k_MgFinePost<<<Gr(g_activeNodes),64,0,stream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgStencil,d_cgH,r,d_multigridFine,z,sums,slot);
}

// Cut edges can change BETWEEN substeps and during final reconstruction.
// Never reuse a pre-cut component map for a coarse solve of separated tissue: the passes run whenever the topology
// revision has moved since the labels were made, and leave the labels of an unchanged topology as they are.
static cudaError_t refreshUnifiedComponents(){
    if(!d_unifiedComponents || g_N<=0)return cudaErrorInvalidValue;
    const unsigned* stale=d_componentRevision+1;
    k_TopologyStale<<<1,1>>>(cut_topology_revision_device(),d_componentRevision);
    k_InitComponents<<<Gr(g_N),64>>>(g_N,d_unifiedComponents,stale);
    if(g_liveNbrIdx)k_JoinComponents<<<Gr(g_N),64>>>(g_N,g_liveNbrIdx,g_liveActive,d_unifiedComponents,stale);
    k_FlattenComponents<<<Gr(g_N),64>>>(g_N,d_unifiedComponents,stale);
    return cudaGetLastError();
}

static cudaError_t refreshFragmentSupport(){
    cudaError_t e=cudaSuccess;
    if(!d_fragmentSupported)e=cudaMalloc(&d_fragmentSupported,g_N*sizeof(int));
    if(e==cudaSuccess && !d_fragmentUnpinnedVolume)e=cudaMalloc(&d_fragmentUnpinnedVolume,sizeof(int));
    if(e!=cudaSuccess)return e;
    cudaMemsetAsync(d_fragmentSupported,0,g_N*sizeof(int));
    cudaMemsetAsync(d_fragmentUnpinnedVolume,0,sizeof(int));
    k_FragmentSupports<<<Gr(g_N),64>>>(g_N,d_unifiedComponents,d_invMass,g_liveActive,d_fragmentSupported);
    if(g_T>0)k_FragmentVolumes<<<Gr(g_T),64>>>(g_T,d_tetIds,d_tetActive,d_unifiedComponents,d_fragmentSupported,d_fragmentUnpinnedVolume);
    return cudaGetLastError();
}

// The eight fragment contact passes. Without a fragment query (k_FragmentQueries) every one of their kernels returns
// at once and they change nothing.
static cudaError_t launchFragmentPasses(cudaStream_t stream,bool prepare,const float3* previous,bool discovery=false,bool refresh=false){
    const float3* positions=discovery && !refresh?d_unifiedTarget:d_pos;
    const float3* start=discovery?d_unifiedTarget:d_fragmentStart;
    if(discovery && !refresh){
        cudaMemsetAsync(d_fragmentDuplicateHeads,0xff,g_N*sizeof(int),stream);
        cudaMemsetAsync(d_fragmentDuplicateAmbiguous,0,g_N*sizeof(int),stream);
        cudaMemsetAsync(d_fragmentUniqueCount,0,sizeof(int),stream);
        k_FragmentDuplicateLinks<<<128,64,0,stream>>>(g_N,g_surfaceCount,d_fragmentQueries,d_fragmentQueryCount,d_surfaceVertices,d_fragmentDuplicateHeads,d_fragmentDuplicateAmbiguous,d_fragmentDuplicateNext);
        k_FragmentUniqueQueries<<<128,64,0,stream>>>(g_N,g_surfaceCount,d_fragmentQueries,d_fragmentQueryCount,d_surfaceVertices,d_fragmentDuplicateHeads,d_fragmentDuplicateAmbiguous,d_fragmentDuplicateNext,d_fragmentUniqueQueries,d_fragmentUniqueCount);
    }
    for(int pass=0;pass<(discovery?1:8);pass++){
        k_FragmentVertices<<<Gr(3*g_surfaceCount),64,0,stream>>>(3*g_surfaceCount,d_surfaceVertices,d_unifiedComponents,positions,previous,start,d_fragmentVertices,d_fragmentQueryCount);

        k_InitFragmentBounds<<<Gr(g_N),64,0,stream>>>(g_N,d_fragmentBounds,d_fragmentQueryCount);
        k_FragmentSurfaceBounds<<<Gr(3*g_surfaceCount),64,0,stream>>>(3*g_surfaceCount,d_fragmentVertices,d_fragmentBounds,d_fragmentQueryCount);
        cudaMemsetAsync(d_fragmentRootCount,0,sizeof(int),stream);
        k_ListFragmentRoots<<<Gr(g_N),64,0,stream>>>(g_N,d_fragmentBounds,d_fragmentRoots,d_fragmentRootCount,d_fragmentQueryCount);
        // Discovery initializes a permutation for this substep. Every refresh
        // refits ALL leaves and ancestors below, so the same permutation is a
        // valid hierarchy even while the tissue moves. Re-sorting Morton keys
        // at each nonlinear refresh repeats work without changing coverage.
        // New substeps/topology still take the full discovery/sort path.
        if(pass==0 && !refresh){
#ifdef ENDO_KIDNEY_CLOSED_RAY
            xpbd_section_mark("fragment: start certificate");
            cudaError_t certified=prepareFragmentClosedCertificate(stream,g_N,g_surfaceCount,d_surfaceVertices,d_unifiedComponents,d_fragmentQueryCount,d_fragmentClosedCanonical,
                d_fragmentKeys,d_fragmentSortedKeys,d_fragmentOrder,d_fragmentSortedOrder,d_fragmentSortTemp,g_fragmentSortBytes);
            if(certified!=cudaSuccess)return certified;
            xpbd_section_mark("fragment: closure certificate");
#endif
            k_FragmentTreeKeys<<<(g_surfaceCount+63)/64,64,0,stream>>>(g_surfaceCount,d_fragmentVertices,d_fragmentBounds,d_fragmentRoots,d_fragmentRootCount,d_fragmentKeys,d_fragmentOrder,d_fragmentQueryCount);
            size_t availableSortBytes=g_fragmentSortBytes;
            cudaError_t sorted=cub::DeviceRadixSort::SortPairs(d_fragmentSortTemp,availableSortBytes,d_fragmentKeys,d_fragmentSortedKeys,d_fragmentOrder,d_fragmentSortedOrder,g_surfaceCount,0,64,stream);
            if(sorted!=cudaSuccess)return sorted;
        }
        k_FragmentTreeLeaves<<<(32*g_fragmentTreeBase+63)/64,64,0,stream>>>(g_surfaceCount,g_fragmentTreeBase,d_fragmentVertices,d_fragmentSortedOrder,d_fragmentTree,d_fragmentQueryCount);
        for(int level=g_fragmentTreeBase/2;level;level/=2)k_FragmentTreeParents<<<(level+63)/64,64,0,stream>>>(level,level,d_fragmentTree,d_fragmentQueryCount);
#ifdef ENDO_KIDNEY_CACHE_TREE_BOUNDS
        k_FragmentTreePrepareQueries<<<Gr(2*g_fragmentTreeBase),64,0,stream>>>(2*g_fragmentTreeBase,d_fragmentTree,d_fragmentQueryCount);
#endif
        xpbd_section_mark("fragment: tree refit and bounds");
        if(discovery){
            FragmentManifold manifold{d_fragmentConstraints,d_fragmentConstraintCount,d_fragmentConstraintDegree,g_liveRestNbr,d_fragmentQuerySlots,d_fragmentDuplicateNext};
            launchFragmentContact(stream,g_surfaceCount,d_fragmentUniqueQueries,d_fragmentUniqueCount,d_unifiedComponents,positions,previous,start,
                d_surfaceVertices,d_fragmentVertices,d_invMass,nullptr,nullptr,d_fragmentBounds,d_fragmentRoots,d_fragmentRootCount,d_fragmentTree,g_fragmentTreeBase,d_fragmentSortedOrder,d_fragmentContacted,manifold,d_fragmentHits);
            xpbd_section_mark("fragment: paired query");
            return cudaGetLastError();
        }
        if(!prepare)k_FragmentEdges<<<Gr(g_N),64,0,stream>>>(g_N,g_liveNbrIdx,g_liveRestNbr,d_unifiedComponents,d_fragmentSupported,d_graspComponents,d_invMass,d_pos,g_p.youngE,g_hFrame,d_fragmentLambda,d_surfaceDelta,d_fragmentDegree,d_fragmentQueryCount);
        // A fixed grid of warps strides over the device query list: no host readback of its length.
        launchFragmentContact(stream,g_surfaceCount,d_fragmentQueries,d_fragmentQueryCount,d_unifiedComponents,d_pos,previous,d_fragmentStart,
            d_surfaceVertices,d_fragmentVertices,d_invMass,d_surfaceDelta,d_fragmentDegree,d_fragmentBounds,d_fragmentRoots,d_fragmentRootCount,d_fragmentTree,g_fragmentTreeBase,d_fragmentSortedOrder,nullptr,{},d_fragmentHits);
        k_ApplyFragmentContact<<<Gr(g_N),64,0,stream>>>(g_N,d_pos,d_surfaceDelta,d_fragmentDegree,d_fragmentQueryCount);
        k_UnsupportedTools<<<Gr(g_N),64,0,stream>>>(g_N,d_fragmentSupported,d_unifiedComponents,d_graspComponents,d_invMass,previous,d_pos,d_coupledTool,g_blade.enabled?d_bladeBody:nullptr,g_jawThickness,g_gp.jawGapClosed,d_fragmentQueryCount);
    }
    return cudaGetLastError();
}
__global__ void k_FragmentPassCondition(cudaGraphConditionalHandle handle,const int* queryCount){
    cudaGraphSetConditional(handle,*queryCount>0?1u:0u);
}
// The launch arguments the fragment pass graph bakes in: a change of any of them builds it again.
struct FragmentPassKey{const void* pointers[41];int surfaceCount,n;float young,h,thickness,gap;};
// The fragment passes of solveFragmentContact (prepare or not) as a graph whose body runs only once a query exists.
// Tissue without detached fragments, the usual case, then pays one graph launch instead of 40 kernel launches that
// return at once.
static cudaGraphExec_t g_fragmentPassGraph[3]={};
static FragmentPassKey g_fragmentPassKey[3]={};
static cudaGraphExec_t g_predictedFragmentGraph=nullptr;
struct PredictedFragmentKey{FragmentPassKey pass;const void* state;const void* revision;};
static PredictedFragmentKey g_predictedFragmentKey={};
struct MaterialPreludeKey{PredictedFragmentKey fragment;const void* pointers[9];int values[9];};
static cudaGraphExec_t g_materialPreludeGraph=nullptr;
static MaterialPreludeKey g_materialPreludeKey={};
static void destroyFragmentPassGraphs(){
    if(g_materialPreludeGraph)cudaGraphExecDestroy(g_materialPreludeGraph);g_materialPreludeGraph=nullptr;
    if(g_predictedFragmentGraph)cudaGraphExecDestroy(g_predictedFragmentGraph);g_predictedFragmentGraph=nullptr;
    for(int k=0;k<3;k++){if(g_fragmentPassGraph[k])cudaGraphExecDestroy(g_fragmentPassGraph[k]);g_fragmentPassGraph[k]=nullptr;}
}
static FragmentPassKey makeFragmentPassKey(const float3* previous){
    FragmentPassKey key;memset(&key,0,sizeof(key));
    const void* pointers[41]={d_surfaceVertices,d_unifiedComponents,d_pos,previous,d_fragmentStart,d_fragmentVertices,d_fragmentQueryCount,
        g_liveNbrIdx,g_liveRestNbr,d_fragmentSupported,d_graspComponents,d_invMass,d_fragmentLambda,d_surfaceDelta,d_fragmentDegree,
        d_fragmentQueries,d_coupledTool,g_blade.enabled?d_bladeBody:nullptr,d_fragmentBounds,d_fragmentRoots,d_fragmentRootCount,d_fragmentTree,d_fragmentKeys,d_fragmentSortedKeys,d_fragmentOrder,d_fragmentSortedOrder,d_fragmentSortTemp,d_fragmentContacted,d_unifiedTarget,d_fragmentConstraints,d_fragmentConstraintCount,d_fragmentConstraintDegree,d_fragmentQuerySlots,d_fragmentUniqueQueries,d_fragmentUniqueCount,d_fragmentDuplicateHeads,d_fragmentDuplicateAmbiguous,d_fragmentDuplicateNext,d_fragmentUnpinnedVolume,d_fragmentHits,d_fragmentClosedCanonical};
    memcpy(key.pointers,pointers,sizeof(pointers));
    key.surfaceCount=g_surfaceCount;key.n=g_N;key.young=g_p.youngE;key.h=g_hFrame;key.thickness=g_jawThickness;key.gap=g_gp.jawGapClosed;
    return key;
}
static cudaError_t runFragmentPasses(bool prepare,const float3* previous,bool discovery=false){
    const int slot=discovery?2:prepare?1:0;
    // Section profiles time the passes kernel by kernel.
    if(g_sectionProfile)return launchFragmentPasses(0,prepare,previous,discovery);
    FragmentPassKey key=makeFragmentPassKey(previous);
    cudaGraphExec_t& exec=g_fragmentPassGraph[slot];
    if(!exec || memcmp(&key,&g_fragmentPassKey[slot],sizeof(key))){
        cudaError_t e=cudaSuccess;
        if(!g_captureStream)e=cudaStreamCreate(&g_captureStream);
        cudaGraph_t graph=nullptr;if(e==cudaSuccess)e=cudaGraphCreate(&graph,0);
        cudaGraphConditionalHandle handle=0;
        if(e==cudaSuccess)e=cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault);
        cudaGraphNode_t condition=nullptr;const int* count=d_fragmentQueryCount;void* arguments[]={&handle,&count};
        cudaKernelNodeParams kernel={};kernel.func=(void*)k_FragmentPassCondition;kernel.gridDim=dim3(1);kernel.blockDim=dim3(1);
        kernel.kernelParams=arguments;
        if(e==cudaSuccess)e=cudaGraphAddKernelNode(&condition,graph,nullptr,0,&kernel);
        cudaGraphNodeParams branch={cudaGraphNodeTypeConditional};
        branch.conditional.handle=handle;branch.conditional.type=cudaGraphCondTypeIf;branch.conditional.size=1;
        cudaGraphNode_t node=nullptr;if(e==cudaSuccess)e=cudaGraphAddNode(&node,graph,&condition,1,&branch);
        if(e==cudaSuccess)e=cudaStreamBeginCaptureToGraph(g_captureStream,branch.conditional.phGraph_out[0],nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal);
        if(e==cudaSuccess){
            cudaError_t passError=launchFragmentPasses(g_captureStream,prepare,previous,discovery);
            cudaGraph_t body=nullptr;e=cudaStreamEndCapture(g_captureStream,&body);if(passError!=cudaSuccess)e=passError;
        }
        if(e==cudaSuccess)e=installRuntimeGraph(exec,graph);
        if(graph)cudaGraphDestroy(graph);
        if(e!=cudaSuccess)return e;
        g_fragmentPassKey[slot]=key;
    }
    return cudaGraphLaunch(exec,0);
}

// Reads the component support of the caller's latest refreshFragmentSupport.
static cudaError_t allocateFragmentContact(){
    if(!d_unifiedComponents || g_surfaceCount<=0)return cudaSuccess;
    cudaError_t e=cudaSuccess;
    if(!g_fragmentRefreshStream)e=cudaStreamCreateWithFlags(&g_fragmentRefreshStream,cudaStreamNonBlocking);
    if(e==cudaSuccess && !d_fragmentUniqueCount)e=cudaMalloc(&d_fragmentUniqueCount,sizeof(int));
    if(e==cudaSuccess && !d_fragmentDuplicateHeads)e=cudaMalloc(&d_fragmentDuplicateHeads,g_N*sizeof(int));
    if(e==cudaSuccess && !d_fragmentDuplicateAmbiguous)e=cudaMalloc(&d_fragmentDuplicateAmbiguous,g_N*sizeof(int));
    if(e==cudaSuccess && !d_fragmentContacted)e=cudaMalloc(&d_fragmentContacted,g_N*sizeof(int));
    if(e==cudaSuccess && !d_fragmentGroups)e=cudaMalloc(&d_fragmentGroups,g_N*sizeof(int));
    if(e==cudaSuccess && !d_fragmentLineSums)e=cudaMalloc(&d_fragmentLineSums,g_N*sizeof(float3));
    if(e==cudaSuccess && !d_fragmentConstraintCount)e=cudaMalloc(&d_fragmentConstraintCount,2*sizeof(int));
    if(e==cudaSuccess && !d_fragmentConstraintDegree)e=cudaMalloc(&d_fragmentConstraintDegree,g_N*sizeof(int));
    if(e==cudaSuccess && g_fragmentConstraintCapacity<g_N+3*g_surfaceCount){
        cudaFree(d_fragmentConstraints);d_fragmentConstraints=nullptr;cudaFree(d_fragmentQuerySlots);d_fragmentQuerySlots=nullptr;g_fragmentConstraintCapacity=0;
        cudaFree(d_fragmentUniqueQueries);d_fragmentUniqueQueries=nullptr;cudaFree(d_fragmentDuplicateNext);d_fragmentDuplicateNext=nullptr;
        cudaFree(d_fragmentHits);d_fragmentHits=nullptr;cudaFree(d_fragmentClosedCanonical);d_fragmentClosedCanonical=nullptr;
        int capacity=1;while(capacity<g_N+3*g_surfaceCount)capacity*=2;
#ifdef ENDO_KIDNEY_CLOSED_RAY
        if(e==cudaSuccess)e=cudaMalloc(&d_fragmentClosedCanonical,capacity*sizeof(int));
#endif
#ifdef ENDO_KIDNEY_SPLIT_FRAGMENT
        if(e==cudaSuccess)e=cudaMalloc(&d_fragmentHits,capacity*sizeof(FragmentHit));
#endif
        if(e==cudaSuccess)e=cudaMalloc(&d_fragmentConstraints,capacity*sizeof(FragmentConstraint));
        if(e==cudaSuccess)e=cudaMalloc(&d_fragmentQuerySlots,capacity*sizeof(int));
        if(e==cudaSuccess)e=cudaMalloc(&d_fragmentUniqueQueries,capacity*sizeof(int));
        if(e==cudaSuccess)e=cudaMalloc(&d_fragmentDuplicateNext,capacity*sizeof(int));
        if(e==cudaSuccess)g_fragmentConstraintCapacity=capacity;
        g_unifiedGraphDirty=g_unifiedLoadedGraphDirty=g_unifiedRecoveryGraphDirty=g_unifiedContactGraphDirty=true;
    }

    g_fragmentTreeBase=1;while(32*g_fragmentTreeBase<g_surfaceCount)g_fragmentTreeBase*=2;
    if(g_fragmentTreeCapacity<2*g_fragmentTreeBase){
        cudaFree(d_fragmentTree);d_fragmentTree=nullptr;g_fragmentTreeCapacity=0;
        e=cudaMalloc(&d_fragmentTree,2*g_fragmentTreeBase*sizeof(FragmentTreeNode));if(e==cudaSuccess)g_fragmentTreeCapacity=2*g_fragmentTreeBase;
    }
    int sortItems=g_surfaceCount;
#ifdef ENDO_KIDNEY_CLOSED_RAY
    sortItems*=3;
#endif
    if(e==cudaSuccess && g_fragmentSortCapacity<sortItems){
        int sortCapacity=1;while(sortCapacity<sortItems)sortCapacity*=2;
        cudaFree(d_fragmentKeys);cudaFree(d_fragmentSortedKeys);cudaFree(d_fragmentOrder);cudaFree(d_fragmentSortedOrder);cudaFree(d_fragmentSortTemp);
        d_fragmentKeys=d_fragmentSortedKeys=nullptr;d_fragmentOrder=d_fragmentSortedOrder=nullptr;d_fragmentSortTemp=nullptr;g_fragmentSortCapacity=0;
        e=cudaMalloc(&d_fragmentKeys,sortCapacity*sizeof(unsigned long long));
        if(e==cudaSuccess)e=cudaMalloc(&d_fragmentSortedKeys,sortCapacity*sizeof(unsigned long long));
        if(e==cudaSuccess)e=cudaMalloc(&d_fragmentOrder,sortCapacity*sizeof(int));
        if(e==cudaSuccess)e=cudaMalloc(&d_fragmentSortedOrder,sortCapacity*sizeof(int));
        g_fragmentSortBytes=0;
        if(e==cudaSuccess)e=cub::DeviceRadixSort::SortPairs(nullptr,g_fragmentSortBytes,d_fragmentKeys,d_fragmentSortedKeys,d_fragmentOrder,d_fragmentSortedOrder,sortCapacity);
        if(e==cudaSuccess)e=cudaMalloc(&d_fragmentSortTemp,g_fragmentSortBytes);
        if(e==cudaSuccess)g_fragmentSortCapacity=sortCapacity;
    }
    if(e==cudaSuccess && !d_fragmentBounds)e=cudaMalloc(&d_fragmentBounds,g_N*sizeof(FragmentBounds));
    if(e==cudaSuccess && !d_fragmentRoots)e=cudaMalloc(&d_fragmentRoots,g_N*sizeof(int));
    if(e==cudaSuccess && !d_fragmentRootCount)e=cudaMalloc(&d_fragmentRootCount,sizeof(int));
    if(e==cudaSuccess && !d_fragmentLambda)e=cudaMalloc(&d_fragmentLambda,6*g_N*sizeof(float));
    if(e==cudaSuccess && !d_fragmentDegree)e=cudaMalloc(&d_fragmentDegree,g_N*sizeof(int));
    if(e==cudaSuccess && !d_fragmentBefore)e=cudaMalloc(&d_fragmentBefore,g_N*sizeof(float3));
    if(e==cudaSuccess && !d_fragmentStart)e=cudaMalloc(&d_fragmentStart,g_N*sizeof(float3));
    if(e==cudaSuccess && !d_fragmentQueryCount)e=cudaMalloc(&d_fragmentQueryCount,sizeof(int));
    if(e==cudaSuccess && g_fragmentCapacity<3*g_surfaceCount){
        cudaFree(d_fragmentVertices);d_fragmentVertices=nullptr;cudaFree(d_fragmentQueries);d_fragmentQueries=nullptr;g_fragmentCapacity=0;
        e=cudaMalloc(&d_fragmentVertices,3*g_surfaceCount*sizeof(FragmentVertex));
        if(e==cudaSuccess)e=cudaMalloc(&d_fragmentQueries,(3*g_surfaceCount+g_N)*sizeof(int));
        if(e==cudaSuccess)g_fragmentCapacity=3*g_surfaceCount;
    }
    if(e!=cudaSuccess)return e;
    return cudaSuccess;
}
static cudaError_t solveFragmentContact(bool prepare=false){
    if(!d_unifiedComponents || g_surfaceCount<=0)return cudaSuccess;
    cudaError_t e=allocateFragmentContact();if(e!=cudaSuccess)return e;
    cudaMemsetAsync(d_fragmentLambda,0,6*g_N*sizeof(float));
    // Support, grasp flags and the surface embedding stay fixed over the passes.
    cudaMemsetAsync(d_fragmentQueryCount,0,sizeof(int));
    k_FragmentQueries<<<Gr(3*g_surfaceCount+g_N),64>>>(g_surfaceCount,g_N,d_unifiedComponents,g_liveActive,d_surfaceVertices,
        d_fragmentSupported,d_graspComponents,d_invMass,d_fragmentQueries,d_fragmentQueryCount,1);
#ifdef ENDO_FRAGMENT_TRACE
    k_FragmentTrace<<<Gr(g_N),64>>>(prepare?0:2,g_N,d_unifiedComponents,d_fragmentSupported,d_graspComponents,d_invMass,d_pos,d_prevPos,d_unifiedTarget,d_vel);
#endif
    if(prepare)cudaMemcpyAsync(d_fragmentBefore,d_pos,g_N*sizeof(float3),cudaMemcpyDeviceToDevice);
    else k_UnsupportedFlight<<<Gr(g_N),64>>>(g_N,d_fragmentSupported,d_unifiedComponents,d_graspComponents,d_invMass,d_unifiedTarget,d_pos);
    cudaMemcpyAsync(d_fragmentStart,d_pos,g_N*sizeof(float3),cudaMemcpyDeviceToDevice);
    const float3* previous=prepare?d_fragmentBefore:d_prevPos;
    // The passes add into d_surfaceDelta and d_fragmentDegree, which k_ApplyFragmentContact leaves cleared.
    cudaMemsetAsync(d_surfaceDelta,0,g_N*sizeof(float3));cudaMemsetAsync(d_fragmentDegree,0,g_N*sizeof(int));
    beginFeasible();
    e=runFragmentPasses(prepare,previous);if(e!=cudaSuccess)return e;
    e=finishFeasible();if(e!=cudaSuccess)return e;
    if(prepare)k_FragmentIncomingVelocity<<<Gr(g_N),64>>>(g_N,d_unifiedComponents,d_fragmentSupported,d_graspComponents,d_fragmentBefore,d_pos,d_vel);
#ifdef ENDO_FRAGMENT_TRACE
    k_FragmentTrace<<<Gr(g_N),64>>>(prepare?1:3,g_N,d_unifiedComponents,d_fragmentSupported,d_graspComponents,d_invMass,d_pos,d_prevPos,d_unifiedTarget,d_vel);
#endif
    return cudaGetLastError();
}

// Detect actual swept/interior fragment contacts at the inertial target. These
// are external forces, so the free-flight COM shortcut is not their nullspace.
static cudaError_t captureFragmentRefresh(cudaStream_t stream,bool refresh=true);
// Only a nonempty immutable query list can produce paired constraints.
// Every consumer of the skipped arrays is guarded by fragmentContactActive.
static cudaError_t launchFragmentDiscovery(cudaStream_t stream){
    cudaMemsetAsync(d_fragmentQuerySlots,0xff,(g_N+3*g_surfaceCount)*sizeof(int),stream);
    cudaMemsetAsync(d_fragmentContacted,0,g_N*sizeof(int),stream);
    cudaMemsetAsync(d_fragmentConstraintDegree,0,g_N*sizeof(int),stream);
    cudaError_t e=launchFragmentPasses(stream,false,d_prevPos,true,false);if(e!=cudaSuccess)return e;
    k_FragmentManifoldDegrees<<<128,64,0,stream>>>(d_fragmentConstraints,d_fragmentConstraintCount,d_fragmentConstraintDegree);
    k_InitFragmentGroups<<<Gr(g_N),64,0,stream>>>(d_unified);
    k_JoinFragmentGroups<<<128,64,0,stream>>>(d_unified);
    k_FlattenFragmentGroups<<<Gr(g_N),64,0,stream>>>(d_unified);
    return cudaGetLastError();
}
static cudaError_t launchPredictedFragmentContact(cudaStream_t stream){
    if(!d_fragmentContacted)return cudaSuccess;
    cudaMemsetAsync(d_fragmentConstraintCount,0,sizeof(int),stream);
    cudaMemcpyAsync(d_fragmentConstraintCount+1,cut_topology_revision_device(),sizeof(int),cudaMemcpyDeviceToDevice,stream);
    cudaMemsetAsync(d_fragmentQueryCount,0,sizeof(int),stream);
    k_FragmentQueries<<<Gr(3*g_surfaceCount+g_N),64,0,stream>>>(g_surfaceCount,g_N,d_unifiedComponents,g_liveActive,d_surfaceVertices,
        d_fragmentSupported,d_graspComponents,d_invMass,d_fragmentQueries,d_fragmentQueryCount,2,d_fragmentUnpinnedVolume);
    k_FragmentAvailability<<<1,1,0,stream>>>(d_unified,d_fragmentQueryCount);
    return captureFragmentRefresh(stream,false);
}

static cudaError_t markPredictedFragmentContact(){
    if(!d_fragmentContacted)return cudaSuccess;
    if(g_sectionProfile)return launchPredictedFragmentContact(0);
    PredictedFragmentKey key{};key.pass=makeFragmentPassKey(d_prevPos);key.state=d_unified;key.revision=cut_topology_revision_device();
    cudaError_t e=cudaSuccess;
    if(!g_predictedFragmentGraph || memcmp(&key,&g_predictedFragmentKey,sizeof(key))){
        if(!g_captureStream)e=cudaStreamCreate(&g_captureStream);if(e!=cudaSuccess)return e;
        e=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(e!=cudaSuccess)return e;
        cudaError_t work=launchPredictedFragmentContact(g_captureStream);cudaGraph_t graph=nullptr;
        e=cudaStreamEndCapture(g_captureStream,&graph);if(work!=cudaSuccess){if(graph)cudaGraphDestroy(graph);return work;}if(e!=cudaSuccess)return e;
        e=installRuntimeGraph(g_predictedFragmentGraph,graph);cudaGraphDestroy(graph);if(e!=cudaSuccess)return e;
        g_predictedFragmentKey=key;
    }
    return cudaGraphLaunch(g_predictedFragmentGraph,0);
}

// Refresh on the accepted material positions while retaining predicted witnesses.
// This continues the same substep; it neither advances time nor changes stiffness.
static cudaError_t launchFragmentRefresh(cudaStream_t stream){
    cudaError_t e=launchFragmentPasses(stream,false,d_prevPos,true,true);if(e!=cudaSuccess)return e;
    cudaMemsetAsync(d_fragmentConstraintDegree,0,g_N*sizeof(int),stream);
    k_FragmentManifoldDegrees<<<128,64,0,stream>>>(d_fragmentConstraints,d_fragmentConstraintCount,d_fragmentConstraintDegree);
    k_InitFragmentGroups<<<Gr(g_N),64,0,stream>>>(d_unified);
    k_JoinFragmentGroups<<<128,64,0,stream>>>(d_unified);
    k_FlattenFragmentGroups<<<Gr(g_N),64,0,stream>>>(d_unified);
    return cudaGetLastError();
}
static cudaError_t captureFragmentRefresh(cudaStream_t stream,bool refresh){
    if(g_sectionProfile)return refresh?launchFragmentRefresh(stream):launchFragmentDiscovery(stream);
    cudaStreamCaptureStatus status;cudaGraph_t graph=nullptr;const cudaGraphNode_t* deps=nullptr;size_t count=0;
    cudaError_t e=cudaStreamGetCaptureInfo_v2(stream,&status,nullptr,&graph,&deps,&count);if(e!=cudaSuccess)return e;
    if(status!=cudaStreamCaptureStatusActive)return cudaErrorStreamCaptureUnsupported;
    cudaGraphConditionalHandle handle=0;e=cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault);if(e!=cudaSuccess)return e;
    const int* queries=d_fragmentQueryCount;void* args[]={&handle,&queries};
    cudaKernelNodeParams p={};p.func=(void*)k_FragmentPassCondition;p.gridDim=p.blockDim=dim3(1);p.kernelParams=args;
    cudaGraphNode_t condition=nullptr;e=cudaGraphAddKernelNode(&condition,graph,deps,count,&p);if(e!=cudaSuccess)return e;
    cudaGraphNodeParams branch={cudaGraphNodeTypeConditional};branch.conditional.handle=handle;branch.conditional.type=cudaGraphCondTypeIf;branch.conditional.size=1;
    cudaGraphNode_t node=nullptr;e=cudaGraphAddNode(&node,graph,&condition,1,&branch);if(e!=cudaSuccess)return e;
    // A separate nonblocking stream populates the child while the parent stream
    // keeps capturing. The conditional dependency joins them before any solve.
    e=cudaStreamBeginCaptureToGraph(g_fragmentRefreshStream,branch.conditional.phGraph_out[0],nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal);if(e!=cudaSuccess)return e;
    cudaError_t work=refresh?launchFragmentRefresh(g_fragmentRefreshStream):launchFragmentDiscovery(g_fragmentRefreshStream);cudaGraph_t body=nullptr;
    e=cudaStreamEndCapture(g_fragmentRefreshStream,&body);if(work!=cudaSuccess)return work;if(e!=cudaSuccess)return e;
    return cudaStreamUpdateCaptureDependencies(stream,&node,1,cudaStreamSetCaptureDependencies);
}
static void launchFragmentCoarseBounds(cudaStream_t stream,CoarseBlock* blocks){
    k_FragmentCoarseFreeze<<<Gr(g_activeNodes),64,0,stream>>>(d_unified,blocks);
    k_FragmentCoarseBound<<<128,64,0,stream>>>(d_unified,blocks);
    k_FragmentCoarseBoundApply<<<Gr(g_activeNodes),64,0,stream>>>(d_unified,blocks);
}
static cudaError_t prepareUnifiedManifold(){
    if(!d_manifoldCount){cudaError_t e=cudaMalloc(&d_manifoldCount,sizeof(int));if(e!=cudaSuccess)return e;}
    int capacity=36*g_surfaceCapacity;
    if(capacity>g_manifoldCapacity){
        cudaFree(d_manifold);cudaFree(d_manifoldIds);cudaFree(d_manifoldOwners);cudaFree(d_manifoldQuadrature);
        d_manifold=nullptr;d_manifoldIds=d_manifoldOwners=nullptr;d_manifoldQuadrature=nullptr;g_manifoldCapacity=0;
        cudaError_t e=cudaMalloc(&d_manifold,capacity*sizeof(SurfaceContact));
        if(e==cudaSuccess)e=cudaMalloc(&d_manifoldIds,capacity*sizeof(int));
        if(e==cudaSuccess)e=cudaMalloc(&d_manifoldOwners,capacity*sizeof(int));
        if(e==cudaSuccess)e=cudaMalloc(&d_manifoldQuadrature,capacity*sizeof(float));
        if(e!=cudaSuccess)return e;g_manifoldCapacity=capacity;
    }
    cudaMemsetAsync(d_manifoldCount,0,sizeof(int));cudaMemsetAsync(d_surfaceDegree,0,g_N*sizeof(int));
    if(g_surfaceCount>0)launchUnifiedManifold(0,d_surfaceContacts,d_surfaceActive,d_surfaceActiveCount,d_surfaceVertices,d_pos,d_prevPos,
        d_coupledTool,g_jawThickness,g_gp.jawGapClosed,g_contactRadius,d_manifold,d_manifoldIds,d_manifoldCount,d_manifoldOwners,d_manifoldQuadrature,d_surfaceDegree,3,3*g_surfaceCount);
    return cudaGetLastError();
}
// A finite-stiffness contact has a nonzero equilibrium residual under load.
// Rebuilding below the existing detection skin can repeatedly solve the same
// equilibrium. Bound reconstruction error by that skin (0.058 mm in the scene),
// while the independent 0.2 mm overlap acceptance remains unchanged.
extern "C" float xpbd_contact_repair_tolerance(){return fmaxf(1e-5f,g_contactRadius);}
extern "C" int xpbd_integrated_contact(){return d_unified && g_frictionContact && d_coupledTool?1:0;}
extern "C" int xpbd_relax_reconstructed_contact(){
    if(!g_bound || !g_frictionContact || !g_toolArmed || g_surfaceCount<=0)return 0;
    beginFeasible();
    // Residual normal contact correction only. Material pressure is integrated
    // inside the time step; rebuilding a surface must not apply another bulk impulse.
    k_BindSurface<<<Gr(3*g_surfaceCount),64>>>(3*g_surfaceCount,g_surfaceTags,g_surfaceExpanded,
        g_surfaceCorners,g_surfaceMasks,g_surfaceConn,d_pos,d_surfaceVertices,g_surfaceRest,g_surfaceMaterialFP,g_surfaceBoundaryMasks);
    cudaMemsetAsync(d_surfaceDegree,0,(size_t)g_N*sizeof(int));
    cudaMemsetAsync(d_surfaceActiveCount,0,sizeof(int));
    k_UpdateSurfacePoints<<<Gr(3*g_surfaceCount),64>>>(3*g_surfaceCount,d_surfaceVertices,g_surfaceMasks,d_pos,d_pos,d_surfaceCurrent,d_surfacePrevious);
    buildContactGroups(g_surfaceCount,d_surfaceCurrent,d_surfacePrevious,d_contactGroups,0);
    classifyContactGroups(g_surfaceCount,3,d_contactGroups,d_contactGroupRejected,d_coupledTool,g_jawThickness,g_gp.jawGapClosed,g_contactRadius,true,false,0);
    if(canBatchQueries(g_surfaceCount,d_coupledTool,2)){
        launchFindSurfaceGroup(g_surfaceCount,0x88u,3,false,0,ContactShape{},ContactShape{},g_contactRadius,
            d_surfaceVertices,g_surfaceMasks,d_pos,d_pos,d_surfaceContacts,d_surfaceDegree,d_surfaceActive,d_surfaceActiveCount,
            d_surfaceCurrent,d_surfacePrevious,d_coupledTool,g_jawThickness,g_gp.jawGapClosed,true,d_contactCacheSlots,d_contactCache,nullptr,3,d_contactGroupRejected);
    }else{for(int kind=0;kind<3;kind++){
        ContactShape shape=contactShape(g_tool,kind,g_jawThickness,g_gp.jawGapClosed);
        k_FindSurface<<<Gr(g_surfaceCount),64>>>(g_surfaceCount,kind,shape,shape,g_contactRadius,
            d_surfaceVertices,g_surfaceMasks,d_pos,d_pos,d_surfaceContacts,d_surfaceDegree,d_surfaceActive,d_surfaceActiveCount,d_surfaceCurrent,d_surfacePrevious,d_coupledTool,g_jawThickness,g_gp.jawGapClosed,true,d_contactCacheSlots,d_contactCache,nullptr,3,false,d_contactGroupRejected+kind*contactGroupCount(g_surfaceCount));
    }}
    // Surface-count changes invalidate captured launch sizes, not the final
    // substep's implicit objective. Never switch to legacy XPBD/projections
    // merely because the execution-graph cache needs to be rebuilt.
    if(d_unified){
        int refreshMax=std::max(std::max(g_N,g_E),std::max(g_B,g_T));
        k_TopologyStale<<<1,1>>>(cut_topology_revision_device(),d_livenessRevision);
        k_RefreshLiveness<<<Gr(refreshMax),64>>>(g_N,g_E,g_B,g_T,g_liveNbrIdx,d_edgeOwnerSlot,d_edgeI,d_edgeJ,
            g_liveBendPairs,d_bendSrcIdx,g_liveActive,g_livePinned,g_liveMass,d_tetIds,d_tetEdgeOwnerSlot,d_edgeActive,d_bendActive,d_invMass,d_tetActive,
            d_livenessRevision+1);
        cudaError_t componentError=refreshUnifiedComponents();
        if(componentError!=cudaSuccess)return -(int)componentError;
        componentError=refreshFragmentSupport();if(componentError!=cudaSuccess)return -(int)componentError;
        if(g_embeddedGrip && g_grasping)refreshGrip();
        cudaError_t manifoldError=prepareUnifiedManifold();if(manifoldError!=cudaSuccess)return -(int)manifoldError;
        k_UnifiedRebind<<<1,1>>>(d_unified,d_manifold,d_manifoldIds,d_gripPoints,g_embeddedGrip&&g_grasping?g_gripCount:0,d_manifoldCount,d_manifoldQuadrature);
        // A cut/reconstruction changes the blade's material witnesses too.
        // Never continue contact against the pre-cut embedding.
        if(g_blade.enabled){
            cudaError_t bladeError=allocateBlade();if(bladeError!=cudaSuccess)return -(int)bladeError;
            findBlade(0,true);
            k_UnifiedRebind<<<1,1>>>(d_bladeState,d_bladeManifold,d_bladeManifoldIds,nullptr,0,d_bladeManifoldCount,d_bladeQuadrature);
        }

        // Component-indexed flags and patch support must use the new roots.
        cudaMemsetAsync(d_graspComponents,0,g_N*sizeof(int));
        k_UnifiedPatchBasis<<<1,32>>>(d_unified);
        k_MarkCoarseContact<<<128,64>>>(d_unified);
        if(g_blade.enabled)k_MarkCoarseContact<<<128,64>>>(d_bladeState,2|MaterialCutterContact);
        k_MarkVolumeRecovery<<<Gr(g_T),64>>>(d_unified);

        // Reconstruction changes the embedded contact footprint. Continue the
        // SAME implicit objective only in the affected neighborhood; neither
        // advance time nor apply an independent position/velocity projection.
        int sites=((g_dimsX+2)/2)*((g_dimsY+2)/2)*((g_dimsZ+2)/2);
        if(!g_unifiedContactGraph || g_unifiedContactGraphDirty){
        g_interactionStats[12]++;
        cudaError_t capture=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(capture!=cudaSuccess)return -(int)capture;
        for(int it=0;it<3;it++){
            // Continue the same regional low-frequency mode after nonlinear
            // surface reconstruction. Vertex-only repair converges very slowly
            // when a stiff captured patch must move with its supporting tissue.
            if(it==0){
                k_CoarseWeights<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,5);
                k_CoarseBegin<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                launchCoarseAccumulateCache(GrAccumulate(g_activeNodes+g_T),g_captureStream,d_unified,d_globalCoarse);
            k_FragmentCoarseTerms<<<128,64,0,g_captureStream>>>(d_unified,d_globalCoarse);
            if(g_blade.enabled)k_BladeCoarse<<<128,64,0,g_captureStream>>>(d_unified,d_bladeState,d_globalCoarse,false);
                k_CoarseSolve<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                launchCoarseBoundCache(GrAccumulate(g_T),g_captureStream,d_unified,d_globalCoarse);
            if(g_blade.enabled)k_BladeCoarse<<<128,64,0,g_captureStream>>>(d_unified,d_bladeState,d_globalCoarse,true);
            launchFragmentCoarseBounds(g_captureStream,d_globalCoarse);
                k_PressureLineClear<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                for(int pass=0;pass<2;pass++){
                    k_FragmentLineClear<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified);
                    launchPressureLine(g_captureStream,Gr(g_activeNodes+g_T),d_unified,d_globalCoarse,pass);
                    k_FragmentLineSum<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse,pass);
                    k_FragmentLineSet<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                }
                k_PressureLineSelect<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                k_CoarseApply<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                k_CoarseWeights<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,0);
            }
            if(it>0 && g_surfaceCount>0){
                cudaMemsetAsync(d_manifoldCount,0,sizeof(int),g_captureStream);
                cudaMemsetAsync(d_surfaceDegree,0,g_N*sizeof(int),g_captureStream);
                launchUnifiedManifold(g_captureStream,d_surfaceContacts,d_surfaceActive,d_surfaceActiveCount,d_surfaceVertices,d_pos,d_prevPos,
                    d_coupledTool,g_jawThickness,g_gp.jawGapClosed,g_contactRadius,d_manifold,d_manifoldIds,d_manifoldCount,d_manifoldOwners,d_manifoldQuadrature,d_surfaceDegree,3,3*g_surfaceCount);
                // Recompute the blade footprint after tissue relaxation too.
                // The cut lip can rebound into a different part of the finite
                // back while the grasper manifold is being refreshed.
                if(g_blade.enabled){
                    k_UpdateSurfacePoints<<<Gr(3*g_surfaceCount),64,0,g_captureStream>>>(3*g_surfaceCount,d_surfaceVertices,g_surfaceMasks,d_pos,d_pos,d_surfaceCurrent,d_surfacePrevious);
                    buildContactGroups(g_surfaceCount,d_surfaceCurrent,d_surfacePrevious,d_contactGroups,g_captureStream);
                    findBlade(g_captureStream,true);
                    k_MarkCoarseContact<<<128,64,0,g_captureStream>>>(d_bladeState,2|MaterialCutterContact);
                }
            }
            cudaMemsetAsync(d_sweepCount,0,8*sizeof(int),g_captureStream);
            k_SweepSites<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,SweepRepair,sites,d_sweepSites,d_sweepCount);
            for(int c=0;c<8;c++){
                k_AssembleUnifiedInteractions<<<ASSEMBLY_BLOCKS,64,0,g_captureStream>>>(d_unified,c);k_AssembleFragmentInteractions<<<ASSEMBLY_BLOCKS,64,0,g_captureStream>>>(d_unified,c);if(g_blade.enabled)k_AssembleUnifiedInteractions<<<ASSEMBLY_BLOCKS,64,0,g_captureStream>>>(d_bladeState,c);
                k_UnifiedVertex<<<g_sweepBlocks,64,0,g_captureStream>>>(d_unified,d_sweepSites+c*sites,d_sweepCount+c);
            }
            k_UnifiedTool<<<1,32,0,g_captureStream>>>(d_unified);
            // Fracture has already consumed this substep's actual blade sweep.
            // Keep that pose during reconstruction repair; advancing it here
            // would skip a segment of CCD before the next predictor resets previous.

            if(it==0){
                k_CoarseBegin<<<1,1,0,g_captureStream>>>(d_unified,d_coarseBlock);
                launchCoarseAccumulateCache(GrAccumulate(g_activeNodes+g_T),g_captureStream,d_unified,d_coarseBlock);
            k_FragmentCoarseTerms<<<128,64,0,g_captureStream>>>(d_unified,d_coarseBlock);
            if(g_blade.enabled)k_BladeCoarse<<<128,64,0,g_captureStream>>>(d_unified,d_bladeState,d_coarseBlock,false);
                k_CoarseSolve<<<1,1,0,g_captureStream>>>(d_unified,d_coarseBlock);
                launchCoarseBoundCache(GrAccumulate(g_T),g_captureStream,d_unified,d_coarseBlock);
            if(g_blade.enabled)k_BladeCoarse<<<128,64,0,g_captureStream>>>(d_unified,d_bladeState,d_coarseBlock,true);
            launchFragmentCoarseBounds(g_captureStream,d_coarseBlock);
            k_CoarseApply<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_coarseBlock);
            }
        }
        cudaGraph_t graph=nullptr;capture=cudaStreamEndCapture(g_captureStream,&graph);if(capture!=cudaSuccess)return -(int)capture;
        // Preserve the executable when reconstruction only changes launch
        // bounds/pointers. Re-instantiating the same contact solve can dominate
        // a closing-jaw frame; every kernel and iteration is still executed.
        if(g_unifiedContactGraph){
            cudaGraphExecUpdateResultInfo info={};capture=cudaGraphExecUpdate(g_unifiedContactGraph,graph,&info);
            if(capture==cudaErrorGraphExecUpdateFailure || (capture==cudaSuccess && info.result!=cudaGraphExecUpdateSuccess)){
                cudaGetLastError();cudaGraphExecDestroy(g_unifiedContactGraph);g_unifiedContactGraph=nullptr;capture=cudaSuccess;
            }
        }
        if(capture==cudaSuccess && !g_unifiedContactGraph)capture=cudaGraphInstantiate(&g_unifiedContactGraph,graph,0);
        cudaGraphDestroy(graph);if(capture!=cudaSuccess)return -(int)capture;g_unifiedContactGraphDirty=false;
        }
        g_interactionStats[13]++;cudaError_t launch=cudaGraphLaunch(g_unifiedContactGraph,0);if(launch!=cudaSuccess)return -(int)launch;
        cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return -(int)e;
        k_UnifiedReaction<<<128,64>>>(d_unified);
        if(g_manifoldCapacity>0){
            k_UnifiedAggregate<<<128,64>>>(d_unified,d_manifoldOwners,d_surfaceContacts,!g_anchorAssist);
            if(!g_anchorAssist)k_UnifiedStick<<<128,64>>>(d_unified,d_surfaceActive,d_surfaceActiveCount,d_surfaceContacts,g_jawThickness,g_gp.jawGapClosed);
        }
        k_EndTool<<<1,1>>>(d_coupledTool);
        e=solveFragmentContact();if(e!=cudaSuccess)return -(int)e;
        k_update_velocity<<<Gr(g_N),64>>>(g_N,1.f/g_hFrame,d_invMass,d_pos,d_prevPos,d_vel);
        // Recompute the final-substep velocity with the SAME drag factor.
        // This is not another time step: recomputing raw displacement velocity
        // without this factor previously undid drag (22% speed jump at alpha=20).
        k_apply_velscale<<<Gr(g_N),64>>>(g_N,g_velScaleF,d_invMass,d_vel);
        return syncSolvedTool();
    }
    // Continue the SAME final-substep constraints after nonlinear reconstruction.
    // The contact repair must not move a held tool independently of its anchor.
    // Multipliers and previous positions are retained; no new time is integrated.
    cudaError_t repairError=solveGrip();if(repairError!=cudaSuccess)return -(int)repairError;
    repairError=repairSurfaceGraph();if(repairError!=cudaSuccess)return -(int)repairError;

    {cudaError_t e=finishFeasible();if(e!=cudaSuccess)return -(int)e;}
    // Reconstruction has no elapsed time. Never run a second instrument-only
    // impulse solve here: that changes the constraint mass metric while reusing
    // multipliers and treats deformable tissue as an immovable obstacle.
    if(d_coupledTool){
        if(d_feasibleStart && g_volumePasses>0){
            k_ContactMobility<<<Gr(g_N),64>>>(d_coupledTool,g_N,d_invMass,d_needFeasible,d_feasibleStart,d_contactMobility,d_pos,d_contactMobility);
            beginFeasible();repairError=solveGrip();if(repairError!=cudaSuccess)return -(int)repairError;
            repairError=repairSurfaceGraph();if(repairError!=cudaSuccess)return -(int)repairError;
            {cudaError_t e=finishFeasible();if(e!=cudaSuccess)return -(int)e;}k_ClearContactMobility<<<1,1>>>(d_coupledTool);
        }
        int rc=syncSolvedTool();if(rc)return rc;
    }
    // The nonlinear surface repair changes the current contact normal. Project
    // normal velocity at that same manifold, so the next substep does not undo
    // the repair and repeatedly bounce into the jaw. Tangential slip is untouched.
    for(int it=0;g_hFrame>0 && it<2;it++){
        cudaMemsetAsync(d_surfaceDelta,0,(size_t)g_N*sizeof(float3));
        if(d_coupledTool)k_ClearToolVelocityDelta<<<1,1>>>(d_coupledTool);
        k_SurfaceVelocity<<<128,64>>>(3*g_surfaceCount,1.f/g_hFrame,d_invMass,d_surfaceDegree,
            d_surfaceContacts,d_vel,d_surfaceDelta,d_surfaceActive,d_surfaceActiveCount,d_coupledTool);
        k_ApplySurface<<<Gr(g_N),64>>>(g_N,d_vel,d_surfaceDelta);
        if(d_coupledTool)k_ApplyToolVelocity<<<1,1>>>(d_coupledTool);
    }
    return -(int)cudaGetLastError();
}


static int unifiedSubstep(float h,XpbdToolState previousTool,XpbdToolState stepTool,bool componentsFresh){
    cudaError_t e=ensureCoarseBoundCache(g_T);if(e!=cudaSuccess)return -(int)e;
    if(g_blade.enabled){
        e=allocateBlade();if(e!=cudaSuccess)return -(int)e;
        if(g_substepIndex==1)cudaMemsetAsync(d_bladeResult,0,6*sizeof(float));
        if(g_scissorsMode)k_ConfigureScissors<<<1,1>>>(d_bladeBody,g_scissors,g_blade.radius,g_scissorsWrist[0],g_scissorsWrist[1],g_scissorsWrist[2]);
        else k_ConfigureKnifeBody<<<1,1>>>(d_bladeBody,g_knifeBody,g_knifeBoxes[0],g_knifeBoxes[1]);
        k_PredictTool<<<1,1>>>(d_bladeBody,g_bladeCommand,g_bladeInitial,h,g_blade.mass,g_blade.driveStiffness,2.f*sqrtf(g_blade.mass*g_blade.driveStiffness),
            g_toolMaxTravel/fmaxf(1.f,(float)g_p.numSubsteps),
            g_scissorsMode?g_scissors.jawMass:0,g_scissorsMode?g_scissors.jawStiffness:0,
            g_scissorsMode?2.f*sqrtf(g_scissors.jawMass*g_scissors.jawStiffness):0,
            g_scissorsMode?g_scissors.jawSpeed:0,g_bladeCommand.jawHinge,g_scissorsMode?g_scissors.gap:0,g_scissorsMode?g_scissors.guided:1);
        if(g_scissorsMode)k_ConfigureScissors<<<1,1>>>(d_bladeBody,g_scissors,g_blade.radius,g_scissorsWrist[0],g_scissorsWrist[1],g_scissorsWrist[2]);
    }


    if(!d_unified)e=cudaMalloc(&d_unified,sizeof(UnifiedState));
    if(e==cudaSuccess && !d_unifiedTarget)e=cudaMalloc(&d_unifiedTarget,(size_t)g_N*sizeof(float3));
    // The coarse kernels visit the active nodes alone (UnifiedState::activeNodes): every other node keeps weight 0.
    if(e==cudaSuccess && !d_coarseWeight){
        e=cudaMalloc(&d_coarseWeight,(size_t)g_N*sizeof(float));
        if(e==cudaSuccess)e=cudaMemset(d_coarseWeight,0,(size_t)g_N*sizeof(float));
    }
    if(e==cudaSuccess && !d_coarseBlock)e=cudaMalloc(&d_coarseBlock,sizeof(CoarseBlock));
    if(e==cudaSuccess && !d_unifiedInteraction)e=cudaMalloc(&d_unifiedInteraction,(size_t)9*g_N*sizeof(InteractionSum));
    // A color holds at most one vertex per site of its parity, the count the sweeps pass as the list capacity.
    if(e==cudaSuccess && !d_sweepSites)e=cudaMalloc(&d_sweepSites,(size_t)8*((g_dimsX+2)/2)*((g_dimsY+2)/2)*((g_dimsZ+2)/2)*sizeof(int));
    if(e==cudaSuccess && !d_sweepCount)e=cudaMalloc(&d_sweepCount,8*sizeof(int));
    if(!g_sweepBlocks){g_sweepBlocks=fullWave(k_UnifiedVertex<true>);g_predictorBlocks=fullWave(k_UnifiedPredictor);}
    const bool cacheRest=g_unifiedTetCapacity<g_T;
    if(e==cudaSuccess && cacheRest){
        cudaFree(d_unifiedPreviousInverse);d_unifiedPreviousInverse=nullptr;
        cudaFree(d_unifiedRestMetric);d_unifiedRestMetric=nullptr;g_unifiedTetCapacity=0;
        e=cudaMalloc(&d_unifiedPreviousInverse,(size_t)3*g_T*sizeof(float3));
        if(e==cudaSuccess)e=cudaMalloc(&d_unifiedRestMetric,(size_t)7*g_T*sizeof(float));
        if(e==cudaSuccess)g_unifiedTetCapacity=g_T;
    }
    // The PCG launches over the active nodes alone (UnifiedState::activeNodes): the other nodes keep the zero
    // stencil row, residual, direction and step set here, which the kernels over all nodes read.
    const bool clearCG=!d_cgStencil;
    if(e==cudaSuccess && !d_cgStencil)e=cudaMalloc(&d_cgStencil,g_N*sizeof(unsigned short));
    if(e==cudaSuccess && !d_cgZ)e=cudaMalloc(&d_cgZ,g_N*sizeof(float3));
    if(e==cudaSuccess && !d_cgMatrix)e=cudaMalloc(&d_cgMatrix,(size_t)72*g_N*sizeof(__half));
#ifdef ENDO_KIDNEY_STENCIL_CACHE
    if(e==cudaSuccess)e=ensureMaterialTetCache(g_T);
#endif
    if(e==cudaSuccess)e=allocateMultigrid();
    if(e==cudaSuccess && !d_cgScale)e=cudaMalloc(&d_cgScale,(size_t)8*g_N*sizeof(float));
    if(e==cudaSuccess && !d_cgR)e=cudaMalloc(&d_cgR,g_N*sizeof(float3));
    if(e==cudaSuccess && !d_cgP)e=cudaMalloc(&d_cgP,g_N*sizeof(float3));
    if(e==cudaSuccess && !d_cgQ)e=cudaMalloc(&d_cgQ,g_N*sizeof(float3));
    if(e==cudaSuccess && !d_cgD)e=cudaMalloc(&d_cgD,g_N*sizeof(float3));
    if(e==cudaSuccess && !d_cgA)e=cudaMalloc(&d_cgA,g_N*sizeof(float3));
    if(e==cudaSuccess && !d_cgH)e=cudaMalloc(&d_cgH,g_N*sizeof(Symmetric3));
    if(e==cudaSuccess && !d_cgS)e=cudaMalloc(&d_cgS,g_N*sizeof(MaterialCGSum));
    if(e==cudaSuccess && !d_cgSearch)e=cudaMalloc(&d_cgSearch,g_N*sizeof(MaterialLineSearch));
    if(e==cudaSuccess && clearCG){
        e=cudaMemset(d_cgStencil,0,g_N*sizeof(unsigned short));
        for(float3* v:{d_cgZ,d_cgR,d_cgP,d_cgQ,d_cgD,d_cgA})if(e==cudaSuccess)e=cudaMemset(v,0,g_N*sizeof(float3));
        if(e==cudaSuccess)e=cudaMemset(d_cgH,0,g_N*sizeof(Symmetric3));
    }
    if(e!=cudaSuccess)return -(int)e;
    // Store the correct inertial target. Component-wide coarse modes below
    // propagate gravity/support loads without requiring thousands of local sweeps.
    cudaMemcpyAsync(d_unifiedTarget,d_pos,(size_t)g_N*sizeof(float3),cudaMemcpyDeviceToDevice);
    if(e==cudaSuccess && !d_unifiedComponents)e=cudaMalloc(&d_unifiedComponents,g_N*sizeof(int));
    if(e==cudaSuccess && !d_graspComponents)e=cudaMalloc(&d_graspComponents,g_N*sizeof(int));
    if(e==cudaSuccess && !d_floorRecovery){
        e=cudaMalloc(&d_floorRecovery,g_N*sizeof(int));if(e==cudaSuccess)e=cudaMemset(d_floorRecovery,0,g_N*sizeof(int));
    }
    if(e==cudaSuccess && !d_safeCluster)e=cudaMalloc(&d_safeCluster,g_N*sizeof(int));
    if(e==cudaSuccess && !d_safeAggregate)e=cudaMalloc(&d_safeAggregate,g_N*sizeof(SafeCoarseCluster));
    if(e==cudaSuccess && !d_safeJoined)e=cudaMalloc(&d_safeJoined,sizeof(int));
    if(e==cudaSuccess && !d_globalCoarse)e=cudaMalloc(&d_globalCoarse,g_N*sizeof(CoarseBlock));
    if(e!=cudaSuccess)return -(int)e;
    // The fragment pre-pass of this substep already refreshed both; nothing
    // has cut tissue or changed a mass since.
    if(!componentsFresh){
        e=refreshUnifiedComponents();if(e!=cudaSuccess)return -(int)e;
        e=refreshFragmentSupport();if(e!=cudaSuccess)return -(int)e;
    }
    if(g_embeddedGrip && g_grasping)refreshGrip();
    cudaMemsetAsync(d_surfaceDegree,0,(size_t)g_N*sizeof(int));cudaMemsetAsync(d_surfaceActiveCount,0,sizeof(int));
    if(g_surfaceCount>0){
        k_UpdateSurfacePoints<<<Gr(3*g_surfaceCount),64>>>(3*g_surfaceCount,d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,d_surfaceCurrent,d_surfacePrevious);
        buildContactGroups(g_surfaceCount,d_surfaceCurrent,d_surfacePrevious,d_contactGroups,0);
        classifyContactGroups(g_surfaceCount,3,d_contactGroups,d_contactGroupRejected,d_coupledTool,g_jawThickness,g_gp.jawGapClosed,g_contactRadius,false,false,0);
        if(canBatchQueries(g_surfaceCount,d_coupledTool,2)){
        launchFindSurfaceGroup(g_surfaceCount,0x88u,3,false,0,ContactShape{},ContactShape{},g_contactRadius,
            d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,d_surfaceContacts,d_surfaceDegree,d_surfaceActive,d_surfaceActiveCount,
            d_surfaceCurrent,d_surfacePrevious,d_coupledTool,g_jawThickness,g_gp.jawGapClosed,false,d_contactCacheSlots,d_contactCache,nullptr,3,d_contactGroupRejected);
    }else{for(int kind=0;kind<3;kind++)k_FindSurface<<<Gr(g_surfaceCount),64>>>(g_surfaceCount,kind,
            contactShape(previousTool,kind,g_jawThickness,g_gp.jawGapClosed),contactShape(stepTool,kind,g_jawThickness,g_gp.jawGapClosed),g_contactRadius,
            d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,d_surfaceContacts,d_surfaceDegree,d_surfaceActive,d_surfaceActiveCount,d_surfaceCurrent,d_surfacePrevious,
            d_coupledTool,g_jawThickness,g_gp.jawGapClosed,false,d_contactCacheSlots,d_contactCache,nullptr,3,false,d_contactGroupRejected+kind*contactGroupCount(g_surfaceCount));}
    }
    if(g_blade.enabled)findBlade();
    e=prepareUnifiedManifold();if(e!=cudaSuccess)return -(int)e;
    xpbd_section_mark("pre: tool, contact discovery, manifold");
    cudaMemcpyAsync(d_pos,d_prevPos,(size_t)g_N*sizeof(float3),cudaMemcpyDeviceToDevice);
    e=allocateFragmentContact();if(e!=cudaSuccess)return -(int)e;
    UnifiedState s={};s.fragmentPossible=true;s.fragmentGroups=d_fragmentGroups;s.fragmentLineSums=d_fragmentLineSums;s.fragmentContacted=d_fragmentContacted;s.fragmentConstraints=d_fragmentConstraints;s.fragmentConstraintCount=d_fragmentConstraintCount;s.fragmentDegree=d_fragmentConstraintDegree;s.fragmentLiveRevision=cut_topology_revision_device();s.componentSupport=d_fragmentSupported;s.coupledReady=true;s.component=d_unifiedComponents;s.graspComponents=d_graspComponents;s.floorRecovery=d_floorRecovery;s.safeCluster=d_safeCluster;s.safeAggregate=d_safeAggregate;s.safeJoined=d_safeJoined;s.n=g_N;s.nx=g_dimsX+1;s.ny=g_dimsY+1;s.nt=g_T;s.grips=g_embeddedGrip&&g_grasping?g_gripCount:0;
    s.kuhn=d_kuhnTets;s.cellBase=d_tetCellBase;s.cells=std::max(g_tetCells,1);s.activeNodes=d_activeNodes;s.activeCount=g_activeNodes;
    s.offsets=d_incidentOffsets;s.incident=d_incidentTets;s.ids=d_tetIds;s.live=d_tetActive;s.contactIds=d_manifoldIds;s.contactCount=d_manifoldCount;
    s.degree=d_surfaceDegree;s.gripDegree=d_gripDegree;s.b0=d_tetBr0;s.b1=d_tetBr1;s.b2=d_tetBr2;s.previous=d_prevPos;s.target=d_unifiedTarget;
    s.attachRest=d_attachRest;s.mu=d_tetMuV;s.bulk=d_tetLambdaV;s.invMass=d_invMass;s.attach=d_attachStiffness;s.pos=d_pos;
    s.interaction=d_unifiedInteraction;s.coarseWeight=d_coarseWeight;s.contacts=d_manifold;s.contactQuadrature=d_manifoldQuadrature;s.points=d_gripPoints;s.tool=d_coupledTool;s.h=h;s.minimum=g_minTetVolume;
    s.damping=g_p.strainDamping;s.hardening=g_tissueHardening;s.contactK=200000.f;s.gripK=1.f/g_gripCompliance;
    s.previousInverse=d_unifiedPreviousInverse;s.restMetric=d_unifiedRestMetric;
    s.minGap=g_gp.jawGapClosed;s.skin=g_contactRadius;s.thickness=g_jawThickness;s.muS=g_muStatic;s.muD=g_muDynamic;
    s.neighbors=g_liveNbrIdx;s.restNeighbors=g_liveRestNbr;s.edgeSupport=d_edgeTetSupport;s.edgeK=g_p.youngE;s.active=g_liveActive;
    if(g_blade.enabled){
        UnifiedState blade=s;blade.tool=d_bladeBody;blade.contacts=d_bladeManifold;blade.contactIds=d_bladeManifoldIds;blade.contactCount=d_bladeManifoldCount;
        blade.contactQuadrature=d_bladeQuadrature;blade.cutContactCount=d_bladeEdgeCount;blade.degree=d_bladeDegree;blade.grips=0;blade.points=nullptr;blade.minGap=blade.skin=blade.thickness=0;
        blade.contactK=10.f*g_blade.driveStiffness;blade.cutPhase=d_bladePhase;blade.cutResistance=g_blade.cutForce;blade.muS=blade.muD=g_blade.friction;
        e=cudaMemcpyAsync(d_bladeState,&blade,sizeof(blade),cudaMemcpyHostToDevice);if(e!=cudaSuccess)return -(int)e;
        // Retain the accepted V21 pressure-blade drive; this change targets the grasper.
        k_UnifiedTargets<<<1,1>>>(d_bladeState,g_scissorsMode);s.secondary=d_bladeState;
    }

    e=cudaMemcpyAsync(d_unified,&s,sizeof(s),cudaMemcpyHostToDevice);if(e!=cudaSuccess)return -(int)e;
    rebuildMultigrid();
    if(!d_substepQuery)e=cudaMalloc(&d_substepQuery,3*sizeof(int));if(e!=cudaSuccess)return -(int)e;
    if(g_runtimeCoreMode&32768){e=ensureRuntimeModeGates();if(e!=cudaSuccess)return -(int)e;}
    int query[5]={};
    const int predictorSites=((s.nx+1)/2)*((s.ny+1)/2)*((g_dimsZ+2)/2);
    // Capture the unchanged prelude as one sequence. All pose/state uploads
    // remain before it; the existing host recovery/query read remains after it.
    auto launchPrelude=[&](cudaStream_t stream)->cudaError_t{
    k_CacheUnifiedTetData<<<Gr(g_T),64,0,stream>>>(d_unified,d_unifiedPreviousInverse,d_unifiedRestMetric,cacheRest);
    if(cacheRest && g_T>0)k_KuhnTable<<<1,6,0,stream>>>(d_unified,d_kuhnTets);
    k_UnifiedTargets<<<1,1,0,stream>>>(d_unified);
    cudaMemsetAsync(d_graspComponents,0,g_N*sizeof(int),stream);
    k_UnifiedPatchBasis<<<1,32,0,stream>>>(d_unified);
    e=launchPredictedFragmentContact(stream);if(e!=cudaSuccess)return e;
    if(g_blade.enabled)k_MarkCoarseContact<<<128,64,0,stream>>>(d_bladeState,2|MaterialCutterContact);
    k_CoarseWeights<<<Gr(g_activeNodes),64,0,stream>>>(d_unified);

    cudaMemsetAsync(d_sweepCount,0,8*sizeof(int),stream);
    k_SweepSites<<<Gr(g_N),64,0,stream>>>(d_unified,SweepPredictor,predictorSites,d_sweepSites,d_sweepCount);
    for(int color=0;color<8;color++)k_UnifiedPredictor<<<g_predictorBlocks,64,0,stream>>>(d_unified,d_sweepSites+color*predictorSites,d_sweepCount+color);
    k_PredictorClear<<<Gr(g_N),64,0,stream>>>(g_N,d_globalCoarse);
    k_PredictorNodeEnergy<<<Gr(g_N),64,0,stream>>>(d_unified,d_globalCoarse);
    k_PredictorTetEnergy<<<Gr(g_T),64,0,stream>>>(d_unified,d_globalCoarse);
    k_PredictorInteractionEnergy<<<128,64,0,stream>>>(d_unified,d_globalCoarse);
    if(g_blade.enabled)k_PredictorInteractionEnergy<<<128,64,0,stream>>>(d_bladeState,d_globalCoarse);
    k_SelectPredictor<<<Gr(g_N),64,0,stream>>>(d_unified,d_globalCoarse);
    // Select the expensive nonlinear recovery graph only for unheld,
    // contact-free components at the determinant guard. A four-byte query
    // avoids running empty global kernels and rebuilding contact on loaded tissue.
    k_MarkCoarseContact<<<128,64,0,stream>>>(d_unified);
    k_MarkVolumeRecovery<<<Gr(g_T),64,0,stream>>>(d_unified);
    // The same query also flags a free component. Within a substep the component flags only gain bits, so when no
    // movable node belongs to a free component here, every PCG iteration beyond MaterialLoadedIterations returns at
    // once for every node, and the loaded graph leaves them out.


    e=cudaMemsetAsync(d_substepQuery,0,2*sizeof(int),stream);if(e!=cudaSuccess)return e;
    k_MaterialRecoveryRequested<<<Gr(g_N),64,0,stream>>>(d_unified,d_substepQuery);
    k_MaterialFreeNodes<<<Gr(g_N),64,0,stream>>>(d_unified,d_substepQuery+1);
    e=cudaMemcpyAsync(d_substepQuery+2,d_fragmentQueryCount,sizeof(int),cudaMemcpyDeviceToDevice,stream);if(e!=cudaSuccess)return e;
    if(g_runtimeCoreMode&32768){e=launchRuntimeModeQueries(stream);if(e!=cudaSuccess)return e;}

        return cudaGetLastError();
    };
    if(g_sectionProfile)e=launchPrelude(0);
    else{
        MaterialPreludeKey key{};key.fragment.pass=makeFragmentPassKey(d_prevPos);key.fragment.state=d_unified;key.fragment.revision=cut_topology_revision_device();
        const void* pointers[9]={d_unifiedPreviousInverse,d_unifiedRestMetric,d_kuhnTets,d_graspComponents,d_bladeState,d_sweepCount,d_sweepSites,d_globalCoarse,d_substepQuery};memcpy(key.pointers,pointers,sizeof(pointers));
        int values[9]={g_T,g_N,g_activeNodes,g_dimsX,g_dimsY,g_dimsZ,g_predictorBlocks,predictorSites,(int)cacheRest};memcpy(key.values,values,sizeof(values));
        if(!g_materialPreludeGraph || memcmp(&key,&g_materialPreludeKey,sizeof(key))){
            if(!g_captureStream)e=cudaStreamCreate(&g_captureStream);if(e!=cudaSuccess)return -(int)e;
            e=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(e!=cudaSuccess)return -(int)e;
            cudaError_t work=launchPrelude(g_captureStream);cudaGraph_t graph=nullptr;
            e=cudaStreamEndCapture(g_captureStream,&graph);if(work!=cudaSuccess){if(graph)cudaGraphDestroy(graph);return -(int)work;}if(e!=cudaSuccess)return -(int)e;
            e=installRuntimeGraph(g_materialPreludeGraph,graph);cudaGraphDestroy(graph);if(e!=cudaSuccess)return -(int)e;
            g_materialPreludeKey=key;
        }
        e=cudaGraphLaunch(g_materialPreludeGraph,0);
    }
    if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(query,(g_runtimeCoreMode&32768)?d_runtimeModeQueries:d_substepQuery,
        ((g_runtimeCoreMode&32768)?5:3)*sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    const bool skipRigid=(g_runtimeCoreMode&32768) && query[3]==0;
    const bool skipTranslation=(g_runtimeCoreMode&32768) && query[4]==0;
    if(g_runtimeCoreMode&32768){g_runtimeModeHostStats[0]++;g_runtimeModeHostStats[1]+=skipRigid;g_runtimeModeHostStats[2]+=skipTranslation;
        g_runtimeModeHostStats[3]+=!skipRigid;g_runtimeModeHostStats[4]+=!skipTranslation;}
    const int recoveryRequested=query[0];const bool loaded=!query[1];
    // A substep cannot gain a free-volume query: this list is fixed before
    // its material graph runs. Reuse the existing synchronous decision read.
    const bool anyFragmentQueries=query[2]>0;
    g_interactionStats[0]++;g_interactionStats[1]+=query[2];g_interactionStats[2]+=query[1];
    g_interactionStats[3]+=recoveryRequested!=0;g_interactionStats[4]+=loaded;g_interactionStats[9]=g_surfaceCount;
    g_interactionStats[15]=std::max(g_interactionStats[15],query[2]);
    const int fragmentGraphSlot=recoveryRequested?2:loaded?1:0;
    cudaGraphExec_t& selectedGraph=recoveryRequested?g_unifiedRecoveryGraph:loaded?g_unifiedLoadedGraph:g_unifiedGraph;
    bool& selectedGraphDirty=recoveryRequested?g_unifiedRecoveryGraphDirty:loaded?g_unifiedLoadedGraphDirty:g_unifiedGraphDirty;
    if(g_fragmentGraphQueries[fragmentGraphSlot]!=(int)anyFragmentQueries)selectedGraphDirty=true;
    const int runtimeModeKey=(skipRigid?1:0)|(skipTranslation?2:0)|(g_queryValidate?4:0);
    if((g_runtimeCoreMode&32768) && g_runtimeModeGraphKey[fragmentGraphSlot]!=runtimeModeKey)selectedGraphDirty=true;
    xpbd_section_mark("pre: predictor, recovery query");
    // Section timing issues the sequence directly, kernel after kernel, instead of capturing it.
    const bool direct=g_sectionProfile;
    cudaStream_t captureStream=g_captureStream;
    if(direct || !selectedGraph || selectedGraphDirty){
        g_interactionStats[5]++;
        if(direct)g_captureStream=0;
        else{
            if(!g_captureStream)e=cudaStreamCreate(&g_captureStream);if(e!=cudaSuccess)return -(int)e;
            e=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(e!=cudaSuccess)return -(int)e;
        }
        int sites=((s.nx+1)/2)*((s.ny+1)/2)*((g_dimsZ+2)/2);
        // Reuse the same swept-contact discovery after every material correction.
        auto refreshJawManifold=[&](){

                cudaMemsetAsync(d_surfaceDegree,0,g_N*sizeof(int),g_captureStream);
                cudaMemsetAsync(d_surfaceActiveCount,0,sizeof(int),g_captureStream);
                k_UpdateSurfacePoints<<<Gr(3*g_surfaceCount),64,0,g_captureStream>>>(3*g_surfaceCount,d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,d_surfaceCurrent,d_surfacePrevious);
                buildContactGroups(g_surfaceCount,d_surfaceCurrent,d_surfacePrevious,d_contactGroups,g_captureStream);
                classifyContactGroups(g_surfaceCount,3,d_contactGroups,d_contactGroupRejected,d_coupledTool,g_jawThickness,g_gp.jawGapClosed,g_contactRadius,false,false,g_captureStream);
                if(canBatchQueries(g_surfaceCount,d_coupledTool,2)){
        launchFindSurfaceGroup(g_surfaceCount,0x88u,3,false,g_captureStream,ContactShape{},ContactShape{},g_contactRadius,
            d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,d_surfaceContacts,d_surfaceDegree,d_surfaceActive,d_surfaceActiveCount,
            d_surfaceCurrent,d_surfacePrevious,d_coupledTool,g_jawThickness,g_gp.jawGapClosed,false,d_contactCacheSlots,d_contactCache,nullptr,3,d_contactGroupRejected);
    }else{for(int kind=0;kind<3;kind++)k_FindSurface<<<Gr(g_surfaceCount),64,0,g_captureStream>>>(g_surfaceCount,kind,
                    contactShape(previousTool,kind,g_jawThickness,g_gp.jawGapClosed),contactShape(stepTool,kind,g_jawThickness,g_gp.jawGapClosed),g_contactRadius,
                    d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,d_surfaceContacts,d_surfaceDegree,d_surfaceActive,d_surfaceActiveCount,d_surfaceCurrent,d_surfacePrevious,
                    d_coupledTool,g_jawThickness,g_gp.jawGapClosed,false,d_contactCacheSlots,d_contactCache,nullptr,3,false,d_contactGroupRejected+kind*contactGroupCount(g_surfaceCount));}
                cudaMemsetAsync(d_surfaceDegree,0,g_N*sizeof(int),g_captureStream);
                cudaMemsetAsync(d_manifoldCount,0,sizeof(int),g_captureStream);
                launchUnifiedManifold(g_captureStream,d_surfaceContacts,d_surfaceActive,d_surfaceActiveCount,d_surfaceVertices,d_pos,d_prevPos,
                    d_coupledTool,g_jawThickness,g_gp.jawGapClosed,g_contactRadius,d_manifold,d_manifoldIds,d_manifoldCount,d_manifoldOwners,d_manifoldQuadrature,d_surfaceDegree,3,3*g_surfaceCount);
                    };
        for(int it=0;it<4;it++){
            if(it==3)k_MaterialReady<<<1,1,0,g_captureStream>>>(d_unified,false);
            if(it>0 && g_surfaceCount>0){
                cudaMemsetAsync(d_manifoldCount,0,sizeof(int),g_captureStream);
                cudaMemsetAsync(d_surfaceDegree,0,g_N*sizeof(int),g_captureStream);
                launchUnifiedManifold(g_captureStream,d_surfaceContacts,d_surfaceActive,d_surfaceActiveCount,d_surfaceVertices,d_pos,d_prevPos,
                    d_coupledTool,g_jawThickness,g_gp.jawGapClosed,g_contactRadius,d_manifold,d_manifoldIds,d_manifoldCount,d_manifoldOwners,d_manifoldQuadrature,d_surfaceDegree,3,3*g_surfaceCount);
            }
            if(g_blade.enabled){findBlade(g_captureStream);k_MarkCoarseContact<<<128,64,0,g_captureStream>>>(d_bladeState,2|MaterialCutterContact);}
            k_MarkCoarseContact<<<128,64,0,g_captureStream>>>(d_unified);
            if(it==0)k_MarkVolumeRecovery<<<Gr(g_T),64,0,g_captureStream>>>(d_unified);
            xpbd_section_mark("loop: manifold, contact marks");
            if(it==0 || it==2)for(int mode=1;mode<=(it==0?4:1);mode++){
                if(anyFragmentQueries)k_CoarseWeights<true><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,mode);else k_CoarseWeights<false><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,mode);
                k_InitSafeCoarse<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified);
                k_JoinSafeCoarse<<<Gr(g_T),64,0,g_captureStream>>>(d_unified);
                k_FlattenSafeCoarse<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified);
                k_AccumulateSafeCoarse<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified);
                k_FinishSafeCoarse<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified);
                k_CoarseBegin<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                launchCoarseAccumulateCache(GrAccumulate(g_activeNodes+g_T),g_captureStream,d_unified,d_globalCoarse);
            if(anyFragmentQueries)k_FragmentCoarseTerms<<<128,64,0,g_captureStream>>>(d_unified,d_globalCoarse);
            if(g_blade.enabled)k_BladeCoarse<<<128,64,0,g_captureStream>>>(d_unified,d_bladeState,d_globalCoarse,false);
                k_CoarseSolve<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                launchCoarseBoundCache(GrAccumulate(g_T),g_captureStream,d_unified,d_globalCoarse);
            if(g_blade.enabled)k_BladeCoarse<<<128,64,0,g_captureStream>>>(d_unified,d_bladeState,d_globalCoarse,true);
            if(anyFragmentQueries)launchFragmentCoarseBounds(g_captureStream,d_globalCoarse);
                if(mode>=1 && mode<=4){
                    k_PressureLineClear<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                    for(int pass=0;pass<2;pass++){
                        if(anyFragmentQueries)k_FragmentLineClear<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified);
                    launchPressureLine(g_captureStream,Gr(g_activeNodes+g_T),d_unified,d_globalCoarse,pass);
                    if(anyFragmentQueries)k_FragmentLineSum<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse,pass);
                    if(anyFragmentQueries)k_FragmentLineSet<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                    }
                    k_PressureLineSelect<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                }
                if(anyFragmentQueries)k_CoarseApply<true><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);else k_CoarseApply<false><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
            }
            xpbd_section_mark("loop: coarse modes");
            // Solve unconstrained material DOFs with contact/anchor witnesses
            // fixed in this correction. Witnesses retain all four V18 sweeps;
            // remaining vertices receive a final nonlinear polish.
            // Free components need the larger budget to resolve gravity-driven
            // hinge motion; loaded components also have V18's coarse/local solve.
            if(it==0){
                // Keep one full correction for the established solve, then at
                // most two relinearizations for free, volume-limited recovery.
                // All share the SAME inertial target, previous shape and h.
                // Rediscover witnesses before each retry; newly contacted or
                // grasped components must return to the ordinary coupled solve.
                for(int nonlinear=0;nonlinear<(recoveryRequested?3:1);nonlinear++){
                if(nonlinear>0){
                    if(g_surfaceCount>0)refreshJawManifold();
                    if(g_blade.enabled){findBlade(g_captureStream);k_MarkCoarseContact<<<128,64,0,g_captureStream>>>(d_bladeState,2|MaterialCutterContact);}
                    k_MarkCoarseContact<<<128,64,0,g_captureStream>>>(d_unified);
                }
                k_MaterialRecoveryOnly<<<1,1,0,g_captureStream>>>(d_unified,nonlinear>0);
                xpbd_section_mark("cg assembly: start");
#ifdef ENDO_KIDNEY_STENCIL_CACHE
                k_MaterialTetCache<<<Gr(g_T),64,0,g_captureStream>>>(d_unified,d_materialTetCache);
                if(anyFragmentQueries)k_MaterialStencilAssemble<true,true><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgR,d_cgH,d_materialTetCache);else k_MaterialStencilAssemble<false,true><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgR,d_cgH,d_materialTetCache);
#else
                if(anyFragmentQueries)k_MaterialStencilAssemble<true><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgR,d_cgH);else k_MaterialStencilAssemble<false><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgR,d_cgH);
#endif
                xpbd_section_mark("cg assembly: stencil and residual");
                if(anyFragmentQueries)k_MaterialStencilActiveMask<true><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgStencil);else k_MaterialStencilActiveMask<false><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgStencil);
                xpbd_section_mark("cg assembly: mask");
                k_MaterialCGClear<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_cgS);
                assembleMultigrid(g_captureStream);
                k_MaterialCGBegin<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgQ,d_cgD);
                applyMultigrid(g_captureStream,d_cgR,d_cgZ,d_cgS,2);
                xpbd_section_mark("cg: assembly");
                const int iterations=nonlinear==0 && !loaded?MaterialFreeIterations:MaterialLoadedIterations;
                for(int cg=0;cg<iterations;cg++){
                    float3* direction=cg%2?d_cgQ:d_cgP;const float3* previous=cg%2?d_cgP:d_cgQ;
                    k_MaterialStencilProduct<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,previous,d_cgZ,direction,d_cgMatrix,d_cgScale,d_cgStencil,d_cgA,d_cgS,cg);
                    k_MaterialCGAdvance<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgStencil,d_cgR,direction,d_cgA,d_cgD,d_cgS,cg);
                    // The last direction is formed by no further product.
                    if(cg+1<iterations)applyMultigrid(g_captureStream,d_cgR,d_cgZ,d_cgS,cg%3);
                }
                xpbd_section_mark("cg: iterations");
                // The trust region and the tangent-cone projection act on free recovering components alone. Without
                // one at the query above none appears in this substep (component flags only gain bits, and the volume
                // flags were raised on the same positions), so both would leave every direction unchanged.
                if(recoveryRequested){
                    k_MaterialRestrictionClear<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_coarseWeight);
                    k_RecoveryTrust<<<Gr(g_T),64,0,g_captureStream>>>(d_unified,d_cgD,d_coarseWeight);
                    k_ApplyMaterialRestriction<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_cgD,d_coarseWeight,nullptr,d_unified);
                }
                cudaMemsetAsync(d_cgA,0,g_N*sizeof(float3),g_captureStream);
                cudaMemsetAsync(d_coarseWeight,0,g_N*sizeof(float),g_captureStream);
                for(int projection=0;recoveryRequested && projection<64;projection++){
                    k_ProjectRecoveryDirection<<<Gr(g_T),64,0,g_captureStream>>>(d_unified,d_cgH,d_cgD,d_cgA,d_coarseWeight);
                    k_ApplyRecoveryDirection<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_cgD,d_cgA,d_coarseWeight,true);
                }
                xpbd_section_mark("cg: trust, projection");
                for(int feasible=0;feasible<8;feasible++){
                    // Reuse two line-search flags before that buffer is cleared.
                    // If the preceding pass changed nothing, every later pass
                    // is a no-op; skip it on-device without a CPU readback.
                    int* changed=&d_cgSearch[feasible&1].invalid;
                    const int* previous=feasible?&d_cgSearch[(feasible-1)&1].invalid:nullptr;
                    cudaMemsetAsync(changed,0,sizeof(int),g_captureStream);
                    k_MaterialRestrictionClear<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_coarseWeight);
                    k_RestrictMaterialDirection<<<Gr(g_T),64,0,g_captureStream>>>(d_unified,d_cgD,d_coarseWeight,previous,changed);
                    k_ApplyMaterialRestriction<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_cgD,d_coarseWeight,changed,d_unified);
                }
                xpbd_section_mark("cg: feasible");
                k_MaterialLineClear<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_cgSearch);
                k_MaterialLineNodes<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgD,d_cgSearch);
                k_MaterialLineTets<<<Gr(g_T),64,0,g_captureStream>>>(d_unified,d_cgD,d_cgSearch);
                k_MaterialLineApply<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgD,d_cgSearch);
                k_MaterialFloorRecovery<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgSearch);
                k_MaterialRecoveryOnly<<<1,1,0,g_captureStream>>>(d_unified,false);
                k_MaterialReady<<<1,1,0,g_captureStream>>>(d_unified,true);
                xpbd_section_mark("cg: line search");
                }
            }
            // Coarse motion can bring previously separated triangles into a jaw.
            // Discover that swept contact before the local descent.
            if((it==0 || it==2) && g_surfaceCount>0){
                cudaMemsetAsync(d_surfaceDegree,0,g_N*sizeof(int),g_captureStream);
                cudaMemsetAsync(d_surfaceActiveCount,0,sizeof(int),g_captureStream);
                k_UpdateSurfacePoints<<<Gr(3*g_surfaceCount),64,0,g_captureStream>>>(3*g_surfaceCount,d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,d_surfaceCurrent,d_surfacePrevious);
                buildContactGroups(g_surfaceCount,d_surfaceCurrent,d_surfacePrevious,d_contactGroups,g_captureStream);
                classifyContactGroups(g_surfaceCount,3,d_contactGroups,d_contactGroupRejected,d_coupledTool,g_jawThickness,g_gp.jawGapClosed,g_contactRadius,false,false,g_captureStream);
                if(canBatchQueries(g_surfaceCount,d_coupledTool,2)){
        launchFindSurfaceGroup(g_surfaceCount,0x88u,3,false,g_captureStream,ContactShape{},ContactShape{},g_contactRadius,
            d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,d_surfaceContacts,d_surfaceDegree,d_surfaceActive,d_surfaceActiveCount,
            d_surfaceCurrent,d_surfacePrevious,d_coupledTool,g_jawThickness,g_gp.jawGapClosed,false,d_contactCacheSlots,d_contactCache,nullptr,3,d_contactGroupRejected);
    }else{for(int kind=0;kind<3;kind++)k_FindSurface<<<Gr(g_surfaceCount),64,0,g_captureStream>>>(g_surfaceCount,kind,
                    contactShape(previousTool,kind,g_jawThickness,g_gp.jawGapClosed),contactShape(stepTool,kind,g_jawThickness,g_gp.jawGapClosed),g_contactRadius,
                    d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,d_surfaceContacts,d_surfaceDegree,d_surfaceActive,d_surfaceActiveCount,d_surfaceCurrent,d_surfacePrevious,
                    d_coupledTool,g_jawThickness,g_gp.jawGapClosed,false,d_contactCacheSlots,d_contactCache,nullptr,3,false,d_contactGroupRejected+kind*contactGroupCount(g_surfaceCount));}
                cudaMemsetAsync(d_surfaceDegree,0,g_N*sizeof(int),g_captureStream);
                cudaMemsetAsync(d_manifoldCount,0,sizeof(int),g_captureStream);
                launchUnifiedManifold(g_captureStream,d_surfaceContacts,d_surfaceActive,d_surfaceActiveCount,d_surfaceVertices,d_pos,d_prevPos,
                    d_coupledTool,g_jawThickness,g_gp.jawGapClosed,g_contactRadius,d_manifold,d_manifoldIds,d_manifoldCount,d_manifoldOwners,d_manifoldQuadrature,d_surfaceDegree,3,3*g_surfaceCount);
            }
            if(g_blade.enabled){findBlade(g_captureStream);k_MarkCoarseContact<<<128,64,0,g_captureStream>>>(d_bladeState,2|MaterialCutterContact);}
            k_MarkCoarseContact<<<128,64,0,g_captureStream>>>(d_unified);
            if(anyFragmentQueries && (it==0 || it==2)){cudaError_t fragmentRefresh=captureFragmentRefresh(g_captureStream);if(fragmentRefresh!=cudaSuccess)return -(int)fragmentRefresh;}
            xpbd_section_mark("loop: contact discovery");
            // A tissue-only regional mode lets pressure move a finite patch
            // away from an operator-held jaw, complementing joint transport.
            if(it==0 || it==2){
                if(anyFragmentQueries)k_CoarseWeights<true><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,5);else k_CoarseWeights<false><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,5);
                k_CoarseBegin<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                launchCoarseAccumulateCache(GrAccumulate(g_activeNodes+g_T),g_captureStream,d_unified,d_globalCoarse);
            if(anyFragmentQueries)k_FragmentCoarseTerms<<<128,64,0,g_captureStream>>>(d_unified,d_globalCoarse);
            if(g_blade.enabled)k_BladeCoarse<<<128,64,0,g_captureStream>>>(d_unified,d_bladeState,d_globalCoarse,false);
                k_CoarseSolve<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                launchCoarseBoundCache(GrAccumulate(g_T),g_captureStream,d_unified,d_globalCoarse);
            if(g_blade.enabled)k_BladeCoarse<<<128,64,0,g_captureStream>>>(d_unified,d_bladeState,d_globalCoarse,true);
            if(anyFragmentQueries)launchFragmentCoarseBounds(g_captureStream,d_globalCoarse);
                k_PressureLineClear<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                for(int pass=0;pass<2;pass++){
                    if(anyFragmentQueries)k_FragmentLineClear<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified);
                    launchPressureLine(g_captureStream,Gr(g_activeNodes+g_T),d_unified,d_globalCoarse,pass);
                    if(anyFragmentQueries)k_FragmentLineSum<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse,pass);
                    if(anyFragmentQueries)k_FragmentLineSet<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                }
                k_PressureLineSelect<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                if(anyFragmentQueries)k_CoarseApply<true><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);else k_CoarseApply<false><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
            }
            xpbd_section_mark("loop: regional mode");
            if(anyFragmentQueries)k_CoarseWeights<true><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,0);else k_CoarseWeights<false><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,0);
            cudaMemsetAsync(d_sweepCount,0,8*sizeof(int),g_captureStream);
            if(anyFragmentQueries)k_SweepSites<true><<<Gr(g_N),64,0,g_captureStream>>>(d_unified,SweepLocal,sites,d_sweepSites,d_sweepCount);else k_SweepSites<false><<<Gr(g_N),64,0,g_captureStream>>>(d_unified,SweepLocal,sites,d_sweepSites,d_sweepCount);
            for(int c=0;c<8;c++){k_AssembleUnifiedInteractions<<<ASSEMBLY_BLOCKS,64,0,g_captureStream>>>(d_unified,c);if(anyFragmentQueries)k_AssembleFragmentInteractions<<<ASSEMBLY_BLOCKS,64,0,g_captureStream>>>(d_unified,c);if(g_blade.enabled)k_AssembleUnifiedInteractions<<<ASSEMBLY_BLOCKS,64,0,g_captureStream>>>(d_bladeState,c);if(anyFragmentQueries)k_UnifiedVertex<true><<<g_sweepBlocks,64,0,g_captureStream>>>(d_unified,d_sweepSites+c*sites,d_sweepCount+c);else k_UnifiedVertex<false><<<g_sweepBlocks,64,0,g_captureStream>>>(d_unified,d_sweepSites+c*sites,d_sweepCount+c);}
            xpbd_section_mark("loop: local sweeps");
            k_UnifiedTool<<<1,32,0,g_captureStream>>>(d_unified);
            if(g_blade.enabled)k_UnifiedTool<<<1,32,0,g_captureStream>>>(d_bladeState);
            // The coupled tool/patch direction needs the same feasible basis
            // as the tissue-only modes. Otherwise one nearly collapsed tet
            // stalls the entire drive, then releases it after a local sweep.
            k_InitSafeCoarse<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified);
            k_JoinSafeCoarse<<<Gr(g_T),64,0,g_captureStream>>>(d_unified);
            k_FlattenSafeCoarse<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified);
            k_AccumulateSafeCoarse<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified);
            k_FinishSafeCoarse<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified);
            k_CoarseBegin<<<1,1,0,g_captureStream>>>(d_unified,d_coarseBlock);
            launchCoarseAccumulateCache(GrAccumulate(g_activeNodes+g_T),g_captureStream,d_unified,d_coarseBlock);
            if(anyFragmentQueries)k_FragmentCoarseTerms<<<128,64,0,g_captureStream>>>(d_unified,d_coarseBlock);
            if(g_blade.enabled)k_BladeCoarse<<<128,64,0,g_captureStream>>>(d_unified,d_bladeState,d_coarseBlock,false);
            k_CoarseSolve<<<1,1,0,g_captureStream>>>(d_unified,d_coarseBlock);
            launchCoarseBoundCache(GrAccumulate(g_T),g_captureStream,d_unified,d_coarseBlock);
            if(g_blade.enabled)k_BladeCoarse<<<128,64,0,g_captureStream>>>(d_unified,d_bladeState,d_coarseBlock,true);
            if(anyFragmentQueries)launchFragmentCoarseBounds(g_captureStream,d_coarseBlock);
            if(anyFragmentQueries)k_CoarseApply<true><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_coarseBlock);else k_CoarseApply<false><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_coarseBlock);
            xpbd_section_mark("loop: tool, safe and global coarse");
                if(anyFragmentQueries)k_CoarseWeights<true><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,7);else k_CoarseWeights<false><<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,7);
                k_CoarseBegin<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
            if(!skipRigid || g_queryValidate){ // Reference proof retains the original stage.
                if(skipRigid && g_queryValidate)cudaMemcpyAsync(d_runtimeModeBefore,d_pos,(size_t)g_N*sizeof(float3),cudaMemcpyDeviceToDevice,g_captureStream);
                k_RigidCenters<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                k_RigidContactKinds<<<128,64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                k_RigidCenterFinish<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                k_RigidNodes<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                k_CoarseSolve<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                k_RigidEnergy<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                k_RigidFeasible<<<Gr(g_T),64,0,g_captureStream>>>(d_unified,d_globalCoarse);
                k_RigidApply<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);

                if(skipRigid && g_queryValidate)k_RuntimeProveNoPositionWrite<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_runtimeModeBefore,d_pos,d_runtimeModeProof,0);
            }
            xpbd_section_mark("loop: rigid modes");
        }
        if(direct)g_captureStream=captureStream;
        else{
            cudaGraph_t graph=nullptr;e=cudaStreamEndCapture(g_captureStream,&graph);if(e!=cudaSuccess)return -(int)e;
            if(selectedGraph){
                g_interactionStats[6]++;cudaGraphExecUpdateResultInfo info={};e=cudaGraphExecUpdate(selectedGraph,graph,&info);
                if(e==cudaErrorGraphExecUpdateFailure || (e==cudaSuccess && info.result!=cudaGraphExecUpdateSuccess)){
                    cudaGetLastError();cudaGraphExecDestroy(selectedGraph);selectedGraph=nullptr;e=cudaSuccess;
                }
            }
            if(e==cudaSuccess && !selectedGraph){g_interactionStats[8]++;e=cudaGraphInstantiate(&selectedGraph,graph,0);}
            cudaGraphDestroy(graph);if(e!=cudaSuccess)return -(int)e;selectedGraphDirty=false;g_fragmentGraphQueries[fragmentGraphSlot]=(int)anyFragmentQueries;g_runtimeModeGraphKey[fragmentGraphSlot]=runtimeModeKey;
        }
    }
    if(!direct){e=cudaGraphLaunch(selectedGraph,0);if(e!=cudaSuccess)return -(int)e;}
    k_MaterialReady<<<1,1>>>(d_unified,false);
    // Finish the translational nullspace solve ONLY for free components. Internal
    // elasticity cannot change their COM momentum. Components with supports,
    // contact or grasp remain in the coupled iteration above; a tissue-only
    // correction there would undo the tool/tissue contact balance.
    if(anyFragmentQueries)k_CoarseWeights<true><<<Gr(g_activeNodes),64>>>(d_unified,6);else k_CoarseWeights<false><<<Gr(g_activeNodes),64>>>(d_unified,6);
    k_CoarseBegin<<<Gr(g_activeNodes),64>>>(d_unified,d_globalCoarse);
    if(!skipTranslation || g_queryValidate){
        if(skipTranslation && g_queryValidate)cudaMemcpyAsync(d_runtimeModeBefore,d_pos,(size_t)g_N*sizeof(float3),cudaMemcpyDeviceToDevice);
    launchCoarseAccumulateCache(GrAccumulate(g_activeNodes+g_T),0,d_unified,d_globalCoarse);
            if(anyFragmentQueries)k_FragmentCoarseTerms<<<128,64,0,0>>>(d_unified,d_globalCoarse);
            if(g_blade.enabled)k_BladeCoarse<<<128,64>>>(d_unified,d_bladeState,d_globalCoarse,false);
    k_CoarseSolve<<<Gr(g_activeNodes),64>>>(d_unified,d_globalCoarse);
    launchCoarseBoundCache(GrAccumulate(g_T),0,d_unified,d_globalCoarse);
            if(g_blade.enabled)k_BladeCoarse<<<128,64>>>(d_unified,d_bladeState,d_globalCoarse,true);
            if(anyFragmentQueries)launchFragmentCoarseBounds(0,d_globalCoarse);
    if(anyFragmentQueries)k_CoarseApply<true><<<Gr(g_activeNodes),64>>>(d_unified,d_globalCoarse);else k_CoarseApply<false><<<Gr(g_activeNodes),64>>>(d_unified,d_globalCoarse);
        if(skipTranslation && g_queryValidate)k_RuntimeProveNoPositionWrite<<<Gr(g_N),64>>>(g_N,d_runtimeModeBefore,d_pos,d_runtimeModeProof,3);
    }
    xpbd_section_mark("post: free-component nullspace");
    e=solveFragmentContact();if(e!=cudaSuccess)return -(int)e;
    profileMark(g_substepIndex-1,1);profileMark(g_substepIndex-1,2);
    xpbd_section_mark("post: fragment contact");
    if(g_blade.enabled){
        k_BladeReaction<<<1,32>>>(d_bladeState,d_bladePhase,g_blade.startForce,d_bladeResult,d_bladeHistory,true,false,d_scissorsLoads,g_scissors.squeeze);
        k_UnifiedTool<<<1,32>>>(d_bladeState);
        k_BladeReaction<<<1,32>>>(d_bladeState,d_bladePhase,g_blade.startForce,d_bladeResult,d_bladeHistory,false,true);
        k_BladeSweep<<<1,1>>>(d_bladeBody,d_bladeSweep,d_bladeHistory);
        k_EndTool<<<1,1>>>(d_bladeBody);
    }

    k_UnifiedReaction<<<128,64>>>(d_unified);
    if(g_manifoldCapacity>0){
        k_UnifiedAggregate<<<128,64>>>(d_unified,d_manifoldOwners,d_surfaceContacts,!g_anchorAssist);
        if(!g_anchorAssist)k_UnifiedStick<<<128,64>>>(d_unified,d_surfaceActive,d_surfaceActiveCount,d_surfaceContacts,g_jawThickness,g_gp.jawGapClosed);
    }
    k_EndTool<<<1,1>>>(d_coupledTool);
    float scale=1.f/(h*h*fmaxf(1.f,(float)g_p.numSubsteps));
    if(s.grips>0)k_GripReaction<<<Gr(s.grips),64>>>(s.grips,d_gripPoints,stepTool,scale,d_contactReaction,d_jawLoad);
    if(g_surfaceCount>0)k_SurfaceReaction<<<Gr(3*g_surfaceCount),64>>>(3*g_surfaceCount,scale,d_surfaceContacts,d_contactReaction,d_jawLoad,
        contactShape(stepTool,1,g_jawThickness,g_gp.jawGapClosed),contactShape(stepTool,2,g_jawThickness,g_gp.jawGapClosed));
    // Report the forces actually applied in this substep before releasing an overloaded witness.
    if(g_trainingFlags)trainingSubstep(h);
    k_update_velocity<<<Gr(g_N),64>>>(g_N,1.f/h,d_invMass,d_pos,d_prevPos,d_vel);
    k_apply_velscale<<<Gr(g_N),64>>>(g_N,g_velScaleF,d_invMass,d_vel);
    if(g_substepIndex>=g_p.numSubsteps)g_toolFrom=g_tool;
    xpbd_section_mark("post: blade and contact reactions, velocity");
    profileMark(g_substepIndex-1,3);return -(int)cudaGetLastError();
}

extern "C" int xpbd_substep(void)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    if (!g_frameReady) return XPBD_ERR_NO_FRAME;        // no begin_frame yet, or invalidated by
                                                        // set_params / re-init / set_bending
    float h = g_hFrame;
    profileMark(g_substepIndex,0);
    float t=fminf(1.f,float(++g_substepIndex)/float(g_p.numSubsteps>0?g_p.numSubsteps:1));
    float previousT=fmaxf(0.f,t-1.f/float(g_p.numSubsteps>0?g_p.numSubsteps:1));
    XpbdToolState previousTool=g_tool;
    previousTool.jawHinge=g_toolFrom.jawHinge*(1-previousT)+g_tool.jawHinge*previousT;
    previousTool.jawAxis=g_toolFrom.jawAxis*(1-previousT)+g_tool.jawAxis*previousT;
    previousTool.jawNormal=g_toolFrom.jawNormal*(1-previousT)+g_tool.jawNormal*previousT;
    previousTool.shaftA=g_toolFrom.shaftA*(1-previousT)+g_tool.shaftA*previousT;
    previousTool.shaftB=g_toolFrom.shaftB*(1-previousT)+g_tool.shaftB*previousT;
    previousTool.jawGap=g_toolFrom.jawGap*(1-previousT)+g_tool.jawGap*previousT;
    XpbdToolState stepTool=g_tool;
    // Interpolate translations and directions per substep instead of teleporting at frame boundaries.
    stepTool.jawHinge=g_toolFrom.jawHinge*(1.f-t)+g_tool.jawHinge*t;
    stepTool.jawAxis=g_toolFrom.jawAxis*(1.f-t)+g_tool.jawAxis*t;
    stepTool.jawNormal=g_toolFrom.jawNormal*(1.f-t)+g_tool.jawNormal*t;
    stepTool.shaftA=g_toolFrom.shaftA*(1.f-t)+g_tool.shaftA*t;
    stepTool.shaftB=g_toolFrom.shaftB*(1.f-t)+g_tool.shaftB*t;
    stepTool.jawGap=g_toolFrom.jawGap*(1.f-t)+g_tool.jawGap*t;
    if(d_coupledTool && g_toolArmed)
        k_PredictTool<<<1,1>>>(d_coupledTool,stepTool,g_toolFrom,h,g_toolMass,g_toolStiffness,g_toolDamping,g_toolMaxTravel/fmaxf(1.f,(float)g_p.numSubsteps),g_jawMass,g_jawStiffness,g_jawDamping,g_jawSpeed,g_tool.jawHinge,g_tool.jawGap,g_guidedInstrument);
    float3 g = mkf3(g_p.gx, g_p.gy, g_p.gz);
    if (g_bound) {
        // §9-(4): refresh liveness + invMass from the LIVE shared flags at the START of every substep,
        // BEFORE predict — a cut.cu sever/debris-freeze committed since the last substep takes effect now.
        // STEP 7: the same refresh re-derives d_tetActive (domain T) from the identical live flags.
        int refreshMax = g_N;
        if (g_E > refreshMax) refreshMax = g_E;
        if (g_B > refreshMax) refreshMax = g_B;
        if (g_T > refreshMax) refreshMax = g_T;
        k_TopologyStale<<<1,1>>>(cut_topology_revision_device(), d_livenessRevision);
        k_RefreshLiveness<<<Gr(refreshMax), 64>>>(g_N, g_E, g_B, g_T,
            g_liveNbrIdx, d_edgeOwnerSlot, d_edgeI, d_edgeJ,
            g_liveBendPairs, d_bendSrcIdx,
            g_liveActive, g_livePinned, g_liveMass,
            d_tetIds, d_tetEdgeOwnerSlot,
            d_edgeActive, d_bendActive, d_invMass, d_tetActive, d_livenessRevision + 1);
    }
    bool fallbackEdges=g_bound && g_doStrainF && g_p.ks<=0.f && d_edgeTetSupport;
    if(fallbackEdges) {
        // Its own record: the repair refreshes the liveness flags without it.
        k_TopologyStale<<<1,1>>>(cut_topology_revision_device(), d_edgeSupportRevision);
        k_ClearTetEdgeSupport<<<Gr(6*g_N),64>>>(6*g_N,d_edgeTetSupport,d_edgeSupportRevision+1);
        k_MarkTetEdgeSupport<<<Gr(g_T),64>>>(g_T,d_tetActive,d_tetEdgeOwnerSlot,d_edgeTetSupport,d_edgeSupportRevision+1);
    }
    // A fresh fragment may already overlap the newly reconstructed cut lip.
    // Repair that starting state before ballistic prediction can carry it
    // entirely through the organ, beyond endpoint containment and swept tests.
    bool componentsFresh=false;
    if(d_unified && d_unifiedComponents && g_surfaceCount>0){
        cudaError_t contactError=refreshUnifiedComponents();
        if(contactError==cudaSuccess)contactError=refreshFragmentSupport();
        if(contactError!=cudaSuccess)return -(int)contactError;
        componentsFresh=true;
        cudaMemsetAsync(d_graspComponents,0,g_N*sizeof(int));
        k_UnifiedPatchBasis<<<1,32>>>(d_unified);
        contactError=solveFragmentContact(true);
        if(contactError!=cudaSuccess)return -(int)contactError;
    }
    beginFeasible();
    k_predict<<<Gr(g_N), 64>>>(g_N, h, g,
                               g_bound ? g_liveExtForce    : nullptr,
                               g_bound && !g_frictionContact ? g_liveHapticForce : nullptr, // avoid a second, proxy-based tissue traction
                               d_invMass, d_pos, d_prevPos, d_vel);
    xpbd_section_mark("substep prologue: liveness, fragment contact, predict");
    if(d_coupledTool && g_toolArmed && d_incidentOffsets && g_frictionContact && d_surfaceActiveCount) return unifiedSubstep(h,previousTool,stepTool,componentsFresh);
    // redesign 2026-07-07: core corners now have NORMAL invMass (see k_RefreshLiveness), so predict no
    // longer skips them — they need no separate pre-step target-force; the compliant k_SolveCorePin,
    // dispatched inside the K-loop below, is their only mover.
    if (g_E > 0)   cudaMemsetAsync(d_lambda,     0, (size_t)g_E * sizeof(float));   // reset multipliers /substep
    if (g_doBendF) cudaMemsetAsync(d_bendLambda, 0, (size_t)g_B * sizeof(float));
    if (g_doStrainF) {
        cudaMemsetAsync(d_tetLambdaDevL, 0, (size_t)g_T * sizeof(float));
        cudaMemsetAsync(d_tetLambdaHydL, 0, (size_t)g_T * sizeof(float));
    }
    if (g_bound && g_toolArmed) {   // CCD gate's multipliers: reset whenever the tool is armed
        cudaMemsetAsync(d_jawContactLambda,   0, (size_t)g_N * sizeof(float));
        cudaMemsetAsync(d_shaftContactLambda, 0, (size_t)g_N * sizeof(float));
    }
    if (g_bound && g_grasping) {     // grasp-only multipliers: reset only while genuinely grasping
        cudaMemsetAsync(d_slabLambda, 0, (size_t)g_N * sizeof(float));
        cudaMemsetAsync(d_coreLambda, 0, (size_t)g_N * sizeof(float3));
    }
    if(g_embeddedGrip && g_grasping){
        refreshGrip(); // cut_ribbon_tick can change owners between substeps
        if(g_gripCount>0){
            GripSolveStep state={stepTool,previousTool,g_gp.jawGapClosed,g_gripCompliance/(h*h),g_gripDamping/h,g_gp.graspMaxDisp};
            k_ResetGrip<<<Gr(g_gripCount),64>>>(g_gripCount,d_gripPoints,d_gripStep,state);
            cudaMemsetAsync(&d_gripPatch->lambda,0,sizeof(float3));
            cudaMemsetAsync(&d_gripPatch->lambdaU,0,2*sizeof(float3));
        }
    }
    if(d_fiberLambda && g_volumePasses>0)cudaMemsetAsync(d_fiberLambda,0,(size_t)6*g_T*sizeof(float));
    if (d_attachLambda) cudaMemsetAsync(d_attachLambda, 0, (size_t)g_N*sizeof(float3));
    if(g_frictionContact && g_toolArmed) {
        cudaMemsetAsync(d_contactNormalLambda,0,(size_t)3*g_N*sizeof(float));
        cudaMemsetAsync(d_contactTangentLambda,0,(size_t)3*g_N*sizeof(float3));
        if(!g_anchorAssist)cudaMemsetAsync(d_grasped,0,(size_t)g_N*sizeof(int));
    }
    // redesign 2026-07-07 (decision 4, user-confirmed): K=2 coupling gates on GENUINE GRASP, not merely
    // "tool armed" — the force-only CCD/contact path (dispatched separately below) needs no
    // reconciliation with the elastic solve; only the core-pin+slab set benefits from a second pass.
    // This is the SECOND framerate fix (alongside retiring the hard contact constraint): before this
    // change K=2 ran unconditionally from the first xpbd_set_tool call, doubling elastic iteration cost
    // even when the tool was merely nearby and not gripping anything.
    // Only legacy core pins need the extra material pass. Surface contact has
    // its own coupled-node iterations; closing the jaws must not double material work.
    int K = (g_bound && g_grasping && !g_frictionContact) ? 2 : 1;
    float3 jawAxis3   = (len3(stepTool.jawAxis)   > 1e-9f) ? stepTool.jawAxis   * (1.f/len3(stepTool.jawAxis))   : mkf3(0,0,1);
    float3 jawNormal3 = (len3(stepTool.jawNormal) > 1e-9f) ? stepTool.jawNormal * (1.f/len3(stepTool.jawNormal)) : mkf3(0,1,0);
    jawNormal3=jawNormal3-jawAxis3*dot3(jawNormal3,jawAxis3);
    float nl=len3(jawNormal3);
    if (nl<1e-7f) jawNormal3=cross3(jawAxis3,fabsf(jawAxis3.x)<0.9f?mkf3(1,0,0):mkf3(0,1,0));
    jawNormal3=jawNormal3*(1.f/len3(jawNormal3));
    float3 stepW=cross3(jawAxis3,jawNormal3);
    float  effGap     = fmaxf(stepTool.jawGap, g_gp.jawGapClosed), gapHalf = 0.5f * effGap;
    for (int k = 0; k < K; k++) {
        for (int it = 0; it < g_itersF; it++) {
            for (int c = 0; c < g_numColors; c++)               // structural distance constraints
                if ((g_p.ks > 0.f || fallbackEdges) && g_colorCnt[c] > 0)
                    k_solve_edges<<<Gr(g_colorCnt[c]), 64>>>(g_colorOff[c], g_colorCnt[c],
                        g_alphaTildeF, g_p.omega, g_betaGammaHSF, d_prevPos,
                        d_edgeI, d_edgeJ, d_restLen, d_edgeActive, d_invMass, d_lambda, d_pos,
                        fallbackEdges?d_edgeTetSupport:nullptr,d_edgeOwnerSlot,
                        fallbackEdges?1.f/(g_p.youngE*h*h):0.f);
            if (g_doBendF)                                       // then bending angle constraints
                for (int c = 0; c < g_bNumColors; c++)
                    if (g_bColorCnt[c] > 0)
                        k_solve_bending<<<Gr(g_bColorCnt[c]), 64>>>(g_bColorOff[c], g_bColorCnt[c],
                            g_alphaBF, g_p.omega, g_betaGammaHBF, d_prevPos,
                            d_bendI, d_bendJ, d_bendK, d_bendTheta0, d_bendActive, d_invMass, d_bendLambda, d_pos);
            if (g_doStrainF) {
                cudaError_t graphError=solveTetGraph(fmaxf(0.f,g_p.strainDamping)/h);
                if(graphError!=cudaSuccess)return -(int)graphError;
            }
        }
        if (d_attachStiffness)
            k_SolveAttachments<<<Gr(g_N),64>>>(g_N,1.f/(h*h),d_attachStiffness,d_attachRest,d_invMass,d_attachLambda,d_pos);
        if (g_bound && g_grasping && (!g_frictionContact || (g_anchorAssist && !g_embeddedGrip))) {
            if(!g_frictionContact)k_SolveSlabCoulomb<<<Gr(g_N), 64>>>(g_N, g_gp.omegaGrasp, h, g_gp.contactAlpha/(h*h), g_gp.mu, g_gp.graspMaxDisp,
                stepTool.jawHinge, jawAxis3, jawNormal3, stepW, stepTool.jawLen, stepTool.jawHalfWidth, gapHalf,
                d_grasped, d_graspCore, d_invMass, d_slabLambdaPrev, d_slabLambda,
                d_coulombAnchorU, d_coulombAnchorW, d_pos);
            k_SolveCorePin<<<Gr(g_N), 64>>>(g_N, g_gp.omegaGrasp, g_gp.coreAlpha/(h*h), g_gp.graspMaxDisp,
                stepTool.jawHinge, jawAxis3, jawNormal3, stepW, d_coreOffU, d_coreOffN, d_coreOffW,
                d_invMass, d_pos, d_coreLambda, d_graspCore,(g_anchorAssist && g_captureWedge)?1:0,(g_anchorAssist && g_captureWedge)?d_coreComplianceScale:nullptr);
        }
    }
    if (g_bound && g_grasping && !g_frictionContact) {   // legacy break check after material/core coupling
                                    // coreLambda (see k_SolveCorePin/k_CoreBreakCheck headers for why this
                                    // must not run once per K-pass).
        k_CoreBreakCheck<<<Gr(g_N), 64>>>(g_N, g_gp.coreBreakForce, h, d_coreLambda, d_coreForceEMA, d_graspCore, d_grasped);
    }
    if (g_bound && g_toolArmed && !g_frictionContact) {   // legacy CCD contact: force/speculative-only, dispatched ONCE, LAST, regardless
                                    // of K or grasp state — "project contact last" (unchanged principle),
                                    // but no longer needs a second coupling pass (redesign 2026-07-07).
        k_SolveCCDContact<<<Gr(g_N), 64>>>(g_N, g_gp.ccdOmega, g_gp.ccdVelThresh, g_gp.ccdMaxDisp, g_gp.contactAlpha/(h*h),
            stepTool.shaftA, stepTool.shaftB, stepTool.shaftR,
            stepTool.jawHinge, jawAxis3, jawNormal3, stepW, stepTool.jawLen, stepTool.jawHalfWidth, gapHalf,
            d_grasped, d_invMass, d_vel, d_pos, d_jawContactLambda, d_shaftContactLambda);
    }
    profileMark(g_substepIndex-1,1);
    if(g_anchorAssist && g_grasping && !g_embeddedGrip)
        k_SolveCorePin<<<Gr(g_N),64>>>(g_N,1.f,g_gp.coreAlpha/(h*h),g_gp.graspMaxDisp,
            stepTool.jawHinge,jawAxis3,jawNormal3,stepW,d_coreOffU,d_coreOffN,d_coreOffW,
            d_invMass,d_pos,d_coreLambda,d_graspCore,(g_anchorAssist && g_captureWedge)?1:0,(g_anchorAssist && g_captureWedge)?d_coreComplianceScale:nullptr);
    cudaError_t gripError=solveGrip();if(gripError!=cudaSuccess)return -(int)gripError;
    if(g_frictionContact && g_toolArmed) {
        if(g_surfaceCount>0){
            cudaMemsetAsync(d_surfaceDegree,0,(size_t)g_N*sizeof(int));
            cudaMemsetAsync(d_surfaceActiveCount,0,sizeof(int));
            k_UpdateSurfacePoints<<<Gr(3*g_surfaceCount),64>>>(3*g_surfaceCount,d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,d_surfaceCurrent,d_surfacePrevious);
            buildContactGroups(g_surfaceCount,d_surfaceCurrent,d_surfacePrevious,d_contactGroups,0);
            classifyContactGroups(g_surfaceCount,3,d_contactGroups,d_contactGroupRejected,d_coupledTool,g_jawThickness,g_gp.jawGapClosed,g_contactRadius,false,false,0);
            if(canBatchQueries(g_surfaceCount,d_coupledTool,2)){
        launchFindSurfaceGroup(g_surfaceCount,0x88u,3,false,0,ContactShape{},ContactShape{},g_contactRadius,
            d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,d_surfaceContacts,d_surfaceDegree,d_surfaceActive,d_surfaceActiveCount,
            d_surfaceCurrent,d_surfacePrevious,d_coupledTool,g_jawThickness,g_gp.jawGapClosed,false,d_contactCacheSlots,d_contactCache,nullptr,3,d_contactGroupRejected);
    }else{for(int kind=0;kind<3;kind++)
                k_FindSurface<<<Gr(g_surfaceCount),64>>>(g_surfaceCount,kind,
                    contactShape(previousTool,kind,g_jawThickness,g_gp.jawGapClosed),
                    contactShape(stepTool,kind,g_jawThickness,g_gp.jawGapClosed),g_contactRadius,
                    d_surfaceVertices,g_surfaceMasks,d_prevPos,d_pos,d_surfaceContacts,d_surfaceDegree,d_surfaceActive,d_surfaceActiveCount,d_surfaceCurrent,d_surfacePrevious,d_coupledTool,g_jawThickness,g_gp.jawGapClosed,false,d_contactCacheSlots,d_contactCache,nullptr,3,false,d_contactGroupRejected+kind*contactGroupCount(g_surfaceCount));}
            cudaError_t contactError=solveSurfaceGraph(g_gp.contactAlpha/(h*h),g_grasping && !g_anchorAssist?1:0);
            if(contactError!=cudaSuccess)return -(int)contactError;
            // Pressure/contact block iterations: let incompressibility displace
            // neighboring material before re-solving the same contact multipliers.
            // Resetting contact lambdas here would discard the tissue's resistance.
            for(int coupling=0;g_volumePasses>0 && coupling<2;coupling++){
                contactError=solvePressureGraph();if(contactError!=cudaSuccess)return -(int)contactError;
                contactError=solveGrip();if(contactError!=cudaSuccess)return -(int)contactError;
                if(g_anchorAssist && g_grasping && !g_embeddedGrip)
                    k_SolveCorePin<<<Gr(g_N),64>>>(g_N,1.f,g_gp.coreAlpha/(h*h),g_gp.graspMaxDisp,
                        stepTool.jawHinge,jawAxis3,jawNormal3,stepW,d_coreOffU,d_coreOffN,d_coreOffW,
                        d_invMass,d_pos,d_coreLambda,d_graspCore,g_captureWedge?1:0,g_captureWedge?d_coreComplianceScale:nullptr);
                contactError=solveSurfaceGraph(g_gp.contactAlpha/(h*h),g_grasping && !g_anchorAssist?1:0);
                if(contactError!=cudaSuccess)return -(int)contactError;
            }

        }else{
            for(int kind=0;kind<3;kind++)
                k_FindToolContacts<<<Gr(g_N),64>>>(g_N,contactShape(previousTool,kind,g_jawThickness,g_gp.jawGapClosed),
                    contactShape(stepTool,kind,g_jawThickness,g_gp.jawGapClosed),kind,g_contactRadius,d_prevPos,d_pos,d_invMass,d_contacts);
            for(int it=0;it<2;it++)
                k_SolveFrictionContacts<<<Gr(g_N),64>>>(g_N,g_gp.contactAlpha/(h*h),g_muStatic,g_muDynamic,
                    g_grasping && !g_anchorAssist?1:0,d_invMass,d_prevPos,d_contacts,d_contactNormalLambda,d_contactTangentLambda,d_grasped,d_pos);
            k_NodeReaction<<<Gr(3*g_N),64>>>(g_N,1.f/(h*h*fmaxf(1.f,(float)g_p.numSubsteps)),d_contacts,d_contactNormalLambda,d_contactTangentLambda,d_contactReaction);
        }
    }
    profileMark(g_substepIndex-1,2);
    // Velocity write, split so the STEP 7 perpendicular cs post-pass runs strictly between the raw write
    if(g_anchorAssist && g_grasping && !g_embeddedGrip){
        // Contact is projected after the anchor so anchors cannot undo nonpenetration.
        if(!g_captureWedge)k_CoreBreakCheck<<<Gr(g_N),64>>>(g_N,g_gp.coreBreakForce,h,d_coreLambda,d_coreForceEMA,d_graspCore,d_grasped);
        k_AnchorReaction<<<Gr(g_N),64>>>(g_N,-1.f/(h*h*fmaxf(1.f,(float)g_p.numSubsteps)),d_graspCore,d_coreLambda,d_contactReaction);
    }

    // and the velScale (Rayleigh/legacy-damping) multiply (design §4[Structural]).
    if(d_coupledTool && g_surfaceCount>0)k_ContactResidual<<<128,64>>>(d_pos,d_surfaceContacts,d_surfaceActive,d_surfaceActiveCount,d_coupledTool,d_contactResidual);
    {cudaError_t e=finishFeasible();if(e!=cudaSuccess)return -(int)e;}
    if(d_coupledTool && g_surfaceCount>0)k_ContactResidual<<<128,64>>>(d_pos,d_surfaceContacts,d_surfaceActive,d_surfaceActiveCount,d_coupledTool,d_contactResidual+1);
    if(d_coupledTool && g_toolArmed){
        // Reconcile after the nonlinear determinant safeguard with BOTH tissue
        // and tool/aperture movable, and the same mass metric/multipliers.
        for(int pass=0;pass<1;pass++){
            if(d_feasibleStart && g_volumePasses>0)
                k_ContactMobility<<<Gr(g_N),64>>>(d_coupledTool,g_N,d_invMass,d_needFeasible,d_feasibleStart,d_contactMobility,d_pos,d_contactMobility);
            beginFeasible();
            cudaError_t e=solveGrip();if(e!=cudaSuccess)return -(int)e;
            if(g_surfaceCount>0){e=solveSurfaceGraph(g_gp.contactAlpha/(h*h),0);if(e!=cudaSuccess)return -(int)e;}
            {cudaError_t e=finishFeasible();if(e!=cudaSuccess)return -(int)e;}
        }
        k_EndTool<<<1,1>>>(d_coupledTool);
    }
    if(d_coupledTool && g_surfaceCount>0)k_ContactResidual<<<128,64>>>(d_pos,d_surfaceContacts,d_surfaceActive,d_surfaceActiveCount,d_coupledTool,d_contactResidual+2);
    if(g_embeddedGrip && g_grasping && g_gripCount>0){
        float scale=1.f/(h*h*fmaxf(1.f,(float)g_p.numSubsteps));
        if(d_coupledTool){
            k_PatchReaction<<<1,1>>>(d_gripPatch,d_coupledTool,scale,d_contactReaction,d_jawLoad);
            k_GripReaction<<<Gr(g_gripCount),64>>>(g_gripCount,d_gripPoints,stepTool,scale,d_contactReaction,d_jawLoad);
        }
        else k_GripReaction<<<Gr(g_gripCount),64>>>(g_gripCount,d_gripPoints,stepTool,scale,d_contactReaction,d_jawLoad);
    }
    if(g_frictionContact && g_toolArmed && g_surfaceCount>0){
            k_SurfaceReaction<<<Gr(3*g_surfaceCount),64>>>(3*g_surfaceCount,1.f/(h*h*fmaxf(1.f,(float)g_p.numSubsteps)),d_surfaceContacts,d_contactReaction,d_jawLoad,
                contactShape(stepTool,1,g_jawThickness,g_gp.jawGapClosed),contactShape(stepTool,2,g_jawThickness,g_gp.jawGapClosed));
    }
    k_update_velocity<<<Gr(g_N), 64>>>(g_N, 1.f / h, d_invMass, d_pos, d_prevPos, d_vel);
    if(g_frictionContact && g_toolArmed && g_surfaceCount>0){
        cudaError_t velocityError=dampSurfaceGraph(h);if(velocityError!=cudaSuccess)return -(int)velocityError;
    }
    if (g_bound && g_grasping)   // snapshot for the NEXT substep's Fn (design §5 item 3 PREV-substep timing)
        cudaMemcpyAsync(d_slabLambdaPrev, d_slabLambda, (size_t)g_N * sizeof(float), cudaMemcpyDeviceToDevice);
    if (g_p.betaS > 0.f)
        for (int c = 0; c < g_numColors; c++)
            if (g_colorCnt[c] > 0)
                k_perp_damp_edges<<<Gr(g_colorCnt[c]), 64>>>(g_colorOff[c], g_colorCnt[c], g_p.betaS, h,
                    d_pos, d_edgeI, d_edgeJ, d_edgeActive, d_invMass, d_vel);
    k_apply_velscale<<<Gr(g_N), 64>>>(g_N, g_velScaleF, d_invMass, d_vel);
    if (g_substepIndex>=g_p.numSubsteps) g_toolFrom=g_tool;
    profileMark(g_substepIndex-1,3);
    return -(int)cudaGetLastError();                        // async launch check; caller syncs at frame end
}

extern "C" int xpbd_step(float dt)
{
    int rc = xpbd_begin_frame(dt);
    if (rc != 0) return rc;
    int N = g_p.numSubsteps > 0 ? g_p.numSubsteps : 1;
    for (int sub = 0; sub < N; sub++) {
        rc = xpbd_substep();
        if (rc != 0) return rc;
    }
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

extern "C" int xpbd_set_attachments(int count, const float* stiffness)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    if (count != g_N || !stiffness) return XPBD_ERR_BAD_ARG;
    for (int i=0;i<count;i++) if (!std::isfinite(stiffness[i]) || stiffness[i]<0.f) return XPBD_ERR_BAD_ARG;
    if (!d_attachStiffness) {
        XK(cudaMalloc(&d_attachStiffness, (size_t)g_N*sizeof(float)));
        XK(cudaMalloc(&d_attachRest, (size_t)g_N*sizeof(float3)));
        XK(cudaMalloc(&d_attachLambda, (size_t)g_N*sizeof(float3)));
    }
    XK(cudaMemcpy(d_attachStiffness, stiffness, (size_t)g_N*sizeof(float), cudaMemcpyHostToDevice));
    XK(cudaMemcpy(d_attachRest, d_pos, (size_t)g_N*sizeof(float3), cudaMemcpyDeviceToDevice));
    XK(cudaMemset(d_attachLambda, 0, (size_t)g_N*sizeof(float3)));
    return 0;
}

extern "C" int xpbd_get_positions(float* out3)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    if (!out3) return XPBD_ERR_BAD_ARG;
    return -(int)cudaMemcpy(out3, d_pos, (size_t)g_N * sizeof(float3), cudaMemcpyDeviceToHost);
}

extern "C" int xpbd_get_velocities(float* out3)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    if (!out3) return XPBD_ERR_BAD_ARG;
    return -(int)cudaMemcpy(out3, d_vel, (size_t)g_N * sizeof(float3), cudaMemcpyDeviceToHost);
}

extern "C" void xpbd_shutdown()
{
    freeRuntimeGripTerms();
    freeRuntimeModeGates();
    freeCoarseBoundCache();freeManifoldValidation();
    freeQueryGroups();
    cudaFree(d_fragmentHits);d_fragmentHits=nullptr;cudaFree(d_fragmentClosedCanonical);d_fragmentClosedCanonical=nullptr;
    cudaFree(d_fragmentVertices);d_fragmentVertices=nullptr;g_fragmentCapacity=0;
    cudaFree(d_fragmentSupported);d_fragmentSupported=nullptr;
    cudaFree(d_fragmentUnpinnedVolume);d_fragmentUnpinnedVolume=nullptr;
    cudaFree(d_fragmentDegree);d_fragmentDegree=nullptr;
    cudaFree(d_fragmentLambda);d_fragmentLambda=nullptr;
    cudaFree(d_fragmentBefore);d_fragmentBefore=nullptr;
    cudaFree(d_fragmentStart);d_fragmentStart=nullptr;
    cudaFree(d_fragmentQueries);d_fragmentQueries=nullptr;cudaFree(d_fragmentQueryCount);d_fragmentQueryCount=nullptr;
    cudaFree(d_toolFeedback);d_toolFeedback=nullptr;
    freeBlade();
    cudaFree(d_toolClearance);d_toolClearance=nullptr;
    cudaFree(d_cgStencil);d_cgStencil=nullptr;cudaFree(d_cgZ);d_cgZ=nullptr;
#ifdef ENDO_KIDNEY_STENCIL_CACHE
    freeMaterialTetCache();
#endif
    cudaFree(d_cgMatrix);d_cgMatrix=nullptr;cudaFree(d_cgScale);d_cgScale=nullptr;freeMultigrid();
    cudaFree(d_cgR);d_cgR=nullptr;cudaFree(d_cgP);d_cgP=nullptr;cudaFree(d_cgQ);d_cgQ=nullptr;cudaFree(d_cgD);d_cgD=nullptr;cudaFree(d_cgA);d_cgA=nullptr;
    cudaFree(d_cgH);d_cgH=nullptr;cudaFree(d_cgS);d_cgS=nullptr;cudaFree(d_cgSearch);d_cgSearch=nullptr;
    if(g_unifiedGraph)cudaGraphExecDestroy(g_unifiedGraph);g_unifiedGraph=nullptr;if(g_unifiedRecoveryGraph)cudaGraphExecDestroy(g_unifiedRecoveryGraph);g_unifiedRecoveryGraph=nullptr;if(g_unifiedLoadedGraph)cudaGraphExecDestroy(g_unifiedLoadedGraph);g_unifiedLoadedGraph=nullptr;cudaFree(d_substepQuery);d_substepQuery=nullptr;if(g_unifiedContactGraph)cudaGraphExecDestroy(g_unifiedContactGraph);g_unifiedContactGraph=nullptr;cudaFree(d_unified);d_unified=nullptr;cudaFree(d_unifiedTarget);d_unifiedTarget=nullptr;cudaFree(d_unifiedPreviousInverse);d_unifiedPreviousInverse=nullptr;cudaFree(d_unifiedRestMetric);d_unifiedRestMetric=nullptr;g_unifiedTetCapacity=0;cudaFree(d_unifiedComponents);d_unifiedComponents=nullptr;cudaFree(d_graspComponents);d_graspComponents=nullptr;cudaFree(d_floorRecovery);d_floorRecovery=nullptr;cudaFree(d_safeCluster);d_safeCluster=nullptr;cudaFree(d_safeAggregate);d_safeAggregate=nullptr;cudaFree(d_safeJoined);d_safeJoined=nullptr;cudaFree(d_globalCoarse);d_globalCoarse=nullptr;cudaFree(d_coarseWeight);d_coarseWeight=nullptr;cudaFree(d_unifiedInteraction);d_unifiedInteraction=nullptr;cudaFree(d_sweepSites);d_sweepSites=nullptr;cudaFree(d_sweepCount);d_sweepCount=nullptr;cudaFree(d_coarseBlock);d_coarseBlock=nullptr;cudaFree(d_manifold);d_manifold=nullptr;cudaFree(d_manifoldIds);d_manifoldIds=nullptr;cudaFree(d_manifoldCount);d_manifoldCount=nullptr;cudaFree(d_manifoldOwners);d_manifoldOwners=nullptr;cudaFree(d_manifoldQuadrature);d_manifoldQuadrature=nullptr;g_manifoldCapacity=0;
    cudaFree(d_feasibleStart);d_feasibleStart=nullptr;cudaFree(d_feasibleTarget);d_feasibleTarget=nullptr;cudaFree(d_incidentOffsets);d_incidentOffsets=nullptr;cudaFree(d_incidentTets);d_incidentTets=nullptr;cudaFree(d_needFeasible);d_needFeasible=nullptr;
    cudaFree(d_captureInside);d_captureInside=nullptr;g_volumePasses=0;g_minTetVolume=0;g_captureWedge=false;
    destroyGripGraph();cudaFree(d_contactResidual);d_contactResidual=nullptr;cudaFree(d_coupledTool);d_coupledTool=nullptr;cudaFree(d_gripError);d_gripError=nullptr;
    destroyTetGraph();destroySurfaceGraph();destroyRepairGraph();if(g_captureStream)cudaStreamDestroy(g_captureStream);g_captureStream=nullptr;
    if(g_fragmentRefreshStream)cudaStreamDestroy(g_fragmentRefreshStream);g_fragmentRefreshStream=nullptr;
    for(auto e:g_profileEvents)cudaEventDestroy(e);g_profileEvents.clear();g_profile=false;
    cudaFree(d_attachStiffness); d_attachStiffness=nullptr;
    cudaFree(d_attachRest); d_attachRest=nullptr;
    cudaFree(d_attachLambda); d_attachLambda=nullptr;
    if (!g_bound) {                       // BOUND mode: d_pos/d_vel ALIAS the shared live buffers
        cudaFree(d_pos);                  // (recon_corner_pos()/physics_vel()) — owned by dc_recon/
        cudaFree(d_vel);                  // physics.cu, NEVER freed here.
    }
    d_pos = nullptr; d_vel = nullptr;
    cudaFree(d_prevPos); d_prevPos = nullptr;
    cudaFree(d_invMass); d_invMass = nullptr;
    cudaFree(d_edgeOwnerSlot); d_edgeOwnerSlot = nullptr;
    g_liveRestNbr=nullptr; g_liveNbrIdx = nullptr; g_liveBendPairs = nullptr; g_liveActive = nullptr; g_livePinned = nullptr;
    cudaFree(d_topologyRecords);
    d_topologyRecords = d_componentRevision = d_livenessRevision = d_edgeSupportRevision = d_multigridRevision = nullptr;
    g_liveMass = nullptr; g_liveExtForce = nullptr; g_liveHapticForce = nullptr;
    g_dimsX = g_dimsY = g_dimsZ = 0; g_bound = false;
    cudaFree(d_edgeI);   d_edgeI = nullptr;
    cudaFree(d_edgeJ);   d_edgeJ = nullptr;
    cudaFree(d_restLen); d_restLen = nullptr;
    cudaFree(d_lambda);  d_lambda = nullptr;
    cudaFree(d_edgeActive); d_edgeActive = nullptr;
    cudaFree(d_edgeSrcIdx); d_edgeSrcIdx = nullptr;
    cudaFree(d_bendI); d_bendI = nullptr; cudaFree(d_bendJ); d_bendJ = nullptr; cudaFree(d_bendK); d_bendK = nullptr;
    cudaFree(d_bendTheta0); d_bendTheta0 = nullptr; cudaFree(d_bendActive); d_bendActive = nullptr;
    cudaFree(d_bendLambda); d_bendLambda = nullptr;
    cudaFree(d_bendSrcIdx); d_bendSrcIdx = nullptr;
    // STEP 7 tet buffers.
    cudaFree(d_tetIds); d_tetIds = nullptr;
    cudaFree(d_tetCellBase); d_tetCellBase = nullptr; cudaFree(d_kuhnTets); d_kuhnTets = nullptr; g_tetCells = 0;
    cudaFree(d_activeNodes); d_activeNodes = nullptr; g_activeNodes = 0;
    cudaFree(d_tetBr0); d_tetBr0 = nullptr; cudaFree(d_tetBr1); d_tetBr1 = nullptr; cudaFree(d_tetBr2); d_tetBr2 = nullptr;
    cudaFree(d_tetMuV); d_tetMuV = nullptr; cudaFree(d_tetLambdaV); d_tetLambdaV = nullptr;
    cudaFree(d_tetAlphaDev); d_tetAlphaDev = nullptr; cudaFree(d_tetAlphaHyd); d_tetAlphaHyd = nullptr;
    cudaFree(d_tetLambdaDevL); d_tetLambdaDevL = nullptr; cudaFree(d_tetLambdaHydL); d_tetLambdaHydL = nullptr;
    cudaFree(d_tetActive); d_tetActive = nullptr;
    cudaFree(d_tetColorOrder); d_tetColorOrder = nullptr; g_tetColorOrder.clear();
    cudaFree(d_colorTetIds); d_colorTetIds = nullptr; cudaFree(d_colorTetDet); d_colorTetDet = nullptr;
    destroyFeasibleGraphs();
    destroyFragmentPassGraphs();
    cudaFree(d_fragmentKeys);cudaFree(d_fragmentSortedKeys);cudaFree(d_fragmentOrder);cudaFree(d_fragmentSortedOrder);cudaFree(d_fragmentSortTemp);
    d_fragmentKeys=d_fragmentSortedKeys=nullptr;d_fragmentOrder=d_fragmentSortedOrder=nullptr;d_fragmentSortTemp=nullptr;g_fragmentSortCapacity=0;g_fragmentSortBytes=0;
    cudaFree(d_fragmentGroups);d_fragmentGroups=nullptr;cudaFree(d_fragmentLineSums);d_fragmentLineSums=nullptr;
    cudaFree(d_fragmentConstraints);d_fragmentConstraints=nullptr;cudaFree(d_fragmentQuerySlots);d_fragmentQuerySlots=nullptr;g_fragmentConstraintCapacity=0;
    cudaFree(d_fragmentUniqueQueries);d_fragmentUniqueQueries=nullptr;cudaFree(d_fragmentUniqueCount);d_fragmentUniqueCount=nullptr;
    cudaFree(d_fragmentDuplicateHeads);d_fragmentDuplicateHeads=nullptr;cudaFree(d_fragmentDuplicateAmbiguous);d_fragmentDuplicateAmbiguous=nullptr;cudaFree(d_fragmentDuplicateNext);d_fragmentDuplicateNext=nullptr;
    cudaFree(d_fragmentConstraintCount);d_fragmentConstraintCount=nullptr;cudaFree(d_fragmentConstraintDegree);d_fragmentConstraintDegree=nullptr;
    cudaFree(d_fragmentContacted);d_fragmentContacted=nullptr;
    cudaFree(d_fragmentTree);d_fragmentTree=nullptr;g_fragmentTreeCapacity=g_fragmentTreeBase=0;
    cudaFree(d_fragmentBounds);d_fragmentBounds=nullptr;cudaFree(d_fragmentRoots);d_fragmentRoots=nullptr;cudaFree(d_fragmentRootCount);d_fragmentRootCount=nullptr;
    if(g_pressureGraph){cudaGraphExecDestroy(g_pressureGraph);g_pressureGraph=nullptr;}
    cudaFree(d_fiberLambda);d_fiberLambda=nullptr;g_tissueHardening=80.f;g_bulkDampingSeconds=.003125f;
    cudaFree(d_tetEdgeOwnerSlot); d_tetEdgeOwnerSlot = nullptr;
    cudaFree(d_edgeTetSupport); d_edgeTetSupport = nullptr;
    cudaFree(d_localTets);d_localTets=nullptr;g_localPasses=0;g_localOff.clear();g_localCount.clear();
    g_T = 0; g_tColorOff.clear(); g_tColorCnt.clear(); g_tNumColors = 0; g_haveTets = false; g_doStrainF = false;
    // STEP 8 grasp/contact buffers.
    cudaFree(d_surfaceVertices);d_surfaceVertices=nullptr;cudaFree(d_surfaceContacts);d_surfaceContacts=nullptr;
    cudaFree(d_contactGroups);cudaFree(d_contactGroupRejected);d_contactGroups=nullptr;d_contactGroupRejected=nullptr;
    cudaFree(d_surfaceCurrent);d_surfaceCurrent=nullptr;cudaFree(d_surfacePrevious);d_surfacePrevious=nullptr;
    cudaFree(d_jawLoad);d_jawLoad=nullptr;
    cudaFree(d_surfaceActive);d_surfaceActive=nullptr;cudaFree(d_surfaceActiveCount);d_surfaceActiveCount=nullptr;
    cudaFree(d_surfaceDegree);d_surfaceDegree=nullptr;cudaFree(d_surfaceDelta);d_surfaceDelta=nullptr;
    cudaFree(d_contactReaction);d_contactReaction=nullptr;g_surfaceCount=0;g_surfaceCapacity=0;
    cudaFree(d_contactCache);d_contactCache=nullptr;cudaFree(d_contactCacheSlots);d_contactCacheSlots=nullptr;
    g_surfaceRest=nullptr;g_surfaceMaterialFP=nullptr;g_surfaceBoundaryMasks=nullptr;
    g_surfaceTags=nullptr;g_surfaceCorners=nullptr;g_surfaceConn=nullptr;g_surfaceMasks=nullptr;g_surfaceExpanded=nullptr;
    cudaFree(d_contacts);d_contacts=nullptr;
    cudaFree(d_contactNormalLambda);d_contactNormalLambda=nullptr;
    cudaFree(d_contactTangentLambda);d_contactTangentLambda=nullptr;
    g_frictionContact=false;
    g_anchorAssist=false;
    cudaFree(d_grasped); d_grasped = nullptr; cudaFree(d_graspCore); d_graspCore = nullptr;
    cudaFree(d_coreOffU); d_coreOffU = nullptr; cudaFree(d_coreOffN); d_coreOffN = nullptr; cudaFree(d_coreOffW); d_coreOffW = nullptr;
    destroyGripGraph();cudaFree(d_gripStep);d_gripStep=nullptr;
    cudaFree(d_gripPoints);d_gripPoints=nullptr;cudaFree(d_gripCounter);d_gripCounter=nullptr;
    cudaFree(d_gripLive);d_gripLive=nullptr;cudaFree(d_gripDegree);d_gripDegree=nullptr;
    cudaFree(d_gripMoments);d_gripMoments=nullptr;g_gripCount=g_gripCapacity=0;g_embeddedGrip=false;
    cudaFree(d_gripPatch);d_gripPatch=nullptr;cudaFree(d_gripPatchWeights);d_gripPatchWeights=nullptr;
    cudaFree(d_coreComplianceScale);d_coreComplianceScale=nullptr;
    cudaFree(d_coreLambda); d_coreLambda = nullptr;
    trainingShutdown();
    cudaFree(d_coreForceEMA); d_coreForceEMA = nullptr;
    cudaFree(d_candScratch); d_candScratch = nullptr;
    cudaFree(d_uScratch); d_uScratch = nullptr; cudaFree(d_vScratch); d_vScratch = nullptr; cudaFree(d_nScratch); d_nScratch = nullptr;
    cudaFree(d_slabLambda); d_slabLambda = nullptr; cudaFree(d_slabLambdaPrev); d_slabLambdaPrev = nullptr;
    cudaFree(d_jawContactLambda); d_jawContactLambda = nullptr; cudaFree(d_shaftContactLambda); d_shaftContactLambda = nullptr;
    cudaFree(d_coulombAnchorU); d_coulombAnchorU = nullptr; cudaFree(d_coulombAnchorW); d_coulombAnchorW = nullptr;
    g_grasping = false; g_tool = XpbdToolState{}; g_toolArmed = false;g_jawStiffness=0;g_openJawGap=0;
    cudaFree(d_contactMobility);d_contactMobility=nullptr;
    g_colorOff.clear(); g_colorCnt.clear(); g_numColors = 0;
    g_bColorOff.clear(); g_bColorCnt.clear(); g_bNumColors = 0;
    g_edgeSlotOfInput.clear(); g_bendSlotOfInput.clear();
    g_N = g_E = g_B = 0; g_ready = false; g_frameReady = false;g_hFrame=0;
}

extern "C" int xpbd_read_pressure_footprint(int count,int* flags){
    if(!g_bound || count!=g_N || count<=0 || !flags)return XPBD_ERR_BAD_ARG;
    if(!d_unified || !g_toolArmed){std::memset(flags,0,count*sizeof(int));return 0;}
    int* device=nullptr;cudaError_t e=cudaMalloc(&device,count*sizeof(int));
    if(e!=cudaSuccess)return -(int)e;
    e=cudaMemset(device,0,count*sizeof(int));
    if(e==cudaSuccess){k_ReadPressureFootprint<<<128,64>>>(d_unified,device);e=cudaGetLastError();}
    if(e==cudaSuccess)e=cudaMemcpy(flags,device,count*sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(device);return -(int)e;
}

extern "C" int xpbd_contact_diagnostic(int triangles,float* values){
    float maximum=0;int rc=xpbd_surface_depth(triangles,&maximum);if(rc)return rc;
    float* d=nullptr;cudaError_t e=cudaMalloc(&d,9*sizeof(float));if(e!=cudaSuccess)return -(int)e;
    cudaMemset(d,0,9*sizeof(float));
    for(int kind=0;kind<3;kind++)k_ContactDiagnostic<<<Gr(triangles),64>>>(triangles,kind,contactShape(g_tool,kind,g_jawThickness,g_gp.jawGapClosed),g_surfaceExpanded,d_pos,d_surfaceContacts,d_coupledTool,maximum,d,g_surfaceTags,d_surfaceActive,d_surfaceActiveCount);
    e=cudaMemcpy(values,d,9*sizeof(float),cudaMemcpyDeviceToHost);cudaFree(d);
    if(e==cudaSuccess && d_contactResidual)e=cudaMemcpy(values+9,d_contactResidual,3*sizeof(float),cudaMemcpyDeviceToHost);return -(int)e;
}

// Preserve LCS_GetSolvedTool's documented centroid error. The region-shape
// diagnostic is separate so a stretched edge cannot masquerade as tool detachment.
extern "C" int xpbd_grasp_tangent_error(float* error){
    if(!error || !d_gripError)return XPBD_ERR_BAD_ARG;
    return -(int)cudaMemcpy(error,d_gripError+2,sizeof(float),cudaMemcpyDeviceToHost);
}

#ifdef ENDO_FRAGMENT_PROBE
#include "../contact/fragment_probe.cuh"
#endif

#ifndef ENDO_FRAGMENT_PROBE
#include "../contact/fragment_snapshots.cuh"
#endif

#ifndef ENDO_FRAGMENT_PROBE
#include "../contact/fragment_paired_probe.cuh"
#endif

// Opt-in, read-only query-set check. Never called by ordinary simulation.
extern "C" LCS_COMBINED_EXPORT int LCS_ValidateFragmentVolumeGate(int* result,int capacity){
    if(!result || capacity<8 || !d_fragmentUnpinnedVolume || !d_fragmentSupported || !d_graspComponents)return -1;
    const int entries=3*g_surfaceCount+g_N;int *reference=nullptr,*gated=nullptr,*counts=nullptr;
    cudaError_t e=cudaMalloc(&reference,std::max(entries,1)*sizeof(int));
    if(e==cudaSuccess)e=cudaMalloc(&gated,std::max(entries,1)*sizeof(int));
    if(e==cudaSuccess)e=cudaMalloc(&counts,2*sizeof(int));
    auto finish=[&](){cudaFree(reference);cudaFree(gated);cudaFree(counts);return e==cudaSuccess?0:-(int)e;};
    if(e!=cudaSuccess)return finish();
    e=cudaMemset(counts,0,2*sizeof(int));
    if(e!=cudaSuccess)return finish();
    k_FragmentQueries<<<Gr(entries),64>>>(g_surfaceCount,g_N,d_unifiedComponents,g_liveActive,d_surfaceVertices,
        d_fragmentSupported,d_graspComponents,d_invMass,reference,counts,2,nullptr);
    k_FragmentQueries<<<Gr(entries),64>>>(g_surfaceCount,g_N,d_unifiedComponents,g_liveActive,d_surfaceVertices,
        d_fragmentSupported,d_graspComponents,d_invMass,gated,counts+1,2,d_fragmentUnpinnedVolume);
    e=cudaGetLastError();int length[2]={},available=0;
    if(e==cudaSuccess)e=cudaMemcpy(length,counts,sizeof(length),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(&available,d_fragmentUnpinnedVolume,sizeof(int),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return finish();
    if(length[0]<0 || length[1]<0 || length[0]>entries || length[1]>entries){e=cudaErrorInvalidValue;return finish();}
    std::vector<int> a(length[0]),b(length[1]),support(g_N);
    if(length[0])e=cudaMemcpy(a.data(),reference,length[0]*sizeof(int),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess && length[1])e=cudaMemcpy(b.data(),gated,length[1]*sizeof(int),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(support.data(),d_fragmentSupported,g_N*sizeof(int),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return finish();
    std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
    int differences=std::abs(length[0]-length[1]),expected=0;
    for(int i=0;i<std::min(length[0],length[1]);i++)differences+=a[i]!=b[i];
    for(int flags:support)expected|=(flags&3)==2;
    result[0]=available;result[1]=length[0];result[2]=length[1];result[3]=differences;
    result[4]=expected;result[5]=entries;result[6]=g_surfaceCount;result[7]=g_N;
    return finish();
}

#include "../training/xpbd_training.cuh"

#include "../training/xpbd_state.cuh"

#include <cmath>
#include <vector>
#include "labels_readback.cuh"

extern "C" int xpbd_batch_query_mode(int mode){
    if(mode<0 || mode>3 || g_ready)return -10001;
    g_queryBatchMode=mode;return 0;
}
extern "C" int xpbd_batch_validation(int enabled){
    cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess)return -(int)e;
    lcs_state::invalidate_xpbd_graphs();
    if(enabled){if(!g_ready || !g_queryBatchMode)return -10002;
        e=allocateQueryValidation(g_N);if(e!=cudaSuccess)return -(int)e;
        e=allocateManifoldValidation(g_N);if(e!=cudaSuccess)return -(int)e;
        if(g_coarseBoundCacheEnabled){e=ensureCoarseBoundStats();if(e!=cudaSuccess)return -(int)e;}
        e=cudaMemset(d_queryValidationStats,0,32*sizeof(int));if(e!=cudaSuccess)return -(int)e;
        e=cudaMemset(d_queryValidationStats+11,0xff,sizeof(int));if(e!=cudaSuccess)return -(int)e;}
    g_queryValidate=enabled!=0;return 0;
}
extern "C" int xpbd_batch_validation_read(int* values){
    if(!values || !d_queryValidationStats)return -10003;
    cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess)return -(int)e;
    return -(int)cudaMemcpy(values,d_queryValidationStats,16*sizeof(int),cudaMemcpyDeviceToHost);
}

extern "C" int xpbd_native_schedule(int assembly,int manifold){
    if(g_ready || (assembly!=64 && assembly!=128 && assembly!=256 && assembly!=512) ||
       (manifold!=64 && manifold!=128 && manifold!=256 && manifold!=512 && manifold!=1024))return -10004;
    ASSEMBLY_BLOCKS=assembly;MANIFOLD_BLOCKS=manifold;return 0;
}
extern "C" int xpbd_native_workload(int* values){
    if(!values || !g_ready)return -10005;
    values[0]=g_N;values[1]=g_T;values[2]=g_surfaceCount;values[3]=ASSEMBLY_BLOCKS;values[4]=MANIFOLD_BLOCKS;
    values[5]=values[6]=values[7]=values[8]=0;
    UnifiedState state{};cudaError_t e=cudaSuccess;
    if(d_unified){e=cudaMemcpy(&state,d_unified,sizeof(state),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
        values[6]=state.grips;if(state.contactCount){e=cudaMemcpy(values+5,state.contactCount,sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;}}
    if(d_bladeState){e=cudaMemcpy(&state,d_bladeState,sizeof(state),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
        values[8]=state.grips;if(state.contactCount){e=cudaMemcpy(values+7,state.contactCount,sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;}}
    return 0;
}

extern "C" int xpbd_full_contact_validation_read(int* values){
    if(!values || !d_queryValidationStats)return -10003;
    cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess)return -(int)e;
    return -(int)cudaMemcpy(values,d_queryValidationStats,32*sizeof(int),cudaMemcpyDeviceToHost);
}

extern "C" int xpbd_query_threads(int threads){
    if(g_ready || (threads!=32 && threads!=64 && threads!=128))return -10006;
    g_queryThreads=threads;return 0;
}

extern "C" int xpbd_set_coarse_bound_cache(int enabled){if(g_ready || (enabled!=0 && enabled!=1))return -10011;g_coarseBoundCacheEnabled=enabled;return 0;}
#include "../training/kidney_contact_probe.cuh"
#ifdef ENDO_KIDNEY_STENCIL_CACHE
#include "../training/material_stencil_probe.cuh"
#endif
#ifdef ENDO_KIDNEY_SPECIALIZE_PRESSURE
#include "../training/pressure_line_probe.cuh"
#endif
#ifdef ENDO_KIDNEY_STAGE_TRACE
#include "../training/kidney_stage_trace.cuh"
#endif
extern "C" int xpbd_coarse_bound_stats(unsigned long long* values){if(!values || !d_coarseBoundStats)return -10012;cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess)return -(int)e;return -(int)cudaMemcpy(values,d_coarseBoundStats,8*sizeof(unsigned long long),cudaMemcpyDeviceToHost);}

// Diagnostic-only positive/negative ordering probe, no live physical buffers.
extern "C" int xpbd_query_order_probe(int stride,unsigned packed,int kinds,const int* reference,const int* candidate,int count,int* out){
    if(!reference || !candidate || !out || stride<=0 || stride>6 || count<=0 || count>64)return -10013;
    int entries=2*stride;for(int i=0;i<count;i++)if(reference[i]<0 || reference[i]>=entries || candidate[i]<0 || candidate[i]>=entries)return -10014;
    struct W{int *a=nullptr,*b=nullptr,*n=nullptr,*freq=nullptr,*bad=nullptr,*stats=nullptr;~W(){cudaDeviceSynchronize();for(void* p:{a,b,n,freq,bad,stats})queryPrivateFree(p);}}w;
    cudaError_t e=cudaSuccess;
    std::pair<int**,int> allocations[]={{&w.a,count},{&w.b,count},{&w.n,1},{&w.freq,2*entries},{&w.bad,1},{&w.stats,32}};
    for(const auto& item:allocations){e=queryPrivateAlloc(item.first,(size_t)item.second*sizeof(int));if(e!=cudaSuccess)return -(int)e;}
    e=cudaMemcpy(w.a,reference,count*sizeof(int),cudaMemcpyHostToDevice);if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(w.b,candidate,count*sizeof(int),cudaMemcpyHostToDevice);if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(w.n,&count,sizeof(int),cudaMemcpyHostToDevice);if(e!=cudaSuccess)return -(int)e;
    cudaMemset(w.freq,0,2*entries*sizeof(int));cudaMemset(w.bad,0,sizeof(int));cudaMemset(w.stats,0,32*sizeof(int));
    k_QueryFrequenciesStrided<<<1,128>>>(entries,stride,packed,kinds,w.a,w.n,w.b,w.n,w.freq,w.bad,w.stats);
    k_CompareQueryFrequencies<<<1,128>>>(entries,w.freq,w.bad,w.stats);
    e=cudaMemcpy(out,w.stats,32*sizeof(int),cudaMemcpyDeviceToHost);return -(int)e;
}

#ifdef ENDO_KIDNEY_CLOSED_STATS
extern "C" LCS_COMBINED_EXPORT int LCS_ClosedContactCounters(unsigned long long* out,int reset){
    if(!out)return -1;cudaError_t e=cudaMemcpyFromSymbol(out,g_fragmentRuntimeStats,6*sizeof(unsigned long long));
    if(e==cudaSuccess && reset){unsigned long long zero[6]={};e=cudaMemcpyToSymbol(g_fragmentRuntimeStats,zero,sizeof(zero));}
    return (int)e;
}
#endif
