// cut.h — host-callable API for Stage 3 cutting (defined in cut.cu, called from plugin_api.cpp).
//
// Realizes paper Fig 3.1 Thread-2 "Cut Voxel" + "Look up in the Connectivity Table" + the cut part of
// "Calculate Feature Points" / "Construct Triangles", and the SeverLinks feedback into Thread-1
// ("changes the connectivity between soft tissue particles"). Ported VERBATIM from Cutting.compute +
// CuttingCommon.hlsl + CutDetector.cs + DualContouring.cs (cut-FP dispatch chain).
//
// Buffer sharing: cut reads dc_recon's cornerPos/isect/isectWorld/voxelIsect* + physics's particleRot,
// and WRITES physics's nbrIdx/bendPairs (SeverLinks). It pushes its own cut buffers to dc_recon
// (recon_set_cut_buffers) so the cut-aware Stitch/Normals/Expand resolve the per-component cut FPs.

#pragma once

// One-time cut description (field order MUST match the C# [StructLayout] CutInitDesc in LiverCudaManager).
struct CutInitDesc
{
    int   voxelCount, cornerCount, gridEdgeCount, isectCount, triCapacity;
    int   dimsX, dimsY, dimsZ;
    float voxelL;
    float originX, originY, originZ;
    int   cutPointCapacity;   // = 4 * gridEdgeCount (C# LiverCudaManager; v3.1 headroom — the raw
                              //   emit counter runs past capacity by design, see cut_get_debug)
    float cutFPInterp;        // retained ABI field, ignored (component motion is unfiltered)
    float tearStretchRatio;   // v5.1 D10 tear law: sever any intact edge stretched beyond this
                              //   ratio x rest length (must exceed max legit gravity strain;
                              //   <1 or NaN disables). Kills the unsevered-rim-bridge strands.
};

// Per-(sub)step swept-plane tool geometry (set by the C# CuttingTool). T1=(Sprev,Eprev,E), T2=(Sprev,E,S).
struct CutToolDesc
{
    float t1v0[3], t1v1[3], t1v2[3];
    float t2v0[3], t2v1[3], t2v2[3];
    float ncut[3];
    float d;
    int   valid;              // 0 = degenerate sweep (skip DetectCut this substep)
    float aabbMin[3], aabbMax[3];
};

// Allocate cut buffers, upload Conn4096 + gridEdges + voxelOccupied, zero the cumulative buffers, cache the
// shared dc_recon/physics pointers, and hand the cut buffers to dc_recon. Returns 0 on success.
//   conn4096      : Conn4096Gpu[4096]            (passed as void*)
//   gridEdges     : GridEdgeGpu2[gridEdgeCount]  (passed as void*)
//   voxelOccupied : int[voxelCount]  (1 = touches tissue)
// cornerInside : int[cornerCount] 1 = phi<0 (Stage-7 air-edge gate: blade crossings in the AIR
//                part of occupied surface voxels are not tissue cuts - audit wf_60781cea A1).
int  cut_init(const CutInitDesc* desc, const void* conn4096, const void* gridEdges, const int* voxelOccupied,
              const int* cornerInside);

// Set the current swept-plane geometry (call once per rod sub-step before cut_detect).
void cut_set_tool(const CutToolDesc* tool);

// DetectCut (Stage 6, paper SS2.1.3 Eq6-8 verbatim): Moller-Trumbore of the swept cutting plane
// (2 WORLD triangles, prev+cur tool frames) against the CURRENT DEFORMED grid edges + SeverLinks.
// Cut points are recorded on the undeformed edge (alongRest, SS2.1.2) and reconstructed via the
// current particle frame. Call once per C# rod substep (rod moved, tissue frozen). Uses the tool
// set by cut_set_tool. Returns 0 on success.
int  cut_detect();

// CCD tick (Stage 6) — called by physics_step after every accepted RK45 substep (tissue moved, rod
// fixed): continuous collision of each MOVING deformed edge against the STATIC blade segment
// (coplanarity is quadratic in tick-time; exact space-time crossing — no tunneling, no inverse map).
// Together with cut_detect this tiles the full space-time sweep (task_plan.md Stage 6).
// No-op until cut_init + the first cut_set_tool. Returns 0 on success.
int  cut_ribbon_tick();

// The cut feature-point chain: ClearCutFP -> LookupConnectivity -> AccumulateCutFP -> ComputeComponentFP
// Run inside the rebuild AFTER DC_FeaturePoints, BEFORE the TriCounter reset.
int  cut_fp_chain();

// BuildCutTriangles — append the cut-wall faces into the SHARED _Tri buffer (after DC_Stitch).
int  cut_build_triangles();

// DC_NormalizeCutFP — normalize the cut-FP normal accumulator (after DC_NormalsScatter).
int  cut_normalize_fp();

// Free all cut device buffers.
void cut_shutdown();

// [DEBUG-CUT] (v4.1 P3) — copy the 16 per-frame debug counters + the 2 RAW atomic counters to host
// and report the runtime cutFPInterp. out18 layout: 0=emaPairs 1=jump>L 2=jump>3L 3=maxDisp(f-as-u)
// 4=slotNew 5=epochInval 6=dangleRefs 7=giantWall 8=giantMixed 9=giantSkin 10=maxEdgeLen(f-as-u)
// 11=rim<3 12=pinch1 13=all6severed 14=frozenThisFrame 15=tearMarks(CUMULATIVE — tick-phase writer,
// so it is excluded from the per-chain zeroing; review wf_c416ae77 R2-m3) 16=rawCutPts 17=rawTriIdx.
// Call AFTER LCS_Finalize each frame; slots 0-14 cover exactly that frame's chain.
int cut_get_debug(unsigned int* out18, float* alphaOut);

// True once cut_init succeeded (dc_recon's stitch uses this to pick the cut-aware vs external-only path).
bool cut_is_ready();
float* cut_edge_fraction(); // [6*cornerCount], first blade fraction; -1 before cutting

// design §6/§9-(6) tear exemption: device int[cornerCount], 1 = grasped-neighbourhood corner whose
// incident edges are EXEMPT from the tau*L tear law. ZERO-FILLED at cut_init (INERT); the step-8 grasp
// module writes it (BFS-dilate d_grasped over nbrIdx). nullptr before cut_init.
int* cut_grasp_ring();

// design §6/§9-(8): recompute the tear-exemption ring from the grasp module's CURRENT captured-corner
// flags (grasped[cornerCount], 1=captured -- both core and ring corners are the epicenter). Dilates by
// nExempt graph-hops over the LIVE nbrIdx (the "1-2-ring dilation"), then carves out any corner that has
// EVER lost a neighbour vs the topology snapshotted at cut_init (rim-bridge carve-out: a cut-rim corner
// keeps the tau*L law regardless of grasp proximity, so grasp-and-cut-the-fold still severs the debris
// strands). Call once per frame (cheap; not per-substep) -- recomputing unconditionally satisfies the
// design's "recompute after each sever event inside the ring" requirement for free. Returns 0 on success.
int cut_recompute_grasp_ring(const int* grasped, int nExempt);

// §9-(6) INERT tear-prep unit test: prove the graspRing early-out in k_TearOverstretch exempts a grasped
// edge (set the flag -> no tear) and tears when cleared. Self-contained (no cut_init needed). Returns
// the number of failed assertions (0 = pass).
int  cut_selftest_tear_exempt();

int cut_surface_revision(unsigned* revision);
