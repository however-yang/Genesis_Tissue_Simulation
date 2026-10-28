// plugin_api.cpp — exported C ABI for the Unity native plugin (Stage 1).
// Thin forwarding layer over the CUDA recon API (cuda/dc_recon.cu). Host C++ only; all kernel
// launches live in the .cu. Mirrors the design spec's P/Invoke surface (Init / GetSurface / Shutdown).
//
// Stage 0's hard-coded-triangle smoke test (plugin.cu) is superseded — the toolchain it validated
// (DLL build/load/P-Invoke/CUDA/render) is exercised by the real recon path below. The Stage 0
// source remains in git history (commit dcaf010).

#include "dc_recon.h"
#include "xpbd/tool_feedback.h"
#include "physics.h"
#include "cut.h"
#include "haptics.h"
#include "xpbd/xpbd_solver.h"   // STEP 8 (design rev-B §9-(8)): XpbdToolState/XpbdGraspParams ABI

#define LCS_API extern "C" __declspec(dllexport)
LCS_API int LCS_ToolFeedbackAbiVersion(){return sizeof(XpbdToolFeedback);}
LCS_API int LCS_GetToolFeedback(XpbdToolFeedback* result){return xpbd_get_tool_feedback(result);}
extern "C" void xpbd_profile_enable(int);
extern "C" int xpbd_profile_read(int,float*);
LCS_API void LCS_ProfileEnable(int enabled){xpbd_profile_enable(enabled);}
LCS_API int LCS_ProfileRead(int steps,float* ms){return xpbd_profile_read(steps,ms);}
extern "C" void xpbd_profile_sections(int);
extern "C" int xpbd_profile_sections_read(char*,int);
LCS_API void LCS_ProfileSections(int enabled){xpbd_profile_sections(enabled);}
LCS_API int LCS_ProfileSectionsRead(char* text,int capacity){return xpbd_profile_sections_read(text,capacity);}
int recon_render_buffer(int,void*,int*,float*,const float*,const float*);
LCS_API int LCS_GetRenderBuffer(int n,void* vertices,int* counts,float* bounds,const float* uvMin,const float* uvSize){return recon_render_buffer(n,vertices,counts,bounds,uvMin,uvSize);}
extern "C" int xpbd_get_jaw_load(float*);
LCS_API int LCS_GetJawLoad(float* load){return xpbd_get_jaw_load(load);}
extern "C" int xpbd_surface_depth(int,float*);
LCS_API int LCS_SurfaceDepth(float* depth){int n=0;int e=recon_get_counts(&n,nullptr);return e?e:xpbd_surface_depth(n/3,depth);}
extern "C" int xpbd_set_knife_geometry(int,const float*);
extern "C" int xpbd_knife_body_depth(int,float*);
LCS_API int LCS_SetKnifeGeometry(int enabled,const float* boxes){return xpbd_set_knife_geometry(enabled,boxes);}
LCS_API int LCS_KnifeBodyDepth(float* depth){int n=0;int e=recon_get_counts(&n,nullptr);return e?e:xpbd_knife_body_depth(n/3,depth);}
extern "C" int cut_read_edge_fractions(float*);
LCS_API int LCS_GetCutEdgeFractions(float* output){return cut_read_edge_fractions(output);}
extern "C" int xpbd_resolve_tool_translation(int,float*);
LCS_API int LCS_ResolveToolTranslation(float* delta){int n=0;int e=recon_get_counts(&n,nullptr);return e?e:xpbd_resolve_tool_translation(n/3,delta);}
extern "C" int xpbd_relax_reconstructed_contact();
extern "C" float xpbd_contact_repair_tolerance();
extern "C" int xpbd_sync_solved_tool();
extern "C" int xpbd_integrated_contact();
LCS_API int LCS_RefineContactSurface(int passes){
    // Reconstructed embedded geometry can reveal new contact footprints.
    // Refine the same implicit objective locally, without adding a time step.
    // Depth is measured against the grasper pose the step solved: a held bite
    // keeps the jaws open while they are commanded shut, and against the command
    // it would read as penetration every frame and rebuild the surface for it.
    int sync=xpbd_sync_solved_tool();if(sync)return sync;
    float previousDepth=1e30f;int stalled=0;
    for(int i=0;i<passes && i<8;i++){
        float depth=0;int e=LCS_SurfaceDepth(&depth);if(e)return e;if(depth<=xpbd_contact_repair_tolerance())return 0;
        // Volume feasibility can make further geometric repair impossible. Avoid
        // repeating identical GPU solves/reconstruction. Residual depth remains
        // visible to diagnostics; early exit does not certify nonpenetration.
        if(i>0){stalled=depth>=previousDepth*.995f?stalled+1:0;if(stalled>=2)return 0;}
        previousDepth=depth;
        e=recon_prepare_contact_surface();if(e)return e;
        e=xpbd_relax_reconstructed_contact();if(e)return e;
        e=physics_compute_rotations();if(e)return e;
        e=recon_rebuild();if(e)return e;
    }
    return 0;
}

// Allocate device buffers, upload the C#-built grid, run the static DC pipeline, sync.
// Returns 0 on success; negative CUDA error code otherwise.
LCS_API int LCS_Init(const ReconInitDesc* desc,
                     const float* cornerPos,
                     const int*   voxelCorner,
                     const int*   voxelIsectOffset,
                     const int*   voxelIsectCount,
                     const void*  isect,
                     const void*  surfaceEdges)
{
    return recon_init(desc, cornerPos, voxelCorner, voxelIsectOffset, voxelIsectCount, isect, surfaceEdges);
}

// vertexCount = voxelCount; indexCount = 3 * triangle count (clamped to capacity).
LCS_API int LCS_GetSurfaceCounts(int* vertexCount, int* indexCount)
{
    return recon_get_counts(vertexCount, indexCount);
}

// Copy the resolved surface (positions, normals, indices) back to host for a Unity Mesh upload.
LCS_API int LCS_GetSurface(float* outPos3, float* outNrm3, int* outIdx)
{
    return recon_get_surface(outPos3, outNrm3, outIdx);
}

// Liver-render design §4.4: per-index render aux channel (rest anchor xyz + wall flag) — UV1 for
// the rest-position triplanar liver shader. Same clamped-count contract as LCS_GetSurface.
LCS_API int LCS_GetSurfaceAux(float* outAux4)
{
    return recon_get_surface_aux(outAux4);
}

// ── Stage 2: mass-spring physics (XPBD; RK45 retired, see physics.cu/physics.h) ─────────────────
// Allocate + upload the physics buffers. Call AFTER LCS_Init (shares dc_recon's corner grid).
LCS_API int LCS_PhysicsAbiVersion() { return (int)sizeof(PhysicsInitDesc); }
LCS_API int LCS_SetGravity(float x, float y, float z) { return physics_set_gravity(x, y, z); }
LCS_API int LCS_SetAttachments(int count, const float* stiffness) { return xpbd_set_attachments(count, stiffness); }
LCS_API int LCS_InitPhysics(const PhysicsInitDesc* desc,
                            const float* mass,
                            const int*   pinned,
                            const int*   active,
                            const float* extForce,
                            const int*   nbrIdx,
                            const float* restNbr,
                            const void*  bendPairs)
{
    return physics_init(desc, mass, pinned, active, extForce, nbrIdx, restNbr, bendPairs);
}

// Advance one outer dt: XPBD substeps deform the shared corner grid (RK45 is retired -- see
// physics.cu's own header; each accepted substep also runs the Stage-6 CCD cut tick inside
// physics_step), then update the current particle frames used by the SS2.1.2 cut-point
// record/reconstruct. NO re-mesh here — rod cut sub-steps run between LCS_Step and LCS_Finalize
// (paper Fig 3.1: physics -> cut -> reconstruct).
// Read from the loaded native binary, so experiment reports cannot silently
// attribute an old in-memory DLL to newly edited source files.
LCS_API int LCS_ConsistencyRevision() { return 1; }
LCS_API int LCS_Step(float dt)
{
    int e = recon_prepare_contact_surface();
    if(e != 0)return e;
    e = physics_step(dt);
    if (e != 0) return e;
    e = xpbd_commit_grip_topology();
    if (e != 0) return e;
    return physics_compute_rotations(); // pre-cut frames for DetectCut / EmitCutPoint local coordinates
}

// ── Stage 3: cutting ───────────────────────────────────────────────────────────────────────────
// Allocate + upload the cut buffers (Conn4096 LUT, gridEdges, occupancy). Call AFTER LCS_Init + LCS_InitPhysics.
LCS_API int LCS_InitCut(const CutInitDesc* desc, const void* conn4096, const void* gridEdges, const int* voxelOccupied,
                        const int* cornerInside)
{
    return cut_init(desc, conn4096, gridEdges, voxelOccupied, cornerInside);
}

// Set the current swept-plane tool geometry (call once per rod sub-step before LCS_DetectCut).
LCS_API void LCS_SetTool(const CutToolDesc* tool)
{
    cut_set_tool(tool);
}

// DetectCut (Stage 6, paper SS2.1.3 Eq6-8 verbatim: swept world cutting-plane triangles vs the
// CURRENT DEFORMED grid edges) + SeverLinks. Call between LCS_Step and LCS_Finalize.
LCS_API int LCS_DetectCut()
{
    return cut_detect();
}

// Recompute particle frames after severing links, then run the cut-aware DC rebuild
// (cut-FP chain + cut walls + normals + expand).
LCS_API int LCS_Finalize()
{
    int e = physics_compute_rotations();
    if (e != 0) return e;
    e = recon_rebuild();
    if (e != 0) return e;
    return xpbd_commit_grip_topology();
}

// ── Diagnostics (Phase-1 evidence: split? gravity moving corners?) ───────────────────────────
// Copy the LIVE post-sever nbrIdx[6*cornerCount] to host (for a connected-components check).
LCS_API int LCS_GetNbrIdx(int* out) { return physics_readback_nbr_idx(out); }
// Copy the LIVE deformed cornerPos[3*cornerCount] floats to host (for a gravity-motion check).
LCS_API int LCS_GetCornerPos(float* out) { return recon_readback_corner_pos(out); }
// [DEBUG-CUT] (v4.1 P3): 16 per-frame debug counters + 2 raw atomic counters + runtime cutFPInterp.
// Call after LCS_Finalize each frame (frame already synced); layout documented in cuda/cut.h.
LCS_API int LCS_GetCutDebug(unsigned int* out18, float* alphaOut) { return cut_get_debug(out18, alphaOut); }

// ── Haptic rendering (paper SS2.3 SDF Finger-Proxy: collision + grasping + friction) ────────────
// ABI guard: C# checks this equals Marshal.SizeOf(HapticParams) BEFORE LCS_InitHaptics, so a stale DLL
// whose struct is a different size cannot silently corrupt outerMax into a billion-iteration GPU hang.
LCS_API int  LCS_HapticAbiVersion() { return (int)sizeof(HapticParams); }
LCS_API int  LCS_InitHaptics(int cornerCount, int triCapacity, const HapticParams* p) { return haptics_init(cornerCount, triCapacity, p); }
LCS_API void LCS_SetPhysicalTool(const float* s3, const float* e3, const float* quat4, const float* linVel3, const float* angVel3, float toolRadius) { haptics_set_physical_tool(s3, e3, quat4, linVel3, angVel3, toolRadius); }
LCS_API void LCS_SetToolMesh(const float* triWorld, int triCount) { haptics_set_tool_mesh(triWorld, triCount); }
LCS_API int  LCS_StepHaptics() { return haptics_step(); }
LCS_API void LCS_GetHapticForce(float* outForce3, float* outTorque3) { haptics_get_force(outForce3, outTorque3); }
LCS_API void LCS_GetProxyTool(float* outPos3, float* outQuat4) { haptics_get_proxy(outPos3, outQuat4); }
LCS_API int  LCS_GetHapticContact() { return haptics_get_contact(); }
LCS_API void LCS_SetGrasp(float grasp01) { haptics_set_grasp(grasp01); }
LCS_API void LCS_ShutdownHaptics() { haptics_shutdown(); }

// ── STEP 8: native two-jaw grasp + contact (design rev-B §4[Contact]/§5/§6/§9-(8)) ───────────────
// Call LCS_SetGraspTool ONCE per frame, BEFORE LCS_Step (constant across that frame's N substeps —
// the same once-per-frame staleness contract §9-(5) established for hapticForce). ABI guard mirrors
// LCS_HapticAbiVersion's pattern (a stale DLL/struct-size mismatch fails loud, not silently).
LCS_API int  LCS_GraspToolAbiVersion() { return (int)sizeof(XpbdToolState); }
LCS_API int  LCS_SetGraspTool(const XpbdToolState* t) { return physics_set_tool(t); }
LCS_API void LCS_SetGraspParams(const XpbdGraspParams* p) { physics_set_grasp_params(p); }
LCS_API int LCS_SetContactModel(int enabled, float muS, float muD, float thickness, float radius)
{ return xpbd_set_contact_model(enabled,muS,muD,thickness,radius); }
LCS_API int LCS_GetContactReaction(float* force, int* contacts)
{ return xpbd_get_contact_reaction(force,contacts); }
extern "C" int xpbd_set_local_refinement(float,float,float,float,int);
extern "C" int xpbd_get_material_quality(float*);
LCS_API int LCS_SetLocalRefinement(float x,float y,float z,float radius,int passes){return xpbd_set_local_refinement(x,y,z,radius,passes);}
LCS_API int LCS_GetMaterialQuality(float* quality){return xpbd_get_material_quality(quality);}
extern "C" int xpbd_get_material_stats(float*);
LCS_API int LCS_GetMaterialStats(float* stats){return xpbd_get_material_stats(stats);}
extern "C" int xpbd_set_tissue_guard(float,int,int,const int*,int);
extern "C" int xpbd_get_grasp_lock(float*);
LCS_API int LCS_SetTissueGuard(float minimum,int passes,int count,const int* inside,int capture){return xpbd_set_tissue_guard(minimum,passes,count,inside,capture);}
LCS_API int LCS_GetGraspLock(float* gap){return xpbd_get_grasp_lock(gap);}
LCS_API int LCS_SetTissueMaterial(float strength,float bulkDamping){return xpbd_set_tissue_material(strength,bulkDamping);}
LCS_API int LCS_GetVolumeError(float* output){return xpbd_get_volume_error(output);}
LCS_API int LCS_SetGraspPatch(float sigma,float compliance,float damping){return xpbd_set_grasp_patch(sigma,compliance,damping);}
extern "C" int xpbd_set_instrument_control(int);
LCS_API int LCS_SetInstrumentControl(int guided){return xpbd_set_instrument_control(guided);}
LCS_API int LCS_SetToolDynamics(float mass,float stiffness,float damping,float maxTravel){return xpbd_set_tool_dynamics(mass,stiffness,damping,maxTravel);}
LCS_API int LCS_GetSolvedTool(float* hinge,float* error,int* live){return xpbd_get_solved_tool(hinge,error,live);}
LCS_API int LCS_SetJawDynamics(float mass,float stiffness,float damping,float speed){return xpbd_set_jaw_dynamics(mass,stiffness,damping,speed);}
LCS_API int LCS_GetSolvedJaw(float* gap){return xpbd_get_solved_jaw(gap);}
extern "C" int xpbd_set_jaw_geometry(const float*);
LCS_API int LCS_SetJawGeometry(const float* bounds){return xpbd_set_jaw_geometry(bounds);}
LCS_API int LCS_ContactModelVersion(){return 28;} // Pressure reentry and feasible elastic release recovery; blade ABI unchanged.
LCS_API int LCS_GetGraspCoreFlags(int count,int* flags){return xpbd_read_grasp_core(count,flags);}
// [DEBUG-grasp1] diagnostic: currently-grasped / currently-core-pinned corner counts (cheap-ish device
// scan, fine for temporary debug polling — NOT called every substep, just once per C# debug tick).
LCS_API void LCS_GetGraspCounts(int* outGrasped, int* outCore)
{
    if (outGrasped) *outGrasped = xpbd_grasp_corner_count();
    if (outCore)     *outCore   = xpbd_grasp_core_count();
}

// Free all device buffers (haptics first, then cut, physics, dc_recon).
LCS_API void LCS_Shutdown()
{
    haptics_shutdown();
    cut_shutdown();
    physics_shutdown();
    recon_shutdown();
}

extern "C" int xpbd_contact_diagnostic(int,float*);
LCS_API int LCS_ContactDiagnostic(float* values){int n=0;int e=recon_get_counts(&n,nullptr);return e?e:xpbd_contact_diagnostic(n/3,values);}
LCS_API int LCS_GetPressureFootprint(int count,int* flags){return xpbd_read_pressure_footprint(count,flags);}

extern "C" int xpbd_grasp_tangent_error(float*);
LCS_API int LCS_GetGraspTangentError(float* error){return xpbd_grasp_tangent_error(error);}


// Pressure cutting is a separate capability; existing grasper ABI remains V18.
LCS_API int LCS_ScissorsAbiVersion(){return (int)sizeof(XpbdScissorsState);}
LCS_API int LCS_ScissorsResultAbiVersion(){return (int)sizeof(XpbdScissorsResult);}
LCS_API int LCS_SetScissors(const XpbdScissorsState* state){return xpbd_set_scissors(state);}
LCS_API int LCS_GetScissors(XpbdScissorsResult* result){return xpbd_get_scissors(result);}
LCS_API int LCS_GetScissorsLoads(float* values){return xpbd_get_scissors_loads(values);}
LCS_API int LCS_SetScissorsWrist(float front,float length,float radius){return xpbd_set_scissors_wrist(front,length,radius);}
LCS_API int LCS_PressureBladeAbiVersion(){return (int)sizeof(XpbdBladeState);}
LCS_API int LCS_SetPressureBlade(const XpbdBladeState* blade){return xpbd_set_blade(blade);}
LCS_API int LCS_GetPressureBlade(float* values){return xpbd_get_blade(values);}
void cut_set_pressure_mode(int enabled);void cut_suspend();
LCS_API void LCS_SetPressureCutMode(int enabled){cut_set_pressure_mode(enabled);}
LCS_API void LCS_SuspendCut(){cut_suspend();}
