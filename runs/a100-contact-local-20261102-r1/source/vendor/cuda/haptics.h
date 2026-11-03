// haptics.h - host API for the paper 2.3 SDF Finger-Proxy force feedback (defined in haptics.cu).
//
// Realizes paper 2.3 (Eq 22-27 + Algorithm 1): a proxy tool is held on the soft-tissue surface by
// the alternating-projection SDF collision (Eq 26), the proxy-vs-physical deviation yields the
// contact + (implicit tangential) friction force/torque (Eq 22-27), and the Newton reaction force
// is scattered onto the nearby triangle particles (physics_haptic_force buffer).
//
// Collision detection + friction are this one 2.3 algorithm (the paper has no separate Coulomb
// term for plain contact): friction = the tangential component of K_p*dP under the proxy-stays-
// on-surface constraint. `haptics_set_grasp`'s g_graspMu below is a friction-cone multiplier on
// THIS proxy-tracking friction only.
//
// STALE-CLAIM FIX (audit-round finding): grasping is NOT the same reaction-force mechanism as the
// line above might suggest -- the real two-jaw grasp (capture, compliant breakable core-pin,
// Coulomb slab friction, CCD anti-tunnelling) lives entirely in xpbd_solver.cu as its own XPBD
// constraint system; it does not route through haptics.cu's Eq27 force at all. See
// xpbd_solver.cu's own header comments (k_SolveCorePin, k_SolveSlabCoulomb) for that system's
// honestly-documented divergence from the paper (grasp mechanics are paper-silent). g_graspMu here
// and xpbd_solver.cu's grasp01 hysteresis both read the SAME upstream jaw-closedness value from
// C#, but are two independent, non-communicating mechanisms -- keep that in mind when retuning
// either one.

#pragma once
#include <vector_types.h>   // float3 (host-includable)

// Tunable coefficients (paper-silent numerics except a=0.1, eps=1e-3 from Algorithm 1).
// contactBeta/contactDispMax: LIVE again as of 2026-07-07 audit-round-2 (briefly marked DEAD earlier
// the same day when Task 1 of an earlier redesign retired the position-update mechanism these drive,
// k_ApplyContactDisp -- the paper SS3.2 "reaction ... applied to UPDATE THE PARTICLE POSITIONS" dent;
// that retirement was reversed after closer paper reading showed this IS the paper's own described
// tissue-deformation mechanism, and the specific instability that motivated retiring it -- racing
// against the since-ALSO-retired hard k_SolveJawShaftContact constraint -- no longer applies). See
// k_ScatterContact/k_ApplyContactDisp's own comments in haptics.cu for the full restoration rationale.
// contactBeta scales penetration depth (clamped to the tool radius, BUG-A fix) into a per-triangle
// displacement share; contactDispMax caps the SUMMED per-corner displacement per frame.
// captureMargin/proxyTrackStep (persistent finger-proxy, diagnosis wf_de18f2f3):
//   captureMargin (D18) — detection-only band widening (sized to per-frame physical displacement) so a
//     fast plunge is still detected past one radius; PER-CALL, never reaches the separation/dent pass.
//   proxyTrackStep (D19) — bounded per-frame attraction of the persistent proxy toward the physical pose
//     (paper prose "approaches in discrete steps" line 787-789; ABSENT from Algorithm 1); < tissue depth.
// muS/muK (friction cone, research weo6zf2bt): static/kinetic Coulomb coefficients. friction = the
// paper's implicit tangential Kp*dP made non-zero by a stick anchor (Chai3D cAlgorithmFingerProxy);
// frictionRadius = normalDepth*mu. Keep muS >= muK or the static stick region vanishes.
struct HapticParams { float kP, dP, kO, dO, a, eps, reactionMax, contactBeta, contactDispMax, captureMargin, proxyTrackStep, muS, muK; int outerMax; };

// Allocate device buffers, cache the shared recon/physics pointers, store the params. Returns 0 on success.
int  haptics_init(int cornerCount, int triCapacity, const HapticParams* p);

// Set the current PHYSICAL tool: capsule endpoints [S,E] (like CuttingTool.S/E), orientation quat
// (x,y,z,w), linear + angular velocity, and the tool radius (capsule radius, or the mesh SKIN radius).
void haptics_set_physical_tool(const float* physS3, const float* physE3, const float* physQuat4,
                              const float* linVel3, const float* angVel3, float toolRadius);

// Upload the tool COLLISION MESH for the SDF (BUG 2): triWorld = 3*3*triCount world-space verts at the
// PHYSICAL pose. Call each frame BEFORE haptics_set_physical_tool; triCount=0 -> capsule fallback.
void haptics_set_tool_mesh(const float* triWorld, int triCount);

// Run one frame of the 2.3 algorithm: Algorithm 1 outer loop (proxy pushed out of penetration),
// Eq 22-27 force/torque, thresholded reaction scatter. Call AFTER the DC rebuild (triangles current).
// Returns 0 on success.
int  haptics_step();

// Read back the Eq 27 force / torque for the device.
void haptics_get_force(float* outForce3, float* outTorque3);

// Read back the proxy pose (pos, quat) for rendering.
void haptics_get_proxy(float* outPos3, float* outQuat4);

// Native contact state (persistent-proxy predicate g_hasProxy, i.e. the PHYSICAL tool is penetrating —
// NOT additionally gated on |dP|>eps; that conjunct was a bug, see haptics.cu's g_inContact assignment
// and diagnosis wf_395c900a BUG3) — 1 in contact, else 0.
int  haptics_get_contact();

// Grasp (jaws) 0=open .. 1=closed — scales the friction cone so a closed grasper grips/drags the tissue.
void haptics_set_grasp(float grasp01);

// Free device buffers.
void haptics_shutdown();
