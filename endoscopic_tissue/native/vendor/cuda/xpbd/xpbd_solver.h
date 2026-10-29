// xpbd_solver.h — host-callable API for the XPBD soft-body solver (migration STEP 1).
//
// This is the round-3 design (docs/superpowers/specs/2026-07-05-xpbd-combination-round3-design.md):
// replace the mass-spring explicit RK45 integrator with FULL-XPBD, keeping the SAME model topology.
// A structural distance constraint with compliance a~ = (1/ks)/h^2 reproduces a Hookean spring of
// stiffness ks EXACTLY (Macklin/Muller/Chentanez, "XPBD", MIG 2016) but is unconditionally stable at
// any ks. Step 1 implements ONLY: the fixed-N small-steps substep loop, structural distance constraints
// (graph-coloured Gauss-Seidel, no atomics), gravity, and pinned (invMass=0) boundary conditions.
//
// SELF-CONTAINED MODULE: lives entirely under cuda/xpbd/ and does NOT touch physics.cu / the existing
// mass-spring solver. Later stages add bending, volume, contact, grasp constraints + wire it to the
// shared cornerPos/nbrIdx buffers; step 1 is a standalone, analytically-verifiable core.
//
// Reference solver structure mirrored from the user's XPBD repo (Simulation/Assets/SurgicalSim/Physics/
// XPBDSolverGPU.cs + XPBDSolver.compute) and github.com/FantasyVR/neohookean_XPBD, reimplemented in CUDA.

#pragma once
#include <cstdint>
#include <vector_types.h>   // float3 (host-includable CUDA type header; same pattern as dc_recon.h)

// ── Status codes. STATE/VALIDATION errors live in a DISJOINT band (-1000..) so the live caller can
// distinguish a recoverable caller-state error from a fatal device fault: CUDA errors are returned as
// -(int)cudaError_t (small negatives, e.g. -1 = cudaErrorInvalidValue, -2 = cudaErrorMemoryAllocation),
// which the old hand-rolled -1/-2/-3 codes ALIASED. 0 = success.
#define XPBD_ERR_NOT_READY       (-1000)   // xpbd_init not (successfully) called
#define XPBD_ERR_NO_FRAME        (-1001)   // xpbd_substep without a valid xpbd_begin_frame (or invalidated
                                           // by xpbd_set_params / re-init / xpbd_set_bending)
#define XPBD_ERR_BAD_DT          (-1002)   // dt does not yield a positive finite substep h (incl. NaN/Inf)
#define XPBD_ERR_COLOR_OVERFLOW  (-1003)   // greedy colouring exhausted 64 colours (vertex degree >= 64):
                                           // FAIL LOUD rather than alias a used colour (no-atomics GS race)
#define XPBD_ERR_BAD_ARG         (-1004)   // null pointer / count / index out of range
#define XPBD_ERR_IRREGULAR_REST  (-1005)   // a rest tet differs from its Kuhn type beyond rounding: the solver bakes one
                                           // rest shape per type, valid on a regular rest lattice only

extern "C" {

// Solver tunables. h = dt / numSubsteps.  compliance a = 1/ks ;  a~ = a / h^2  (per substep).
struct XpbdParams
{
    int   numSubsteps;   // N — fixed small-steps count per outer dt (Macklin "Small Steps" SCA 2019)
    int   iters;         // constraint iterations per substep (1 = pure small-steps; >1 accumulates lambda)
    float omega;         // SOR under-relaxation, applied CONSISTENTLY to lambda AND position (so the
                         // compliant equilibrium DL=mg/ks is omega-invariant — omega only changes the
                         // convergence RATE, not the effective stiffness). omega=1.0 is permitted for
                         // the STANDALONE step-1-3 self-test ONLY (design §3.5); the FULL-constraint-set
                         // default is 0.2-0.5 (production reference XPBDSolverGPU.cs:450 uses 0.2 — the
                         // near-hard contact/grasp compliances are the worst over-relaxation regime).
                         // Committing omega=1 beyond steps 1-3 REQUIRES the §3.5 stability test at
                         // ks=7.5e4 with structural+bending+contact+grasp.
    float damping;       // legacy global post-solve velocity damping in [0,1): v *= (1 - damping) PER
                         // SUBSTEP. NOTE its per-FRAME retention is (1 - damping)^N, so its strength is
                         // N-DEPENDENT (unlike alpha, which is re-derived per-N for N-invariant decay).
                         // It cannot move the static equilibrium (DL=mg/ks), but RETUNE it if N changes;
                         // prefer alpha (Rayleigh) for physical damping. e.g. N=20, damping=0.02 -> 0.98^20
                         // ~= 0.67 per frame (a 33% per-frame velocity kill).
    float ks;            // structural stiffness (paper SS3.4 = 7.5e4) -> compliance 1/ks
    float gx, gy, gz;    // gravity acceleration (e.g. 0, -9.81, 0)
    float kb;            // bending stiffness (paper SS3.4 = 2e4) -> compliance 1/kb (step 2; 0 = no bending)
    // STEP 3 damping (paper SS3.4). alpha = global mass-proportional Rayleigh (1/s), applied as the
    // per-substep implicit velocity term v *= 1/(1+alpha*h). betaS/betaB = Kelvin-Voigt DAMPING STIFFNESSES
    // for the structural / bending constraints (Macklin XPBD 2016 damped-constraint term gamma = a~*beta*h;
    // beta is a damping STIFFNESS, NOT an inverse/compliance). The XPBD damped constraint reproduces a
    // dashpot force f = beta * (relative velocity along the constraint gradient), so to match the paper's
    // cs*(vi-vj) / cb*theta_dot set betaS = cs and betaB = cb DIRECTLY (do NOT divide by ks/kb). 0 = undamped.
    float alpha;         // Rayleigh (global velocity), 1/s
    float betaS;         // structural KV damping STIFFNESS = cs (the dashpot coefficient, NOT cs/ks)
    float betaB;         // bending KV damping STIFFNESS = cb
    // Isochoric/volumetric Neo-Hookean: mu=E/(2*(1+nu)), kappa=E/(3*(1-2*nu)).
    // Coupled dev/hyd XPBD blocks vanish at rest. Build-time parameters:
    // set BEFORE xpbd_build_tets; changing them later requires rebuilding the rest mesh.
    float youngE;        // Young's modulus E. <=0 disables the strain constraint (tets still built/
                         // tracked for the cut lifecycle, but alpha->+inf so dev/hyd never correct).
    float poisson;       // Poisson ratio nu, design start 0.45.
    float strainDamping; // material Kelvin-Voigt retardation time in seconds; 0 = elastic
};

// Allocate + upload. Particles: pos3[3*particleCount], invMass[particleCount] (0 = pinned/infinite mass).
// Edges: edges[2*edgeCount] (i,j vertex indices), restLen[edgeCount]. The host builds a greedy edge
// colouring (no two edges in a colour share a vertex) so each colour is one conflict-free GS dispatch.
// Returns 0 on success, XPBD_ERR_* or -(int)cudaError_t on error. FAILURE SEMANTICS: bad args / colour
// overflow are detected BEFORE any existing instance is torn down (a failed re-init with bad inputs
// preserves the running instance); a DEVICE alloc/upload failure during re-init, however, leaves the
// module shut down (no half-allocated state).
int  xpbd_init(int particleCount, const float* pos3, const float* invMass,
               int edgeCount, const int* edges, const float* restLen);

// Set/replace the solver tunables (safe to call any time after init).
void xpbd_set_params(const XpbdParams* p);
// Soft ligament attachment to the current REST position. stiffness[N] in kg/s^2,
// zero means unattached. Call before stepping; pinned/inactive nodes are ignored.
int xpbd_set_attachments(int count, const float* stiffness);

// STEP 2: register the bending elements (paper Eq13-16). Each element is a triple (apex i, arm j, arm k)
// with rest angle theta0 = angle at i between (xi-xj) and (xi-xk). Reproduces the k_AccBending model as an
// XPBD angle constraint C = theta - theta0 with compliance (1/kb)/h^2. ijk = 3*bendCount ints; theta0 =
// bendCount floats. Call AFTER xpbd_init; bendCount 0 (or never called) = structural-only (step 1). The
// bending elements get their own greedy colouring (no two sharing a vertex in a colour). Returns 0,
// XPBD_ERR_*, or -(int)cudaError_t. FAILURE SEMANTICS: the colouring runs BEFORE the existing bend set is
// freed (a colour-overflow failure preserves the old set); a DEVICE alloc/upload failure leaves the
// module bending-less (structural-only), never half-allocated. Invalidates the current frame (§9-(4)
// begin/substep contract) — call xpbd_begin_frame again.
int  xpbd_set_bending(int bendCount, const int* ijk, const float* theta0);

// Number of bending colour groups (diagnostic / test).
int  xpbd_bend_color_count();

// Advance ONE outer frame `dt` by N internal XPBD substeps. Deforms the device position buffer in place.
// Returns 0 on success, negative CUDA error otherwise. (Thin wrapper over the M2 split below —
// byte-equivalent stepping, kept for the standalone self-test.)
int  xpbd_step(float dt);

// M2 API SPLIT (design rev-B §9-(4)): the live caller (physics_step) must OWN the substep loop so
// cut_ribbon_tick() can fire BETWEEN substeps (one CCD tick per committed sub-interval — the R-cut
// contract). xpbd_begin_frame(dt) computes ALL h-derived coefficients once (a~ ~1/h^2, gamma ~1/h,
// velScale ~h — §12 R-compliance: recomputed together); xpbd_substep() advances exactly ONE substep
// (predict -> lambda reset -> colour sweeps -> velocity write) and returns an ASYNC launch-error check
// (the caller syncs at frame end). Calling xpbd_set_params / re-init invalidates the frame: call
// xpbd_begin_frame again. Loop shape: begin_frame(dt); for n in 1..N { substep(); cut_ribbon_tick(); }
int  xpbd_begin_frame(float dt);
int  xpbd_substep(void);

// ── REBIND / GO-LIVE (design rev-B §9-(4)) ───────────────────────────────────────────────────────
// Layout-compatible copy of physics.cu's BendPairGpu — the codebase convention (cut.cu declares its own
// BendPairGpuC the same way); keeps module isolation (no physics.h include). 32 bytes; layout MUST match.
struct XpbdBendPairGpu { int i, j, k; float theta0; int alive, _p0, _p1, _p2; };

// Bind the solver to the LIVE shared device buffers and derive its topology from them. After a
// successful bind the module projects IN-PLACE on `cornerPos` (= recon_corner_pos(), the buffer
// cut/haptics/recon/particleRot all read) and writes `vel` (= physics_vel()) — NO private pos/vel
// buffers on the live path (they remain only for the standalone self-test via xpbd_init).
//   - EDGE LIST: derived ON DEVICE-READBACK from nbrIdx by the canonical dedup — each lattice edge is
//     emitted ONCE from its POSITIVE slot s = 2*axis+1 (slot order: 0=-x,1=+x,2=-y,3=+y,4=-z,5=+z),
//     restLen = length3(restNbr[6p+s]). Per colour-ordered edge the OWNER SLOT (6p+s) is persisted on
//     device; per colour-ordered bend element the SOURCE INDEX into the live bendPairs array.
//   - SEVER PROPAGATION: a k_RefreshLiveness kernel at the START of every substep derives
//     edgeActive[e] = (nbrIdx[edgeOwnerSlot[e]] >= 0) && active[i] && active[j]  (BOTH k_AccStructural
//     gates in physics.cu k_AccStructural — an edge to debris-frozen corners must drop too), bendActive[b] =
//     bendPairs[bendSrcIdx[b]].alive (k_AccBending's only gate; a frozen member is handled by w=0),
//     and invMass[c] = (pinned||!active) ? 0 : 1/mass[c] — from the LIVE shared flags (cut.cu writes
//     nbrIdx=-1 / alive=0 / active=0 at runtime). invMass is NEVER baked. Zero cut.cu changes; a sever
//     is picked up the very next substep. Colouring stays STATIC (a dead edge early-outs).
//   - BIND-TIME STATE: d_prevPos is initialized = cornerPos (no first-frame velocity kick) and the
//     CURRENT `vel` contents are adopted (a moving liver is not zeroed).
//   - PREDICT (bound mode): the acceleration is (extForce[c] + hapticForce[c]) * invMass[c] — gravity
//     PLUS the SS2.3 felt haptic reaction (migration STEP 5), the same body-force sum the retired RK45
//     slope formed. Both are LIVE shared buffers; hapticForce is the one haptics.cu scatters into.
//   - Bend elements DEAD at bind time (alive==0) are not emitted (a sever is permanent).
// Only bendPairs entries [0, bendCapacity) are read. dims is carried for the step-7 tet builder.
// Replaces any prior xpbd_init/xpbd_bind state. Returns 0, XPBD_ERR_*, or -(int)cudaError_t.
int xpbd_bind(float3* cornerPos, float3* vel,
              const int* nbrIdx, const float3* restNbr,
              const void* bendPairs, int bendCapacity,
              const int* active, const int* pinned,
              const float* mass,
              const float3* extForce, const float3* hapticForce,
              int dimsX, int dimsY, int dimsZ, int cornerCount);

// M1 SEVER/LIVENESS NOTIFY PATH (design rev-B §9-(4)): the input-edge/element -> colour-slot permutation
// is PERSISTED (host inverse map + device src-index arrays), so a runtime sever propagates to the
// colour-ordered active flags — the mechanism a cut needs to DROP a constraint (and a debris-freeze
// needs to zero a mass) after go-live. Indices are the CALLER's original input indices (xpbd_init edge
// order / xpbd_set_bending element order). At the rebind step these are superseded by k_RefreshLiveness
// (deriving liveness from the LIVE shared nbrIdx/bendPairs/active/mass every substep); they remain for
// the standalone self-test + targeted host control. Return 0 / negative on error.
int  xpbd_set_edge_active(int inputEdgeIndex, int active);      // 0 = severed (constraint dropped)
int  xpbd_set_bend_active(int inputBendIndex, int active);      // 0 = severed (element dropped)
int  xpbd_set_particle_inv_mass(int particle, float invMass);   // 0 = pinned/frozen (debris freeze)

// Read back the current particle positions to host (out3[3*particleCount]). Returns 0 on success.
int  xpbd_get_positions(float* out3);

// Read back the current particle velocities (out3[3*particleCount]) — for the step-3 damping/settle
// (residual kinetic-energy) verification. Returns 0 on success.
int  xpbd_get_velocities(float* out3);

// Number of colour groups the greedy colouring produced (diagnostic / test).
int  xpbd_color_count();

// Diagnostics for the §9-(4) sever-propagation VERIFY ("readback d_edgeActive"): the number of edges
// and a snapshot of the COLOUR-ORDERED per-edge active flags (out[edgeCount]). In bound mode these are
// re-derived by k_RefreshLiveness each substep, so a readback taken after a substep reflects any sever
// committed before it. Returns count / 0, or XPBD_ERR_* on error.
int  xpbd_get_edge_count();
int  xpbd_readback_edge_active(int* out);

// ── STEP 7: per-tet isochoric/volumetric Neo-Hookean strain (design rev-B §4[Strain]/§9-(7)) ────────────────────
// Builds a 6-tet conforming Freudenthal decomposition of every LIVE voxel cell (all 8 corners active
// at build time) from the buffers captured by a prior successful xpbd_bind (dims/cornerPos/active/
// nbrIdx). BOUND MODE ONLY (tets need the voxel-grid structure the standalone self-test has no use
// for). Computes rest Dm^-1 (3 rows) + RestVolume=|det(Dm)|/6 per tet (host, one-time; degenerate tets
// V<eps are skipped), a 4-vertex greedy tet colouring, and per-tet OWNER-EDGE SLOTS (the 3 member axis
// edges, i.e. the cell edges the tet actually spans — face/body diagonals are not severable
// primitives) for LIVE deactivation: k_RefreshLiveness re-derives d_tetActive[t] every substep from the
// SAME live nbrIdx/active flags edges/bending already read — a tet dies PERMANENTLY the substep any of
// its 3 member axis edges is severed (nbrIdx<0) or any of its 4 corners goes inactive, exactly
// mirroring the edge/bend liveness mechanism (zero cut.cu changes). Set XpbdParams.youngE/poisson
// BEFORE calling (xpbd_set_params) — the material bake reads them at build time. Returns 0,
// XPBD_ERR_*, or -(int)cudaError_t. Call ONCE, after xpbd_bind, before the first xpbd_begin_frame.
int xpbd_build_tets();

// Diagnostics: tet count / tet colour-group count (0 before a successful xpbd_build_tets).
int xpbd_tet_count();
int xpbd_tet_color_count();

// Copy the tetrahedron node IDs (grouped by Kuhn tet type, each type in cell order) for topology diagnostics.
// out4T contains 4*xpbd_tet_count() ints. Shared-face conformity tests must inspect
// these IDs rather than assuming a particular alternating diagonal pattern.
int xpbd_readback_tet_ids(int* out4T);

// ── STEP 8: contact + two-jaw grasp (design rev-B §4[Contact]/§5/§6/§9-(8)) ────────────────────────
// Per-frame tool/grasp geometry, uploaded ONCE per outer frame (constant across its N substeps — the
// SAME once-per-frame staleness contract §9-(5) established for hapticForce; the tool pose is a C#/
// haptics input that does not change mid-frame today). shaftA/shaftB/shaftR: the shaft CAPSULE (Issue
// D, unchanged). jawHinge/jawAxis/jawMid/jawNormal/jawHalfWidth/jawLen/jawGap: the analytic "JawSlabWorld"
// (design §5 — a NEW GrasperRig helper, Assets/ReconGridDC/Cuda/GrasperRig.cs JawSlabWorld()). **Flagged
// simplification** (design-sync note, not silently substituted): §4[Contact] says "jaw-mesh ∪ shaft-
// capsule" (Issue D's real triangle soup); this reuses the SAME analytic slab primitive the design
// itself uses for jaw geometry everywhere else in §5 (mid/normal/gap) for the jaw's non-penetration
// surface too, rather than porting a second independent triangle-closest-point routine alongside
// haptics.cu's (which stays byte-unchanged). The jaw is modelled as two finite plates +-jawGap/2 along
// jawNormal, spanning [0,jawLen] along jawAxis from jawHinge and +-jawHalfWidth along the derived width
// axis (jawAxis x jawNormal) — an oriented box, not an infinite plane.
struct XpbdToolState
{
    float3 shaftA, shaftB; float shaftR;
    float3 jawHinge, jawAxis, jawMid, jawNormal;
    float jawHalfWidth, jawLen, jawGap;
    float grasp01;
};
// Uploads the tool state and, on a grasp01 threshold CROSSING (design §5 item 1 hysteresis — see
// xpbd_set_grasp_params), performs CAPTURE (rising edge: latch the corner set once, split core/ring,
// store each corner's jaw-local anchor) or RELEASE (falling edge: damp velocity x0.1, clear
// grasped/graspCore -- no invMass save/restore is needed at all, since invMass is unconditioned on
// graspCore and re-derived every substep from ONLY pinned/active; see k_RefreshLiveness).
// Does NOT touch cut.cu (module self-containment) -- the caller (physics.cu) reads xpbd_grasp_flags()
// afterward and calls cut_recompute_grasp_ring() itself, exactly as it already owns cut_ribbon_tick().
// Returns 0, XPBD_ERR_*, or -(int)cudaError_t.
//
// CALLER CONTRACT -- KNOWN LIMITATION (audit-round finding, real and unresolved as of 2026-07-07): while
// grasping, repositioning the jaw (this call's jawHinge etc.) FASTER than ~0.15 world-units/second was
// confirmed to destabilize the compliant core-pin (an underdamped resonance with the structural solve;
// see k_CoreBreakCheck's kernel comment in xpbd_solver.cu for the full investigation). This is well below
// this project's own ordinary tool-drive speed (LiverCudaManager.cs's hapticMoveSpeed, ~4 u/s) -- callers
// driving a grasp through fast, continuous tool motion should expect the grasp to reliably detach before
// this is fixed at the solver level (candidate fixes: a genuine rigid-body solve for the captured patch,
// or a caller-side rate limit on jaw motion while grasping — neither implemented yet).
int xpbd_set_tool(const XpbdToolState* t);

// Grasp/contact tunables (design §5/§6/§11 user-decision knobs).
struct XpbdGraspParams
{
    float graspThreshold, releaseThreshold;  // capture/release hysteresis (design §5 item 1; e.g. 0.5/0.3)
    float jawGapClosed;                      // §11 item 5 floor: min slab half-gap, never fully collapses
    float coreEllipseFrac;                   // normalized-ellipse core fraction (0.55 start, §5)
    float mu;                                // Coulomb friction coefficient (ring tangential stick/slip)
    float contactAlpha; // physical compliance (world-unit/native-force), scaled by 1/h^2 per substep
    float contactSkin;                       // capture-band widening for [Contact] (design §4[Contact])
    // ── redesign 2026-07-07 (collision-contact-redesign.md, Option A) ──────────────────────────────
    float ccdVelThresh;   // corner speed (world units/s) TOWARD the tool above which the CCD gate fires;
                          // a statically-embedded corner (near-zero approach velocity) gets ZERO
                          // correction by design (rule a) -- STARTING ANCHOR 0.5, paper-silent, tune
                          // against the video's fast-plunge test.
    float ccdMaxDisp;     // absolute per-substep correction cap in world units (rule b) -- caller
                          // computes this as a fraction of L (design range 0.1-0.25*L); STARTING
                          // ANCHOR passed in already-scaled (native module has no L of its own).
    float ccdOmega;       // CCD gate's OWN under-relaxation (rule c), independent of xp.omega and
                          // omegaGrasp -- STARTING ANCHOR 0.25 (design's own explicit bound "<=0.25").
    float omegaGrasp;     // separate SOR under-relaxation for the core-pin + slab/Coulomb constraint
                          // set (redesign: under-relax the SOFT grasp block, not the stiff structural
                          // one) -- STARTING ANCHOR 0.25.
    float coreAlpha; // total patch compliance in wedge mode, per-node in legacy; scaled by 1/h^2
    float coreBreakForce; // threshold (native force = N/metersPerUnit) above which a core corner detaches PERMANENTLY for this
                          // grasp session. NOTE: the comparator is a persistent per-substep EMA of
                          // |coreLambda|/h^2 (k_CoreBreakCheck's coreForceEMA), NOT the raw instantaneous
                          // value -- see that kernel's comment for why (a raw per-substep reading false-
                          // triggers on ordinary target reposition). STARTING ANCHOR 150 (real-scene-scale
                          // audit fix: this codebase's h/coreAlpha/omegaGrasp are the same at real scale
                          // and test-fixture scale, so the same empirical derivation transfers directly:
                          // ordinary hold settles ~9.8-13N, a genuine hard pull settles roughly equal to
                          // the applied load; 150 sits with real margin on both sides).
    float graspMaxDisp;   // audit fix (spec Stability Measure D names BOTH grasp and CCD constraints,
                          // not CCD alone): absolute per-substep correction cap in world units for
                          // k_SolveCorePin AND k_SolveSlabCoulomb's normal AND STICK branches -- caller
                          // computes this as a fraction of L, same convention as ccdMaxDisp. STARTING
                          // ANCHOR passed in already-scaled; without this, a sudden large elastic
                          // imbalance on a grasped corner could move it more than 0.1-0.25*L in one
                          // frame before omegaGrasp/coreAlpha alone bring it back down, or before the
                          // core-pin's break threshold has a chance to fire.
    int   nExemptRings;   // graspRing BFS dilation depth (1-2, design §6)
};
void xpbd_set_grasp_params(const XpbdGraspParams* p);
// Mode 0: legacy proxy contact; 1: surface contact and pure Coulomb friction;
// 2: same surface contact plus persistent compliant grasp anchors.
// Call after bind/init. muDynamic <= muStatic; geometry in native model units.
int xpbd_set_contact_model(int enabled, float muStatic, float muDynamic, float jawThickness, float particleRadius);
// Configure after building tets and before the first step. With passes>0, use
// finite log-volume elasticity and local material/contact coupling, followed by
// a minimum-J safeguard on the live tetrahedral topology. It is a numerical
// inversion safeguard, not a target volume band or exact incompressibility.
// It requires initially positive elements and does not certify self-contact.
// captureWedge enables sub-cell material capture; xpbd_set_grasp_patch selects
// the Gaussian region attachment. Capture NEVER locks the aperture.
// Legacy pressure-gated/breakable grasp remains when captureWedge is zero.
int xpbd_set_tissue_guard(float minimumVolume, int passes, int count, const int* inside, int captureWedge);
// Historical ABI name: returns held state (1/0), and CURRENT actual jaw gap.
// The output is not a minimum gap and must not be used as an aperture latch.
int xpbd_get_grasp_lock(float* currentJawGap);
// Commit post-cut ownership at physical step/finalize boundaries, never in getters.
int xpbd_commit_grip_topology();
// GPU-resident DC surface. Bind before a frame; each vertex is mapped only to its own cut component.
// Optional fixed material embedding for the intact surface; pointers live until shutdown.
int xpbd_set_contact_material(const float3* rest,const float4* featurePoints);
int xpbd_set_contact_surface(int triangleCount, const int* triangles, const float3* expanded,
    const int* voxelCorners, const unsigned* cutMasks, const int* connectivity);
int xpbd_get_contact_reaction(float* force3, int* contacts);

// Diagnostics (VERIFY / tests): counts + the raw per-corner grasp flag (for cut.cu's graspRing BFS —
// physics.cu bridges this to cut_recompute_grasp_ring via physics_grasp_flags()).
int  xpbd_grasp_corner_count();          // number of currently-grasped corners (core + ring)
int  xpbd_grasp_core_count();            // number of currently-core-pinned corners
int  xpbd_read_grasp_core(int count,int* flags); // diagnostics: exact stable material IDs for lift validation
int* xpbd_grasp_flags();                 // device int[cornerCount], 1 = grasped (core OR ring); nullptr
                                          // before a successful xpbd_bind
int* xpbd_grasp_core_flags();            // device int[cornerCount], 1 = hard-pinned core (subset of
                                          // xpbd_grasp_flags()); nullptr before a successful xpbd_bind
float* xpbd_grasp_core_force_ema();      // device float[cornerCount], k_CoreBreakCheck's smoothed
                                          // break-detection signal (see coreBreakForce's comment above);
                                          // nullptr before a successful xpbd_bind

// Free all device buffers.
void xpbd_shutdown();

} // extern "C"

// 10 floats: global then near-jaw [mean abs J error, RMS, fraction above .2, p95 upper bin, rest weight].
extern "C" int xpbd_get_volume_error(float* output);
extern "C" int xpbd_set_tissue_material(float tensileStrength,float bulkDampingSeconds);

// Persistent embedded Gaussian grip. Sigma uses native length units; compliance
// is inverse patch stiffness (m/N, equivalently native length/native force).
// Call after binding and before the first step. Captured material coordinates
// persist until release/cut invalidation. With a coupled tool, constrain the
// Gaussian region center in 3D while allowing strain within the captured region.
extern "C" int xpbd_set_grasp_patch(float sigma,float compliance,float dampingSeconds);
// Finite-mass translation shared by contact, grasp and the rendered instrument.
// Setup before the first step; mass kg, stiffness N/m, damping Ns/m; maxTravel
// bounds command-endpoint travel in native units per outer tick; it never clips
// force-generated physical motion or its reaction impulse.
extern "C" int xpbd_set_tool_dynamics(float mass,float stiffness,float damping,float maxTravel);
// Error is the actual Gaussian material-center/target residual in native units,
// not a per-triangle error, penetration depth or a rigid-patch guarantee.
extern "C" int xpbd_get_solved_tool(float* hinge,float* patchCenterError,int* liveGripPoints);
// Aperture is a coupled scalar DOF; contact and anchor Jacobians exchange its
// impulse with the tissue in the same substep. Speed uses native length/second.
extern "C" int xpbd_set_jaw_dynamics(float mass,float stiffness,float damping,float speed);
extern "C" int xpbd_get_solved_jaw(float* gap);

// Independent force-driven cutting rod, SI force divided by metres/world-unit.
struct XpbdBladeState {float3 a,b;float radius,startForce,cutForce,driveStiffness,mass,friction;int enabled;};
// Closed edge endpoints are in (tip, opening, width) coordinates about the hinge.
// Scissors occupy the pressure tool slot and remain independent of the grasper.
struct XpbdScissorsBackBox {float3 center,halfSize,axisX,axisY;};
struct XpbdScissorsState {
    float3 hinge,axis,normal,leftA,leftB,rightA,rightB;
    float length,gap,maxGap,jawSpeed,startForce,cutForce,driveStiffness,mass,friction,jawMass,jawStiffness;
    int enabled,cutting,guided;
    XpbdScissorsBackBox leftBack,rightBack;
    int backContact;
    // Jaw closure at the tips that counts as a squeeze of the tissue (see
    // k_BladeReaction).
    float squeeze;
};
struct XpbdScissorsResult {
    float3 hinge,axis,normal,leftA,leftB,rightA,rightB;
    float gap,radius,pressure,mass,jawMass;
    float3 force;
    int phase,contactCount;
};
static_assert(sizeof(XpbdScissorsState)==244,"Scissors command ABI");
static_assert(sizeof(XpbdScissorsResult)==124,"Scissors result ABI");
extern "C" int xpbd_set_scissors(const XpbdScissorsState* state);
extern "C" int xpbd_get_scissors(XpbdScissorsResult* result);
// Current left/right blade loads, then left/right/total load at the latest
// fracture and the fracture count, in startForce units, then 1 while both
// blades have loaded the tissue in the current pressing gesture.
extern "C" int xpbd_get_scissors_loads(float* values7);
// Wrist capsule of the scissors, in world units along the hinge axis: from
// `front` ahead of the hinge to `length` behind it, of `radius` (0: no wrist
// contact). It collides like a blade back and never cuts. Set before enabling.
extern "C" int xpbd_set_scissors_wrist(float front,float length,float radius);
extern "C" int xpbd_blade_sweep_count();
static_assert(sizeof(XpbdBladeState)==52,"Pressure blade ABI must match C# sequential layout");
extern "C" int xpbd_set_blade(const XpbdBladeState* state);
extern "C" int xpbd_get_blade(float* values10);
extern "C" const int* xpbd_blade_phase_device();

extern "C" const float3* xpbd_blade_sweep_device();
// cut.cu: the count of intact links severed since cut_init, on the device; null before cut_init.
extern "C" const unsigned* cut_topology_revision_device();

// Optional CPU diagnostic, one flag per material node (count must match N).
// Bits 1/2/4: compressive shaft/upper/lower jaw contact in the accepted state.
// No blade/grasp flags and no simulation mutation; not called by the frame loop.
extern "C" int xpbd_read_pressure_footprint(int count,int* flags);

extern "C" int xpbd_rebind_contact_surface();
