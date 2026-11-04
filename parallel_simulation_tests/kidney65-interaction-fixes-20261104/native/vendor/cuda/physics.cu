#define LCS_STATE_TRACK_ALLOCATIONS
#include "training/state_archive.h"
// physics.cu — host API for the tissue solver (Stage 2). SOLVER = XPBD small-steps since migration
// step 4 (design rev-B §9-(4), commit a170233): physics_init binds cuda/xpbd/xpbd_solver.cu to the
// SHARED buffers and physics_step runs its substep loop, interleaving the Stage-6 CCD cut tick.
//
// XPBD owns integration and compliant structural/bending constraints. Particle frames remain
// here because reconstruction and cutting share them independently of the integrator.
// The shared state is dc_recon's cornerPos plus physics-owned velocity/topology buffers.

#include "common.cuh"     // float3 ops, length3/dot3/cross3, operators
#include "physics.h"
#include <cmath>
#include "dc_recon.h"     // recon_corner_pos(), recon_corner_count(), recon_dims()
#include "cut.h"          // cut_is_ready() + cut_ribbon_tick() (Stage 4 per-substep ribbon)
#include "xpbd/xpbd_solver.h"  // XPBD GO-LIVE (design rev-B 9-(4)): xpbd_bind + begin_frame/substep
extern "C" void xpbd_section_mark(const char* name);
extern "C" void xpbd_section_flush();
#include <stdlib.h>
#include <math.h>

// ── GPU structs — strides MUST match C# ReconBuffers (BendPairGpu 32, RotGpu 36) ─────────────
struct BendPairGpu { int i, j, k; float theta0; int alive, _p0, _p1, _p2; };           // 32 bytes
static_assert(sizeof(BendPairGpu) == 32, "BendPairGpu must be 32 bytes");
static_assert(sizeof(XpbdBendPairGpu) == sizeof(BendPairGpu),
              "xpbd module's layout-compatible BendPair copy drifted from physics.cu's");
struct RotGpu      { float3 r0, r1, r2; };                                              // 36 bytes (rows of R)
static_assert(sizeof(RotGpu) == 36, "RotGpu must be 36 bytes");
// ── 3x3 matrix (3 float3 ROWS) helpers — mirror Physics.compute Det3x3/Inverse3x3/transpose ──
struct Mat3 { float3 r0, r1, r2; };

__device__ __forceinline__ float Det3x3(Mat3 m)
{
    return m.r0.x * (m.r1.y * m.r2.z - m.r1.z * m.r2.y)
         - m.r0.y * (m.r1.x * m.r2.z - m.r1.z * m.r2.x)
         + m.r0.z * (m.r1.x * m.r2.y - m.r1.y * m.r2.x);
}

__device__ __forceinline__ Mat3 Inverse3x3(Mat3 m)
{
    float invDet = 1.0f / Det3x3(m);
    Mat3 adj;   // adjugate = transpose(cofactors); inverse = adj/det
    adj.r0 = make_float3( (m.r1.y * m.r2.z - m.r1.z * m.r2.y),
                         -(m.r0.y * m.r2.z - m.r0.z * m.r2.y),
                          (m.r0.y * m.r1.z - m.r0.z * m.r1.y));
    adj.r1 = make_float3(-(m.r1.x * m.r2.z - m.r1.z * m.r2.x),
                          (m.r0.x * m.r2.z - m.r0.z * m.r2.x),
                         -(m.r0.x * m.r1.z - m.r0.z * m.r1.x));
    adj.r2 = make_float3( (m.r1.x * m.r2.y - m.r1.y * m.r2.x),
                         -(m.r0.x * m.r2.y - m.r0.y * m.r2.x),
                          (m.r0.x * m.r1.y - m.r0.y * m.r1.x));
    Mat3 r;
    r.r0 = adj.r0 * invDet;
    r.r1 = adj.r1 * invDet;
    r.r2 = adj.r2 * invDet;
    return r;
}

__device__ __forceinline__ Mat3 Transpose3(Mat3 m)
{
    Mat3 t;
    t.r0 = make_float3(m.r0.x, m.r1.x, m.r2.x);
    t.r1 = make_float3(m.r0.y, m.r1.y, m.r2.y);
    t.r2 = make_float3(m.r0.z, m.r1.z, m.r2.z);
    return t;
}

// ── Device buffers (physics-owned). cornerPos is SHARED from dc_recon (recon_corner_pos()). ──
static float3*       d_vel         = nullptr;  // [cornerCount]
static float*        d_mass        = nullptr;  // [cornerCount]
static int*          d_pinned      = nullptr;  // [cornerCount]
static int*          d_active      = nullptr;  // [cornerCount]
static float3*       d_extForce    = nullptr;  // [cornerCount]
static float3*       d_hapticForce = nullptr;  // [cornerCount] paper SS2.3 reaction force (haptics.cu scatters + zeroes; NOT zeroed here)
static int*          d_nbrIdx      = nullptr;  // [6*cornerCount]
static float3*       d_restNbr     = nullptr;  // [6*cornerCount]
static BendPairGpu*  d_bendPairs   = nullptr;  // [bendPairCount]
static RotGpu*       d_particleRot = nullptr;  // [cornerCount]
static RotGpu*       d_particleRotPrev = nullptr;  // [cornerCount] previous-frame frames (Berndt [22] Fig 4c)

static PhysicsInitDesc g_pdesc = {};
static bool            g_phys_ready = false;
static int             g_graspNExempt = 1;   // STEP 8: graspRing BFS-dilation depth (design §6: 1-2 rings);
                                             // mirrors XpbdGraspParams.nExemptRings, defaulted here too so
                                             // cut_recompute_grasp_ring has a sane value even if the C#
                                             // side never calls physics_set_grasp_params (grasp01 tuning
                                             // is a §11 user-decision item; 1 ring is the conservative start)

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Kernels — one-to-one ports of Physics.compute
// ─────────────────────────────────────────────────────────────────────────────────────────────

// component accessor + world direction of a frame's local axis k (= column k of R, rows r0/r1/r2).
__device__ __forceinline__ float compF3(float3 v, int k) { return (k == 0) ? v.x : ((k == 1) ? v.y : v.z); }
__device__ __forceinline__ float3 RotColumn(const RotGpu& R, int k)
{
    return make_float3(compF3(R.r0, k), compF3(R.r1, k), compF3(R.r2, k));
}

// ComputeParticleRot — the particle coordinate frame per Berndt [22] Fig 4 (paper SS2.1.2 "The calculation
// of the new coordinate system is based on the method described in [22]"). For each of the 3 axes:
//   (b) a direction from the CURRENT neighbour-edge vectors (the through-particle edge +k neighbour minus
//       -k neighbour; one-sided when only one neighbour survives);
//   (c) blended (averaged) with the neighbours' PREVIOUS-frame axis-k (temporal/spatial smoothing that
//       "avoids abrupt visual perturbation" — the reason [22] keeps the iso-surface on the same cell edges);
//   (d) when BOTH neighbours of an axis were removed by cutting, that axis "cannot be computed accurately"
//       from edges, so it is recovered as the CROSS PRODUCT of the other two axes (Fig 4d).
// The three axes then form matrix M (columns = the axes); R = polar(M) is the nearest proper rotation, so
// the cut-point record/reconstruct round-trip (R^T then R, SS2.1.2) stays valid. pinned/inactive or a
// degenerate/reflected frame (>=2 axes lost, det<=eps) -> identity.
__global__ void k_ComputeParticleRot(int cc, const float3* cornerPos, const int* nbrIdx,
                                     const int* active, const int* pinned,
                                     const RotGpu* particleRotPrev, RotGpu* particleRot)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cc) return;

    RotGpu identity;
    identity.r0 = make_float3(1, 0, 0); identity.r1 = make_float3(0, 1, 0); identity.r2 = make_float3(0, 0, 1);

    if (pinned[c] != 0 || active[c] == 0) { particleRot[c] = identity; return; }

    float3 cPos = cornerPos[c];
    float3 axis[3]; bool have[3] = { false, false, false };

    for (int k = 0; k < 3; k++)
    {
        int np = nbrIdx[6 * c + (2 * k + 1)];   // +k neighbour (slot 2k+1)
        int nm = nbrIdx[6 * c + (2 * k)];       // -k neighbour (slot 2k)
        bool okp = (np >= 0 && active[np] != 0);
        bool okm = (nm >= 0 && active[nm] != 0);

        float3 e;
        if (okp && okm)      e = cornerPos[np] - cornerPos[nm];   // (b) through-particle edge, points +k
        else if (okp)        e = cornerPos[np] - cPos;            // one-sided +k
        else if (okm)        e = cPos - cornerPos[nm];            // one-sided +k
        else                 continue;                            // both gone -> (d) cross-product later
        float len = length3(e);
        if (len < 1e-12f) continue;
        float3 dir = e / len;

        // (c) average with the neighbours' previous-frame axis-k (world column k), all surviving neighbours.
        float3 psum = make_float3(0.f, 0.f, 0.f); int pc = 0;
        for (int s = 0; s < 6; s++)
        {
            int n = nbrIdx[6 * c + s];
            if (n >= 0 && active[n] != 0) { psum = psum + RotColumn(particleRotPrev[n], k); pc++; }
        }
        if (pc > 0)
        {
            float pl = length3(psum);
            if (pl > 1e-6f)
            {
                float3 blended = dir + psum / pl;      // average of current edge dir + mean prev axis (both unit)
                float dl = length3(blended);
                // Robustness: only accept the blend when it is well-formed. Near-cancellation (a ~180deg
                // frame flip between frames) would normalize a tiny vector into a GARBAGE direction, so in
                // that case trust the current-edge direction instead.
                if (dl > 0.25f) dir = blended / dl;
            }
        }
        axis[k] = dir; have[k] = true;
    }

    // (d) recover a single fully-severed axis from the cross product of the other two (right-handed).
    int miss = -1, mc = 0;
    for (int k = 0; k < 3; k++) if (!have[k]) { miss = k; mc++; }
    if (mc == 1)
    {
        float3 a = axis[(miss + 1) % 3], b = axis[(miss + 2) % 3];
        float3 cx = cross3(a, b); float l = length3(cx);
        if (l > 1e-9f) { axis[miss] = cx / l; have[miss] = true; }
    }
    if (!(have[0] && have[1] && have[2])) { particleRot[c] = identity; return; }

    // Matrix M has the three axes as COLUMNS: column k = (M.r0[k], M.r1[k], M.r2[k]) = axis[k]. R = polar(M)
    // is then the nearest proper rotation whose column k is the world direction of local axis k (what
    // RotColumn / mulRv read). det(M)<=eps rejects a reflected/near-collapsed (axes ~parallel) frame.
    Mat3 M;
    M.r0 = make_float3(axis[0].x, axis[1].x, axis[2].x);
    M.r1 = make_float3(axis[0].y, axis[1].y, axis[2].y);
    M.r2 = make_float3(axis[0].z, axis[1].z, axis[2].z);
    if (Det3x3(M) <= 1e-4f) { particleRot[c] = identity; return; }

    // R = polar(M) via Higham R_{k+1}=0.5*(R+R^-T). 16 iters converges even for skewed M (det~0.02);
    // 6 was insufficient (||R^T R - I|| ~ 0.1 near det~0.03) and would break the R^T/R round-trip.
    Mat3 R = M;
    for (int it = 0; it < 16; it++)
    {
        Mat3 RinvT = Transpose3(Inverse3x3(R));
        R.r0 = 0.5f * (R.r0 + RinvT.r0);
        R.r1 = 0.5f * (R.r1 + RinvT.r1);
        R.r2 = 0.5f * (R.r2 + RinvT.r2);
    }

    // Orthonormality safety: if Higham did not fully converge (pathological M), fall back to identity so
    // EmitCutPoint/AccumulateCutFP keep an exact R^T==R^-1 round-trip (SS2.1.2). Columns of R = c0,c1,c2.
    float3 c0 = make_float3(R.r0.x, R.r1.x, R.r2.x);
    float3 c1 = make_float3(R.r0.y, R.r1.y, R.r2.y);
    float3 c2 = make_float3(R.r0.z, R.r1.z, R.r2.z);
    float ortho = fmaxf(fmaxf(fabsf(dot3(c0, c0) - 1.f), fabsf(dot3(c1, c1) - 1.f)), fabsf(dot3(c2, c2) - 1.f));
    ortho = fmaxf(ortho, fmaxf(fabsf(dot3(c0, c1)), fmaxf(fabsf(dot3(c0, c2)), fabsf(dot3(c1, c2)))));
    if (ortho > 1e-3f) { particleRot[c] = identity; return; }

    RotGpu rg; rg.r0 = R.r0; rg.r1 = R.r1; rg.r2 = R.r2;
    particleRot[c] = rg;
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Host API
// ─────────────────────────────────────────────────────────────────────────────────────────────
static int Gp(int n) { int g = (n + 63) / 64; return g < 1 ? 1 : g; }

#define PK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { physics_shutdown(); return -(int)_e; } } while (0)

int physics_init(const PhysicsInitDesc* desc,
                 const float* mass, const int* pinned, const int* active, const float* extForce,
                 const int* nbrIdx, const float* restNbr, const void* bendPairs)
{
    if (g_phys_ready) physics_shutdown();
    if (!desc) return -2000;

    if (!std::isfinite(desc->youngE) || desc->youngE < 0.f ||
        !std::isfinite(desc->poisson) || desc->poisson < 0.f || desc->poisson >= 0.5f ||
        !std::isfinite(desc->strainDamping) || desc->strainDamping < 0.f) return -2004;
    g_pdesc = *desc;
    int cc  = desc->cornerCount;
    int bpc = desc->bendPairCount;
    if (cc <= 0) return -2001;
    if (cc != recon_corner_count()) return -2002;   // must share dc_recon's grid

    int bpcA = bpc > 0 ? bpc : 1;

    PK(cudaMalloc(&d_vel,         (size_t)cc      * sizeof(float3)));
    PK(cudaMalloc(&d_mass,        (size_t)cc      * sizeof(float)));
    PK(cudaMalloc(&d_pinned,      (size_t)cc      * sizeof(int)));
    PK(cudaMalloc(&d_active,      (size_t)cc      * sizeof(int)));
    PK(cudaMalloc(&d_extForce,    (size_t)cc      * sizeof(float3)));
    PK(cudaMalloc(&d_hapticForce, (size_t)cc      * sizeof(float3)));   // SS2.3 reaction; zeroed by haptics_step
    PK(cudaMemset(d_hapticForce, 0, (size_t)cc     * sizeof(float3)));
    PK(cudaMalloc(&d_nbrIdx,      (size_t)6 * cc  * sizeof(int)));
    PK(cudaMalloc(&d_restNbr,     (size_t)6 * cc  * sizeof(float3)));
    PK(cudaMalloc(&d_bendPairs,   (size_t)bpcA    * sizeof(BendPairGpu)));
    PK(cudaMalloc(&d_particleRot,     (size_t)cc  * sizeof(RotGpu)));
    PK(cudaMalloc(&d_particleRotPrev, (size_t)cc  * sizeof(RotGpu)));

    // Upload topology / properties.
    PK(cudaMemcpy(d_mass,     mass,     (size_t)cc     * sizeof(float),  cudaMemcpyHostToDevice));
    PK(cudaMemcpy(d_pinned,   pinned,   (size_t)cc     * sizeof(int),    cudaMemcpyHostToDevice));
    PK(cudaMemcpy(d_active,   active,   (size_t)cc     * sizeof(int),    cudaMemcpyHostToDevice));
    PK(cudaMemcpy(d_extForce, extForce, (size_t)cc     * sizeof(float3), cudaMemcpyHostToDevice));
    PK(cudaMemcpy(d_nbrIdx,   nbrIdx,   (size_t)6 * cc * sizeof(int),    cudaMemcpyHostToDevice));
    PK(cudaMemcpy(d_restNbr,  restNbr,  (size_t)6 * cc * sizeof(float3), cudaMemcpyHostToDevice));
    if (bpc > 0)
        PK(cudaMemcpy(d_bendPairs, bendPairs, (size_t)bpc * sizeof(BendPairGpu), cudaMemcpyHostToDevice));

    // Particles start at rest.
    PK(cudaMemset(d_vel,       0, (size_t)cc     * sizeof(float3)));

    // ParticleRot = identity (persists; consumed at Stage 3).
    RotGpu* idn = (RotGpu*)malloc((size_t)cc * sizeof(RotGpu));
    if (!idn) { physics_shutdown(); return -2003; }
    for (int i = 0; i < cc; i++)
    {
        idn[i].r0 = make_float3(1, 0, 0); idn[i].r1 = make_float3(0, 1, 0); idn[i].r2 = make_float3(0, 0, 1);
    }
    cudaError_t e = cudaMemcpy(d_particleRot, idn, (size_t)cc * sizeof(RotGpu), cudaMemcpyHostToDevice);
    if (e == cudaSuccess)
        e = cudaMemcpy(d_particleRotPrev, idn, (size_t)cc * sizeof(RotGpu), cudaMemcpyHostToDevice);   // [22] Fig 4c seed = identity
    free(idn);
    if (e != cudaSuccess) { physics_shutdown(); return -(int)e; }

    // ── XPBD GO-LIVE (design rev-B §9-(4)): configure + bind the XPBD solver to the SHARED buffers. ──
    // Params map from the SAME PhysicsInitDesc the RK45 consumed (C# StructLayout ABI unchanged):
    //   numSubsteps = maxSubsteps REPURPOSED as the fixed small-steps count N (hInit/hMax are DEAD);
    //   betaS/betaB = the paper's cs/cb — Kelvin-Voigt dashpot STIFFNESSES, used DIRECTLY (§4, NOT c/k);
    //   alpha       = global Rayleigh drag (implicit 1/(1+a*h));
    //   damping     = 0 (no legacy scalar velocity kill on the live path — alpha + cs/cb damp);
    //   iters=1, omega=1 — the steps-1-3-VALIDATED configuration for THIS constraint set (structural +
    //   bending + damping; §3.5 mandates re-evaluating omega before the contact/grasp set at step 8);
    //   gx/gy/gz = 0 — in BOUND mode gravity enters predict per-corner as extForce[c]/mass[c] (the
    //   same d_extForce the RK45 slope consumed).
    XpbdParams xp = {};
    xp.numSubsteps = (g_pdesc.maxSubsteps > 0) ? g_pdesc.maxSubsteps : 20;
    xp.iters   = 1;
    xp.omega   = 1.0f;
    xp.damping = 0.f;
    xp.ks      = g_pdesc.ks;
    xp.gx = 0.f; xp.gy = 0.f; xp.gz = 0.f;
    xp.kb      = g_pdesc.kb;
    xp.alpha   = g_pdesc.alpha;
    xp.betaS   = g_pdesc.cs;
    xp.betaB   = g_pdesc.cb;
    xp.youngE = g_pdesc.youngE;
    xp.poisson = g_pdesc.poisson;
    xp.strainDamping = g_pdesc.strainDamping;
    xpbd_set_params(&xp);
    int dx = 0, dy = 0, dz = 0;
    recon_dims(&dx, &dy, &dz);
    int xrc = xpbd_bind(recon_corner_pos(), d_vel,
                        d_nbrIdx, d_restNbr, d_bendPairs, bpc,
                        d_active, d_pinned, d_mass,
                        d_extForce, d_hapticForce,
                        dx, dy, dz, cc);
    if (xrc != 0) { physics_shutdown(); return xrc; }

    // STEP 7 GO-LIVE: build the tet mesh from the just-bound live buffers (dims/cornerPos/active — the
    // REST configuration, since no substep has run yet). Tet CUT/TEAR lifecycle (design §4[Strain]) is
    // derived every substep by k_RefreshLiveness from the SAME live nbrIdx/active flags edges/bending
    // already read — zero cut.cu changes, exactly mirroring the sever-propagation mechanism §9-(4)
    // established for edges/bending.
    int trc = xpbd_build_tets();
    if (trc != 0) { physics_shutdown(); return trc; }

    g_phys_ready = true;
    return 0;
}

__global__ void k_SetGravity(int count, const float* mass, float3 acceleration, float3* force)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) force[i] = acceleration * mass[i];
}

int physics_set_gravity(float gx, float gy, float gz)
{
    if (!g_phys_ready) return XPBD_ERR_NOT_READY;
    if (!std::isfinite(gx) || !std::isfinite(gy) || !std::isfinite(gz)) return XPBD_ERR_BAD_ARG;
    k_SetGravity<<<Gp(g_pdesc.cornerCount), 64>>>(g_pdesc.cornerCount, d_mass, make_float3(gx,gy,gz), d_extForce);
    return -(int)cudaGetLastError();
}

int physics_step(float dt)
{
    if (!g_phys_ready) return -1;
    if (!recon_corner_pos()) return -2;

    // ── XPBD small-steps body (design rev-B §9-(4) rebind + §9-(6) cut-tick interleave). The RK45
    // adaptive-h loop is RETIRED (deviation-ledger D3/D4/D6 VOID, superseded by XPBD small-steps
    // substepping — the solver stays inside the paper's own cited PBD family, refs [7]/[22]).
    // physics_step OWNS the N-loop so the Stage-6 CCD cut tick fires BETWEEN substeps (§9-(6)) exactly
    // as it rode the RK45 substeps: one tick per committed sub-interval, post-position-commit, with the
    // d_prevCorner roll inside the tick (cut.cu). The solve projects the SHARED recon_corner_pos() in
    // place and writes d_vel via the bound module (xpbd_bind in physics_init) — recon/cut/haptics/
    // particleRot all read the SAME deformed buffer.
    //   N = maxSubsteps, REPURPOSED as the fixed small-steps count (== the module's numSubsteps, from the
    // same field in physics_init). Design §9-(6): N = max(N_cut, N_elastic). N_cut (no-tunnel: one
    // substep's TISSUE travel < voxelL, worst case fast-falling severed debris) ∈ [14, 200] at
    // physicsDt=0.02 (bounds map from the old adaptive g_h∈[1e-4, 1.5e-3]); N_elastic (convergence) ~ 10.
    // The shipped N=20 (h=1e-3, near the old hMax) sits at N_cut's low end — a sound first live value;
    // raise maxSubstepsPerFrame if fast debris tunnels past the per-substep CCD tick. PATH B (the
    // rod-phase quad, cut_detect between LCS_Step and LCS_Finalize) runs per C# rod substep, independent
    // of N.
    xpbd_section_mark("step start");
    int rc = xpbd_begin_frame(dt);
    if (rc != 0) return rc;
    int N = (g_pdesc.maxSubsteps > 0) ? g_pdesc.maxSubsteps : 20;
    int tickErr = 0;   // first ribbon-tick failure (surfaced after the loop — review R3-m4)
    for (int n = 0; n < N; n++)
    {
        xpbd_section_mark("begin frame, cut tick and rebuild between substeps");
        rc = xpbd_substep();
        if (rc != 0) return rc;

        // Stage 6 CCD tick: continuous collision of every moving deformed edge vs the static blade
        // segment after EVERY committed substep — exact space-time crossings, so fast tissue motion
        // under gravity cannot tunnel between per-frame snapshots. No-op until cutting is active.
        if (cut_is_ready())
        {
            unsigned before=0,after=0;
            int r=cut_surface_revision(&before);if(r)return r;
            r = cut_ribbon_tick();
            if (r != 0 && tickErr == 0) tickErr = r;
            r=cut_surface_revision(&after);if(r)return r;
            if(after!=before && n+1<N){
                // A newly detached fragment needs its own contact embedding
                // before the next gravity prediction, not at the next frame.
                r=physics_compute_rotations();if(r)return r;
                r=recon_rebuild();if(r)return r;
                r=recon_prepare_contact_surface();if(r)return r;
                r=xpbd_rebind_contact_surface();if(r)return r;
            }
        }
    }

    // NOTE: ComputeParticleRot lives in physics_compute_rotations(). Stage 3 calls it once after physics
    // for pre-cut material frames, and again after detect+sever so cut-FP reconstruction uses post-sever
    // connectivity.
    cudaError_t le = cudaGetLastError();
    if (le == cudaSuccess) le = cudaDeviceSynchronize();   // commit the frame (replaces the errMax sync)
    xpbd_section_mark("begin frame, cut tick and rebuild between substeps");
    xpbd_section_flush();
    if (le != cudaSuccess) return -(int)le;
    if (tickErr != 0) return tickErr;   // surface ribbon-tick failures to C# (review R3-m4)
    return 0;
}

// physics_compute_rotations — k_ComputeParticleRot: Berndt [22] Fig 4 particle frame (paper SS2.1.2).
// Snapshots the current frames into d_particleRotPrev first (Fig 4c previous-orientation average). Stage 3
// uses it both before DetectCut (material-frame hit recording) and after SeverLinks (cut-FP reconstruction).
int physics_compute_rotations()
{
    if (!g_phys_ready) return -1;
    int cc = g_pdesc.cornerCount;
    float3* cornerPos = recon_corner_pos();
    if (!cornerPos) return -2;
    // Snapshot the current frames as "previous" for [22] Fig 4c (neighbours' previous-orientation average).
    cudaError_t ce = cudaMemcpy(d_particleRotPrev, d_particleRot, (size_t)cc * sizeof(RotGpu), cudaMemcpyDeviceToDevice);
    if (ce != cudaSuccess) return -(int)ce;
    k_ComputeParticleRot<<<Gp(cc), 64>>>(cc, cornerPos, d_nbrIdx, d_active, d_pinned, d_particleRotPrev, d_particleRot);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

// Stage 3 shared accessors (read by cut.cu): particleRot (RotGpu*), nbrIdx (int*), bendPairs (BendPairGpu*).
void* physics_particle_rot() { return d_particleRot; }
int*  physics_nbr_idx()      { return d_nbrIdx; }
void* physics_bend_pairs()   { return d_bendPairs; }
int*  physics_active()       { return d_active; }   // v4.1 P1b: cut.cu freezes fully-severed debris
int*  physics_pinned()       { return d_pinned; }   // SS2.3 contact: haptics skips these in the position update (anchored BC)
void* physics_haptic_force() { return d_hapticForce; }   // SS2.3: float3[cornerCount] reaction buffer (haptics.cu scatters)
void* physics_vel()          { return d_vel; }           // SS2.3 contact patch: velocity damping along the push direction

// Diagnostic readback: copy the LIVE (post-sever) nbrIdx[6*cornerCount] to host. Returns 0 on success.
int physics_readback_nbr_idx(int* hostOut)
{
    if (!g_phys_ready || !hostOut) return -1;
    int cc = g_pdesc.cornerCount;
    cudaError_t e = cudaMemcpy(hostOut, d_nbrIdx, (size_t)6 * cc * sizeof(int), cudaMemcpyDeviceToHost);
    return (e == cudaSuccess) ? 0 : -(int)e;
}

// STEP 8 (design rev-B §9-(8)): tool/grasp per-frame bridge. xpbd_set_tool runs the module's own
// capture/release state machine; graspRing is then recomputed from whatever it just captured/released
// (cheap — once per frame, not per-substep — and correctly re-arms the cut.cu tear exemption the SAME
// frame a capture happens, satisfying the "recompute after each sever event inside the ring" rule for
// free since it always reflects the CURRENT set).
int physics_set_tool(const void* toolState)
{
    if (!g_phys_ready || !toolState) return -1;
    int rc = xpbd_set_tool((const XpbdToolState*)toolState);
    if (rc != 0) return rc;
    // Grasp/contact must not hard-depend on the CUTTING subsystem being initialized (they are separate
    // pipelines — a scene with no cutting tool configured still grasps/contacts fine). Only recompute the
    // tear-exemption ring when cut_init actually ran; the ring is otherwise never consumed anyway
    // (k_TearOverstretch itself no-ops until cut_is_ready() && g_toolSet, cut.cu:cut_ribbon_tick).
    if (!cut_is_ready()) return 0;
    return cut_recompute_grasp_ring(xpbd_grasp_flags(), g_graspNExempt);
}

void physics_set_grasp_params(const void* graspParams)
{
    if (!graspParams) return;
    const XpbdGraspParams* p = (const XpbdGraspParams*)graspParams;
    g_graspNExempt = p->nExemptRings;
    xpbd_set_grasp_params(p);
}

void physics_shutdown()
{
    xpbd_shutdown();   // unbind FIRST: the XPBD module aliases cornerPos/d_vel and reads these buffers
    cudaFree(d_vel);         d_vel = nullptr;
    cudaFree(d_mass);        d_mass = nullptr;
    cudaFree(d_pinned);      d_pinned = nullptr;
    cudaFree(d_active);      d_active = nullptr;
    cudaFree(d_extForce);    d_extForce = nullptr;
    cudaFree(d_hapticForce); d_hapticForce = nullptr;
    cudaFree(d_nbrIdx);      d_nbrIdx = nullptr;
    cudaFree(d_restNbr);     d_restNbr = nullptr;
    cudaFree(d_bendPairs);   d_bendPairs = nullptr;
    cudaFree(d_particleRot); d_particleRot = nullptr;
    cudaFree(d_particleRotPrev); d_particleRotPrev = nullptr;
    g_phys_ready = false;
}

#include "training/physics_state.cuh"
