#define LCS_STATE_TRACK_ALLOCATIONS
#include "training/state_archive.h"
// haptics.cu - native CUDA implementation of the paper 2.3 SDF Finger-Proxy force feedback.
//
// Paper mapping (all verbatim unless ledgered D10-D15):
//   Eq 26 / Algorithm 1 inner loop : k_ClosestPerTriangle (alternating projection on the capsule AXIS)
//   Algorithm 1 line 16 selection  : min ||D|| (= min penetration depth) among PENETRATING triangles
//   Algorithm 1 outer loop         : haptics_step (P_t += a*v_ms until no penetration / ||v_ms||<eps)
//   Eq 22-25 deviation             : dP, dQ, theta, axisK in haptics_step
//   Eq 27 force/torque             : g_force = kP*dP - dP*v ; g_torque = kO*theta*k - dO*w (contact-gated)
//   Reaction to particles (SS3.2)  : k_CountPenetrating -> k_ScatterContact (depth-weighted force,
//                                    atomicAdd onto the nearest deformable corner, PLUS a bounded
//                                    position-displacement accumulator) -> k_ApplyContactDisp (clamped
//                                    position write + spring-back velocity damping).
//
// RESTORATION NOTE (2026-07-07, audit-round 2): Task 1 of an earlier redesign this same day RETIRED
// this position-update half ("Option A: force-only"), reasoning it was a deliberate deviation from
// SS3.2, not a fidelity improvement. Reading the paper's own text closely (SS2.3 line ~917-919: "the
// reaction force is applied to the particles of the nearby triangle faces"; SS3.2 line ~985-987: "a
// reaction force threshold is set and applied to update the particle positions, ensuring smooth
// collision forces without penetration") shows this framing had it backwards: this position update
// IS the paper's own described tissue-deformation mechanism, not an add-on hack. Task 1's retirement
// was motivated by a SPECIFIC, real instability (redesign spec root-cause item 4: this position write
// raced against `k_SolveJawShaftContact`, a hard, frame-1-armed, rigid ejection constraint that was
// the ACTUAL primary cause of the original catastrophic bug) -- that constraint is ALSO retired now
// (replaced by the velocity-gated, capped `k_SolveCCDContact`), so the specific racing condition that
// justified removing this mechanism no longer exists. Empirically, force-only alone was confirmed to
// produce no visible tissue deformation at any gravity-safe (non-tearing) structural stiffness (see
// LiverCudaManager.cs's own long-standing comment on hapticKp: "at a gravity-safe stiff Ks the
// reaction alone cannot visibly squeeze... the VISIBLE deformation is the Ks-independent bounded
// dent"). Restored verbatim from the last validated, BUG-A-fixed implementation (commit 869cd4d),
// reusing the SAME already-present (previously dead) `contactBeta`/`contactDispMax` HapticParams
// fields -- no ABI change. The paper gives no explicit numbered equation for this position update
// (Eq27 is the paper's last numbered equation) -- this remains, as it always was, a physically-
// motivated realization of the paper's qualitative description, not a literal formula transcription.

#include "haptics.h"
#include "physics.h"
#include "dc_recon.h"
#include "common.cuh"     // float3 ops: dot3/cross3/length3, operators
#include <math.h>
#include <string.h>

#define HK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) return -(int)_e; } while (0)

// ============================================================================
// Task 3: SDF closest-point primitives (Eq 26)
// ============================================================================

// SDF(Triangle, P): nearest point on triangle abc to p (Ericson region test).
__device__ __forceinline__ float3 ClosestOnTriangle(float3 a, float3 b, float3 c, float3 p)
{
    float3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0.f && d2 <= 0.f) return a;
    float3 bp = p - b; float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0.f && d4 <= d3) return b;
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) return a + ab * (d1 / (d1 - d3));
    float3 cp = p - c; float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0.f && d5 <= d6) return c;
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) return a + ac * (d2 / (d2 - d6));
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f)
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    float den = 1.f / (va + vb + vc); float v = vb * den, w = vc * den;
    return a + ab * v + ac * w;
}

// SDF(Tool, A): nearest point on the capsule AXIS segment [s,e] to p. The capsule's directed
// distance field is dist_to_axis - radius, so the alternating projection runs on the AXIS and the
// radius is applied analytically (ledger D15). Projecting on the surface degenerates under penetration.
__device__ __forceinline__ float3 ClosestOnSegment(float3 s, float3 e, float3 p)
{
    float3 se = e - s;
    float t = fminf(fmaxf(dot3(p - s, se) / fmaxf(dot3(se, se), 1e-12f), 0.f), 1.f);
    return s + se * t;
}

// SDF(Tool MESH, A): nearest point on the tool collision MESH (toolTri = 3*count world verts at the
// UPLOAD pose) to p, with the mesh translated by `shift` (q_B=q_A, D14, so the proxy tool is the uploaded
// physical tool merely translated by proxyPos-physPos). The tool surface is the mesh inflated by the skin
// radius (Minkowski), applied analytically like the capsule radius — keeps the exact jaw/shaft SHAPE the
// user wants while staying watertight against the coarse tissue mesh (diagnosis wvucv57f6 leak fix).
__device__ __forceinline__ float3 ClosestOnToolMesh(const float3* toolTri, int toolCount, float3 shift, float3 p)
{
    float3 ps = p - shift;                      // query into the uploaded tool frame
    float3 best = ps; float bd = 1e30f;
    for (int k = 0; k < toolCount; k++) {
        float3 q = ClosestOnTriangle(toolTri[3*k+0], toolTri[3*k+1], toolTri[3*k+2], ps);
        float3 d = q - ps; float dd = dot3(d, d);
        if (dd < bd) { bd = dd; best = q; }
    }
    return best + shift;                        // back to world
}

// Quaternion helpers (float4 stored as (x,y,z,w)).
__host__ __device__ __forceinline__ float4 quatConj(float4 q)
{ return make_float4(-q.x, -q.y, -q.z, q.w); }
__host__ __device__ __forceinline__ float4 quatMul(float4 a, float4 b)   // Hamilton product a (x) b
{
    return make_float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

// ============================================================================
// Device buffers + host state
// ============================================================================
static float4*       d_triVms = nullptr;   // [triCapacity] per-triangle (vms.xyz, key)
static unsigned int* d_vmsKey = nullptr;   // [1] atomicMin reduction of the non-negative key
static int*          d_vmsTri = nullptr;   // [1] argmin triangle index
static float3*       d_contactDisp = nullptr; // [cornerCount] contact-patch displacement accumulator
                                               // (RESTORED, see k_ScatterContact/k_ApplyContactDisp headers)
static int*          d_penCount = nullptr;    // [1] penetrating-triangle count
static float*        d_depthSum = nullptr;    // [1] sum of penetration depths
static int           g_triCap = 0;
// Tool COLLISION MESH (BUG 2, diagnosis wvucv57f6): the tool is the low-poly grasper collision soup, not
// a capsule. C# uploads it in world space at the physical pose each frame; the SDF uses ClosestOnToolMesh.
static float3*       d_toolTri     = nullptr;  // [3*maxToolTris] world verts at the upload (physical) pose
static int           g_toolTriCap  = 0;
static int           g_toolTriCount = 0;       // 0 -> fall back to the capsule [s,e] (backward compatible)
static float3        g_toolCenter  = { 0, 0, 0 };   // upload-pose tool centroid (for the query shift + cull)
static float         g_toolBoundR  = 0.f;      // bounding-sphere radius about g_toolCenter (AABB-lite cull)

static HapticParams  g_hp = {};
static int           g_cornerCount = 0;
static float3        g_physS, g_physE, g_physPos;   // physical capsule endpoints + midpoint (P_A)
static float4        g_physQuat  = { 0, 0, 0, 1 };  // q_A
static float3        g_linVel, g_angVel;
static float3        g_proxyS, g_proxyE, g_proxyPos_dev;   // proxy capsule + P_B (= P_t) — PERSISTENT
static float4        g_proxyQuat = { 0, 0, 0, 1 };  // q_B
static bool          g_hasProxy = false;            // persistent proxy is latched in contact (D13 fix)
static float3        g_prevPhysPos = { 0, 0, 0 };   // last frame's P_A (per-frame displacement -> band)
static bool          g_havePrev = false;
// Friction cone (Chai3D cAlgorithmFingerProxy, research weo6zf2bt): a persistent world-space contact
// anchor the proxy sticks to tangentially until the tangential pull exceeds mu*normalForce, then slips.
static float3        g_stickAnchor = { 0, 0, 0 };   // world contact point (tangential stick)
static bool          g_slipping    = false;         // stick/slip hysteresis latch
static float3        g_lastNormal  = { 0, 1, 0 };   // outward surface normal at contact (cone axis)
static float         g_graspMu     = 1.f;           // grasp multiplier on mu (jaws closed -> strong grip)
static float         g_toolRadius = 0.f;
static float3        g_force, g_torque;
static bool          g_inContact = false;
static int           g_minTri = -1;

static float3*       c_cornerPos   = nullptr;   // recon_corner_pos() (live deformed corners)
static float3*       c_hapticForce = nullptr;   // physics_haptic_force()
static int*          c_active      = nullptr;   // physics_active()
static int*          c_pinned      = nullptr;   // physics_pinned() (anchored BC — never displaced by contact)
static float3*       c_vel         = nullptr;   // physics_vel() (contact-patch velocity damping)
static bool          g_ready = false;

// SDF(Tool, A): the tool MESH if uploaded (BUG 2), else the capsule AXIS segment (backward compatible).
__device__ __forceinline__ float3 ClosestOnTool(const float3* toolTri, int toolCount, float3 shift,
                                                float3 s, float3 e, float3 A)
{
    return (toolCount > 0) ? ClosestOnToolMesh(toolTri, toolCount, shift, A) : ClosestOnSegment(s, e, A);
}

// ============================================================================
// Task 4: per-triangle alternating projection (Eq 26 / Algorithm 1 inner loop)
// ============================================================================
__global__ void k_ClosestPerTriangle(int triCount, const float3* soup, const float4* aux,
                                     const float3* nrm, float3 s, float3 e, float radius,
                                     float captureMargin, float eps,
                                     const float3* toolTri, int toolCount, float3 toolShift,
                                     float3 toolQueryCenter, float boundR, float4* triVms)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= triCount) return;
    // SKIN ONLY (ledger D12): skip cut-wall triangles (any vertex carries the bit30 wall flag).
    if (aux[3*t+0].w > 0.5f || aux[3*t+1].w > 0.5f || aux[3*t+2].w > 0.5f)
    { triVms[t] = make_float4(0.f, 0.f, 0.f, 1e30f); return; }

    float3 a = soup[3*t+0], b = soup[3*t+1], c = soup[3*t+2];
    float3 ctr = (a + b + c) * (1.f / 3.f);
    // AABB-lite cull (BUG 2 perf, diagnosis wvucv57f6): only tissue tris near the tool bounding sphere run
    // the expensive mesh loop; the tool occupies a tiny slab of the liver so this drops ~30k -> a few 100.
    // Include the tissue-tri CIRCUMRADIUS so a tri whose CENTROID is just outside but a VERTEX penetrates
    // is NOT false-culled (review wysyvnnmd MAJOR: else the jaw tip tunnels through a straddling tri).
    if (toolCount > 0)
    {
        float triR = fmaxf(fmaxf(length3(a - ctr), length3(b - ctr)), length3(c - ctr));
        if (length3(ctr - toolQueryCenter) > boundR + radius + captureMargin + triR)
        { triVms[t] = make_float4(0.f, 0.f, 0.f, 1e30f); return; }
    }

    float3 P = ClosestOnTool(toolTri, toolCount, toolShift, s, e, ctr);   // P0 on the tool
    float3 A = a, Pi = P;
    #pragma unroll 1
    for (int i = 0; i < 100; i++)                    // Algorithm 1 line 10
    {
        A = ClosestOnTriangle(a, b, c, P);           // A_n = SDF(Triangle, P_n)
        float3 Pn = ClosestOnTool(toolTri, toolCount, toolShift, s, e, A);   // P_{n+1} = SDF(Tool, A_n)
        float dPrev = length3(P - A);
        float dNew  = length3(Pn - A);
        Pi = P;                                       // capture P_i BEFORE the update (line 15)
        P = Pn;
        if (fabsf(dPrev - dNew) < eps) break;        // |Delta| < eps
    }
    // SIGNED constraint-plane distance (diagnosis wik7z4vrv MODE 1/2 — the 穿模 root cause). The paper's
    // SDF(Tool,A) is a SIGNED field; the axis-UNSIGNED |Pi-A| can't tell inside from outside, so a tool
    // deeper than one radius read "no penetration" -> proxy tunneled + HUD "free". Sign it with the
    // OUTWARD face normal: signedDist = dot(Pi-A, fn) is + outside / - inside the tissue. The tool
    // (radius r) penetrates when its surface crosses the face plane: signedDist < r. depth = r-signedDist
    // grows past r once the axis is inside; the outward push is the face normal (the classic god-object
    // constraint half-space — also gives tangential surface-sliding when combined with the attraction).
    // Requires OUTWARD-wound skin (recon convention, D15). WINDING GUARD (review MAJOR 2): sign fn
    // against the recon's smoothed vertex normal (outward — matches the correctly-shaded render) so an
    // isolated inward-wound stitch can't flip the sign and drive the proxy INTO the tissue.
    float3 fn = cross3(b - a, c - a); float fl = length3(fn);
    fn = (fl > 1e-9f) ? fn / fl : make_float3(0.f, 1.f, 0.f);
    float3 vn = nrm[3*t+0] + nrm[3*t+1] + nrm[3*t+2];
    if (dot3(fn, vn) < 0.f) fn = make_float3(-fn.x, -fn.y, -fn.z);
    float signedDist = dot3(Pi - A, fn);
    float depth = radius - signedDist;                 // >0 penetrating; >radius once the axis is inside
    float3 vms = fn * depth;                            // outward push; |vms| == depth (the dent reads it)
    // SELECT by the paper's ||D|| = ||Pi-A|| (true point distance, Algorithm 1 line 13-17), NOT the
    // plane offset (review MAJOR 1): when A clamps to an edge/vertex, a FAR tangent face has a small
    // plane offset but large ||Pi-A|| and must not outrank the true-nearest face. Detection is still the
    // SIGNED test (signedDist < radius + margin, D18). Key = ||Pi-A|| among penetrating faces.
    float axisToTri = length3(Pi - A);
    float key = (signedDist < radius + captureMargin) ? axisToTri : 1e30f;
    triVms[t] = make_float4(vms.x, vms.y, vms.z, key);
}

// GPU argmin of the NON-NEGATIVE key (__float_as_uint is order-preserving for >= 0).
__global__ void k_ReduceVmsKey(int triCount, const float4* triVms, unsigned int* outKey)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= triCount) return;
    atomicMin(outKey, __float_as_uint(triVms[t].w));
}
__global__ void k_PickVmsTri(int triCount, const float4* triVms, unsigned int winKey, int* outTri)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= triCount) return;
    if (__float_as_uint(triVms[t].w) == winKey) atomicMin(outTri, t);   // lowest tri index wins ties
}

// Point-in-body detection (diagnosis wpoirpnc5 #1 / review w13ag9ld8): the penetration key (gated by
// signedDist<radius) OVER-REPORTS far faces behind their infinite planes on a closed body, so "any
// penetrating face" is true almost everywhere. Instead find the geometrically NEAREST skin face
// (min ||Pi-A||, UNGATED) and store its SIGNED distance; the physical is inside/in-contact iff the
// NEAREST face reports it on the inside within a radius (signedDist < radius). Output: (.x=signedDist,
// .w=axisToTri key for the argmin). Correct point-in-body test for a convex-ish liver.
__global__ void k_NearestSigned(int triCount, const float3* soup, const float4* aux, const float3* nrm,
                                float3 s, float3 e, const float3* toolTri, int toolCount, float3 toolShift,
                                float4* out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= triCount) return;
    if (aux[3*t+0].w > 0.5f || aux[3*t+1].w > 0.5f || aux[3*t+2].w > 0.5f)
    { out[t] = make_float4(1e30f, 0.f, 0.f, 1e30f); return; }        // skin only; cut walls never nearest
    float3 a = soup[3*t+0], b = soup[3*t+1], c = soup[3*t+2];
    float3 P = ClosestOnTool(toolTri, toolCount, toolShift, s, e, (a + b + c) * (1.f / 3.f));
    float3 A = a, Pi = P;
    #pragma unroll 1
    for (int i = 0; i < 100; i++) {
        A = ClosestOnTriangle(a, b, c, P);
        float3 Pn = ClosestOnTool(toolTri, toolCount, toolShift, s, e, A);
        float dPrev = length3(P - A), dNew = length3(Pn - A);
        Pi = P; P = Pn;
        if (fabsf(dPrev - dNew) < 1e-3f) break;
    }
    float3 fn = cross3(b - a, c - a); float fl = length3(fn);
    fn = (fl > 1e-9f) ? fn / fl : make_float3(0.f, 1.f, 0.f);
    float3 vn = nrm[3*t+0] + nrm[3*t+1] + nrm[3*t+2];
    if (dot3(fn, vn) < 0.f) fn = make_float3(-fn.x, -fn.y, -fn.z);
    out[t] = make_float4(dot3(Pi - A, fn), 0.f, 0.f, length3(Pi - A));   // signedDist, key=axisToTri
}

// ============================================================================
// Contact patch (paper SS2.3 line 917-919 "particles of the nearby triangle FACES" plural +
// SS3.2 line 985-987 "reaction ... applied to UPDATE THE PARTICLE POSITIONS"). Diagnosis
// wf_c7c6efc2 D1: the force-into-slope reaction alone is invisible at any gravity-safe structural
// stiffness (static dent settles well under a visible threshold); the paper's dent is the
// POSITION-level update, penetration-scaled, RESTORED 2026-07-07 audit-round-2 (see file header).
// Pass 1: count penetrating triangles + sum their depths (for depth-weighted force split).
// Pass 2: per penetrating triangle, for each of its 3 verts -> nearest ACTIVE corner:
//   (a) FORCE: atomicAdd reaction * (depth_t/sumDepth) / 3 into hapticForce (feeds the XPBD predict
//       step's external force accumulator -- an independent, gradual elastic-propagation effect);
//   (b) POSITION: atomicAdd disp = contactBeta * depth_t * (-nOut_t) / 3 into a displacement
//       accumulator (applied clamped by k_ApplyContactDisp -- the visible, IMMEDIATE dent, a
//       separate, bounded non-penetration effect). Both (a) and (b) coexist, matching the last
//       validated pre-retirement architecture (commit 869cd4d) -- (a) is not double-counting (b);
//       they are two distinct effects (gradual elastic settling vs. immediate bounded surface push).
// ============================================================================
__global__ void k_CountPenetrating(int triCount, const float4* triVms, float proximityMax,
                                   int* outCount, float* outDepthSum)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= triCount) return;
    float4 vt = triVms[t];
    // Gate on PROXIMITY (key = ||Pi-A|| < radius+margin), NOT just "penetrating" (diagnosis wpoirpnc5 /
    // review w13ag9ld8): the signed key marks FAR faces behind their planes as penetrating on a closed
    // body, so the dent would scatter onto the OPPOSITE side of the liver and starve the near dent. Only
    // faces the tool actually touches (within radius) dent.
    if (vt.w >= proximityMax) return;
    atomicAdd(outCount, 1);
    float depth = length3(make_float3(vt.x, vt.y, vt.z)); // depth = |vms| (key is now the selection dist)
    atomicAdd(outDepthSum, depth);
}

__global__ void k_ScatterContact(int triCount, const float4* triVms, const float3* soup,
                                 float3 reaction, float depthSum, float contactBeta, float proximityMax,
                                 float dentDepthMax,
                                 const float3* cornerPos, const int* active, const int* pinned,
                                 int cornerCount, float3* hapticForce, float3* contactDisp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;       // one thread per (triangle, vertex-lane)
    int t = i / 3, lane = i % 3;
    if (t >= triCount) return;
    float4 vt = triVms[t];
    if (vt.w >= proximityMax) return;                    // proximity gate (only near faces dent; see above)
    float3 vms = make_float3(vt.x, vt.y, vt.z);
    float vlen = length3(vms);
    float depth = vlen;                                  // depth = |vms| (the key is the selection dist)
    float3 nOut = (vlen > 1e-9f) ? vms / vlen : make_float3(0.f, 1.f, 0.f);

    float3 v = soup[3 * t + lane];
    int best = -1; float bd = 1e30f;
    for (int cptr = 0; cptr < cornerCount; cptr++) {
        // Skip frozen/severed (slope discards them) AND pinned corners: pinned particles are the
        // anchored boundary condition (freezes their position and never restores it), so a position
        // displacement there would PLASTICALLY drift the anchor. Route both the force and the dent
        // to the nearest DEFORMABLE corner (D11 "nearest active corner"), which the physics then
        // elastically recovers.
        if (active[cptr] == 0 || pinned[cptr] != 0) continue;
        float3 d = cornerPos[cptr] - v; float dd = dot3(d, d);
        if (dd < bd) { bd = dd; best = cptr; }
    }
    if (best < 0) return;

    // (a) depth-weighted thresholded reaction force (paper's Eq27 atomic-add rule) -- feeds the XPBD
    // predict step's external force (gradual elastic propagation through the surrounding mesh).
    float w = (depthSum > 1e-9f) ? (depth / depthSum) : 0.f;
    float3 fShare = reaction * (w / 3.f);
    atomicAdd(&hapticForce[best].x, fShare.x);
    atomicAdd(&hapticForce[best].y, fShare.y);
    atomicAdd(&hapticForce[best].z, fShare.z);

    // (b) position update: the paper's SS3.2 BOUNDED non-penetration constraint (update particle
    // positions to PREVENT penetration), NOT an unbounded plastic crater (BUG-A, diagnosis wf_6703c058).
    // depth = radius - signedDist grows PAST the radius once the physical axis is inside, so scaling the
    // dent by the raw depth punched a single-frame hole that, with recovery killed + soft Ks, carved a
    // growing crater far deeper/wider than the thin jaw. Clamp the DENT depth to the tool radius so the
    // per-frame push is bounded and the recon-rebuild feedback converges the surface to the tool
    // penetration WITHOUT overshoot. (The FORCE weighting w above keeps the true depth -- force is
    // separately capped by reactionMax.)
    float dentDepth = fminf(depth, dentDepthMax);
    float3 dShare = nOut * (-contactBeta * dentDepth / 3.f);
    atomicAdd(&contactDisp[best].x, dShare.x);
    atomicAdd(&contactDisp[best].y, dShare.y);
    atomicAdd(&contactDisp[best].z, dShare.z);
}

// Apply the accumulated contact displacement to the LIVE corner positions, clamped to
// contactDispMax per frame; damp the velocity component along the push so springs do not
// slingshot the patch back ("ensuring smooth collision forces", SS3.2 line 986-987).
__global__ void k_ApplyContactDisp(int cornerCount, const float3* contactDisp, float dispMax,
                                   const int* pinned, float3* cornerPos, float3* vel)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cornerCount) return;
    if (pinned[c] != 0) return;                          // anchored BC: never displaced by contact (defense-in-depth)
    float3 d = contactDisp[c];
    float len = length3(d);
    if (len < 1e-9f) return;
    if (len > dispMax) d = d * (dispMax / len);          // per-frame clamp
    cornerPos[c] = cornerPos[c] + d;
    // Make the dent PERSISTENT (diagnosis wpoirpnc5 deform #2): a stiff spring can over-restore a dent
    // and REBOUND it back through the surface -- violent jitter. Killing only the opposing velocity
    // component left the tangential velocity to rebuild it. Strongly damp the WHOLE contact-corner
    // velocity so the dent holds and reads as a clean dent.
    vel[c] = vel[c] * 0.15f;
}

// ============================================================================
// Host API
// ============================================================================
static int Gh(int n) { int g = (n + 63) / 64; return g < 1 ? 1 : g; }

int haptics_init(int cornerCount, int triCapacity, const HapticParams* p)
{
    if (g_ready) haptics_shutdown();
    if (!p || cornerCount <= 0 || triCapacity <= 0) return -4000;
    g_hp = *p;
    g_cornerCount = cornerCount;
    g_triCap = triCapacity;

    HK(cudaMalloc(&d_triVms, (size_t)triCapacity * sizeof(float4)));
    HK(cudaMalloc(&d_vmsKey, sizeof(unsigned int)));
    HK(cudaMalloc(&d_vmsTri, sizeof(int)));
    HK(cudaMalloc(&d_contactDisp, (size_t)cornerCount * sizeof(float3)));
    HK(cudaMemset(d_contactDisp, 0, (size_t)cornerCount * sizeof(float3)));
    HK(cudaMalloc(&d_penCount, sizeof(int)));
    HK(cudaMalloc(&d_depthSum, sizeof(float)));
    g_toolTriCap = 1024;   // grasper collision soup is ~224 tris; cap generously
    HK(cudaMalloc(&d_toolTri, (size_t)3 * g_toolTriCap * sizeof(float3)));
    g_toolTriCount = 0; g_toolCenter = make_float3(0, 0, 0); g_toolBoundR = 0.f;

    c_cornerPos   = recon_corner_pos();
    c_hapticForce = (float3*)physics_haptic_force();
    c_active      = physics_active();
    c_pinned      = physics_pinned();
    c_vel         = (float3*)physics_vel();
    if (!c_cornerPos || !c_hapticForce || !c_active || !c_pinned || !c_vel) { haptics_shutdown(); return -4001; }

    g_physQuat = make_float4(0, 0, 0, 1); g_proxyQuat = make_float4(0, 0, 0, 1);
    g_force = make_float3(0, 0, 0); g_torque = make_float3(0, 0, 0);
    g_inContact = false; g_minTri = -1;
    g_hasProxy = false; g_havePrev = false;   // persistent-proxy latch + prev-pose start fresh
    g_slipping = false; g_graspMu = 1.f;      // friction cone starts stuck, jaws open
    g_ready = true;
    return 0;
}

void haptics_set_physical_tool(const float* physS3, const float* physE3, const float* physQuat4,
                              const float* linVel3, const float* angVel3, float toolRadius)
{
    if (!g_ready) return;
    g_physS   = make_float3(physS3[0], physS3[1], physS3[2]);
    g_physE   = make_float3(physE3[0], physE3[1], physE3[2]);
    g_physPos = (g_physS + g_physE) * 0.5f;                          // P_A = capsule midpoint
    g_physQuat = make_float4(physQuat4[0], physQuat4[1], physQuat4[2], physQuat4[3]);  // q_A
    g_linVel  = make_float3(linVel3[0], linVel3[1], linVel3[2]);
    g_angVel  = make_float3(angVel3[0], angVel3[1], angVel3[2]);
    g_toolRadius = toolRadius;
}

// Upload the tool COLLISION MESH (BUG 2): triWorld = 3*3*triCount floats = world-space verts at the
// PHYSICAL pose (C# GrasperRig transforms the low-poly collision soup by the same mount as the display
// grasper). Also compute the bounding sphere (centroid + radius) for the per-frame AABB-lite cull. Pass
// triCount=0 to fall back to the capsule. Call BEFORE haptics_set_physical_tool each frame.
void haptics_set_tool_mesh(const float* triWorld, int triCount)
{
    if (!g_ready) return;
    if (triCount <= 0 || triCount > g_toolTriCap || !triWorld) { g_toolTriCount = 0; return; }
    int nv = 3 * triCount;
    double cx = 0, cy = 0, cz = 0;
    for (int i = 0; i < nv; i++) { cx += triWorld[3*i+0]; cy += triWorld[3*i+1]; cz += triWorld[3*i+2]; }
    cx /= nv; cy /= nv; cz /= nv;
    float r2 = 0.f;
    for (int i = 0; i < nv; i++) {
        float dx = triWorld[3*i+0]-(float)cx, dy = triWorld[3*i+1]-(float)cy, dz = triWorld[3*i+2]-(float)cz;
        float dd = dx*dx + dy*dy + dz*dz; if (dd > r2) r2 = dd;
    }
    g_toolCenter = make_float3((float)cx, (float)cy, (float)cz);
    g_toolBoundR = sqrtf(r2);
    if (cudaMemcpy(d_toolTri, triWorld, (size_t)nv * sizeof(float3), cudaMemcpyHostToDevice) != cudaSuccess)
    { g_toolTriCount = 0; return; }
    g_toolTriCount = triCount;
}

// Detection at the PHYSICAL pose with a per-frame capture band (D18): true iff some skin triangle is
// within (radius + margin) of the physical axis. Detection-only widening; separation passes margin 0.
static bool haptics_detect(const float3* soup, const float4* aux, const float3* nrm, int triCount, float margin)
{
    // Tool mesh was uploaded at the PHYSICAL pose -> query at the physical pose has zero shift; the cull
    // sphere is the uploaded mesh centroid g_toolCenter (BUG 2).
    k_ClosestPerTriangle<<<Gh(triCount), 64>>>(triCount, soup, aux, nrm,
        g_physS, g_physE, g_toolRadius, margin, g_hp.eps,
        d_toolTri, g_toolTriCount, make_float3(0,0,0), g_toolCenter, g_toolBoundR, d_triVms);
    unsigned int initKey = 0xFFFFFFFFu;
    cudaMemcpy(d_vmsKey, &initKey, sizeof(unsigned int), cudaMemcpyHostToDevice);
    k_ReduceVmsKey<<<Gh(triCount), 64>>>(triCount, d_triVms, d_vmsKey);
    unsigned int winKey = 0; cudaMemcpy(&winKey, d_vmsKey, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    float wk; memcpy(&wk, &winKey, sizeof(float));
    // wk = min ||Pi-A|| among faces the signed test flagged. Require the NEAREST such face to be within
    // the contact band (review wtij2sh5k MAJOR 1): the signed test alone marks FAR faces behind their
    // planes on a closed body, so "any penetrating face" false-triggers a rising edge while HOVERING in
    // free space -> phantom buzz on approach. Proximity gate kills that; a fast plunge still passes via
    // the widened margin (max(captureMargin,|physDisp|)) the caller provides.
    return wk < g_toolRadius + margin;
}

// Point-in-body test at the PHYSICAL pose: the NEAREST skin face's signed distance < radius (review
// w13ag9ld8 blocker fix). Correct for a convex-ish liver: a deep-inside physical -> nearest face
// signedDist<0 (in contact at any depth); a physical dragged out the side / plunged THROUGH to below ->
// its nearest face reports it OUTSIDE (signedDist>radius) -> release, killing the phantom-force latch.
static bool haptics_phys_inside(const float3* soup, const float4* aux, const float3* nrm, int triCount)
{
    k_NearestSigned<<<Gh(triCount), 64>>>(triCount, soup, aux, nrm, g_physS, g_physE,
        d_toolTri, g_toolTriCount, make_float3(0,0,0), d_triVms);   // physical pose -> zero shift
    unsigned int initKey = 0xFFFFFFFFu; int initTri = 0x7FFFFFFF;
    cudaMemcpy(d_vmsKey, &initKey, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vmsTri, &initTri, sizeof(int), cudaMemcpyHostToDevice);
    k_ReduceVmsKey<<<Gh(triCount), 64>>>(triCount, d_triVms, d_vmsKey);      // min axisToTri = nearest face
    unsigned int winKey = 0; cudaMemcpy(&winKey, d_vmsKey, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    k_PickVmsTri<<<Gh(triCount), 64>>>(triCount, d_triVms, winKey, d_vmsTri);
    int minTri = 0; cudaMemcpy(&minTri, d_vmsTri, sizeof(int), cudaMemcpyDeviceToHost);
    if (minTri < 0 || minTri >= triCount) return false;                     // no skin -> outside
    float4 win; cudaMemcpy(&win, &d_triVms[minTri], sizeof(float4), cudaMemcpyDeviceToHost);
    return win.x < g_toolRadius;   // nearest-face signedDist < radius -> physical is inside/in-contact
}

// Algorithm 1 outer separation loop on the CURRENT (persistent) proxy pose, at the true radius
// (captureMargin=0). Pushes the proxy out by a*v_ms until ||v_ms||<eps or outerMax. Sets g_minTri.
// Returns true if it reached the surface (converged / no penetration), false if it hit the cap (fast
// motion or lost support after a cut) -> caller re-seeds to physical and separates again.
static bool haptics_separate(const float3* soup, const float4* aux, const float3* nrm, int triCount)
{
    g_minTri = -1;
    for (int outer = 0; outer < g_hp.outerMax; outer++)   // Algorithm 1 while True (D10 cap)
    {
        // Proxy pose: the tool is the uploaded (physical-pose) mesh translated by (proxy - physical),
        // since q_B=q_A (D14). Shift the query + the cull sphere by that same delta.
        float3 pshift = g_proxyPos_dev - g_physPos;
        k_ClosestPerTriangle<<<Gh(triCount), 64>>>(triCount, soup, aux, nrm,
            g_proxyS, g_proxyE, g_toolRadius, 0.f, g_hp.eps,
            d_toolTri, g_toolTriCount, pshift, g_toolCenter + pshift, g_toolBoundR, d_triVms);
        unsigned int initKey = 0xFFFFFFFFu; int initTri = 0x7FFFFFFF;
        cudaMemcpy(d_vmsKey, &initKey, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vmsTri, &initTri, sizeof(int), cudaMemcpyHostToDevice);
        k_ReduceVmsKey<<<Gh(triCount), 64>>>(triCount, d_triVms, d_vmsKey);
        unsigned int winKey = 0; cudaMemcpy(&winKey, d_vmsKey, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        k_PickVmsTri<<<Gh(triCount), 64>>>(triCount, d_triVms, winKey, d_vmsTri);
        int minTri = 0; cudaMemcpy(&minTri, d_vmsTri, sizeof(int), cudaMemcpyDeviceToHost);
        if (minTri < 0 || minTri >= triCount) return true;      // no supporting tri -> proxy is free/out
        float4 win; cudaMemcpy(&win, &d_triVms[minTri], sizeof(float4), cudaMemcpyDeviceToHost);
        if (win.w >= 1e29f) return true;                        // no penetration -> on the surface
        g_minTri = minTri;
        float3 vMs = make_float3(win.x, win.y, win.z);
        float vl = length3(vMs);
        if (vl < g_hp.eps) return true;                         // converged ||v_ms|| < eps
        g_lastNormal = vMs * (1.f / vl);                        // outward surface normal (cone axis, friction)
        float3 step = vMs * g_hp.a;                             // P_t += a * v_ms
        g_proxyS = g_proxyS + step; g_proxyE = g_proxyE + step; g_proxyPos_dev = g_proxyPos_dev + step;
    }
    return false;   // hit outerMax without reaching the surface
}

int haptics_step()
{
    if (!g_ready) return 0;
    // Buffer lifecycle (BLOCKER fix): haptics OWNS the zeroing - clear last frame's reaction now,
    // AFTER the previous frame's LCS_Step consumed it, BEFORE this frame's scatter.
    HK(cudaMemset(c_hapticForce, 0, (size_t)g_cornerCount * sizeof(float3)));

    int vc = 0, idxCount = 0;
    recon_get_counts(&vc, &idxCount);                 // existing API - NOT a duplicate getter
    int triCount = idxCount / 3;
    if (triCount > g_triCap) triCount = g_triCap;

    // PERSISTENT finger-proxy (D13 fix): the proxy is a god-object carried frame-to-frame that always
    // stays on the surface (paper lines 788, 915-916). Each frame: detect contact at the physical pose
    // (widened band, D18), then FREE -> proxy tracks physical (dP=0); RISING EDGE -> seed + separate to
    // the surface; IN CONTACT -> bounded attraction toward physical (D19) then Algorithm 1 separation.
    const float3* soup = recon_expand_pos();
    const float4* aux  = recon_expand_aux();
    const float3* nrm  = recon_expand_nrm();   // per-vertex outward normal (winding guard, review MAJOR 2)
    g_minTri = -1;
    g_proxyQuat = g_physQuat;    // q_B = q_A (D14: Algorithm 1 outputs position only, no orient advance)

    if (triCount > 0)
    {
        if (!g_hasProxy)
        {
            // FREE: detect approach at the physical pose with the per-frame widened band (D18) — this
            // is the ONLY use of physical-pose detection (rising edge). Once latched, persistence (not
            // re-detection) holds contact, so a DEEP physical (which reads FREE past one radius) does
            // NOT drop contact.
            float physDisp = g_havePrev ? length3(g_physPos - g_prevPhysPos) : 0.f;
            float margin = fmaxf(g_hp.captureMargin, physDisp);   // D18: a fast plunge is always captured
            if (!haptics_detect(soup, aux, nrm, triCount, margin)) {
                g_proxyS = g_physS; g_proxyE = g_physE; g_proxyPos_dev = g_physPos;   // free: dP=0 -> F=0
            } else {
                // Rising edge: seed on the NEAR side — last frame's physical, which was OUTSIDE the band
                // (we were free), NOT the current (possibly >1-radius-deep) physical. A seed at a
                // fast-plunge physical would be interior-beyond-radius: haptics_separate finds no
                // penetrating triangle and cannot move it out -> stranded interior (review A1 major).
                // From the near-side seed the bounded attraction then walks the proxy in over frames,
                // always on the surface, never tunneling.
                float3 seedC = g_havePrev ? g_prevPhysPos : g_physPos;
                float3 off = seedC - g_physPos;
                g_proxyS = g_physS + off; g_proxyE = g_physE + off; g_proxyPos_dev = seedC;
                haptics_separate(soup, aux, nrm, triCount);
                g_hasProxy = true;
                // Birth of the friction contact anchor (Chai3D): stick to the seed point, not slipping.
                g_stickAnchor = g_proxyPos_dev; g_slipping = false;
                float3 dp0 = g_proxyPos_dev - g_physPos; float dl0 = length3(dp0);
                if (dl0 > 1e-6f) g_lastNormal = dp0 * (1.f / dl0);   // rising-edge normal fallback
            }
        }
        else
        {
            // IN CONTACT (persistent god-object + FRICTION CONE, Chai3D research weo6zf2bt): instead of
            // attracting the proxy straight to the physical (which leaves dP normal-only = frictionless),
            // stick a world-space contact anchor tangentially and attract to the CONE-LIMITED goal, so dP
            // keeps a tangential (friction) component = the paper's implicit tangential Kp*dP. Then the
            // usual bounded step + Algorithm 1 separation put the proxy back on the surface.
            float3 pbS = g_proxyS, pbE = g_proxyE, pbC = g_proxyPos_dev;   // last good surface proxy
            float3 n = g_lastNormal;                                       // outward surface normal (cone axis)
            float3 D = g_physPos - g_stickAnchor;                          // anchor -> physical goal
            float dn = dot3(D, n);                                         // physical is inside -> dn < 0
            float3 Dn = n * dn;                                            // normal follow
            float3 Dt = D - Dn;                                            // tangential offset from the anchor
            float rt = length3(Dt);                                       // tangential slip distance
            // Normal penetration = the LIVE proxy(on surface)-vs-physical depth, NOT the drifting world
            // anchor (review w5i8uv8tw minor): the proxy tracks the dented surface so grip does not creep.
            float depth = fmaxf(dot3(g_proxyPos_dev - g_physPos, n), 0.f);
            float theta = atan2f(rt, fmaxf(depth, 1e-6f));                // cone angle
            float muEff = g_graspMu;                                       // jaws closed -> stronger grip
            if (g_slipping) { if (theta < atanf(g_hp.muK * muEff) * 0.6f) g_slipping = false; }  // hysteresis
            else            { if (theta > atanf(g_hp.muS * muEff))        g_slipping = true;  }
            float mu = (g_slipping ? g_hp.muK : g_hp.muS) * muEff;
            // Cap the normal force the cone sees at one tool radius (diagnosis wpoirpnc5 #2): past a
            // radius the axis is fully inside, so grip must SATURATE, not grow with depth — else a deep
            // tool gets a superhuman ~depth*mu stick radius and dP explodes to several units.
            float cappedDepth = fminf(depth, g_toolRadius);
            float frictionRadius = cappedDepth * mu;                      // Coulomb cone radius (bounded)
            if (rt > frictionRadius && rt > 1e-6f)                        // SLIP: chase the goal, leave
                g_stickAnchor = g_stickAnchor + Dt * (1.f - frictionRadius / rt);  // frictionRadius behind
            // (STICK: rt <= frictionRadius -> anchor frozen -> the tool grips/drags the tissue.)
            float3 constrainedGoal = g_stickAnchor + Dn;                  // full normal follow + limited tangential

            float3 d = constrainedGoal - g_proxyPos_dev; float dl = length3(d);
            if (dl > 1e-6f) {
                float3 sA = d * (fminf(1.f, g_hp.proxyTrackStep / dl) * 0.5f);   // 0.5 relaxation (chatter, #3)
                g_proxyS = g_proxyS + sA; g_proxyE = g_proxyE + sA; g_proxyPos_dev = g_proxyPos_dev + sA;
            }
            if (!haptics_separate(soup, aux, nrm, triCount)) {
                // Did not reach the surface (fast motion / supporting tri cut away) -> re-seed to the
                // LAST GOOD SURFACE proxy (near side) + re-anchor, then separate again.
                g_proxyS = pbS; g_proxyE = pbE; g_proxyPos_dev = pbC;
                haptics_separate(soup, aux, nrm, triCount);
                g_stickAnchor = g_proxyPos_dev; g_slipping = false;
            }
            // EXIT to free ONLY when the PHYSICAL is no longer penetrating the tissue (paper Algorithm 1 +
            // prose 915-916: the proxy is a PERSISTENT god-object that stays on the surface for the FULL
            // duration the physical is inside; the ONLY free/coincide condition is the physical leaving the
            // body). Diagnosis wf_395c900a (3 paper derivations high-conf, adversarial verify 2/2) — THE
            // shared root cause of BUG1/BUG2/BUG3: the old `|| |proxyPos-physPos| < eps` disjunct released
            // the god-object whenever |dP| momentarily collapsed WHILE the physical was still buried (a
            // straight deep press makes dP nearly pure-normal and small; the D19 attraction can over-track
            // the proxy onto the physical). Releasing there snapped the proxy onto the DEEP physical ->
            // dP=0 => F=0 forever (BUG1; FREE cannot re-detect a >1-radius-deep physical), the displayed
            // proxy tunneled onto the physical (BUG2), and the |dP|-gated dent never ran (BUG3). Release is
            // now the physical-inside test ALONE — never a dP magnitude. When |dP| collapses while the
            // physical is still inside we KEEP the proxy (the separation above already re-pinned it to the
            // surface) so dP re-inflates to the true penetration depth next frame.
            bool physInside = haptics_phys_inside(soup, aux, nrm, triCount);   // NEAREST-face test (not any-face)
            if (!physInside) {
                g_proxyS = g_physS; g_proxyE = g_physE; g_proxyPos_dev = g_physPos;
                g_hasProxy = false; g_slipping = false;
            }
        }
    }
    else {
        g_proxyS = g_physS; g_proxyE = g_physE; g_proxyPos_dev = g_physPos; g_hasProxy = false;
    }

    // Eq 22-25
    float3 dP = g_proxyPos_dev - g_physPos;                            // Eq 22
    // Contact predicate for FORCE/DAMPING/DENT: the PERSISTENT god-object is "in contact" for as long as
    // the proxy is held on the surface (i.e. the physical is penetrating) — NOT gated on |dP|>eps
    // (diagnosis wf_395c900a, BUG3). A momentarily small |dP| (proxy transiently coinciding with a still-
    // buried physical) must NOT zero the force AND the dent. F = Kp*dP is naturally 0 when dP==0, so no
    // phantom force results; the reaction/dent below re-evaluates the true penetration at the physical
    // pose independent of |dP|. hasProxy is true iff the physical is inside (release is !physInside only),
    // so this is exactly the paper's "reaction applied while the tool penetrates" (SS2.3 917-919 / SS3.2).
    g_inContact = g_hasProxy;
    float4 dQ = quatMul(g_proxyQuat, quatConj(g_physQuat));            // Eq 23
    float theta = 2.f * acosf(fminf(fmaxf(dQ.w, -1.f), 1.f));          // Eq 24
    float3 axisK = make_float3(0, 0, 0);                               // Eq 25
    float sh = sinf(theta * 0.5f);
    if (fabsf(sh) > 1e-6f) axisK = make_float3(dQ.x, dQ.y, dQ.z) * (1.f / sh);
    // Eq 27 (damping only in contact; no contact => dP==0 => F==0)
    float dpd = g_inContact ? g_hp.dP : 0.f;
    float dod = g_inContact ? g_hp.dO : 0.f;
    g_force  = dP * g_hp.kP - g_linVel * dpd;
    g_torque = axisK * (g_hp.kO * theta) - g_angVel * dod;
    // Reaction to the tissue (only in contact) over the WHOLE contact patch — plural "nearby
    // triangle faces" (SS2.3 line 917-919): thresholded force (depth-weighted) into the RK45-era
    // slope term (now the XPBD predict step's external force) + the paper's SS3.2 position update
    // (the visible dent), applied clamped with spring-back velocity damping. RESTORED 2026-07-07
    // audit-round-2 (see file header for why the earlier Task-1 retirement no longer applies).
    if (g_inContact && triCount > 0)
    {
        float3 reaction = make_float3(-g_force.x, -g_force.y, -g_force.z);
        float rlen = length3(reaction);
        if (rlen > g_hp.reactionMax && rlen > 1e-9f) reaction = reaction * (g_hp.reactionMax / rlen);

        // Re-evaluate penetration at the PHYSICAL tool pose (captureMargin=0, true radius): the dent
        // must track the PHYSICAL tool's penetration (dent depth ~ |dP|, diagnosis D1). The widened
        // detection band NEVER reaches this pass — the margin is a per-call arg (D18).
        k_ClosestPerTriangle<<<Gh(triCount), 64>>>(triCount, recon_expand_pos(), recon_expand_aux(),
            recon_expand_nrm(), g_physS, g_physE, g_toolRadius, 0.f, g_hp.eps,
            d_toolTri, g_toolTriCount, make_float3(0,0,0), g_toolCenter, g_toolBoundR, d_triVms);   // physical pose

        // The dent follows the ENTRY patch to ANY depth (review wtij2sh5k MAJOR 2) without scattering onto
        // the far side: gate on ||Pi-A|| within a band of the NEAREST penetrating face, not a fixed
        // radius cutoff. minDist = the nearest penetrating face's distance (the entry surface); a face
        // dents iff ||Pi-A|| < minDist + radius+margin. Shallow press -> minDist~radius (near patch); deep
        // press -> minDist=depth so the entry patch still dents while the opposite face (depth+thickness)
        // stays excluded.
        unsigned int initK = 0xFFFFFFFFu;
        HK(cudaMemcpy(d_vmsKey, &initK, sizeof(unsigned int), cudaMemcpyHostToDevice));
        k_ReduceVmsKey<<<Gh(triCount), 64>>>(triCount, d_triVms, d_vmsKey);
        unsigned int wkD = 0; HK(cudaMemcpy(&wkD, d_vmsKey, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        float minDist; memcpy(&minDist, &wkD, sizeof(float));
        // Footprint band tightened to the JAW (BUG-A): only faces within one tool radius of the nearest
        // penetrating face dent, so the crater tracks the thin jaw instead of a captureMargin-widened
        // patch. (captureMargin is a fast-plunge DETECTION widening, D18 — it must not spread the dent.)
        float proximityMax = (minDist < 1e29f ? minDist : 0.f) + g_toolRadius;
        HK(cudaMemset(d_penCount, 0, sizeof(int)));
        HK(cudaMemset(d_depthSum, 0, sizeof(float)));
        k_CountPenetrating<<<Gh(triCount), 64>>>(triCount, d_triVms, proximityMax, d_penCount, d_depthSum);
        float depthSum = 0.f;
        HK(cudaMemcpy(&depthSum, d_depthSum, sizeof(float), cudaMemcpyDeviceToHost));

        HK(cudaMemset(d_contactDisp, 0, (size_t)g_cornerCount * sizeof(float3)));
        int lanes = 3 * triCount;
        k_ScatterContact<<<Gh(lanes), 64>>>(triCount, d_triVms, recon_expand_pos(),
                                            reaction, depthSum, g_hp.contactBeta, proximityMax,
                                            g_toolRadius,   // BUG-A: dent depth clamp = one tool radius
                                            c_cornerPos, c_active, c_pinned, g_cornerCount,
                                            c_hapticForce, d_contactDisp);
        k_ApplyContactDisp<<<Gh(g_cornerCount), 64>>>(g_cornerCount, d_contactDisp,
                                                      g_hp.contactDispMax, c_pinned, c_cornerPos, c_vel);
    }
    g_prevPhysPos = g_physPos; g_havePrev = true;   // for next frame's per-frame displacement band (D18)
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

// Native contact state (persistent-proxy predicate) — read by C# instead of |F|>eps (A1.6).
int haptics_get_contact() { return g_inContact ? 1 : 0; }

// Grasp input (jaws): 0 = open (normal friction), 1 = closed (strong grip). Scales the friction mu so a
// closed grasper sticks harder and drags the tissue (grasping = the same §2.3 reaction, boosted friction).
void haptics_set_grasp(float grasp01)
{
    float g = grasp01 < 0.f ? 0.f : (grasp01 > 1.f ? 1.f : grasp01);
    g_graspMu = 1.f + g * 5.f;   // closed jaws -> 6x friction cone -> firm grip/drag
}

void haptics_get_force(float* outForce3, float* outTorque3)
{
    if (outForce3)  { outForce3[0]  = g_force.x;  outForce3[1]  = g_force.y;  outForce3[2]  = g_force.z;  }
    if (outTorque3) { outTorque3[0] = g_torque.x; outTorque3[1] = g_torque.y; outTorque3[2] = g_torque.z; }
}

void haptics_get_proxy(float* outPos3, float* outQuat4)
{
    if (outPos3)  { outPos3[0] = g_proxyPos_dev.x; outPos3[1] = g_proxyPos_dev.y; outPos3[2] = g_proxyPos_dev.z; }
    if (outQuat4) { outQuat4[0] = g_proxyQuat.x; outQuat4[1] = g_proxyQuat.y; outQuat4[2] = g_proxyQuat.z; outQuat4[3] = g_proxyQuat.w; }
}

void haptics_shutdown()
{
    cudaFree(d_triVms); d_triVms = nullptr;
    cudaFree(d_vmsKey); d_vmsKey = nullptr;
    cudaFree(d_vmsTri); d_vmsTri = nullptr;
    cudaFree(d_contactDisp); d_contactDisp = nullptr;
    cudaFree(d_penCount); d_penCount = nullptr;
    cudaFree(d_depthSum); d_depthSum = nullptr;
    cudaFree(d_toolTri); d_toolTri = nullptr; g_toolTriCount = 0; g_toolTriCap = 0;
    c_cornerPos = nullptr; c_hapticForce = nullptr; c_active = nullptr; c_pinned = nullptr; c_vel = nullptr;
    g_ready = false;
}

#include "training/haptics_state.cuh"
