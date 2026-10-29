#include "cutting/pressure_cut_state.cuh"
// cut.cu — native CUDA port of the Stage 3 cutting kernels (Cutting.compute).
//
// CUDA implementation of the Stage 3 cutting kernels.
// Dispatch order mirrors CutDetector.cs (DetectCut->SeverLinks) and DualContouring.cs (the cut-FP chain).
// Paper SS2.1.1 (Fig 2.4/2.5 cut surface) + SS2.1.3 (Fig 2.8 / Eq6-8 Moller-Trumbore).

#include "cut.cuh"
#include "cut.h"
#include "dc_recon.h"     // recon_* getters (cornerPos/isect/.../tri) + recon_set_cut_buffers
#include "physics.h"      // physics_* getters (particleRot/nbrIdx/bendPairs)
#include "common.cuh"     // IsectGpu, VoxId
#include <stdlib.h>
#include <string.h>
#include <vector>

// Bending pair (32B) — mirrors physics.cu BendPairGpu; SeverLinks writes alive=0.
struct BendPairGpuC { int i, j, k; float theta0; int alive, _p0, _p1, _p2; };
static_assert(sizeof(BendPairGpuC) == 32, "BendPairGpuC must be 32 bytes");

// Tool geometry passed by value to k_DetectCut (built per cut_detect from the host CutToolDesc).
struct ToolGpu
{
    float3 t1v0, t1v1, t1v2, t2v0, t2v1, t2v2;
    float3 ncut, aabbMin, aabbMax;
    float  d;
    int    valid;
};

// ── Stage 6 detection (paper SS2.1.3 + SS2.1.2; task_plan.md Stage 6) ─────────────────────────
// TWO exact world-space cadences replace the Stage-4 material-map (whose inverse tore at real folds).
// OPERATOR-SPLITTING NOTE (re-adjudication wf_dbf90148): the blade is frozen at frame granularity
// during the physics phase, so the cut sheet's PLACEMENT carries a splitting displacement bounded by
// one frame of relative blade-tissue motion (no holes, no separation failure; shrinks with frame
// rate — the same splitting error accepted engine-wide).
//   rod phase  (cut_detect, tissue frozen): Moller-Trumbore of the DEFORMED edge vs the swept world
//              quad — the paper's SS2.1.3 detection, proven in zero-g.
//   tick phase (cut_ribbon_tick, rod frozen): CONTINUOUS COLLISION of the MOVING deformed edge
//              against the static blade segment — the exact space-time crossing (coplanarity is
//              QUADRATIC in t for a static blade), so falling/oscillating tissue cannot tunnel and
//              there is no global inverse map to tear. SS2.1.2 recording (alongRest) unchanged.

// ── Cut-owned device buffers ─────────────────────────────────────────────────────────────────
static GridEdgeGpu2*  d_gridEdges      = nullptr;  // [gridEdgeCount]
static float3*        d_prevCorner     = nullptr;  // [cornerCount] corner positions at the previous tick (CCD)
static Conn4096Gpu*   d_conn4096       = nullptr;  // [4096]
static unsigned int*  d_voxelOccupied  = nullptr;  // [voxelCount] (1/0)
static unsigned int*  d_voxelCutMask   = nullptr;  // [voxelCount] (cumulative, 12-bit)
static CutPointGpu*   d_cutPoint       = nullptr;  // [cutPointCapacity]
static unsigned int*  d_cutPointCounter= nullptr;  // [1] (cumulative)
static float4*        d_cutFP          = nullptr;  // [8*voxelCount]
static float*         d_edgeCutT       = nullptr;  // first cut fraction in each directed edge
__device__ float*     edgeCutTDevice   = nullptr;
__device__ const float* edgeSkinTDevice = nullptr;
__device__ const int* edgeInsideDevice = nullptr;
__device__ const float3* materialRestDevice = nullptr;
float* cut_edge_fraction(){return d_edgeCutT;}
static int*           d_cornerInside   = nullptr;  // [cornerCount] 1 = phi<0 (Stage 7 air-edge gate)
static float*         d_cutFPAccumPos  = nullptr;  // [3*8*voxelCount]
static int*           d_cutFPAccumCnt  = nullptr;  // [8*voxelCount]
static unsigned int*  d_voxelFPCount   = nullptr;  // [voxelCount]
static int*           d_cutFPNormal    = nullptr;  // [3*8*voxelCount]
static float3*        d_cutFPNormalF   = nullptr;  // [8*voxelCount]
// v4.1 buffers (plan review wf_b08c3fc2):
static int*           d_cutNrmAccum    = nullptr;  // [3*8*voxelCount] per-slot world cut-normal accumulator (P2 kerf)
static unsigned int*  d_dbg            = nullptr;  // [16] [DEBUG-CUT] counters (P3): slots 0-14 zeroed each cut_fp_chain, 15 cumulative.
static int*           d_graspRing      = nullptr;  // [cornerCount] design §6/§9-(6) tear exemption: 1 = a corner
                                                   // in the GRASPED neighbourhood -> any incident edge is EXEMPT
                                                   // from the tau*L tear law. ZERO-FILLED at cut_init (INERT until
                                                   // step-8 grasp BFS-dilates d_grasped into it via cut_grasp_ring());
                                                   // k_TearOverstretch reads it.
static int*           d_graspRingTmp   = nullptr;  // [cornerCount] ping-pong scratch for the BFS dilation hops
static int*           d_nbrIdxAtInit   = nullptr;  // [6*cornerCount] topology SNAPSHOT at cut_init (design §6
                                                   // rim-bridge carve-out: a corner that has EVER lost a
                                                   // neighbour vs this snapshot is a cut-rim corner and must
                                                   // keep the tau*L law regardless of grasp-ring proximity)
static int*           d_edgeWasExempt  = nullptr;  // [gridEdgeCount] design §6 release-decay: 1 = this edge was
                                                   // EVER exempted by the ring (persists after release until
                                                   // the edge relaxes below 0.8*tau*L or the safety timeout)
static int*           d_edgeDecayTicks = nullptr;  // [gridEdgeCount] ticks spent decaying since release (reset
                                                   // to 0 while actively in-ring; ~2s safety timeout, design §6)
static const int      TEAR_DECAY_TIMEOUT_TICKS = 2000;  // ~2s at the shipped N=20/physicsDt=0.02 (design §6:
                                                   // "a safety timeout of ~2s in case an edge never relaxes")
// d_dbg slot map: 0=emaPairs 1=jump>L 2=jump>3L 3=maxDisp(f-as-u) 4=slotNew 5=epochInval 6=dangleRefs
//                 7=giantWall 8=giantMixed 9=giantSkin 10=maxEdgeLen(f-as-u) 11=rim<3 12=pinch1
//                 13=all6severed 14=frozenThisFrame 15=tearMarks(CUMULATIVE — incremented during
//                 LCS_Step's ribbon ticks, so the per-chain memset covers slots 0-14 ONLY; zeroing
//                 15 mid-frame would erase every tick-phase tear before readback, wf_c416ae77 R2-m3)

// ── Cached shared pointers (from dc_recon + physics; stable after their init) ─────────────────
static float3*       c_cornerPos       = nullptr;
static const float3* c_cornerRest      = nullptr;   // immutable rest positions
static IsectGpu*     c_isect           = nullptr;
static float3*       c_isectWorld      = nullptr;
static int*          c_voxelIsectOffset= nullptr;
static int*          c_voxelIsectCount = nullptr;
static int*          c_voxelCorner     = nullptr;   // 8 corner ids/voxel (cut-FP Eq5 box-SDF clamp)
static int*          c_tri             = nullptr;
static unsigned int* c_triCounter      = nullptr;
static RotGpu*       c_particleRot     = nullptr;
static int*          c_nbrIdx          = nullptr;
static BendPairGpuC* c_bendPairs       = nullptr;
static int*          c_active          = nullptr;   // live material flags

static CutInitDesc g_cdesc = {};
static int3        g_cdims = {};
static CutToolDesc g_tool  = {};
extern "C" const int* xpbd_blade_phase_device();
extern "C" const float3* xpbd_blade_sweep_device();
static int g_pressureCut=0;
static float4* d_pendingCrossings=nullptr;
// Material piece of every corner, and the (links, piece) owner key of the envelope release (k_SpreadShellOwners).
static int* d_materialPieces=nullptr;
static unsigned long long* d_shellOwner=nullptr;
extern "C" int xpbd_blade_sweep_count();
// Geometry immediately before this pressure gesture first contacted tissue.
// The blade can push the skin ahead of its centreline without crossing it.
// Record that entry path in material coordinates, then commit ONLY at fracture.
static float3* d_entryCorner=nullptr;
static RotGpu* d_entryRotation=nullptr;
static int* d_entryState=nullptr; // 0: tracking free tissue, 1: preload, 2: consumed
// Scissors cut level of every material column, the lattice line along the
// column axis: the rest coordinate on that axis at which a snip severed it,
// stored at the column's corner of index 0 on the axis. COLUMN_UNSET marks a
// column no snip has severed. The column axis is the lattice axis nearest the
// shear plane normal of the first snip, -1 before it.
static unsigned* d_columnLevel=nullptr;
static int* d_columnAxis=nullptr;
#define COLUMN_UNSET 0x7F7F7F7Fu
extern "C" const int* xpbd_blade_had_contact_device();
__global__ void k_PreparePressureEntry(int* state,const int* phase,const int* hadContact){
    if(*phase==BladeWithdrawing || *phase==BladeClearStroke)*state=0;
    else if(*state==0 && hadContact && *hadContact)*state=1;
}
__global__ void k_RefreshPressureEntry(int n,const int* state,const float3* pos,const RotGpu* rot,float3* entry,RotGpu* entryRot){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && *state==0){entry[i]=pos[i];entryRot[i]=rot[i];}
}
__global__ void k_FinishPressureEntry(int* state,const int* phase,bool keepEntry){
    if(*state==1 && bladeCommits(*phase) && !keepEntry)*state=2;
}
void cut_set_pressure_mode(int enabled){
    int next=enabled!=0;
    if(next!=g_pressureCut && d_pendingCrossings)
        cudaMemset(d_pendingCrossings,0,g_cdesc.gridEdgeCount*sizeof(float4));
    if(next!=g_pressureCut && d_entryState){
        cudaMemset(d_entryState,0,sizeof(int));
        if(next && c_cornerPos && c_particleRot){
            cudaMemcpy(d_entryCorner,c_cornerPos,g_cdesc.cornerCount*sizeof(float3),cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_entryRotation,c_particleRot,g_cdesc.cornerCount*sizeof(RotGpu),cudaMemcpyDeviceToDevice);
        }
    }
    g_pressureCut=next;
}

static bool        g_cut_ready = false;

extern "C" int cut_read_edge_fractions(float* output){
    if(!g_cut_ready || !output)return -1;
    return -(int)cudaMemcpy(output,d_edgeCutT,(size_t)6*g_cdesc.cornerCount*sizeof(float),cudaMemcpyDeviceToHost);
}

// ── Stage-6 host state ────────────────────────────────────────────────────────────────────────
static bool   g_toolSet        = false;          // at least one cut_set_tool received
static bool   g_havePrevCorner = false;          // d_prevCorner holds the previous tick's positions
static float3 g_emitNcut  = { 0.f, 1.f, 0.f };   // last VALID world cut normal (for the ±D/2 side)
static float  g_tearLen2  = 0.f;                 // (tau*L)^2, 0 = tear law disabled (v5.1 D10)

static int Gc(int n) { int g = (n + 63) / 64; return g < 1 ? 1 : g; }
static float3 mk3(const float* a) { return make_float3(a[0], a[1], a[2]); }

// ─────────────────────────────────────────────────────────────────────────────────────────────
// EmitCutPoint (device) — record one cut point in the owner's MATERIAL frame (paper SS2.1.2).
//
// Per SS2.1.2 the cut-point local coordinate is recorded "relative to the two vertices of the edge
// ... based on the condition that the edge has NOT undergone any deformation." So we store the
// UNDEFORMED (rest) offset from the owner along the edge, `alongRest = t_owner * L * axisDir`
// (a scalar t along the particle's coordinate axis), PLUS the +-D/2 cut-wall gap expressed in the
// owner's material frame (R^T * side*D/2*n_cut). Reconstruction (AccumulateCutFP) applies the CURRENT
// rotation: world = cornerPos[owner] + R * localOffset — "the new cut point is computed using the
// corresponding coordinate axes" (SS2.1.2). The +-D/2 gap is along the world cut normal AT EMISSION
// TIME and thereafter tracks the owner's frame (R_recon*R_emit^T*ncut) — intended SS2.1.2 behavior;
// note mid-frame ribbon ticks record with the previous frame's R (sub-voxel cosmetic skew, D<<L).
// ─────────────────────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void EmitCutPoint(float3 P_hit, float3 alongRest, int p, int edgeId,
                                             float3 ncut, float D,
                                             const float3* cornerPos, const RotGpu* particleRot,
                                             CutPointGpu* cutPoint, unsigned int* cutPointCounter,
                                             int cutPointCapacity)
{
    float3 pPos = cornerPos[p];
    float  side = signf(dot3(pPos - P_hit, ncut));        // which side of the WORLD cut plane the owner is on
    if (side == 0.0f) side = 1.0f;                        // on the plane -> +n_cut side

    unsigned int slot = atomicAdd(&cutPointCounter[0], 1u);
    if (slot >= (unsigned int)cutPointCapacity) return;   // capacity guard

    RotGpu R = particleRot[p];
    CutPointGpu cp;
    // UNDEFORMED-edge local coord + cut-gap in owner material frame (SS2.1.2). See header note.
    cp.localOffset   = alongRest + mulRtv(R, (side * (D * 0.5f)) * ncut);
    cp.ownerParticle = p;
    cp.edgeId        = edgeId;
    // v4.1 P2: material-frame OUTWARD cut normal (side-signed, unit). All cut points of one
    // (voxel,component) slot share the owner's side, so the per-slot world sum is coherent for a
    // single stroke; multi-stroke cancellation is guarded at the projection site.
    cp.matNrm        = mulRtv(R, side * ncut);
    cutPoint[slot]   = cp;
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// MarkAndEmit — shared mark+record tail for BOTH detection cadences: set the cut bit in all 4
// OCCUPIED sharing voxels (RC2); on the FIRST cut of the edge, record the two SS2.1.2 cut points
// on the undeformed edge (alongRest = t*L*axis / (t-1)*L*axis, reconstructed via the current
// particle axes by AccumulateCutFP).
// ─────────────────────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void MarkAndEmit(int e, int axis, int3 baseC, int3 axisDir, int idA, int idB,
                                            float t_ray, float3 P_hit, float3 ncut, float toolD,
                                            float voxelL, int3 dims, const float3* cornerPos,
                                            unsigned int* voxelCutMask, const unsigned int* voxelOccupied,
                                            CutPointGpu* cutPoint, unsigned int* cutPointCounter,
                                            int cutPointCapacity, const RotGpu* particleRot)
{
    bool firstCut = false, sawAnyVox = false, edgeOccupied = false;
    for (int s = 0; s < 4; s++)
    {
        int3 vc = baseC + ES_VoxelOffset(axis, s);
        if (!VoxInBounds(vc, dims)) continue;
        int vid = VoxId(vc, dims);
        if (voxelOccupied[vid] == 0u) continue;                 // RC2: no cut bits in empty voxels
        edgeOccupied = true;
        unsigned int cutBit = (unsigned int)(1 << ES_LocalEdge(axis, s));
        unsigned int old = atomicOr(&voxelCutMask[vid], cutBit);
        if (!sawAnyVox) { sawAnyVox = true; firstCut = (old & cutBit) == 0u; }
    }
    if (firstCut && edgeOccupied)
    {
        if(edgeCutTDevice){
            edgeCutTDevice[DirectedEdgeSlot(idA,idB,dims)]=t_ray;
            edgeCutTDevice[DirectedEdgeSlot(idB,idA,dims)]=1.f-t_ray;
        }
        // Keep connectivity marks in boundary voxels, but do not create tissue
        // surface samples for a blade crossing the AIR interval of a lattice edge.
        if(edgeSkinTDevice && edgeInsideDevice && !CutHitsMaterial(t_ray,
            edgeSkinTDevice[DirectedEdgeSlot(idA,idB,dims)],edgeInsideDevice[idA],edgeInsideDevice[idB]))return;
        float3 axisF  = make_float3((float)axisDir.x, (float)axisDir.y, (float)axisDir.z);
        float3 restEdge = materialRestDevice ? materialRestDevice[idB]-materialRestDevice[idA] : axisF*voxelL;
        float3 alongA = restEdge*t_ray;
        float3 alongB = restEdge*(t_ray-1.f);
        EmitCutPoint(P_hit, alongA, idA, e, ncut, toolD, cornerPos, particleRot, cutPoint, cutPointCounter, cutPointCapacity);
        EmitCutPoint(P_hit, alongB, idB, e, ncut, toolD, cornerPos, particleRot, cutPoint, cutPointCounter, cutPointCapacity);
    }
}

// Save only actual geometric crossings while pressure is below threshold.
// Material edge coordinates survive deformation until fracture is committed.
// Withdrawal / free-space exit discards the uncommitted stroke.
__device__ inline void RecordPressureCrossing(int e,int a,int b,int3 dims,float t,float3 normal,
 const RotGpu* rotations,float4* pending){
    bool replace=pending[e].w==0;
    if(!replace && edgeSkinTDevice && edgeInsideDevice){
        float skin=edgeSkinTDevice[DirectedEdgeSlot(a,b,dims)];
        // A preload may first sweep the air part of a boundary edge and only
        // later cross tissue. Do not let that early air witness suppress the
        // real material incision when all pending crossings are committed.
        replace=!CutHitsMaterial(pending[e].w-1.f,skin,edgeInsideDevice[a],edgeInsideDevice[b]) &&
                 CutHitsMaterial(t,skin,edgeInsideDevice[a],edgeInsideDevice[b]);
    }
    if(replace){float3 n=mulRtv(rotations[a],normal);pending[e]=make_float4(n.x,n.y,n.z,t+1.f);}
}
__device__ inline void PressureMarkAndEmit(int e,int axis,int3 baseC,int3 axisDir,int idA,int idB,float t,float3 hit,float3 normal,float D,float L,int3 dims,
 const float3* pos,unsigned* mask,const unsigned* occupied,CutPointGpu* points,unsigned* count,int capacity,const RotGpu* rotations,
 const int* phase,float4* pending){
    if(phase && !bladeCommits(*phase)){
        if(*phase==0 && pending)RecordPressureCrossing(e,idA,idB,dims,t,normal,rotations,pending);
        return;
    }
    MarkAndEmit(e,axis,baseC,axisDir,idA,idB,t,hit,normal,D,L,dims,pos,mask,occupied,points,count,capacity,rotations);
}
__global__ void k_CommitPending(int n,const int* phase,float4* pending,const GridEdgeGpu2* edges,int3 dims,float D,float L,const float3* pos,
 unsigned* mask,const unsigned* occupied,CutPointGpu* points,unsigned* count,int capacity,const RotGpu* rotations){
    int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=n||!phase||!pending)return;
    if(*phase==2||*phase==4){pending[e]=make_float4(0,0,0,0);return;}
    float4 h=pending[e];if(!bladeCommits(*phase)||h.w==0)return;
    GridEdgeGpu2 ge=edges[e];int3 bc=make_int3(ge.bcX,ge.bcY,ge.bcZ),dir=make_int3(ge.axis==0,ge.axis==1,ge.axis==2);
    int a=CorIdCuda(bc,dims),b=CorIdCuda(bc+dir,dims);float t=h.w-1.f;
    float3 normal=mulRv(rotations[a],make_float3(h.x,h.y,h.z));float length=length3(normal);if(length>1e-10f)normal=normal/length;
    MarkAndEmit(e,ge.axis,bc,dir,a,b,t,pos[a]+t*(pos[b]-pos[a]),normal,D,L,dims,pos,mask,occupied,points,count,capacity,rotations);
    pending[e]=make_float4(0,0,0,0);
}
// Whether the scissor edge from a to b (closed hinge frame: tip, opening) has
// come within `reach` of the in-plane point (x, y) of the present blade frame,
// or passed over it, while it closed from the opening angle `open` where the
// stroke began down to `now`. The blade turns by side * opening angle.
__device__ inline bool scissorsSwept(float x,float y,float3 a,float3 b,float side,float now,float open,float reach){
    float dx=b.x-a.x,dy=b.y-a.y,r2=x*x+y*y,A=dx*dx+dy*dy,B=2*(a.x*dx+a.y*dy),C=a.x*a.x+a.y*a.y-r2;
    float disc=B*B-4*A*C;if(A<1e-12f || disc<0)return false;
    float root=sqrtf(disc),q=(-B+root)/(2*A);
    if(q<0 || q>1){q=(-B-root)/(2*A);if(q<0 || q>1)return false;}
    float angle=side*(atan2f(y,x)-atan2f(a.y+q*dy,a.x+q*dx));
    return angle>=now-reach*rsqrtf(r2) && angle<=open;
}
__device__ inline int ColumnOf(int3 corner,int axis,int3 dims){
    if(axis==0)corner.x=0;else if(axis==1)corner.y=0;else corner.z=0;
    return CorIdCuda(corner,dims);
}
__device__ inline float AlongAxis(float3 v,int axis){return axis==0?v.x:axis==1?v.y:v.z;}
// The crack crosses a link where the blade plane does. A crossing exactly on a
// node would put the level on that node, between two links of the column, and
// neither would part: it stays a thousandth of the link inside.
__device__ inline float CrackFraction(float t){return fminf(fmaxf(t,1e-3f),1.f-1e-3f);}
// Scissors cut levels (see cut_ribbon_tick). plane: the present shear plane
// (normal, point); frame: sweep[10..17] of the blade.
__global__ void k_ColumnAxis(const float3* plane,const int* phase,const int* entryState,int* axis){
    float3 n=plane[0];float x=fabsf(n.x),y=fabsf(n.y),z=fabsf(n.z);
    if(*axis<0 && bladeCommits(*phase) && *entryState==1)*axis=x>=y && x>=z?0:y>=z?1:2;
}
// A material link along the column axis, with a node inside the material,
// sets its column's level once its present segment crosses the present shear
// plane at a point an edge has swept, unless the column already has one.
__global__ void k_LevelScissorsColumns(int n,int3 dims,float reach,const GridEdgeGpu2* edges,const int* nbrIdx,const int* inside,
 const float3* rest,const float3* pos,const float3* plane,const float3* frame,const int* phase,const int* entryState,
 const int* columnAxis,unsigned* columnLevel){
    int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=n || !bladeCommits(*phase) || *entryState!=1)return;
    GridEdgeGpu2 ge=edges[e];int axis=*columnAxis;if(ge.axis!=axis)return;
    int3 bc=make_int3(ge.bcX,ge.bcY,ge.bcZ),dir=make_int3(axis==0,axis==1,axis==2);
    int a=CorIdCuda(bc,dims),b=CorIdCuda(bc+dir,dims);
    if(nbrIdx[6*a+2*axis+1]<0 || (!inside[a] && !inside[b]))return;
    float da=dot3(pos[a]-plane[1],plane[0]),db=dot3(pos[b]-plane[1],plane[0]);
    if((da>0)==(db>0))return;
    float t=da/(da-db);
    float3 m=pos[a]+t*(pos[b]-pos[a])-frame[0];
    float length=frame[3].y,now=asinf(fminf(.95f,fmaxf(0.f,frame[3].x)/(2*length))),open=asinf(fminf(.95f,frame[3].z/(2*length)));
    float x=dot3(m,frame[1]),y=dot3(m,frame[2]);
    if(!scissorsSwept(x,y,frame[4],frame[5],1.f,now,open,reach) && !scissorsSwept(x,y,frame[6],frame[7],-1.f,now,open,reach))return;
    float ya=AlongAxis(rest[a],axis),yb=AlongAxis(rest[b],axis),level=ya+CrackFraction(t)*(yb-ya);
    // A level in the last material link of a run of material nodes along the column cuts the end node off alone. It
    // keeps only its links across the column, and where the neighbouring columns are cut a layer away, those part at
    // the steps too: the node becomes an island that the envelope release sets free. The level moves into the
    // envelope link beyond that node instead, at the material surface, so the node stays with its run.
    if(inside[a] && inside[b]){
        int3 above=bc+dir+dir,below=bc-dir;
        if(above.x<=dims.x && above.y<=dims.y && above.z<=dims.z && !inside[CorIdCuda(above,dims)] && nbrIdx[6*b+2*axis+1]>=0)
            level=yb+CrackFraction(0.f)*(AlongAxis(rest[CorIdCuda(above,dims)],axis)-yb);
        else if(below.x>=0 && below.y>=0 && below.z>=0 && !inside[CorIdCuda(below,dims)] && nbrIdx[6*CorIdCuda(below,dims)+2*axis+1]>=0){
            float yc=AlongAxis(rest[CorIdCuda(below,dims)],axis);level=yc+CrackFraction(1.f)*(ya-yc);
        }
    }
    atomicCAS(columnLevel+ColumnOf(bc,axis,dims),COLUMN_UNSET,__float_as_uint(level));
}
// A skin column, the ring of lattice corners just outside the material, takes
// the level of a neighbouring material column, side by side or diagonal across
// the column axis, where that level lies on one of its skin links. A skin
// column beside the material thus mirrors its neighbour and both crack walls
// reach the skin in the same voxel. Levels never pass from skin to skin: the
// skin wraps the whole tissue, and the crack would run around it.
__global__ void k_SpreadScissorsLevels(int n,int3 dims,const GridEdgeGpu2* edges,const int* nbrIdx,const int* inside,
 const float3* rest,const int* phase,const int* entryState,const int* columnAxis,unsigned* columnLevel){
    int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=n || !bladeCommits(*phase) || *entryState!=1)return;
    GridEdgeGpu2 ge=edges[e];int axis=*columnAxis;if(ge.axis!=axis)return;
    int3 bc=make_int3(ge.bcX,ge.bcY,ge.bcZ),dir=make_int3(axis==0,axis==1,axis==2);
    int a=CorIdCuda(bc,dims),b=CorIdCuda(bc+dir,dims);
    int column=ColumnOf(bc,axis,dims);
    if(nbrIdx[6*a+2*axis+1]<0 || inside[a] || inside[b] || columnLevel[column]!=COLUMN_UNSET)return;
    float ya=AlongAxis(rest[a],axis),yb=AlongAxis(rest[b],axis);
    int p=(axis+1)%3,q=(axis+2)%3;
    for(int du=-1;du<=1;du++)for(int dv=-1;dv<=1;dv++){
        if(!du && !dv)continue;
        int3 c=bc;
        if(p==0)c.x+=du;else if(p==1)c.y+=du;else c.z+=du;
        if(q==0)c.x+=dv;else if(q==1)c.y+=dv;else c.z+=dv;
        if(c.x<0 || c.y<0 || c.z<0 || c.x>dims.x || c.y>dims.y || c.z>dims.z)continue;
        int ca=CorIdCuda(c,dims),cb=CorIdCuda(c+dir,dims);
        if(!inside[ca] && !inside[cb])continue;
        unsigned level=columnLevel[ColumnOf(c,axis,dims)];float h=__uint_as_float(level);
        if(level!=COLUMN_UNSET && h>ya && h<yb){atomicCAS(columnLevel+column,COLUMN_UNSET,level);return;}
    }
}
// Every link the column levels pass through parts: one along the column axis
// where its column's level lies on it, and one across it where its rest layer
// lies between the levels of the two columns it joins, which is where the
// wound steps from one material layer to another. Such a step wall faces along
// the link it crosses.
__global__ void k_CutScissorsLevels(int n,int3 dims,const GridEdgeGpu2* edges,const int* nbrIdx,const float3* rest,
 const float3* pos,const float3* plane,const int* phase,const int* entryState,const RotGpu* rotations,const int* columnAxis,
 const unsigned* columnLevel,float4* pending){
    int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=n || !bladeCommits(*phase) || *entryState!=1)return;
    int axis=*columnAxis;if(axis<0)return;
    GridEdgeGpu2 ge=edges[e];int3 bc=make_int3(ge.bcX,ge.bcY,ge.bcZ),dir=make_int3(ge.axis==0,ge.axis==1,ge.axis==2);
    int a=CorIdCuda(bc,dims),b=CorIdCuda(bc+dir,dims);
    unsigned levelA=columnLevel[ColumnOf(bc,axis,dims)];
    if(nbrIdx[6*a+2*ge.axis+1]<0 || levelA==COLUMN_UNSET)return;
    float la=__uint_as_float(levelA);
    if(ge.axis==axis){
        float ya=AlongAxis(rest[a],axis),yb=AlongAxis(rest[b],axis);
        if(la>ya && la<yb)RecordPressureCrossing(e,a,b,dims,(la-ya)/(yb-ya),plane[0],rotations,pending);
        return;
    }
    unsigned levelB=columnLevel[ColumnOf(bc+dir,axis,dims)];if(levelB==COLUMN_UNSET)return;
    float lb=__uint_as_float(levelB),h=AlongAxis(rest[a],axis);
    if((h-la)*(h-lb)>=0)return;
    float3 along=pos[b]-pos[a];
    RecordPressureCrossing(e,a,b,dims,CrackFraction((h-la)/(lb-la)),along*rsqrtf(fmaxf(dot3(along,along),1e-20f)),rotations,pending);
}
__device__ inline ToolGpu PressureSweep(ToolGpu tool,const float3* sweep){
    if(!sweep)return tool;
    tool.t1v0=tool.t2v0=sweep[0];tool.t1v1=sweep[1];tool.t1v2=tool.t2v1=sweep[3];tool.t2v2=sweep[2];
    float3 n=cross3(sweep[1]-sweep[0],sweep[3]-sweep[0]);float length=length3(n);
    tool.valid=length>1e-8f;tool.ncut=tool.valid?n/length:make_float3(0,0,0);
    float3 lo=sweep[0],hi=lo;for(int j=1;j<4;j++){float3 v=sweep[j];lo=make_float3(fminf(lo.x,v.x),fminf(lo.y,v.y),fminf(lo.z,v.z));hi=make_float3(fmaxf(hi.x,v.x),fmaxf(hi.y,v.y),fmaxf(hi.z,v.z));}
    tool.aabbMin=lo;tool.aabbMax=hi;return tool;
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// DetectCutQuad — ROD-SUBSTEP phase (tissue frozen; called via cut_detect). Paper SS2.1.3 verbatim:
// Moller-Trumbore (Eq6-8) of the swept cutting plane (2 world triangles, prev+cur tool frames)
// against the CURRENT DEFORMED grid edges. This is the architecture proven by the perfect zero-g
// cut. On hit t: alongRest = t*L*axis is the SS2.1.2 undeformed-edge record.
// ─────────────────────────────────────────────────────────────────────────────────────────────
__global__ void k_DetectCutQuad(int gridEdgeCount, ToolGpu tool, float voxelL, int3 dims,
                                int cutPointCapacity, const int* cornerInside,
                                const float3* cornerPos, const GridEdgeGpu2* gridEdges,
                                unsigned int* voxelCutMask, const unsigned int* voxelOccupied,
                                CutPointGpu* cutPoint, unsigned int* cutPointCounter,
                                const RotGpu* particleRot,const int* phase=nullptr,float4* pending=nullptr,const float3* sweep=nullptr,const int* entryState=nullptr)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= gridEdgeCount || (phase && !bladeDetects(*phase))) return;
    if(entryState && *entryState!=1)return;
    tool=PressureSweep(tool,sweep);
    if (tool.valid == 0) return;

    GridEdgeGpu2 ge = gridEdges[e];
    int axis = ge.axis;
    int3 baseC = make_int3(ge.bcX, ge.bcY, ge.bcZ);
    int3 axisDir = make_int3(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0, axis == 2 ? 1 : 0);

    int idA = CorIdCuda(baseC, dims);
    int idB = CorIdCuda(baseC + axisDir, dims);

    // Mark boundary-voxel connectivity for every crossed edge. MarkAndEmit records
    // physical cut samples only in the material interval, using the original skin
    // crossing; reconstruction and skin stitching use that same interval ownership.

    // SS2.1.3 / Eq6: ray = the CURRENT DEFORMED grid edge, tris = the world swept quad.
    float3 O    = cornerPos[idA];
    float3 Dvec = cornerPos[idB] - O;

    // cull vs the tool AABB (pre-built by MakeToolDesc; inflated by L)
    float pad = voxelL;
    float3 eMin = make_float3(fminf(O.x, O.x + Dvec.x), fminf(O.y, O.y + Dvec.y), fminf(O.z, O.z + Dvec.z));
    float3 eMax = make_float3(fmaxf(O.x, O.x + Dvec.x), fmaxf(O.y, O.y + Dvec.y), fmaxf(O.z, O.z + Dvec.z));
    if (eMax.x < tool.aabbMin.x - pad || eMin.x > tool.aabbMax.x + pad ||
        eMax.y < tool.aabbMin.y - pad || eMin.y > tool.aabbMax.y + pad ||
        eMax.z < tool.aabbMin.z - pad || eMin.z > tool.aabbMax.z + pad) return;

    float t1, u1, v1, t2, u2, v2;
    bool hit1 = MollerTrumbore(O, Dvec, tool.t1v0, tool.t1v1, tool.t1v2, t1, u1, v1);
    bool hit2 = MollerTrumbore(O, Dvec, tool.t2v0, tool.t2v1, tool.t2v2, t2, u2, v2);
    if (!hit1 && !hit2) return;

    float t_ray = (hit1 && hit2) ? fminf(t1, t2) : (hit1 ? t1 : t2);
    t_ray = fminf(fmaxf(t_ray, 0.f), 1.f);
    float3 P_hit = O + t_ray * Dvec;          // world hit ON the deformed edge (no offset)

    if(entryState){
        RecordPressureCrossing(e,idA,idB,dims,t_ray,tool.ncut,particleRot,pending);
        return;
    }
    PressureMarkAndEmit(e, axis, baseC, axisDir, idA, idB, t_ray, P_hit, tool.ncut, tool.d, voxelL, dims,
                cornerPos, voxelCutMask, voxelOccupied, cutPoint, cutPointCounter, cutPointCapacity,
                particleRot,phase,pending);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// DetectCutCCD — PHYSICS-TICK phase (rod frozen; called via cut_ribbon_tick after every accepted
// RK45 substep). Continuous collision of the MOVING deformed edge against the STATIC blade segment
// [S,E]: with the blade static and edge endpoints linear in tick-time t, the coplanarity function
//   f(t) = ((a(t)-S) x (b(t)-S)) . (E-S)
// is QUADRATIC in t — solved exactly, so falling/oscillating tissue cannot tunnel between ticks
// (the Stage-4 D2-blocker fix, now WITHOUT a global inverse map that tears at folds). At the root
// t*, validate the segment-segment intersection params (u along edge, v along blade) in the common
// plane. SS2.1.2 recording (alongRest = u*L*axis) unchanged.
// ─────────────────────────────────────────────────────────────────────────────────────────────
__global__ void k_DetectCutCCD(int gridEdgeCount, float3 S, float3 E, float3 emitNcut, float toolD,
                               float voxelL, int3 dims, int cutPointCapacity,
                               const float3* prevPos, const float3* curPos,
                               const GridEdgeGpu2* gridEdges, const int* nbrIdx, const int* cornerInside,
                               unsigned int* voxelCutMask, const unsigned int* voxelOccupied,
                               CutPointGpu* cutPoint, unsigned int* cutPointCounter,
                               const RotGpu* particleRot,const int* phase=nullptr,float4* pending=nullptr,const float3* sweep=nullptr)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= gridEdgeCount || (phase && !bladeDetects(*phase))) return;

    if(sweep){S=sweep[0];E=sweep[1];}
    GridEdgeGpu2 ge = gridEdges[e];
    int axis = ge.axis;
    int3 baseC = make_int3(ge.bcX, ge.bcY, ge.bcZ);
    int3 axisDir = make_int3(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0, axis == 2 ? 1 : 0);

    int idA = CorIdCuda(baseC, dims);
    int idB = CorIdCuda(baseC + axisDir, dims);

    // Already-severed edges cannot be cut again (wf_65901910 B2 minor): skipping them stops the CCD
    // from endlessly re-marking a fallen piece streaming past the blade (cut points/tris growth).
    if (nbrIdx[6 * idA + 2 * axis + 1] < 0) return;
    // AIR-EDGE RULE v3 — paper-literal mark AND emit (see k_DetectCutQuad).

    float3 a0 = prevPos[idA], a1 = curPos[idA];
    float3 b0 = prevPos[idB], b1 = curPos[idB];

    // cull: swept-edge AABB vs blade-segment AABB (+L pad)
    float pad = voxelL;
    float3 sMin = make_float3(fminf(S.x, E.x) - pad, fminf(S.y, E.y) - pad, fminf(S.z, E.z) - pad);
    float3 sMax = make_float3(fmaxf(S.x, E.x) + pad, fmaxf(S.y, E.y) + pad, fmaxf(S.z, E.z) + pad);
    float3 eMin = make_float3(fminf(fminf(a0.x, a1.x), fminf(b0.x, b1.x)),
                              fminf(fminf(a0.y, a1.y), fminf(b0.y, b1.y)),
                              fminf(fminf(a0.z, a1.z), fminf(b0.z, b1.z)));
    float3 eMax = make_float3(fmaxf(fmaxf(a0.x, a1.x), fmaxf(b0.x, b1.x)),
                              fmaxf(fmaxf(a0.y, a1.y), fmaxf(b0.y, b1.y)),
                              fmaxf(fmaxf(a0.z, a1.z), fmaxf(b0.z, b1.z)));
    if (eMax.x < sMin.x || eMin.x > sMax.x || eMax.y < sMin.y || eMin.y > sMax.y ||
        eMax.z < sMin.z || eMin.z > sMax.z) return;

    // f(t) = ((P0 + t*Pv) x (Q0 + t*Qv)) . D  with P=a(t)-S, Q=b(t)-S, D=E-S  — quadratic in t.
    float3 D  = E - S;
    float3 P0 = a0 - S, Pv = a1 - a0;
    float3 Q0 = b0 - S, Qv = b1 - b0;
    float  c0 = dot3(cross3(P0, Q0), D);
    float  c1 = dot3(cross3(P0, Qv) + cross3(Pv, Q0), D);
    float  c2 = dot3(cross3(Pv, Qv), D);

    // roots of c2 t^2 + c1 t + c0 = 0 in [0,1] (stable quadratic; degenerate -> linear).
    float roots[2]; int nr = 0;
    if (fabsf(c2) < 1e-14f)
    {
        if (fabsf(c1) > 1e-14f) { float r = -c0 / c1; if (r >= 0.f && r <= 1.f) roots[nr++] = r; }
        // c1,c2 ~ 0: f constant. Static edges are the rod phase's job. A PERSISTENTLY-COPLANAR edge
        // sliding through the blade within its own plane for a whole tick is skipped — a measure-zero
        // configuration under generic motion (any transverse perturbation breaks coplanarity next
        // tick); documented limitation (review S1-m2).
    }
    else
    {
        float disc = c1 * c1 - 4.f * c2 * c0;
        if (disc >= 0.f)
        {
            float sq = sqrtf(disc);
            float q  = -0.5f * (c1 + (c1 >= 0.f ? sq : -sq));
            float r0 = q / c2, r1 = (fabsf(q) > 1e-20f) ? c0 / q : r0;
            if (r0 > r1) { float tmp = r0; r0 = r1; r1 = tmp; }
            if (r0 >= 0.f && r0 <= 1.f) roots[nr++] = r0;
            if (r1 >= 0.f && r1 <= 1.f && r1 != r0) roots[nr++] = r1;
        }
    }

    for (int ri = 0; ri < nr; ri++)
    {
        float tc = roots[ri];
        float3 a = a0 + tc * Pv;                 // edge endpoints at crossing time
        float3 b = b0 + tc * Qv;
        float3 e1 = b - a;
        float3 n2 = cross3(e1, D);
        float  n2len2 = dot3(n2, n2);
        if (n2len2 < 1e-16f) continue;           // edge parallel to blade at t*
        float3 w = S - a;
        float u = dot3(cross3(w, D),  n2) / n2len2;   // param along the edge a->b
        float v = dot3(cross3(w, e1), n2) / n2len2;   // param along the blade S->E
        // Along the blade a small tolerance keeps its ends cutting. Along the edge the
        // crossing must lie on the edge, at its far node at most (see MollerTrumbore):
        // a node passing the blade line would otherwise part from all its neighbours.
        const float EPS = 2e-2f;
        if (u <= 0.f || u > 1.f || v < -EPS || v > 1.f + EPS) continue;

        float t_ray = u;
        // world hit for the ±D/2 side sign: the point on the CURRENT edge geometry.
        float3 P_hit = a1 + t_ray * (b1 - a1);
        // Emit normal (review S1-m1/S3-m3): the CCD cut wall's normal = blade direction x relative
        // tissue motion (the plane actually being cut) — correct even before any VALID rod sweep;
        // degenerate motion falls back to the last valid sweep normal.
        float3 relV = 0.5f * (Pv + Qv);
        float3 nHit = cross3(D, relV);
        float  nLen = length3(nHit);
        float3 nEmit = (nLen > 1e-8f) ? nHit / nLen : emitNcut;
        PressureMarkAndEmit(e, axis, baseC, axisDir, idA, idB, t_ray, P_hit, nEmit, toolD, voxelL, dims,
                    curPos, voxelCutMask, voxelOccupied, cutPoint, cutPointCounter, cutPointCapacity,
                    particleRot,phase,pending);
        return;                                   // first valid crossing cuts the edge
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// TearOverstretch (v5.1, deviation D10 — plan review wf_c416ae77 R1 approve / R2 amendments) —
// one thread per grid edge, launched per ribbon tick AFTER k_DetectCutCCD and BEFORE k_SeverLinks.
// THE 藕断丝连 FIX: rim bridge edges the swept ribbon never crossed have NO fracture law — the
// falling piece (restrained negligibly by 2-3 springs) drags their voxel chains 10-50L into the
// strand/fan artifacts (log: giantWall 0->45, maxEdge 2.15L->51L growing with fall). Tissue beyond
// its elastic limit TEARS: an intact active-active edge stretched past tau*L is marked+emitted via
// the NORMAL MarkAndEmit path (tear plane ⊥ the edge, hit at the midpoint), so conn4096 splits,
// FPs recover to the lips, the torn face renders a real wall, and the same tick's k_SeverLinks
// severs the spring+bends. tau default 4.0 (R2-m1: must clear the max legit gravity strain with
// margin; bridges blow through ANY tau within a frame or two at terminal velocity). Gated on
// g_toolSet (host) so an uncut liver can never tear.
// ─────────────────────────────────────────────────────────────────────────────────────────────
__global__ void k_TearOverstretch(int gridEdgeCount, float tearLen2, int3 dims, float voxelL,
                                  float toolD, const GridEdgeGpu2* gridEdges, const int* nbrIdx,
                                  const int* active, const int* graspRing, const float3* cornerPos,
                                  unsigned int* voxelCutMask, const unsigned int* voxelOccupied,
                                  CutPointGpu* cutPoint, unsigned int* cutPointCounter,
                                  int cutPointCapacity, const RotGpu* particleRot, unsigned int* dbg,
                                  int* edgeWasExempt, int* edgeDecayTicks)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= gridEdgeCount) return;

    GridEdgeGpu2 ge = gridEdges[e];
    int axis = ge.axis;
    int3 baseC = make_int3(ge.bcX, ge.bcY, ge.bcZ);
    int3 axisDir = make_int3(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0, axis == 2 ? 1 : 0);
    int idA = CorIdCuda(baseC, dims);
    int idB = CorIdCuda(baseC + axisDir, dims);

    if (nbrIdx[6 * idA + 2 * axis + 1] < 0) return;            // already severed
    if (active[idA] == 0 || active[idB] == 0) return;          // frozen/inactive corners carry no spring
    // Already mask-marked (e.g. by k_DetectCutCCD earlier in this SAME tick, before k_SeverLinks
    // ran): the edge is already a cut — MarkAndEmit would be a pure no-op (firstCut=false) and the
    // tears counter would over-count (final review W1-m2/W2-m2). Behavior-identical early-out.
    if (EdgeIsCut(axis, baseC, dims, voxelCutMask)) return;

    // R1-m3: an edge with NO occupied stencil voxel can never receive a mask bit, so k_SeverLinks
    // could never sever it and the tear would refire every tick — skip it entirely.
    bool anyOcc = false;
    for (int s = 0; s < 4; s++)
    {
        int3 vc = baseC + ES_VoxelOffset(axis, s);
        if (VoxInBounds(vc, dims) && voxelOccupied[VoxId(vc, dims)] != 0u) { anyOcc = true; break; }
    }
    if (!anyOcc) return;

    float3 A = cornerPos[idA], B = cornerPos[idB];
    float3 d = B - A;
    float  len2 = dot3(d, d);
    if(materialRestDevice){float3 r=materialRestDevice[idB]-materialRestDevice[idA];tearLen2*=dot3(r,r)/(voxelL*voxelL);}

    // design §6 guard 1: an edge in the GRASPED neighbourhood is deliberate large strain, NOT fracture —
    // exempt it from the tau*L tear law so a grasp tent does not decapitate at the grip / the next ring.
    // graspRing is ZERO until step-8 grasp populates it, so this branch is INERT (never taken) at step 6.
    // [audit rev-B RELEASE-DECAY]: an edge that was EVER in-ring keeps its exemption per-EDGE after
    // release until its length relaxes below 0.8*tau*L (decayLen2), or a ~2s safety timeout elapses (a
    // snagged edge then resumes the normal law and tears honestly) — else the memoryless per-substep law
    // would sever the just-released tent on its very first post-release ticks instead of springing back.
    bool inRingNow = (graspRing[idA] != 0 || graspRing[idB] != 0);
    if (inRingNow)
    {
        edgeWasExempt[e] = 1; edgeDecayTicks[e] = 0;           // reset the decay clock while actively held
        return;
    }
    if (edgeWasExempt[e] != 0)
    {
        float decayLen2 = 0.64f * tearLen2;                    // (0.8*tau*L)^2 = 0.64*(tau*L)^2
        if (len2 > decayLen2)
        {
            int ticks = edgeDecayTicks[e] + 1;
            edgeDecayTicks[e] = ticks;
            if (ticks < TEAR_DECAY_TIMEOUT_TICKS) return;      // still decaying: exempt
            // timeout — a snagged edge never relaxed; fall through to the normal law below.
        }
        edgeWasExempt[e] = 0; edgeDecayTicks[e] = 0;           // decayed clean (or timed out): clear the flag
    }

    if (len2 <= tearLen2) return;

    float3 ncut  = d / sqrtf(len2);                            // tear plane ⊥ the overstretched edge
    float3 P_hit = A + 0.5f * d;                               // midpoint: side=-1 for A, +1 for B (R1 verified)
    if (dbg) atomicAdd(&dbg[15], 1u);                          // CUMULATIVE tear counter (see slot map)
    MarkAndEmit(e, axis, baseC, axisDir, idA, idB, 0.5f, P_hit, ncut, toolD, voxelL, dims,
                cornerPos, voxelCutMask, voxelOccupied, cutPoint, cutPointCounter, cutPointCapacity,
                particleRot);
}

// ── graspRing BFS dilation + rim-bridge carve-out (design §6/§9-(8)) ───────────────────────────
// One hop: newRing[c] = ring[c] || (any LIVE neighbour n with ring[n]!=0). nExempt sequential hops
// (1-2 per design) dilate the grasped set by that many graph edges. Ping-ponged host-side (cut_recompute_
// grasp_ring) so no in-place read/write race across threads.
__global__ void k_DilateRingHop(int cornerCount, const int* nbrIdx, const int* ring, int* newRing)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cornerCount) return;
    if (ring[c] != 0) { newRing[c] = 1; return; }
    for (int s = 0; s < 6; s++)
    {
        int n = nbrIdx[6 * c + s];
        if (n >= 0 && ring[n] != 0) { newRing[c] = 1; return; }
    }
    newRing[c] = 0;
}

// Rim-bridge carve-out: a corner that has EVER lost a neighbour vs the cut_init topology snapshot is a
// cut-rim corner (the tear law's OTHER job — killing 藕断丝连 debris strands) and must NOT be shielded by
// ring proximity, even if it is graph-close to the grasp. Zero it out of the ring unconditionally.
__global__ void k_CarveRimBridge(int cornerCount, const int* nbrIdxNow, const int* nbrIdxAtInit, int* ring)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cornerCount) return;
    if (ring[c] == 0) return;
    for (int s = 0; s < 6; s++)
        if (nbrIdxAtInit[6 * c + s] >= 0 && nbrIdxNow[6 * c + s] < 0) { ring[c] = 0; return; }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// SeverLinks — one thread per grid edge. For cut edges: sever structural (NbrIdx=-1 both ways) +
// bending pairs whose arm crosses the edge (alive=0). Paper SS2.1.2 line 367.
// ─────────────────────────────────────────────────────────────────────────────────────────────
// Parts the link from idA to idB, the next corner along the axis, and the bend pairs across it.
__device__ inline void SeverLink(int idA, int idB, int axis, int* nbrIdx, BendPairGpuC* bendPairs)
{
    nbrIdx[6 * idA + 2 * axis + 1] = -1;   // idA's slot toward idB
    nbrIdx[6 * idB + 2 * axis] = -1;       // idB's slot toward idA
    for (int p = 0; p < 15; p++)
    {
        int ia = 15 * idA + p; BendPairGpuC a = bendPairs[ia];
        if (a.alive != 0 && (a.j == idB || a.k == idB)) { a.alive = 0; bendPairs[ia] = a; }
        int ib = 15 * idB + p; BendPairGpuC b = bendPairs[ib];
        if (b.alive != 0 && (b.j == idA || b.k == idA)) { b.alive = 0; bendPairs[ib] = b; }
    }
}
__global__ void k_SeverLinks(int gridEdgeCount, int3 dims, const GridEdgeGpu2* gridEdges,
                             const unsigned int* voxelCutMask, int* nbrIdx, BendPairGpuC* bendPairs)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= gridEdgeCount) return;

    GridEdgeGpu2 ge = gridEdges[e];
    int axis = ge.axis;
    int3 baseC = make_int3(ge.bcX, ge.bcY, ge.bcZ);
    if (!EdgeValid(axis, baseC, dims)) return;
    if (!EdgeIsCut(axis, baseC, dims, voxelCutMask)) return;

    int3 axisDir = make_int3(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0, axis == 2 ? 1 : 0);
    SeverLink(CorIdCuda(baseC, dims), CorIdCuda(baseC + axisDir, dims), axis, nbrIdx, bendPairs);
}

// Count isolated particles for diagnostics. A detached fragment remains dynamic.
__global__ void k_CountIsolated(int cornerCount, const int* nbrIdx, unsigned int* dbg)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cornerCount) return;
    const int* nb = &nbrIdx[6 * p];
    if (nb[0] >= 0 || nb[1] >= 0 || nb[2] >= 0 || nb[3] >= 0 || nb[4] >= 0 || nb[5] >= 0) return;
    if (dbg) atomicAdd(&dbg[13], 1u);
    // Diagnostic only: isolated material is still a valid moving fragment.

}

// ── ClearCutFP — per slot (8*VC): zero accum/cnt/CutFP.w/CutFPNormal (Stage-7 single FP) ──
__global__ void k_ClearCutFP(int voxelCount, float* cutFPAccumPos, int* cutFPAccumCnt, float4* cutFP,
                             int* cutFPNormal, unsigned int* voxelFPCount, int* cutNrmAccum)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= 8 * voxelCount) return;
    cutFPAccumPos[3 * s + 0] = 0; cutFPAccumPos[3 * s + 1] = 0; cutFPAccumPos[3 * s + 2] = 0;
    cutFPAccumCnt[s] = 0;
    cutFP[s] = make_float4(0.f, 0.f, 0.f, 0.f);
    cutFPNormal[3 * s + 0] = 0; cutFPNormal[3 * s + 1] = 0; cutFPNormal[3 * s + 2] = 0;
    cutNrmAccum[3 * s + 0] = 0; cutNrmAccum[3 * s + 1] = 0; cutNrmAccum[3 * s + 2] = 0;   // v4.1 P2
    if (s < voxelCount) voxelFPCount[s] = 0u;
}

// ── LookupConnectivity — per voxel: voxelFPCount = (cut & occupied) ? conn[mask].compCount : 0 ──
__global__ void k_LookupConnectivity(int voxelCount, const unsigned int* voxelCutMask,
                                     const unsigned int* voxelOccupied, const Conn4096Gpu* conn4096,
                                     unsigned int* voxelFPCount, unsigned int* dbg)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= voxelCount) return;
    unsigned int mask = voxelCutMask[v] & 0xFFFu;
    if (mask == 0u || voxelOccupied[v] == 0u) { voxelFPCount[v] = 0u; return; }
    int compCount = conn4096[mask].compCount;
    voxelFPCount[v] = (unsigned int)compCount;
    // [DEBUG-CUT] pinch counter (P3): cut voxel whose corner graph is still ONE component — the seam
    // pinches here (0-g roughness disambiguator).
    if (dbg && compCount == 1) atomicAdd(&dbg[12], 1u);
}

// ── AccumulateCutFP — per cut point: scatter world pos into the owner-component slots (centroid sum) ──
__global__ void k_AccumulateCutFP(int cutPointCapacity,
                                  const unsigned int* cutPointCounter, const CutPointGpu* cutPoint,
                                  const RotGpu* particleRot, const float3* cornerPos, int3 dims,
                                  const GridEdgeGpu2* gridEdges, const unsigned int* voxelOccupied,
                                  const unsigned int* voxelCutMask, const Conn4096Gpu* conn4096,
                                  float* cutFPAccumPos, int* cutFPAccumCnt,
                                  const int* active, int* cutNrmAccum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // OOB guard (diagnosis D1 minor): the emit counter increments BEFORE the capacity check, so
    // counter[0] can exceed capacity; without this clamp threads read d_cutPoint out of bounds.
    if (i >= (int)cutPointCounter[0] || i >= cutPointCapacity) return;

    CutPointGpu cp = cutPoint[i];
    // Skip only non-material nodes; cutting never deactivates material particles.
    if (active[cp.ownerParticle] == 0) return;
    RotGpu R = particleRot[cp.ownerParticle];
    float3 nrmW  = mulRv(R, cp.matNrm);                                      // v4.1 P2: world cut normal
    int3 ownerCoord = CornerCoordCuda(cp.ownerParticle, dims);

    GridEdgeGpu2 ge = gridEdges[cp.edgeId];
    int axis = ge.axis;
    int3 baseC = make_int3(ge.bcX, ge.bcY, ge.bcZ);

    for (int s = 0; s < 4; s++)
    {
        int3 vc = baseC + ES_VoxelOffset(axis, s);
        if (!VoxInBounds(vc, dims)) continue;
        int v = VoxId(vc, dims);
        if (voxelOccupied[v] == 0u) continue;
        unsigned int mask = voxelCutMask[v] & 0xFFFu;
        if (mask == 0u) continue;
        int lc = LocalCornerOf(ownerCoord, vc);
        if (lc < 0) continue;
        int comp = Conn4096Comp(conn4096[mask], lc);
        int slot = 8 * v + comp;
        // Accumulate relative to a canonical corner of THIS component. Absolute fixed-point
        // positions overflow for distant fragments and lose translation invariance.
        int root=0; while(root<8 && Conn4096Comp(conn4096[mask],root)!=comp) ++root;
        float3 origin=cornerPos[CorIdCuda(vc+CornerOffsetCuda(root),dims)];
        float3 local=(cornerPos[cp.ownerParticle]-origin)+mulRv(R,cp.localOffset);
        atomicAdd(&cutFPAccumPos[3 * slot + 0],local.x);
        atomicAdd(&cutFPAccumPos[3 * slot + 1],local.y);
        atomicAdd(&cutFPAccumPos[3 * slot + 2],local.z);
        atomicAdd(&cutFPAccumCnt[slot], 1);
        atomicAdd(&cutNrmAccum[3 * slot + 0], (int)rintf(nrmW.x * POS_SCALE));   // v4.1 P2
        atomicAdd(&cutNrmAccum[3 * slot + 1], (int)rintf(nrmW.y * POS_SCALE));
        atomicAdd(&cutNrmAccum[3 * slot + 2], (int)rintf(nrmW.z * POS_SCALE));
    }
}

// ── ComputeComponentFP — per voxel (loops components). Stage 7 (paper-exact; audit wf_60781cea A2):
// ONE feature point per (voxel, connected component), seeded by ALL of the component's crossing
// points — its ISOSURFACE INTERSECTIONS and its CUT POINTS together (SS2.1.1: "Each edge that
// contains a vertex will participate in the calculation, and the cut edges will be calculated
// twice"), refined by the Fig 2.2 particle-movement iteration over the component's iso
// intersections (Eq3/D1 gradient, Eq4 decaying step) and clamped by Eq5/D5. Skin quads (k_Stitch)
// and wall quads (k_BuildCutTriangles) SHARE this vertex, so the cut margin is watertight by
// construction. The paper has NO rim band and NO dual FP families — Table 3's External/Internal
// feature points are POSITIONAL labels of this same single entity (Stage-5's dual-family split was
// a misreading and caused the skin-parallel-cut durian/shadow artifacts).
__global__ void k_ComputeComponentFP(int voxelCount, int3 dims, const unsigned int* voxelCutMask,
                                     const unsigned int* voxelOccupied, const Conn4096Gpu* conn4096,
                                     const int* voxelIsectOffset, const int* voxelIsectCount,
                                     const IsectGpu* isect, const float3* isectWorld,
                                     const int* cutFPAccumCnt, const float* cutFPAccumPos, float4* cutFP,
                                     const int* voxelCorner, const float3* cornerPos, float voxelL,
                                     int qefIters, const int* cornerInside, const int* cutNrmAccum,const float* edgeCutT=nullptr)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= voxelCount) return;

    unsigned int mask = voxelCutMask[v] & 0xFFFu;
    if (mask == 0u || voxelOccupied[v] == 0u) return;

    int3 vCoord = make_int3(v % dims.x, (v / dims.x) % dims.y, v / (dims.x * dims.y));
    Conn4096Gpu conn = conn4096[mask];
    int compCount = conn.compCount;
    int off = voxelIsectOffset[v];
    int cnt = voxelIsectCount[v];

    float3 half=make_float3(.5f*voxelL,.5f*voxelL,.5f*voxelL);
    if(materialRestDevice){
        float3 extent=materialRestDevice[voxelCorner[8*v+6]]-materialRestDevice[voxelCorner[8*v]];
        half=make_float3(fabsf(extent.x)*.5f,fabsf(extent.y)*.5f,fabsf(extent.z)*.5f);
    }

    for (int comp = 0; comp < compCount; comp++)
    {
        int slot = 8 * v + comp;

        // Per-COMPONENT Eq5/D5 box (plan v3.1 item 3, diagnosis D2 blocker): center = mean of the
        // COMPONENT'S OWN deformed corners. After a piece falls many*L, a voxel straddling both
        // fragments must clamp each side's FP toward ITS side, not toward the void between them
        // (the all-8 mean caused the post-separation striated strands). Reduces to the all-8 mean
        // pre-cut (single comp). Never borrow a plane from another component.
        float3 bc = make_float3(0.f, 0.f, 0.f);
        int nCorn = 0;
        float3 componentOrigin=make_float3(0,0,0);
        for (int lc = 0; lc < 8; lc++)
        {
            if (Conn4096Comp(conn, lc) != comp) continue;
            int cid = voxelCorner[8 * v + lc];
            if(nCorn==0) componentOrigin=cornerPos[cid];
            bc = bc + cornerPos[cid];
            nCorn++;

        }
        if (nCorn > 0) bc = bc / (float)nCorn;

        // Combined seed: sum of the COMPONENT's iso intersection points ...
        float3 sum = make_float3(0.f, 0.f, 0.f);
        int nIncl = 0;
        for (int s = 0; s < cnt; s++)
        {
            IsectGpu it = isect[off + s];
            int insideCorner = SurfaceOwner(it,dims,edgeCutT);
            int3 icc = CornerCoordCuda(insideCorner, dims);
            int lc = LocalCornerOf(icc, vCoord);
            if (lc < 0) continue;
            if (Conn4096Comp(conn, lc) != comp) continue;
            sum = sum + isectWorld[off + s];
            nIncl++;
        }
        // ... plus its cut points ("the cut edges will be calculated twice" = both endpoint owners).
        int ccnt = cutFPAccumCnt[slot];
        if (ccnt > 0)
            sum = sum + make_float3((float)cutFPAccumPos[3 * slot + 0],
                                    (float)cutFPAccumPos[3 * slot + 1],
                                    cutFPAccumPos[3 * slot + 2]) + componentOrigin*(float)ccnt;
        int total = nIncl + ccnt;
        if (total == 0) { cutFP[slot] = make_float4(0.f, 0.f, 0.f, 0.f); continue; }
        float3 x = sum / (float)total;                       // Eq2 seed over ALL crossing points

        // Fig 2.2 particle-movement refinement over the component's iso intersections (Eq3/D1 + Eq4).
        if (nIncl > 0)
        {
            for (int it2 = 0; it2 < qefIters; it2++)
            {
                float3 F = make_float3(0.f, 0.f, 0.f);
                for (int s2 = 0; s2 < cnt; s2++)
                {
                    IsectGpu it = isect[off + s2];
                    int insideCorner = SurfaceOwner(it,dims,edgeCutT);
                    int3 icc = CornerCoordCuda(insideCorner, dims);
                    int lc = LocalCornerOf(icc, vCoord);
                    if (lc < 0 || Conn4096Comp(conn, lc) != comp) continue;
                    float3 p_i = isectWorld[off + s2];
                    float3 n_i = it.normal;                   // rest normal (PAPER-SILENT frozen)
                    F = F + n_i * dot3(n_i, p_i - x);         // DEVIATION D1
                }
                float a = 0.1f * (1.0f - (float)it2 / (float)qefIters);   // Eq4
                float3 xn = x + a * F;
                if (fabsf(xn.x-bc.x)>=half.x || fabsf(xn.y-bc.y)>=half.y || fabsf(xn.z-bc.z)>=half.z) break;                 // D5 stop
                x = xn;
            }
        }

        // DEVIATION D5 / Eq5 clamp: keep the FP inside its deformed voxel box.
        x.x = fminf(fmaxf(x.x, bc.x - half.x), bc.x + half.x);
        x.y = fminf(fmaxf(x.y, bc.y - half.y), bc.y + half.y);
        x.z = fminf(fmaxf(x.z, bc.z - half.z), bc.z + half.z);

        // Never project an air-side component onto intersections owned by another component.
        // Such a projection couples a detached crumb back to its old voxel's stationary skin.

        // v4.1 P2 (kerf fix, diagnosis S2 + plan review wf_b08c3fc2 R1-m1 ordering): wall-plane
        // projection LAST, with NO re-clamp after it — the wall term has the final word on the
        // along-n_cut coordinate. Without this, iso intersections (~0.5L lateral) dominate the seed
        // and the Fig-2.2 refinement has no cut-plane term, so FPs settle ~0.22L from the plane and
        // the horizontal kerf reads ~0.44L x 1/sin(theta) instead of D=0.16L. The component's OWN
        // cut-point centroid already sits ON its ±D/2 wall (EmitCutPoint bakes the offset), so
        // projecting x onto the plane through that centroid ⊥ the mean recorded cut normal pulls
        // the lip back to the paper's D-wide slit.
        // Degenerate guard (R1-m2/R2-m3): skip when the accumulated normals disagree badly (multi-
        // stroke opposing planes can cancel the sum -> normalize(~0) would NaN-poison the FP).
        if (ccnt > 0)
        {
            float3 nAcc = make_float3((float)cutNrmAccum[3 * slot + 0],
                                      (float)cutNrmAccum[3 * slot + 1],
                                      (float)cutNrmAccum[3 * slot + 2]) / POS_SCALE;
            float nLenW = length3(nAcc);
            if (nLenW > 1e-6f && nLenW > 0.25f * (float)ccnt)   // coherent sum of ccnt unit normals ≈ ccnt
            {
                float3 nbar = nAcc / nLenW;
                float3 cutC = make_float3((float)cutFPAccumPos[3 * slot + 0],
                                          (float)cutFPAccumPos[3 * slot + 1],
                                          cutFPAccumPos[3 * slot + 2]) / (float)ccnt + componentOrigin;
                x = x - dot3(nbar, x - cutC) * nbar;   // onto the comp's own ±D/2 wall plane
            }
        }

        cutFP[slot] = make_float4(x.x, x.y, x.z, 1.0f);
    }
}


__device__ __forceinline__ void EmitCutQuad(bool ok[4], bool shared[4], int tag[4], int n, bool flip,
                                            const float4* cutFP, int triCapacity,
                                            int* tri, unsigned int* triCounter, float voxelL)
{
    if (n < 3) return;
    int ord[4]; int m = 0;
    for (int s = 0; s < 4; s++) if (ok[s]) { ord[m] = s; m++; }
    // Opposing crack tips can share corners 0 and 2 while corners 1 and 3
    // are separated. A fan through the shared diagonal would attach FOUR
    // wall faces to that edge. Triangulate through the separated corners on
    // both sides instead, preserving the two independent crack surfaces.
    if(m==4 && shared[ord[0]] && shared[ord[2]] && !shared[ord[1]] && !shared[ord[3]]){
        int first=ord[0];ord[0]=ord[1];ord[1]=ord[2];ord[2]=ord[3];ord[3]=first;
    }
    // At the crack front, opposing endpoints may still belong to the SAME
    // voxel component. A triangle made entirely of shared feature points is
    // an internal duplicate, not a newly exposed cut wall.
    float3 q[4];
    for (int k=0;k<m;k++) {
        float4 f=cutFP[(unsigned int)tag[ord[k]] & 0x3FFFFFFFu];
        q[k]=make_float3(f.x,f.y,f.z);
    }

    for(int k=1;k<m-1;k++){
        if(shared[ord[0]] && shared[ord[k]] && shared[ord[k+1]])continue;
        float3 normal=cross3(q[k]-q[0],q[k+1]-q[0]);
        // Coincident feature points collapse a quad into a triangle. Retain
        // its nonzero half; a zero-area face has no surface or contact normal.
        if(dot3(normal,normal)<1e-16f)continue;
        // Winding follows the rest edge stencil. A detached opposite node
        // may move behind this face; it must never flip this component's normals.
        unsigned baseIdx=atomicAdd(&triCounter[0],3u);
        if(baseIdx+3u>(unsigned)(triCapacity*3))return;
        tri[baseIdx]=tag[ord[0]];
        tri[baseIdx+1]=tag[ord[flip?k+1:k]];
        tri[baseIdx+2]=tag[ord[flip?k:k+1]];
    }
}

// ── BuildCutTriangles — one thread per particle: gather component FPs over incident cut edges ->
// cut-wall faces. Stage 7: NO rim band (the paper's margin is the SHARED single FP per component —
// skin quads end at the same vertex the wall starts from; audit wf_60781cea A2).
__global__ void k_BuildCutTriangles(int cornerCount, int triCapacity, int3 dims, float voxelL,
                                    const unsigned int* voxelCutMask, const unsigned int* voxelOccupied,
                                    const Conn4096Gpu* conn4096, const float4* cutFP,
                                    const float3* cornerPos,
                                    int* tri, unsigned int* triCounter, unsigned int* dbg)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cornerCount) return;
    int3 c = CornerCoordCuda(p, dims);

    int  axes[6] = {0, 0, 1, 1, 2, 2};
    int3 bs[6] = { c, c - make_int3(1,0,0), c, c - make_int3(0,1,0), c, c - make_int3(0,0,1) };

    for (int e = 0; e < 6; e++)
    {
        int axis = axes[e];
        int3 baseC = bs[e];
        if (!EdgeValid(axis, baseC, dims)) continue;
        if (!EdgeIsCut(axis, baseC, dims, voxelCutMask)) continue;

        // outward reference: from this particle toward the severed opposite endpoint
        int3 axisDir = make_int3(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0, axis == 2 ? 1 : 0);
        bool basedHere = (baseC.x == c.x && baseC.y == c.y && baseC.z == c.z);
        int3 qc = basedHere ? c + axisDir : c - axisDir;
        int other=CorIdCuda(qc,dims),edgeSlot=DirectedEdgeSlot(p,other,dims);
        if(edgeCutTDevice && edgeSkinTDevice && edgeInsideDevice &&
            !CutHitsMaterial(edgeCutTDevice[edgeSlot],edgeSkinTDevice[edgeSlot],edgeInsideDevice[p],edgeInsideDevice[other]))continue;


        int tag[4]; bool ok[4], shared[4]; int n = 0;
        for (int s = 0; s < 4; s++)
        {
            ok[s] = false;
            int3 vc = baseC + ES_VoxelOffset(axis, s);
            int v = VoxId(vc, dims);
            if (voxelOccupied[v] == 0u) continue;
            unsigned int mask = voxelCutMask[v] & 0xFFFu;
            if (mask == 0u) continue;
            int lc = LocalCornerOf(c, vc);
            if (lc < 0) continue;
            int comp = Conn4096Comp(conn4096[mask], lc);
            shared[s] = comp == Conn4096Comp(conn4096[mask], LocalCornerOf(qc,vc));
            int slot = 8 * v + comp;
            if (cutFP[slot].w < 0.5f) continue;
            // bit31 = component cut FP; bit30 = WALL-EMITTER marker (liver-render design §3.3:
            // bit31 alone cannot distinguish wall from skin — StitchVertId shares bit31 tags with
            // cut-voxel skin stitches; bit30 is transparent to every consumer: all idx decodes
            // mask & 0x3FFFFFFF, all cut-FP tests use bit31 only — verified wf_9078a66a V1).
            tag[s] = (int)(0xC0000000u | (unsigned int)slot);
            ok[s] = true;
            n++;
        }
        // [DEBUG-CUT] rim<3 (P3): cut edge with 1-2 present FPs -> no wall quad emitted (hole
        // candidate). basedHere gates to ONE endpoint thread so each edge counts once (R2-m4).
        if (dbg && basedHere && n > 0 && n < 3) atomicAdd(&dbg[11], 1u);
        EmitCutQuad(ok, shared, tag, n, !basedHere, cutFP, triCapacity, tri, triCounter, voxelL);
    }
}


// ── DC_NormalizeCutFP — per slot: normalize the cut-FP normal accumulator ──
__global__ void k_NormalizeCutFP(int voxelCount, const int* cutFPNormal, float3* cutFPNormalF)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= 8 * voxelCount) return;
    float3 nv = make_float3((float)cutFPNormal[3 * s + 0], (float)cutFPNormal[3 * s + 1], (float)cutFPNormal[3 * s + 2]) / NORMAL_SCALE;
    float len = length3(nv);
    cutFPNormalF[s] = (len > 1e-6f) ? nv / len : make_float3(0.f, 0.f, 0.f);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Host API
// ─────────────────────────────────────────────────────────────────────────────────────────────
#define CK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { cut_shutdown(); return -(int)_e; } } while (0)

int cut_init(const CutInitDesc* desc, const void* conn4096, const void* gridEdges, const int* voxelOccupied,
             const int* cornerInside)
{
    if (g_cut_ready) cut_shutdown();
    if (!desc) return -3000;
    g_cdesc = *desc;
    g_cdims = make_int3(desc->dimsX, desc->dimsY, desc->dimsZ);

    int vc  = desc->voxelCount;
    int gec = desc->gridEdgeCount;
    int cpc = desc->cutPointCapacity > 0 ? desc->cutPointCapacity : 1;
    int slot = 8 * vc;
    if (vc <= 0 || desc->cornerCount <= 0) return -3001;
    int gecA = gec > 0 ? gec : 1;

    CK(cudaMalloc(&d_gridEdges,       (size_t)gecA   * sizeof(GridEdgeGpu2)));
    CK(cudaMalloc(&d_edgeCutT,(size_t)6*desc->cornerCount*sizeof(float)));
    std::vector<float> initialCutT(6*desc->cornerCount,-1.f);
    CK(cudaMemcpy(d_edgeCutT,initialCutT.data(),initialCutT.size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_prevCorner,      (size_t)desc->cornerCount * sizeof(float3)));   // CCD tick snapshot
    CK(cudaMalloc(&d_conn4096,        (size_t)4096   * sizeof(Conn4096Gpu)));
    CK(cudaMalloc(&d_voxelOccupied,   (size_t)vc     * sizeof(unsigned int)));
    CK(cudaMalloc(&d_voxelCutMask,    (size_t)vc     * sizeof(unsigned int)));
    CK(cudaMalloc(&d_cutPoint,        (size_t)cpc    * sizeof(CutPointGpu)));
    CK(cudaMalloc(&d_cutPointCounter,                  sizeof(unsigned int)));
    CK(cudaMalloc(&d_cutFP,           (size_t)slot   * sizeof(float4)));
    CK(cudaMalloc(&d_cornerInside,    (size_t)desc->cornerCount * sizeof(int)));
    CK(cudaMalloc(&d_cutFPAccumPos,   (size_t)3*slot * sizeof(float)));
    CK(cudaMalloc(&d_cutFPAccumCnt,   (size_t)slot   * sizeof(int)));
    CK(cudaMalloc(&d_voxelFPCount,    (size_t)vc     * sizeof(unsigned int)));
    CK(cudaMalloc(&d_cutFPNormal,     (size_t)3*slot * sizeof(int)));
    CK(cudaMalloc(&d_cutFPNormalF,    (size_t)slot   * sizeof(float3)));
    CK(cudaMalloc(&d_cutNrmAccum,     (size_t)3*slot * sizeof(int)));            // v4.1 P2
    CK(cudaMalloc(&d_dbg,             (size_t)16     * sizeof(unsigned int)));   // v4.1 P3
    CK(cudaMalloc(&d_graspRing,       (size_t)desc->cornerCount * sizeof(int))); // §6 tear exemption (INERT: zero-filled)
    CK(cudaMemset(d_graspRing, 0,     (size_t)desc->cornerCount * sizeof(int)));
    CK(cudaMalloc(&d_graspRingTmp,    (size_t)desc->cornerCount * sizeof(int))); // §9-(8) BFS ping-pong scratch
    CK(cudaMalloc(&d_nbrIdxAtInit,    (size_t)6 * desc->cornerCount * sizeof(int))); // §6 rim-bridge snapshot
    CK(cudaMalloc(&d_edgeWasExempt,   (size_t)gecA * sizeof(int)));             // §6 release-decay per-edge state
    CK(cudaMalloc(&d_edgeDecayTicks,  (size_t)gecA * sizeof(int)));
    CK(cudaMemset(d_edgeWasExempt,  0, (size_t)gecA * sizeof(int)));
    CK(cudaMemset(d_edgeDecayTicks, 0, (size_t)gecA * sizeof(int)));

    CK(cudaMemcpy(d_conn4096, conn4096, (size_t)4096 * sizeof(Conn4096Gpu), cudaMemcpyHostToDevice));
    if (gec > 0) CK(cudaMemcpy(d_gridEdges, gridEdges, (size_t)gec * sizeof(GridEdgeGpu2), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_voxelOccupied, voxelOccupied, (size_t)vc * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Zero the cumulative + scratch buffers (cut state starts uncut).
    CK(cudaMemset(d_voxelCutMask,    0, (size_t)vc     * sizeof(unsigned int)));
    CK(cudaMemset(d_cutPointCounter, 0,                  sizeof(unsigned int)));
    CK(cudaMemset(d_cutFP,           0, (size_t)slot   * sizeof(float4)));     // w=0 invalid
    CK(cudaMemcpy(d_cornerInside, cornerInside, (size_t)desc->cornerCount * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_cutFPAccumPos,   0, (size_t)3*slot * sizeof(float)));
    CK(cudaMemset(d_cutFPAccumCnt,   0, (size_t)slot   * sizeof(int)));
    CK(cudaMemset(d_voxelFPCount,    0, (size_t)vc     * sizeof(unsigned int)));
    CK(cudaMemset(d_cutFPNormal,     0, (size_t)3*slot * sizeof(int)));
    CK(cudaMemset(d_cutFPNormalF,    0, (size_t)slot   * sizeof(float3)));
    CK(cudaMemset(d_cutNrmAccum,     0, (size_t)3*slot * sizeof(int)));
    CK(cudaMemset(d_dbg,             0, (size_t)16     * sizeof(unsigned int)));

    // Cache the shared dc_recon + physics pointers (stable after their init).
    c_cornerPos        = recon_corner_pos();
    c_isect            = (IsectGpu*)recon_isect();
    c_isectWorld       = recon_isect_world();
    c_voxelIsectOffset = recon_voxel_isect_offset();
    c_voxelIsectCount  = recon_voxel_isect_count();
    c_voxelCorner      = recon_voxel_corner();
    c_tri              = recon_tri();
    c_triCounter       = recon_tri_counter();
    c_particleRot      = (RotGpu*)physics_particle_rot();
    c_nbrIdx           = physics_nbr_idx();
    c_bendPairs        = (BendPairGpuC*)physics_bend_pairs();
    c_active           = physics_active();   // live material flags
    const float* edgeSkinT=recon_surface_edge_t();
    const float3* materialRest=recon_corner_rest();
    c_cornerRest=materialRest;
    CK(cudaMemcpyToSymbol(materialRestDevice,&materialRest,sizeof(float3*)));
    CK(cudaMemcpyToSymbol(edgeCutTDevice,&d_edgeCutT,sizeof(float*)));
    CK(cudaMemcpyToSymbol(edgeSkinTDevice,&edgeSkinT,sizeof(float*)));
    CK(cudaMemcpyToSymbol(edgeInsideDevice,&d_cornerInside,sizeof(int*)));
    if (!c_cornerPos || !c_particleRot || !c_nbrIdx || !c_active) { cut_shutdown(); return -3002; }
    // §6 rim-bridge snapshot: the topology AT THIS MOMENT (before any cut) is the "had a neighbour"
    // baseline a later graspRing recompute diffs against to detect cut-rim corners.
    CK(cudaMemcpy(d_nbrIdxAtInit, c_nbrIdx, (size_t)6 * desc->cornerCount * sizeof(int), cudaMemcpyDeviceToDevice));

    // Hand the cut buffers to dc_recon so the cut-aware Stitch / Normals / Expand resolve cut FPs.
    recon_set_cut_buffers(d_voxelCutMask, d_voxelOccupied, d_conn4096, d_cutFP, d_cutFPNormalF, d_cutFPNormal);

    g_toolSet = false; g_havePrevCorner = false;
    g_emitNcut = make_float3(0.f, 1.f, 0.f);
    // v5.1 D10: sanitize tau (R1-m5: a stale-DLL/struct mismatch must degrade to tear-OFF, never a
    // garbage threshold). tau < 1 would tear edges at rest length; NaN fails the >= compare.
    {
        float tau = g_cdesc.tearStretchRatio;
        g_tearLen2 = (tau >= 1.0f && tau == tau) ? (tau * desc->voxelL) * (tau * desc->voxelL) : 0.f;
    }

    g_cut_ready = true;
    return 0;
}

void cut_set_tool(const CutToolDesc* tool)
{
    if (!tool) return;
    g_tool = *tool;
    if (tool->valid != 0) g_emitNcut = mk3(tool->ncut);   // remember the last VALID world cut normal
    // Seed the CCD snapshot at ACTIVATION (review S1-m3): without this the first physics substep of
    // the next frame is consumed by the seed pass and its tissue motion is never CCD-checked.
    if (!g_toolSet && g_cut_ready && c_cornerPos != nullptr)
    {
        if (cudaMemcpy(d_prevCorner, c_cornerPos, (size_t)g_cdesc.cornerCount * sizeof(float3),
                       cudaMemcpyDeviceToDevice) == cudaSuccess)
            g_havePrevCorner = true;
    }
    if(!g_toolSet && d_entryCorner){
        cudaMemcpy(d_entryCorner,c_cornerPos,(size_t)g_cdesc.cornerCount*sizeof(float3),cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_entryRotation,c_particleRot,(size_t)g_cdesc.cornerCount*sizeof(RotGpu),cudaMemcpyDeviceToDevice);
    }
    g_toolSet = true;
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Stage-6 detection host orchestration (task_plan.md Stage 6).
//   cut_detect()      — per C# rod substep (rod moved, tissue frozen): SS2.1.3 world quad M-T.
//   cut_ribbon_tick() — per accepted RK45 substep (tissue moved, rod frozen): per-edge CCD of the
//                       moving deformed edge vs the static blade segment; then roll the corner
//                       snapshot so consecutive ticks tile the full space-time sweep.
// ─────────────────────────────────────────────────────────────────────────────────────────────
int cut_detect()
{
    if (!g_cut_ready || !g_toolSet || g_pressureCut) return 0;
    int gec = g_cdesc.gridEdgeCount;
    if (gec <= 0) return 0;

    ToolGpu t;
    t.t1v0 = mk3(g_tool.t1v0); t.t1v1 = mk3(g_tool.t1v1); t.t1v2 = mk3(g_tool.t1v2);
    t.t2v0 = mk3(g_tool.t2v0); t.t2v1 = mk3(g_tool.t2v1); t.t2v2 = mk3(g_tool.t2v2);
    t.ncut = mk3(g_tool.ncut); t.aabbMin = mk3(g_tool.aabbMin); t.aabbMax = mk3(g_tool.aabbMax);
    t.d = g_tool.d; t.valid = g_tool.valid;

    k_DetectCutQuad<<<Gc(gec), 64>>>(gec, t, g_cdesc.voxelL, g_cdims, g_cdesc.cutPointCapacity,
                                     d_cornerInside, c_cornerPos, d_gridEdges, d_voxelCutMask, d_voxelOccupied,
                                     d_cutPoint, d_cutPointCounter, c_particleRot);
    k_SeverLinks<<<Gc(gec), 64>>>(gec, g_cdims, d_gridEdges, d_voxelCutMask, c_nbrIdx, c_bendPairs);

    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

// cut_ribbon_tick — CCD pass per accepted RK45 substep. The blade segment is the CURRENT rod
// [S=t2v2, E=t1v2] (rod is frozen during the physics phase); prev corner snapshot rolls after
// every pass so each tick covers exactly one tissue sub-interval (linear endpoint motion).
int cut_surface_revision(unsigned* revision){
    if(!d_cutPointCounter){*revision=0;return 0;}
    return -(int)cudaMemcpy(revision,d_cutPointCounter,sizeof(unsigned),cudaMemcpyDeviceToHost);
}

// Union-find over corners (unified_vertex.cuh), here joined through the links between material corners alone.
__global__ void k_InitComponents(int n,int* parent);
__global__ void k_JoinComponents(int n,const int* neighbors,const int* active,int* parent);
__global__ void k_FlattenComponents(int n,int* parent);
// The envelope corners around the material carry no tissue, yet their links and the envelope tets between them hold
// as tissue does. Where two material pieces part close to each other, such as under the overhang of a tumor cut off
// at its neck, the envelope between them keeps them tied. Each corner belongs to the material piece it reaches in the
// fewest intact links: the key (links << 32 | piece) of a material corner is its piece, and every pass lowers the key
// of a neighbour to one more link with atomicMin.
__global__ void k_SeedShellOwners(int n,const int* inside,const int* pieces,unsigned long long* owner){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)owner[i]=inside[i]?(unsigned long long)(unsigned)pieces[i]:~0ull;
}
__global__ void k_SpreadShellOwners(int n,const int* nbrIdx,unsigned long long* owner){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;unsigned long long key=owner[i];if(key==~0ull)return;
    for(int a=0;a<6;a++){int j=nbrIdx[6*i+a];if(j>=0)atomicMin(owner+j,key+(1ull<<32));}
}
// Severs the intact links whose ends belong to different material pieces, along the link at its middle. A single
// material piece leaves every key on one piece, and nothing parts. A link whose four cells hold no tissue has no bit
// in the voxel cut mask, which tracks cells with tissue alone (MarkAndEmit): it parts here directly, and no surface
// passes through it.
__global__ void k_SeverPieceSeams(int n,const int* phase,const GridEdgeGpu2* edges,int3 dims,float D,float L,const float3* pos,
 int* nbrIdx,BendPairGpuC* bendPairs,const unsigned long long* owner,unsigned* mask,const unsigned* occupied,
 CutPointGpu* points,unsigned* count,int capacity,const RotGpu* rotations){
    int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=n || !bladeCommits(*phase))return;
    GridEdgeGpu2 ge=edges[e];int3 bc=make_int3(ge.bcX,ge.bcY,ge.bcZ),dir=make_int3(ge.axis==0,ge.axis==1,ge.axis==2);
    int a=CorIdCuda(bc,dims),b=CorIdCuda(bc+dir,dims);if(nbrIdx[6*a+2*ge.axis+1]<0)return;
    unsigned long long ka=owner[a],kb=owner[b];if(ka==~0ull || kb==~0ull || (unsigned)ka==(unsigned)kb)return;
    bool tissue=false;
    for(int k=0;k<4;k++){int3 vc=bc+ES_VoxelOffset(ge.axis,k);tissue|=VoxInBounds(vc,dims) && occupied[VoxId(vc,dims)]!=0u;}
    if(!tissue){SeverLink(a,b,ge.axis,nbrIdx,bendPairs);return;}
    float3 along=pos[b]-pos[a];along=along*rsqrtf(fmaxf(dot3(along,along),1e-20f));
    MarkAndEmit(e,ge.axis,bc,dir,a,b,.5f,pos[a]+.5f*(pos[b]-pos[a]),along,D,L,dims,pos,mask,occupied,points,count,capacity,rotations);
}
int cut_ribbon_tick()
{
    if (!g_cut_ready || !g_toolSet) return 0;
    int gec = g_cdesc.gridEdgeCount;
    int cc  = g_cdesc.cornerCount;
    if (gec <= 0 || cc <= 0) return 0;
    const int* phase=g_pressureCut?xpbd_blade_phase_device():nullptr;
    const float3* sweep=g_pressureCut?xpbd_blade_sweep_device():nullptr;
    if(g_pressureCut && (!phase||!sweep))return 0;
    if(g_pressureCut && !d_pendingCrossings){
        cudaError_t err=cudaMalloc(&d_pendingCrossings,gec*sizeof(float4));if(err!=cudaSuccess)return -(int)err;
        err=cudaMemset(d_pendingCrossings,0,gec*sizeof(float4));if(err!=cudaSuccess)return -(int)err;
    }
    int blades=g_pressureCut?xpbd_blade_sweep_count():1;
    // Scissors cut the tissue where their edges pass through it, as it is
    // deformed at that moment. A material column, the lattice line of nodes
    // along the column axis (the one nearest the shear plane normal), takes its
    // cut level once, while the stroke is fractured: the rest coordinate where
    // the present segment of one of its material links crosses the present shear
    // plane, at a point an edge has passed over on its way from the opening
    // where the stroke began to its present angle, or comes within half a cell
    // of. The blades touch the reconstructed surface, up to half a cell outside
    // the material nodes, and their backs stop that surface just behind the
    // edge: a pressing edge never reaches the nodes of the link it cuts into,
    // and a link cannot resolve an edge closer than half a cell. The level never
    // changes: the closing blades squeeze the tissue and the scissors sink under
    // load, so a column swept again in this or a later snip crosses the plane in
    // another layer, and a second cut there would carve a sliver out of it. The
    // skin columns just outside the material copy the levels beside them, so a
    // crack never ends between the skin and the first material layer, a
    // non-manifold pinch. The links the levels pass through then part: along a
    // column at its level, and across columns where the wound steps between
    // layers, so the wound is one surface that follows the deformed tissue.
    // Before fracture nothing is recorded: a crossing pending from the pressing
    // phase would part tissue the blades have since pushed away.
    const bool scissors=g_pressureCut && blades==2;
    if(g_pressureCut && (!d_entryCorner || !d_entryRotation || !d_entryState || !d_columnLevel || !d_columnAxis)){
        cudaError_t err=cudaSuccess;if(!d_entryCorner)err=cudaMalloc(&d_entryCorner,cc*sizeof(float3));if(err!=cudaSuccess)return -(int)err;
        if(!d_columnLevel){
            err=cudaMalloc(&d_columnLevel,cc*sizeof(unsigned));if(err!=cudaSuccess)return -(int)err;
            err=cudaMemset(d_columnLevel,0x7F,cc*sizeof(unsigned));if(err!=cudaSuccess)return -(int)err;
        }
        if(!d_columnAxis){
            err=cudaMalloc(&d_columnAxis,sizeof(int));if(err!=cudaSuccess)return -(int)err;
            err=cudaMemset(d_columnAxis,0xFF,sizeof(int));if(err!=cudaSuccess)return -(int)err;
        }
        if(!d_entryRotation)err=cudaMalloc(&d_entryRotation,cc*sizeof(RotGpu));if(err!=cudaSuccess)return -(int)err;
        if(!d_entryState)err=cudaMalloc(&d_entryState,sizeof(int));if(err!=cudaSuccess)return -(int)err;
        err=cudaMemset(d_entryState,0,sizeof(int));if(err!=cudaSuccess)return -(int)err;
        err=cudaMemcpy(d_entryCorner,g_havePrevCorner?d_prevCorner:c_cornerPos,cc*sizeof(float3),cudaMemcpyDeviceToDevice);if(err!=cudaSuccess)return -(int)err;
        err=cudaMemcpy(d_entryRotation,c_particleRot,cc*sizeof(RotGpu),cudaMemcpyDeviceToDevice);if(err!=cudaSuccess)return -(int)err;
    }
    if(g_pressureCut){
        k_PreparePressureEntry<<<1,1>>>(d_entryState,phase,xpbd_blade_had_contact_device());
        if(scissors){
            k_ColumnAxis<<<1,1>>>(sweep+8,phase,d_entryState,d_columnAxis);
            k_LevelScissorsColumns<<<Gc(gec),64>>>(gec,g_cdims,.5f*g_cdesc.voxelL,d_gridEdges,c_nbrIdx,d_cornerInside,c_cornerRest,
                c_cornerPos,sweep+8,sweep+10,phase,d_entryState,d_columnAxis,d_columnLevel);
            k_SpreadScissorsLevels<<<Gc(gec),64>>>(gec,g_cdims,d_gridEdges,c_nbrIdx,d_cornerInside,c_cornerRest,phase,d_entryState,
                d_columnAxis,d_columnLevel);
            k_CutScissorsLevels<<<Gc(gec),64>>>(gec,g_cdims,d_gridEdges,c_nbrIdx,c_cornerRest,c_cornerPos,sweep+8,phase,d_entryState,
                c_particleRot,d_columnAxis,d_columnLevel,d_pendingCrossings);
        }else{
        ToolGpu entryTool={};entryTool.d=g_tool.d;
        k_DetectCutQuad<<<Gc(gec),64>>>(gec,entryTool,g_cdesc.voxelL,g_cdims,g_cdesc.cutPointCapacity,d_cornerInside,d_entryCorner,
            d_gridEdges,d_voxelCutMask,d_voxelOccupied,d_cutPoint,d_cutPointCounter,d_entryRotation,phase,d_pendingCrossings,sweep,d_entryState);
        }
    }
    if(g_pressureCut)k_CommitPending<<<Gc(gec),64>>>(gec,phase,d_pendingCrossings,d_gridEdges,g_cdims,g_tool.d,g_cdesc.voxelL,c_cornerPos,
        d_voxelCutMask,d_voxelOccupied,d_cutPoint,d_cutPointCounter,g_cdesc.cutPointCapacity,c_particleRot);


    if (!g_havePrevCorner)
    {
        if (cudaMemcpy(d_prevCorner, c_cornerPos, (size_t)cc * sizeof(float3), cudaMemcpyDeviceToDevice) != cudaSuccess)
            return -3200;
        g_havePrevCorner = true;
        return 0;                       // seed pass: nothing swept yet
    }

    float3 S = mk3(g_tool.t2v2);        // current rod endpoints (CuttingTool: T2V2=S, T1V2=E)
    float3 E = mk3(g_tool.t1v2);

    if(!scissors){
    k_DetectCutCCD<<<Gc(gec), 64>>>(gec, S, E, g_emitNcut, g_tool.d, g_cdesc.voxelL, g_cdims,
                                    g_cdesc.cutPointCapacity, d_prevCorner, c_cornerPos,
                                    d_gridEdges, c_nbrIdx, d_cornerInside, d_voxelCutMask, d_voxelOccupied,
                                    d_cutPoint, d_cutPointCounter, c_particleRot,phase,d_pendingCrossings,sweep);
    if(g_pressureCut){
        ToolGpu tool={};tool.d=g_tool.d;
        k_DetectCutQuad<<<Gc(gec),64>>>(gec,tool,g_cdesc.voxelL,g_cdims,g_cdesc.cutPointCapacity,d_cornerInside,c_cornerPos,
            d_gridEdges,d_voxelCutMask,d_voxelOccupied,d_cutPoint,d_cutPointCounter,c_particleRot,phase,d_pendingCrossings,sweep);
    }
    }
    // v5.1 D10 tear law — BEFORE k_SeverLinks so the same tick severs the newly marked tears.
    if (g_tearLen2 > 0.f)
        k_TearOverstretch<<<Gc(gec), 64>>>(gec, g_tearLen2, g_cdims, g_cdesc.voxelL, g_tool.d,
                                           d_gridEdges, c_nbrIdx, c_active, d_graspRing, c_cornerPos,
                                           d_voxelCutMask, d_voxelOccupied, d_cutPoint,
                                           d_cutPointCounter, g_cdesc.cutPointCapacity,
                                           c_particleRot, d_dbg, d_edgeWasExempt, d_edgeDecayTicks);
    k_SeverLinks<<<Gc(gec), 64>>>(gec, g_cdims, d_gridEdges, d_voxelCutMask, c_nbrIdx, c_bendPairs);
    if(scissors){
        // Release the envelope between material pieces this snip has parted (k_SeedShellOwners). Envelope corners lie
        // within a few links of the material, and four passes give each of them its owner.
        cudaError_t err=cudaSuccess;
        if(!d_materialPieces)err=cudaMalloc(&d_materialPieces,cc*sizeof(int));if(err!=cudaSuccess)return -(int)err;
        if(!d_shellOwner)err=cudaMalloc(&d_shellOwner,cc*sizeof(unsigned long long));if(err!=cudaSuccess)return -(int)err;
        k_InitComponents<<<Gc(cc),64>>>(cc,d_materialPieces);
        k_JoinComponents<<<Gc(cc),64>>>(cc,c_nbrIdx,d_cornerInside,d_materialPieces);
        k_FlattenComponents<<<Gc(cc),64>>>(cc,d_materialPieces);
        k_SeedShellOwners<<<Gc(cc),64>>>(cc,d_cornerInside,d_materialPieces,d_shellOwner);
        for(int pass=0;pass<4;pass++)k_SpreadShellOwners<<<Gc(cc),64>>>(cc,c_nbrIdx,d_shellOwner);
        k_SeverPieceSeams<<<Gc(gec),64>>>(gec,phase,d_gridEdges,g_cdims,g_tool.d,g_cdesc.voxelL,c_cornerPos,c_nbrIdx,c_bendPairs,
            d_shellOwner,d_voxelCutMask,d_voxelOccupied,d_cutPoint,d_cutPointCounter,g_cdesc.cutPointCapacity,c_particleRot);
        k_SeverLinks<<<Gc(gec),64>>>(gec,g_cdims,d_gridEdges,d_voxelCutMask,c_nbrIdx,c_bendPairs);
    }

    if(g_pressureCut){
        k_FinishPressureEntry<<<1,1>>>(d_entryState,phase,scissors);
        k_RefreshPressureEntry<<<Gc(cc),64>>>(cc,d_entryState,c_cornerPos,c_particleRot,d_entryCorner,d_entryRotation);
    }
    if (cudaMemcpy(d_prevCorner, c_cornerPos, (size_t)cc * sizeof(float3), cudaMemcpyDeviceToDevice) != cudaSuccess)
        return -3201;

    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}


int cut_fp_chain()
{
    if (!g_cut_ready) return 0;
    int vc = g_cdesc.voxelCount, slot = 8 * vc, cpc = g_cdesc.cutPointCapacity;
    int cc = g_cdesc.cornerCount;
    // [DEBUG-CUT] window = one chain (one frame); the C# side reads EVERY frame and aggregates
    // host-side (plan review R1-m5/R2-m2: 1 Hz sampling of a per-frame window misses transients).
    // Slots 0-14 ONLY: dbg[15] (tearMarks) is written during LCS_Step's ribbon ticks, which run
    // BEFORE this chain in the same frame — zeroing it here would erase every tear before the
    // readback (wf_c416ae77 R2-m3). It stays cumulative.
    if (cudaMemset(d_dbg, 0, 15 * sizeof(unsigned int)) != cudaSuccess) return -3300;
    // Count, but do not freeze or remove, isolated fragments.
    k_CountIsolated<<<Gc(cc), 64>>>(cc, c_nbrIdx, d_dbg);
    k_ClearCutFP<<<Gc(slot), 64>>>(vc, d_cutFPAccumPos, d_cutFPAccumCnt, d_cutFP, d_cutFPNormal, d_voxelFPCount,
                                   d_cutNrmAccum);
    k_LookupConnectivity<<<Gc(vc), 64>>>(vc, d_voxelCutMask, d_voxelOccupied, d_conn4096, d_voxelFPCount, d_dbg);
    k_AccumulateCutFP<<<Gc(cpc), 64>>>(cpc, d_cutPointCounter, d_cutPoint, c_particleRot, c_cornerPos, g_cdims,
                                       d_gridEdges, d_voxelOccupied, d_voxelCutMask, d_conn4096,
                                       d_cutFPAccumPos, d_cutFPAccumCnt, c_active, d_cutNrmAccum);
    k_ComputeComponentFP<<<Gc(vc), 64>>>(vc, g_cdims, d_voxelCutMask, d_voxelOccupied, d_conn4096,
                                         c_voxelIsectOffset, c_voxelIsectCount, c_isect, c_isectWorld,
                                         d_cutFPAccumCnt, d_cutFPAccumPos, d_cutFP,
                                         c_voxelCorner, c_cornerPos, g_cdesc.voxelL,
                                         recon_qef_iters(), d_cornerInside, d_cutNrmAccum,d_edgeCutT);
    // Rebuild every component from current material frames; no world-space EMA or jump gate.
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

int cut_build_triangles()
{
    if (!g_cut_ready) return 0;
    k_BuildCutTriangles<<<Gc(g_cdesc.cornerCount), 64>>>(g_cdesc.cornerCount, g_cdesc.triCapacity, g_cdims,
                                                         g_cdesc.voxelL,
                                                         d_voxelCutMask, d_voxelOccupied, d_conn4096, d_cutFP,
                                                         c_cornerPos, c_tri, c_triCounter, d_dbg);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

// ── cut_get_debug (v4.1 P3) — fill out18 with the 16 [DEBUG-CUT] counters + the RAW (unclamped)
// cutPoint/tri atomic counters (exact demand meters — both increment past capacity by design), and
// report the RUNTIME cutFPInterp so the log line settles the alpha ambiguity (S1-f4). Runs the
// giant-triangle scan (dc_recon owns d_tri/FetchPos — cross-TU hook per R2-m4) before the readback.
int cut_get_debug(unsigned int* out18, float* alphaOut)
{
    if (!g_cut_ready || !out18) return -1;
    int r = recon_debug_scan_tris(d_dbg, g_cdesc.voxelL);
    if (r != 0) return r;
    unsigned int dbg[16];
    if (cudaMemcpy(dbg, d_dbg, sizeof(dbg), cudaMemcpyDeviceToHost) != cudaSuccess) return -3310;
    unsigned int cutRaw = 0, triRaw = 0;
    if (cudaMemcpy(&cutRaw, d_cutPointCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess) return -3311;
    if (c_triCounter &&
        cudaMemcpy(&triRaw, c_triCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess) return -3312;
    memcpy(out18, dbg, sizeof(dbg));
    out18[16] = cutRaw;   // raw cut points emitted (cumulative) vs cutPointCapacity
    out18[17] = triRaw;   // raw tri indices this frame vs 3*triCapacity
    if (alphaOut) *alphaOut = 1.0f;
    return 0;
}

int cut_normalize_fp()
{
    if (!g_cut_ready) return 0;
    int vc = g_cdesc.voxelCount;
    k_NormalizeCutFP<<<Gc(8 * vc), 64>>>(vc, d_cutFPNormal, d_cutFPNormalF);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

bool cut_is_ready() { return g_cut_ready; }

// design §6/§9-(6): the tear-exemption ring buffer (int[cornerCount], 1 = grasped-neighbourhood corner
// -> incident edges exempt from the tau*L tear law). ZERO-FILLED at cut_init; step-8 grasp writes it
// (BFS-dilate d_grasped over nbrIdx). nullptr until cut_init. Read only by k_TearOverstretch.
int* cut_grasp_ring() { return d_graspRing; }

// design §6/§9-(8): recompute the ring from the grasp module's CURRENT captured-corner flags. See cut.h.
int cut_recompute_grasp_ring(const int* grasped, int nExempt)
{
    if (!g_cut_ready || !grasped) return -3010;
    int cc = g_cdesc.cornerCount;
    if (nExempt < 0) nExempt = 0;
    if (cudaMemcpy(d_graspRing, grasped, (size_t)cc * sizeof(int), cudaMemcpyDeviceToDevice) != cudaSuccess)
        return -3011;
    for (int hop = 0; hop < nExempt; hop++)
    {
        k_DilateRingHop<<<Gc(cc), 64>>>(cc, c_nbrIdx, d_graspRing, d_graspRingTmp);
        int* t = d_graspRing; d_graspRing = d_graspRingTmp; d_graspRingTmp = t;   // ping-pong swap
    }
    k_CarveRimBridge<<<Gc(cc), 64>>>(cc, c_nbrIdx, d_nbrIdxAtInit, d_graspRing);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

// ── §9-(6) INERT tear-prep UNIT TEST: prove the graspRing early-out exempts (set the flag -> no tear;
// clear it -> tears). Self-contained — builds a tiny synthetic grid with ONE over-stretched X edge and
// runs the REAL k_TearOverstretch twice; the tear-mark counter dbg[15] is the observable. Does NOT need
// cut_init (no conn4096 / no FP chain). Returns the number of failed assertions (0 = pass). ────────────
__global__ void k_TearTestSetup(int3 dims, float voxelL, GridEdgeGpu2* gridEdges, int* nbrIdx,
                                int* active, float3* cornerPos, unsigned int* voxelOccupied,
                                RotGpu* particleRot, int* outIds)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    int3 baseC = make_int3(1, 1, 1); int axis = 0; int3 axisDir = make_int3(1, 0, 0);
    int idA = CorIdCuda(baseC, dims), idB = CorIdCuda(baseC + axisDir, dims);
    gridEdges[0].axis = axis; gridEdges[0].bcX = 1; gridEdges[0].bcY = 1; gridEdges[0].bcZ = 1;
    nbrIdx[6 * idA + 2 * axis + 1] = idB;                          // +x slot live (NOT severed)
    active[idA] = 1; active[idB] = 1;                              // both active
    cornerPos[idA] = make_float3(0.f, 0.f, 0.f);
    cornerPos[idB] = make_float3(10.f * voxelL, 0.f, 0.f);         // |edge| = 10*voxelL >> tau*L -> overstretch
    voxelOccupied[VoxId(baseC, dims)] = 1u;                        // a live stencil voxel (anyOcc = true)
    RotGpu I; I.r0 = make_float3(1,0,0); I.r1 = make_float3(0,1,0); I.r2 = make_float3(0,0,1);
    particleRot[idA] = I; particleRot[idB] = I;
    outIds[0] = idA; outIds[1] = idB;
}

int cut_selftest_tear_exempt()
{
    const int3 dims = make_int3(4, 4, 4);
    const int  cc   = (dims.x + 1) * (dims.y + 1) * (dims.z + 1);   // 125
    const int  vc   = dims.x * dims.y * dims.z;                     // 64
    const float voxelL = 0.2f;
    const float tau = 2.0f;
    const float tearLen2 = (tau * voxelL) * (tau * voxelL);         // edge (10*voxelL) tears; grasped is exempt
    const int   cap = 16;
    int fails = 0;   // print-free (cut.cu is shipping code): the test harness reports; returns the count.

    GridEdgeGpu2* ge; int* nbr; int* act; float3* pos; unsigned* occ; unsigned* mask;
    CutPointGpu* cp; unsigned* cpc; RotGpu* rot; unsigned* dbg; int* ring; int* outIds;
    int* ewe; int* edt;   // §6 release-decay per-edge state (1 grid edge in this synthetic harness)
    #define TA(p, n) do { if (cudaMalloc(&(p), (n)) != cudaSuccess) { return fails + 1; } } while (0)
    TA(ge, sizeof(GridEdgeGpu2)); TA(nbr, (size_t)6*cc*sizeof(int)); TA(act, (size_t)cc*sizeof(int));
    TA(pos, (size_t)cc*sizeof(float3)); TA(occ, (size_t)vc*sizeof(unsigned)); TA(mask, (size_t)vc*sizeof(unsigned));
    TA(cp, (size_t)cap*sizeof(CutPointGpu)); TA(cpc, sizeof(unsigned)); TA(rot, (size_t)cc*sizeof(RotGpu));
    TA(dbg, 16*sizeof(unsigned)); TA(ring, (size_t)cc*sizeof(int)); TA(outIds, 2*sizeof(int));
    TA(ewe, sizeof(int)); TA(edt, sizeof(int));
    #undef TA
    cudaMemset(nbr, 0, (size_t)6*cc*sizeof(int)); cudaMemset(act, 0, (size_t)cc*sizeof(int));
    cudaMemset(pos, 0, (size_t)cc*sizeof(float3)); cudaMemset(occ, 0, (size_t)vc*sizeof(unsigned));
    cudaMemset(rot, 0, (size_t)cc*sizeof(RotGpu));
    k_TearTestSetup<<<1, 1>>>(dims, voxelL, ge, nbr, act, pos, occ, rot, outIds);
    int ids[2] = {0,0}; cudaMemcpy(ids, outIds, 2*sizeof(int), cudaMemcpyDeviceToHost);
    int idA = ids[0];

    auto runTear = [&]()->unsigned {
        cudaMemset(mask, 0, (size_t)vc*sizeof(unsigned)); cudaMemset(cpc, 0, sizeof(unsigned));
        cudaMemset(dbg, 0, 16*sizeof(unsigned));
        k_TearOverstretch<<<1, 64>>>(1, tearLen2, dims, voxelL, voxelL, ge, nbr, act, ring, pos,
                                     mask, occ, cp, cpc, cap, rot, dbg, ewe, edt);
        cudaDeviceSynchronize();
        unsigned tears = 0; cudaMemcpy(&tears, dbg + 15, sizeof(unsigned), cudaMemcpyDeviceToHost);
        return tears;
    };
    // (1) EXEMPT: graspRing[idA]=1 -> the over-stretched edge must NOT tear (dbg[15] stays 0), and the
    // release-decay bookkeeping activates (edgeWasExempt flips to 1).
    cudaMemset(ring, 0, (size_t)cc*sizeof(int)); cudaMemset(ewe, 0, sizeof(int)); cudaMemset(edt, 0, sizeof(int));
    int one = 1; cudaMemcpy(ring + idA, &one, sizeof(int), cudaMemcpyHostToDevice);
    if (runTear() != 0u) fails++;                                  // exempt but it tore -> FAIL
    int eweVal = 0; cudaMemcpy(&eweVal, ewe, sizeof(int), cudaMemcpyDeviceToHost);
    if (eweVal != 1) fails++;                                      // decay bookkeeping did not activate -> FAIL
    // (2) RELEASE-DECAY: ring cleared but the edge is STILL far past 0.8*tau*L -> stays exempt (does NOT
    // instantly resume the tau law) — else a just-released tent would shred on its first post-release tick.
    cudaMemset(ring, 0, (size_t)cc*sizeof(int));
    if (runTear() != 0u) fails++;                                  // decaying but it tore -> FAIL
    // (3) TIMEOUT: force the decay clock to the safety cutoff (simulates ~2s of a SNAGGED edge that
    // never relaxed) -> the normal law resumes and it tears honestly.
    int atTimeout = TEAR_DECAY_TIMEOUT_TICKS; cudaMemcpy(edt, &atTimeout, sizeof(int), cudaMemcpyHostToDevice);
    if (runTear() != 1u) fails++;                                  // timed out but did NOT tear -> FAIL
    // (4) BASE LAW: a FRESH edge that was NEVER in a ring (bookkeeping cleared) tears immediately —
    // preserves the original "not exempt -> tears" guarantee for edges the grasp module never touched.
    cudaMemset(ring, 0, (size_t)cc*sizeof(int)); cudaMemset(ewe, 0, sizeof(int)); cudaMemset(edt, 0, sizeof(int));
    if (runTear() != 1u) fails++;                                  // never grasped but did NOT tear -> FAIL

    cudaFree(ge); cudaFree(nbr); cudaFree(act); cudaFree(pos); cudaFree(occ); cudaFree(mask);
    cudaFree(cp); cudaFree(cpc); cudaFree(rot); cudaFree(dbg); cudaFree(ring); cudaFree(outIds);
    cudaFree(ewe); cudaFree(edt);
    return fails;
}

void cut_shutdown()
{
    cudaFree(d_entryCorner);d_entryCorner=nullptr;cudaFree(d_entryRotation);d_entryRotation=nullptr;
    cudaFree(d_columnLevel);d_columnLevel=nullptr;cudaFree(d_columnAxis);d_columnAxis=nullptr;
    cudaFree(d_entryState);d_entryState=nullptr;

    cudaFree(d_pendingCrossings);d_pendingCrossings=nullptr;g_pressureCut=0;
    cudaFree(d_materialPieces);d_materialPieces=nullptr;cudaFree(d_shellOwner);d_shellOwner=nullptr;
    cudaFree(d_gridEdges);       d_gridEdges = nullptr;
    cudaFree(d_prevCorner);      d_prevCorner = nullptr;
    g_toolSet = false; g_havePrevCorner = false; g_tearLen2 = 0.f;
    cudaFree(d_conn4096);        d_conn4096 = nullptr;
    cudaFree(d_voxelOccupied);   d_voxelOccupied = nullptr;
    cudaFree(d_voxelCutMask);    d_voxelCutMask = nullptr;
    cudaFree(d_cutPoint);        d_cutPoint = nullptr;
    cudaFree(d_cutPointCounter); d_cutPointCounter = nullptr;
    cudaFree(d_cutFP);           d_cutFP = nullptr;
    cudaFree(d_edgeCutT);        d_edgeCutT = nullptr;
    cudaMemcpyToSymbol(edgeCutTDevice,&d_edgeCutT,sizeof(float*));
    cudaMemcpyToSymbol(edgeSkinTDevice,&d_edgeCutT,sizeof(float*));
    cudaMemcpyToSymbol(edgeInsideDevice,&d_edgeCutT,sizeof(float*));
    const float3* noRest=nullptr;cudaMemcpyToSymbol(materialRestDevice,&noRest,sizeof(float3*));c_cornerRest=nullptr;
    cudaFree(d_cornerInside);    d_cornerInside = nullptr;
    cudaFree(d_cutFPAccumPos);   d_cutFPAccumPos = nullptr;
    cudaFree(d_cutFPAccumCnt);   d_cutFPAccumCnt = nullptr;
    cudaFree(d_voxelFPCount);    d_voxelFPCount = nullptr;
    cudaFree(d_cutFPNormal);     d_cutFPNormal = nullptr;
    cudaFree(d_cutFPNormalF);    d_cutFPNormalF = nullptr;
    cudaFree(d_cutNrmAccum);     d_cutNrmAccum = nullptr;
    cudaFree(d_dbg);             d_dbg = nullptr;
    cudaFree(d_graspRing);       d_graspRing = nullptr;
    cudaFree(d_graspRingTmp);    d_graspRingTmp = nullptr;
    cudaFree(d_nbrIdxAtInit);    d_nbrIdxAtInit = nullptr;
    cudaFree(d_edgeWasExempt);   d_edgeWasExempt = nullptr;
    cudaFree(d_edgeDecayTicks);  d_edgeDecayTicks = nullptr;
    c_active = nullptr;
    if (g_cut_ready) recon_set_cut_buffers(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    g_cut_ready = false;
}

// Disabling cutting also disables tissue-motion CCD against the previous blade.
void cut_suspend(){if(d_entryState)cudaMemset(d_entryState,0,sizeof(int));g_toolSet=false;g_havePrevCorner=false;if(d_pendingCrossings)cudaMemset(d_pendingCrossings,0,g_cdesc.gridEdgeCount*sizeof(float4));}
