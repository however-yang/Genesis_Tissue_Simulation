// Generated from the reviewed inventory. Do not hand edit.
#include "state_archive.h"
#include <cstddef>
namespace lcs_state {
void collect_cut(Fields& f){
    f.pod("cut.d_gridEdges",d_gridEdges);
    f.pod("cut.d_prevCorner",d_prevCorner);
    f.pod("cut.d_conn4096",d_conn4096);
    f.pod("cut.d_voxelOccupied",d_voxelOccupied);
    f.pod("cut.d_voxelCutMask",d_voxelCutMask);
    f.pod("cut.d_surfaceBoundaryMask",d_surfaceBoundaryMask);
    f.pod("cut.d_airBoundaryMask",d_airBoundaryMask);
    f.pod("cut.g_materialBoundaryGraph",g_materialBoundaryGraph);
    f.pod("cut.d_cutPoint",d_cutPoint);
    f.pod("cut.d_cutPointCounter",d_cutPointCounter);
    f.pod("cut.d_topologyRevision",d_topologyRevision);
    f.pod("cut.d_cutFP",d_cutFP);
    f.pod("cut.d_edgeCutT",d_edgeCutT);
    f.symbol("cut.edgeCutTDevice",sizeof(edgeCutTDevice),Layout{sizeof(void*),{0}},
        [](void* out){check(cudaMemcpyFromSymbol(out,edgeCutTDevice,sizeof(edgeCutTDevice)));},
        [](const void* in,size_t bytes){check(cudaMemcpyToSymbol(edgeCutTDevice,in,bytes));});
    f.symbol("cut.edgeSkinTDevice",sizeof(edgeSkinTDevice),Layout{sizeof(void*),{0}},
        [](void* out){check(cudaMemcpyFromSymbol(out,edgeSkinTDevice,sizeof(edgeSkinTDevice)));},
        [](const void* in,size_t bytes){check(cudaMemcpyToSymbol(edgeSkinTDevice,in,bytes));});
    f.symbol("cut.edgeInsideDevice",sizeof(edgeInsideDevice),Layout{sizeof(void*),{0}},
        [](void* out){check(cudaMemcpyFromSymbol(out,edgeInsideDevice,sizeof(edgeInsideDevice)));},
        [](const void* in,size_t bytes){check(cudaMemcpyToSymbol(edgeInsideDevice,in,bytes));});
    f.symbol("cut.materialRestDevice",sizeof(materialRestDevice),Layout{sizeof(void*),{0}},
        [](void* out){check(cudaMemcpyFromSymbol(out,materialRestDevice,sizeof(materialRestDevice)));},
        [](const void* in,size_t bytes){check(cudaMemcpyToSymbol(materialRestDevice,in,bytes));});
    f.pod("cut.d_cornerInside",d_cornerInside);
    f.pod("cut.d_cutFPAccumPos",d_cutFPAccumPos);
    f.pod("cut.d_cutFPAccumCnt",d_cutFPAccumCnt);
    f.pod("cut.d_voxelFPCount",d_voxelFPCount);
    f.pod("cut.d_cutFPNormal",d_cutFPNormal);
    f.pod("cut.d_cutFPNormalF",d_cutFPNormalF);
    f.pod("cut.d_cutNrmAccum",d_cutNrmAccum);
    f.pod("cut.d_dbg",d_dbg);
    f.pod("cut.d_graspRing",d_graspRing);
    f.pod("cut.d_graspRingTmp",d_graspRingTmp);
    f.pod("cut.d_nbrIdxAtInit",d_nbrIdxAtInit);
    f.pod("cut.d_edgeWasExempt",d_edgeWasExempt);
    f.pod("cut.d_edgeDecayTicks",d_edgeDecayTicks);
    f.pod("cut.c_cornerPos",c_cornerPos);
    f.pod("cut.c_cornerRest",c_cornerRest);
    f.pod("cut.c_isect",c_isect);
    f.pod("cut.c_isectWorld",c_isectWorld);
    f.pod("cut.c_voxelIsectOffset",c_voxelIsectOffset);
    f.pod("cut.c_voxelIsectCount",c_voxelIsectCount);
    f.pod("cut.c_voxelCorner",c_voxelCorner);
    f.pod("cut.c_tri",c_tri);
    f.pod("cut.c_triCounter",c_triCounter);
    f.pod("cut.c_particleRot",c_particleRot);
    f.pod("cut.c_nbrIdx",c_nbrIdx);
    f.pod("cut.c_bendPairs",c_bendPairs);
    f.pod("cut.c_active",c_active);
    f.pod("cut.g_cdesc",g_cdesc);
    f.pod("cut.g_cdims",g_cdims);
    f.pod("cut.g_tool",g_tool);
    f.pod("cut.g_pressureCut",g_pressureCut);
    f.pod("cut.d_pendingCrossings",d_pendingCrossings);
    f.pod("cut.d_materialPieces",d_materialPieces);
    f.pod("cut.d_shellOwner",d_shellOwner);
    f.pod("cut.d_entryCorner",d_entryCorner);
    f.pod("cut.d_entryRotation",d_entryRotation);
    f.pod("cut.d_entryState",d_entryState);
    f.pod("cut.d_columnLevel",d_columnLevel);
    f.pod("cut.d_columnAxis",d_columnAxis);
    f.pod("cut.g_cut_ready",g_cut_ready);
    f.pod("cut.g_toolSet",g_toolSet);
    f.pod("cut.g_havePrevCorner",g_havePrevCorner);
    f.pod("cut.g_emitNcut",g_emitNcut);
    f.pod("cut.d_trainingTearCause",d_trainingTearCause);
    f.pod("cut.d_trainingGraspNeighborhood",d_trainingGraspNeighborhood);
    f.pod("cut.g_tearLen2",g_tearLen2);
}
}
