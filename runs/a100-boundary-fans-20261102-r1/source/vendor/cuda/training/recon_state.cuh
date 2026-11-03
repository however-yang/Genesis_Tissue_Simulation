// Generated from the reviewed inventory. Do not hand edit.
#include "state_archive.h"
#include <cstddef>
namespace lcs_state {
void collect_recon(Fields& f){
    f.pod("recon.d_boundaryEdges",d_boundaryEdges);
    f.pod("recon.d_boundaryStats",d_boundaryStats);
    f.pod("recon.d_featureBindings",d_featureBindings);
    f.pod("recon.g_boundaryCapacity",g_boundaryCapacity);
    f.pod("recon.g_featureBindingCapacity",g_featureBindingCapacity);
    f.pod("recon.d_cornerPos",d_cornerPos);
    f.pod("recon.d_voxelCorner",d_voxelCorner);
    f.pod("recon.d_voxelIsectOffset",d_voxelIsectOffset);
    f.pod("recon.d_voxelIsectCount",d_voxelIsectCount);
    f.pod("recon.d_isect",d_isect);
    f.pod("recon.d_surfaceEdgeT",d_surfaceEdgeT);
    f.pod("recon.d_surfaceEdges",d_surfaceEdges);
    f.pod("recon.d_isectWorld",d_isectWorld);
    f.pod("recon.d_voxelExternalFP",d_voxelExternalFP);
    f.pod("recon.d_tri",d_tri);
    f.pod("recon.d_triCounter",d_triCounter);
    f.pod("recon.d_extNormalFixed",d_extNormalFixed);
    f.pod("recon.d_extNormalF",d_extNormalF);
    f.pod("recon.d_expandPos",d_expandPos);
    f.pod("recon.d_expandNrm",d_expandNrm);
    f.pod("recon.d_voxelExternalFPRest",d_voxelExternalFPRest);
    f.pod("recon.d_cornerPosRest",d_cornerPosRest);
    f.pod("recon.d_expandAux",d_expandAux);
    f.pod("recon.d_cut_voxelCutMask",d_cut_voxelCutMask);
    f.pod("recon.d_cut_voxelOccupied",d_cut_voxelOccupied);
    f.pod("recon.d_cut_conn4096",d_cut_conn4096);
    f.pod("recon.d_cut_cutFP",d_cut_cutFP);
    f.pod("recon.d_cut_cutFPNormalF",d_cut_cutFPNormalF);
    f.pod("recon.d_cut_cutFPNormal",d_cut_cutFPNormal);
    f.pod("recon.g_cutActive",g_cutActive);
    f.pod("recon.g_desc",g_desc);
    f.pod("recon.g_dims",g_dims);
    f.pod("recon.g_ready",g_ready);
    f.pod("recon.g_haveRestSurface",g_haveRestSurface);
    f.pod("recon.d_render",d_render);
    f.pod("recon.d_renderNormal",d_renderNormal);
    f.pod("recon.d_renderTangent",d_renderTangent);
    f.pod("recon.d_renderBounds",d_renderBounds);
    f.pod("recon.d_renderCounts",d_renderCounts);
    f.pod("recon.g_renderCapacity",g_renderCapacity);
    f.pod("recon.d_faceTable",d_faceTable);
    f.pod("recon.d_cleanTriangles",d_cleanTriangles);
    f.pod("recon.g_faceCapacity",g_faceCapacity);
    f.pod("recon.g_cleanCapacity",g_cleanCapacity);
}
}
