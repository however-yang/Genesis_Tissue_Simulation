#pragma once
#include "cut.cuh"
// Transient kernel arguments. Ownership remains in cut/recon; these are not
// additional persistent pointers and are not a second physics state.
struct BoundaryInputs {
    int3 dims;
    int voxelCount,cornerCount,gridEdgeCount,cutPointCount;
    const unsigned* cutPointCounter;
    const CutPointGpu* cutPoints;
    const GridEdgeGpu2* gridEdges;
    const RotGpu* rotations;
    const float3* positions;
    const float3* rest;
    const int* voxelCorners;
    const Conn4096Gpu* connectivity;
    const unsigned* boundaryMasks;
    const int* isectCounts;
    const int* active;
    const int* inside;
};
BoundaryInputs cut_boundary_inputs();
