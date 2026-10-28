"""Sever envelope seam links in all-air cells directly: the voxel cut mask holds no bits for cells without tissue."""
from pathlib import Path

cut = Path(r"D:\Simulation\genesis-world\endoscopic_tissue\native\vendor\cuda\cut.cu")
s = cut.read_text(encoding="utf8")


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


# One severing routine for the mask-driven pass and the release of air cells.
s = replace_once(s, """__global__ void k_SeverLinks(int gridEdgeCount, int3 dims, const GridEdgeGpu2* gridEdges,
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
    int idA = CorIdCuda(baseC, dims);
    int idB = CorIdCuda(baseC + axisDir, dims);
    int sPlus  = 2 * axis + 1;    // idA's slot toward idB
    int sMinus = 2 * axis;        // idB's slot toward idA

    nbrIdx[6 * idA + sPlus]  = -1;
    nbrIdx[6 * idB + sMinus] = -1;

    for (int p = 0; p < 15; p++)
    {
        int ia = 15 * idA + p; BendPairGpuC a = bendPairs[ia];
        if (a.alive != 0 && (a.j == idB || a.k == idB)) { a.alive = 0; bendPairs[ia] = a; }
        int ib = 15 * idB + p; BendPairGpuC b = bendPairs[ib];
        if (b.alive != 0 && (b.j == idA || b.k == idA)) { b.alive = 0; bendPairs[ib] = b; }
    }
}
""", """// Parts the link from idA to idB, the next corner along the axis, and the bend pairs across it.
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
""")
s = replace_once(s, """// Severs the intact links whose ends belong to different material pieces, along the link at its middle. A single
// material piece leaves every key on one piece, and nothing parts.
__global__ void k_SeverPieceSeams(int n,const int* phase,const GridEdgeGpu2* edges,int3 dims,float D,float L,const float3* pos,
 const int* nbrIdx,const unsigned long long* owner,unsigned* mask,const unsigned* occupied,CutPointGpu* points,unsigned* count,
 int capacity,const RotGpu* rotations){
    int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=n || !bladeCommits(*phase))return;
    GridEdgeGpu2 ge=edges[e];int3 bc=make_int3(ge.bcX,ge.bcY,ge.bcZ),dir=make_int3(ge.axis==0,ge.axis==1,ge.axis==2);
    int a=CorIdCuda(bc,dims),b=CorIdCuda(bc+dir,dims);if(nbrIdx[6*a+2*ge.axis+1]<0)return;
    unsigned long long ka=owner[a],kb=owner[b];if(ka==~0ull || kb==~0ull || (unsigned)ka==(unsigned)kb)return;
    float3 along=pos[b]-pos[a];along=along*rsqrtf(fmaxf(dot3(along,along),1e-20f));
    MarkAndEmit(e,ge.axis,bc,dir,a,b,.5f,pos[a]+.5f*(pos[b]-pos[a]),along,D,L,dims,pos,mask,occupied,points,count,capacity,rotations);
}
""", """// Severs the intact links whose ends belong to different material pieces, along the link at its middle. A single
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
""")
s = replace_once(s, """        k_SeverPieceSeams<<<Gc(gec),64>>>(gec,phase,d_gridEdges,g_cdims,g_tool.d,g_cdesc.voxelL,c_cornerPos,c_nbrIdx,d_shellOwner,
            d_voxelCutMask,d_voxelOccupied,d_cutPoint,d_cutPointCounter,g_cdesc.cutPointCapacity,c_particleRot);""",
"""        k_SeverPieceSeams<<<Gc(gec),64>>>(gec,phase,d_gridEdges,g_cdims,g_tool.d,g_cdesc.voxelL,c_cornerPos,c_nbrIdx,c_bendPairs,
            d_shellOwner,d_voxelCutMask,d_voxelOccupied,d_cutPoint,d_cutPointCounter,g_cdesc.cutPointCapacity,c_particleRot);""")
cut.write_text(s, encoding="utf8")
print("air-cell seams sever directly")
