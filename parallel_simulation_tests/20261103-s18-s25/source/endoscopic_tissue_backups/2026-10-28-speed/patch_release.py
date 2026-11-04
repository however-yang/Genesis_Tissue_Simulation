"""Sever the envelope links between material pieces once a scissors cut separates them."""
from pathlib import Path

cut = Path(r"D:\Simulation\genesis-world\endoscopic_tissue\native\vendor\cuda\cut.cu")
s = cut.read_text(encoding="utf8")


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


s = replace_once(s, "static float4* d_pendingCrossings=nullptr;\n", """static float4* d_pendingCrossings=nullptr;
// Material piece of every corner, and the (links, piece) owner key of the envelope release (k_SpreadShellOwners).
static int* d_materialPieces=nullptr;
static unsigned long long* d_shellOwner=nullptr;
""")

kernels = r'''// Union-find over corners (unified_vertex.cuh), here joined through the links between material corners alone.
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
'''
s = replace_once(s, "int cut_ribbon_tick()\n{", kernels + "int cut_ribbon_tick()\n{")
s = replace_once(s, """    k_SeverLinks<<<Gc(gec), 64>>>(gec, g_cdims, d_gridEdges, d_voxelCutMask, c_nbrIdx, c_bendPairs);

    if(g_pressureCut){""", """    k_SeverLinks<<<Gc(gec), 64>>>(gec, g_cdims, d_gridEdges, d_voxelCutMask, c_nbrIdx, c_bendPairs);
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
        k_SeverPieceSeams<<<Gc(gec),64>>>(gec,phase,d_gridEdges,g_cdims,g_tool.d,g_cdesc.voxelL,c_cornerPos,c_nbrIdx,d_shellOwner,
            d_voxelCutMask,d_voxelOccupied,d_cutPoint,d_cutPointCounter,g_cdesc.cutPointCapacity,c_particleRot);
        k_SeverLinks<<<Gc(gec),64>>>(gec,g_cdims,d_gridEdges,d_voxelCutMask,c_nbrIdx,c_bendPairs);
    }

    if(g_pressureCut){""")
cut.write_text(s, encoding="utf8")
print("release step written")
