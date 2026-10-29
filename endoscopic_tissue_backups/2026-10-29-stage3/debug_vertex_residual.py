"""Add, for a debug build only, a measurement of how far the substep's solve is from converged: at the end of every
unified substep, the Newton step each free node would take in one more vertex sweep (with all other nodes fixed), as
the RMS and largest step and the count of nodes whose step exceeds 1 um, printed to stderr as

    RESID <substep index> <rms step cm> <max step cm> <nodes> <nodes above 1 um>

Usage: python debug_vertex_residual.py apply   (from endoscopic_tissue/native/vendor/cuda; undo with git checkout)
"""
import sys
from pathlib import Path


def patch(path, old, new):
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, old[:80])
    Path(path).write_text(text.replace(old, new), encoding="utf-8")


assert sys.argv[1] == "apply"
VERTEX = "solver/material/unified_vertex.cuh"
patch(VERTEX, "__device__ inline void unifiedVertexSolve(const UnifiedState& s,int i,int lane){",
      "__device__ inline void unifiedVertexSolve(const UnifiedState& s,int i,int lane,float* measure=nullptr){")
patch(VERTEX, """    for(int offset=16;offset;offset/=2)scale=fminf(scale,__shfl_xor_sync(0xffffffff,scale,offset));
    // Same-color vertices share neither a lattice tetrahedron nor an axial
""", """    for(int offset=16;offset;offset/=2)scale=fminf(scale,__shfl_xor_sync(0xffffffff,scale,offset));
    // Same-color vertices share neither a lattice tetrahedron nor an axial
    if(measure){
        if(lane==0 && isfinite(step.x+step.y+step.z)){
            float l=len3(step*scale);atomicAdd(measure,l*l);atomicMax((unsigned*)measure+1,__float_as_uint(l));
            atomicAdd(measure+2,1.f);if(l>1e-4f)atomicAdd(measure+3,1.f);
        }
        return;
    }
""")
patch(VERTEX, """__global__ void k_UnifiedVertex(const UnifiedState* state,const int* sites,const int* count){""",
      """__global__ void k_DebugResidual(const UnifiedState* state,float* measure){
    const UnifiedState& s=*state;int lane=threadIdx.x%32,warps=gridDim.x*blockDim.x/32;
    for(int k=(blockIdx.x*blockDim.x+threadIdx.x)/32;k<s.activeCount;k+=warps){
        int i=s.activeNodes[k];if(s.invMass[i]>0)unifiedVertexSolve(s,i,lane,measure);
    }
}
__global__ void k_UnifiedVertex(const UnifiedState* state,const int* sites,const int* count){""")
patch("xpbd/xpbd_solver.cu", """
    k_UnifiedReaction<<<128,64>>>(d_unified);
    if(g_manifoldCapacity>0){
        k_UnifiedAggregate<<<128,64>>>(d_unified,d_manifoldOwners,d_surfaceContacts,!g_anchorAssist);
        if(!g_anchorAssist)k_UnifiedStick<<<128,64>>>(d_unified,d_surfaceActive,d_surfaceActiveCount,d_surfaceContacts,g_jawThickness,g_gp.jawGapClosed);
    }
    k_EndTool<<<1,1>>>(d_coupledTool);""", """
    {   // DEBUG: the Newton step of every free node in one more sweep, at the end of the substep.
        static float* d_measure=nullptr;if(!d_measure)cudaMalloc(&d_measure,4*sizeof(float));
        cudaMemsetAsync(d_measure,0,4*sizeof(float));
        int sites=((s.nx+1)/2)*((s.ny+1)/2)*((g_dimsZ+2)/2);
        cudaMemsetAsync(d_sweepCount,0,8*sizeof(int));
        k_SweepSites<<<Gr(g_N),64>>>(d_unified,SweepLocal,sites,d_sweepSites,d_sweepCount);
        for(int c=0;c<8;c++){k_AssembleUnifiedInteractions<<<ASSEMBLY_BLOCKS,64>>>(d_unified,c);if(g_blade.enabled)k_AssembleUnifiedInteractions<<<ASSEMBLY_BLOCKS,64>>>(d_bladeState,c);}
        k_DebugResidual<<<1024,64>>>(d_unified,d_measure);
        float m[4];cudaMemcpy(m,d_measure,sizeof(m),cudaMemcpyDeviceToHost);
        fprintf(stderr,"RESID %d %.6e %.6e %.0f %.0f\\n",g_substepIndex,sqrtf(m[0]/fmaxf(1.f,m[2])),m[1],m[2],m[3]);
    }
    k_UnifiedReaction<<<128,64>>>(d_unified);
    if(g_manifoldCapacity>0){
        k_UnifiedAggregate<<<128,64>>>(d_unified,d_manifoldOwners,d_surfaceContacts,!g_anchorAssist);
        if(!g_anchorAssist)k_UnifiedStick<<<128,64>>>(d_unified,d_surfaceActive,d_surfaceActiveCount,d_surfaceContacts,g_jawThickness,g_gp.jawGapClosed);
    }
    k_EndTool<<<1,1>>>(d_coupledTool);""")
print("applied")
