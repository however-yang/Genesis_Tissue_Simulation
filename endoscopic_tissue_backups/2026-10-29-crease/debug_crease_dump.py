"""Add, for a debug build only, a per-node dump at the end of the last substep of chosen frames: positions, the step
one more local vertex solve would take before and after its volume bound, the assembled contact and anchor
interaction (gradient and Hessian), and flags (active, fixed in the material solve, contact witness, anchor witness,
movable). Frames are counted from 1 per LCS_Step, listed in LCS_DEBUG_FRAMES (comma separated); files go to
LCS_DEBUG_DIR as crease-<frame>.bin: int32 n and frame, then float32 positions (n, 3), raw steps (n, 3), bound
scales (n, 3, first column), interactions (n, 9) and int32 flags (n).

Usage: python debug_crease_dump.py apply   (from endoscopic_tissue/native/vendor/cuda; undo with git checkout)
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
      "__device__ inline void unifiedVertexSolve(const UnifiedState& s,int i,int lane,float3* debugOut=nullptr){")
patch(VERTEX, """    for(int offset=16;offset;offset/=2)scale=fminf(scale,__shfl_xor_sync(0xffffffff,scale,offset));
    // Same-color vertices share neither a lattice tetrahedron nor an axial
""", """    for(int offset=16;offset;offset/=2)scale=fminf(scale,__shfl_xor_sync(0xffffffff,scale,offset));
    if(debugOut){if(lane==0){debugOut[2*i]=step;debugOut[2*i+1]=mkf3(scale,0,0);}return;}
    // Same-color vertices share neither a lattice tetrahedron nor an axial
""")
patch(VERTEX, """__global__ void k_UnifiedVertex(const UnifiedState* state,const int* sites,const int* count){""",
      """__global__ void k_DebugCrease(const UnifiedState* state,float3* out,int* flags){
    const UnifiedState& s=*state;int lane=threadIdx.x%32,warps=gridDim.x*blockDim.x/32;
    for(int k=(blockIdx.x*blockDim.x+threadIdx.x)/32;k<s.activeCount;k+=warps){
        int i=s.activeNodes[k];
        if(lane==0)flags[i]=1|(materialCgFixed(s,i)?2:0)|((s.degree[i]>0 || (s.secondary && s.secondary->degree[i]>0))?4:0)
            |((s.grips>0 && s.gripDegree[i]>0)?8:0)|(s.invMass[i]>0?16:0);
        if(s.invMass[i]>0)unifiedVertexSolve(s,i,lane,out);
    }
}
__global__ void k_UnifiedVertex(const UnifiedState* state,const int* sites,const int* count){""")
SOLVER = "xpbd/xpbd_solver.cu"
patch(SOLVER, "static int g_substepIndex = 0;\n", """static int g_substepIndex = 0;
static int g_debugFrame = 0;
static bool debugDumpFrame(int frame){
    static int frames[256];static int count=-1;
    if(count<0){count=0;const char* list=getenv("LCS_DEBUG_FRAMES");
        while(list && *list && count<256){frames[count++]=atoi(list);while(*list && *list!=',')list++;if(*list==',')list++;}}
    for(int k=0;k<count;k++)if(frames[k]==frame)return true;
    return false;
}
""")
patch(SOLVER, "    g_substepIndex = 0;\n    if(d_contactResidual)", "    g_substepIndex = 0;g_debugFrame++;\n    if(d_contactResidual)")
patch(SOLVER, """
    k_UnifiedReaction<<<128,64>>>(d_unified);
    if(g_manifoldCapacity>0){
        k_UnifiedAggregate<<<128,64>>>(d_unified,d_manifoldOwners,d_surfaceContacts,!g_anchorAssist);
        if(!g_anchorAssist)k_UnifiedStick<<<128,64>>>(d_unified,d_surfaceActive,d_surfaceActiveCount,d_surfaceContacts,g_jawThickness,g_gp.jawGapClosed);
    }
    k_EndTool<<<1,1>>>(d_coupledTool);""", """
    if(g_substepIndex==g_p.numSubsteps && debugDumpFrame(g_debugFrame)){
        // DEBUG: the local vertex step, interactions and solve flags of every node at the end of the frame.
        static float3* d_debugOut=nullptr;static int* d_debugFlags=nullptr;
        if(!d_debugOut){cudaMalloc(&d_debugOut,2*sizeof(float3)*g_N);cudaMalloc(&d_debugFlags,sizeof(int)*g_N);}
        cudaMemset(d_debugOut,0,2*sizeof(float3)*g_N);cudaMemset(d_debugFlags,0,sizeof(int)*g_N);
        int sites=((s.nx+1)/2)*((s.ny+1)/2)*((g_dimsZ+2)/2);
        cudaMemsetAsync(d_sweepCount,0,8*sizeof(int));
        k_SweepSites<<<Gr(g_N),64>>>(d_unified,SweepLocal,sites,d_sweepSites,d_sweepCount);
        for(int c=0;c<8;c++){k_AssembleUnifiedInteractions<<<ASSEMBLY_BLOCKS,64>>>(d_unified,c);if(g_blade.enabled)k_AssembleUnifiedInteractions<<<ASSEMBLY_BLOCKS,64>>>(d_bladeState,c);}
        k_DebugCrease<<<1024,64>>>(d_unified,d_debugOut,d_debugFlags);
        std::vector<float> hostPos(3*(size_t)g_N),hostOut(6*(size_t)g_N),hostInteraction(9*(size_t)g_N);std::vector<int> hostFlags(g_N);
        cudaMemcpy(hostPos.data(),d_pos,3*sizeof(float)*g_N,cudaMemcpyDeviceToHost);
        cudaMemcpy(hostOut.data(),d_debugOut,6*sizeof(float)*g_N,cudaMemcpyDeviceToHost);
        cudaMemcpy(hostInteraction.data(),d_unifiedInteraction,9*sizeof(float)*g_N,cudaMemcpyDeviceToHost);
        cudaMemcpy(hostFlags.data(),d_debugFlags,sizeof(int)*g_N,cudaMemcpyDeviceToHost);
        const char* dir=getenv("LCS_DEBUG_DIR");char path[1024];snprintf(path,sizeof(path),"%s/crease-%04d.bin",dir?dir:".",g_debugFrame);
        FILE* file=fopen(path,"wb");
        if(file){int header[2]={g_N,g_debugFrame};fwrite(header,sizeof(int),2,file);fwrite(hostPos.data(),sizeof(float),hostPos.size(),file);
            std::vector<float> raw(3*(size_t)g_N),bound(3*(size_t)g_N);
            for(int i=0;i<g_N;i++)for(int a=0;a<3;a++){raw[3*i+a]=hostOut[6*i+a];bound[3*i+a]=hostOut[6*i+3+a];}
            fwrite(raw.data(),sizeof(float),raw.size(),file);fwrite(bound.data(),sizeof(float),bound.size(),file);
            fwrite(hostInteraction.data(),sizeof(float),hostInteraction.size(),file);fwrite(hostFlags.data(),sizeof(int),hostFlags.size(),file);fclose(file);}
    }
    k_UnifiedReaction<<<128,64>>>(d_unified);
    if(g_manifoldCapacity>0){
        k_UnifiedAggregate<<<128,64>>>(d_unified,d_manifoldOwners,d_surfaceContacts,!g_anchorAssist);
        if(!g_anchorAssist)k_UnifiedStick<<<128,64>>>(d_unified,d_surfaceActive,d_surfaceActiveCount,d_surfaceContacts,g_jawThickness,g_gp.jawGapClosed);
    }
    k_EndTool<<<1,1>>>(d_coupledTool);""")
print("applied")
