"""Add, for a debug build of the floor-recovery fix only, one stderr line per substep in which some component is in
floor recovery: FLOOR <frame> <substep> <components in recovery> <components entering it>. Frames count LCS_Step calls.

Usage: python debug_floor_count.py apply   (from endoscopic_tissue/native/vendor/cuda; undo with git checkout)
"""
import sys
from pathlib import Path


def patch(path, old, new):
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, old[:80])
    Path(path).write_text(text.replace(old, new), encoding="utf-8")


assert sys.argv[1] == "apply"
SEARCH = "solver/material/material_line_search.cuh"
text = Path(SEARCH).read_text(encoding="utf-8")
Path(SEARCH).write_text(text + """
__global__ void k_DebugFloorCount(const UnifiedState* state,int* count){
    const UnifiedState& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0 || s.component[i]!=i)return;
    int left=s.floorRecovery[i]&~MaterialFloorRestricted;
    if(left>0)atomicAdd(count,1);if(left==MaterialFloorRecoverySubsteps)atomicAdd(count+1,1);
}
""", encoding="utf-8")
SOLVER = "xpbd/xpbd_solver.cu"
patch(SOLVER, "static int g_substepIndex = 0;\n", "static int g_substepIndex = 0;\nstatic int g_debugFloorFrame = 0;\n")
patch(SOLVER, "    g_substepIndex = 0;\n    if(d_contactResidual)", "    g_substepIndex = 0;g_debugFloorFrame++;\n    if(d_contactResidual)")
patch(SOLVER, """
    k_UnifiedReaction<<<128,64>>>(d_unified);
    if(g_manifoldCapacity>0){""", """
    {   // DEBUG: components in floor recovery.
        static int* d_count=nullptr;if(!d_count)cudaMalloc(&d_count,2*sizeof(int));
        cudaMemsetAsync(d_count,0,2*sizeof(int));k_DebugFloorCount<<<Gr(g_activeNodes),64>>>(d_unified,d_count);
        int count[2];cudaMemcpy(count,d_count,sizeof(count),cudaMemcpyDeviceToHost);
        if(count[0])fprintf(stderr,"FLOOR %d %d %d %d%c",g_debugFloorFrame,g_substepIndex,count[0],count[1],10);
    }
    k_UnifiedReaction<<<128,64>>>(d_unified);
    if(g_manifoldCapacity>0){""")
print("applied")
