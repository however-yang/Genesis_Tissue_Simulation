"""Variant for a test build: a component with a tet at the volume floor, grasped or not, gets the recovery trust region
and the tangent-cone projection of its material direction in every solve graph. In a grasped component the trust
region weighs only the tets within twice the floor, so it binds only where the floor does.

Usage: python variant_trust_projection.py apply   (from endoscopic_tissue/native/vendor/cuda; undo with git checkout)
"""
import sys
from pathlib import Path


def patch(path, old, new):
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, old[:80])
    Path(path).write_text(text.replace(old, new), encoding="utf-8")


assert sys.argv[1] == "apply"
SEARCH = "solver/material/material_line_search.cuh"
patch(SEARCH, "int id[4];unifiedTetIds(s,t,id);int flags=s.graspComponents[s.component[id[0]]];if(!(flags&8)||(flags&7))return;",
      "int id[4];unifiedTetIds(s,t,id);int flags=s.graspComponents[s.component[id[0]]];if(!(flags&8))return;")
patch(SEARCH, " int id[4];unifiedTetIds(s,t,id);int root=s.component[id[0]],flags=s.graspComponents[root];if(!(flags&8)||(flags&7))return;\n",
      """ int id[4];unifiedTetIds(s,t,id);int root=s.component[id[0]],flags=s.graspComponents[root];if(!(flags&8))return;
 if(flags&7){float3 p0=s.pos[id[0]];float J=dot3(s.pos[id[1]]-p0,cross3(s.pos[id[2]]-p0,s.pos[id[3]]-p0))*unifiedKuhn(s,t).metric[6];
  if(J>=2*s.minimum)return;}
""")
patch(SEARCH, "\n// Bound the *trial* strain before tangent projection;", """
__global__ void k_ApplyRecoveryTrust(int n,float3* direction,const float* scale,const UnifiedState* state){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int root=state->component[i];
    if(!(state->graspComponents[root]&8))return;float a=scale[root];if(a<1)direction[i]=direction[i]*a;
}
// Bound the *trial* strain before tangent projection;""")
SOLVER = "xpbd/xpbd_solver.cu"
patch(SOLVER, """                if(recoveryRequested){
                    k_MaterialRestrictionClear<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_coarseWeight);
                    k_RecoveryTrust<<<Gr(g_T),64,0,g_captureStream>>>(d_unified,d_cgD,d_coarseWeight);
                    k_ApplyMaterialRestriction<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_cgD,d_coarseWeight,nullptr,d_unified);
                }""", """                k_MaterialRestrictionClear<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_coarseWeight);
                k_RecoveryTrust<<<Gr(g_T),64,0,g_captureStream>>>(d_unified,d_cgD,d_coarseWeight);
                k_ApplyRecoveryTrust<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_cgD,d_coarseWeight,d_unified);""")
patch(SOLVER, "for(int projection=0;recoveryRequested && projection<64;projection++){", "for(int projection=0;projection<64;projection++){")
print("applied")
