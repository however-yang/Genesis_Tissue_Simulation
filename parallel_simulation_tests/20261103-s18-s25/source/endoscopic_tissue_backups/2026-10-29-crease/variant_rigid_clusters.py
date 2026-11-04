"""Variant for a test build: in a grasped component, every cluster of nodes joined by tets at the volume floor
(k_JoinSafeCoarse) takes the mean material direction of its nodes, so it translates and its tets keep their volume;
a cluster holding a contact or anchor witness keeps still. Applied after the CG iterations, before the restriction.

Usage: python variant_rigid_clusters.py apply   (from endoscopic_tissue/native/vendor/cuda; undo with git checkout)
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
__global__ void k_ClusterDirectionClear(const UnifiedState* state,float3* sum,float* count){
    int i=unifiedActiveNode(*state,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;sum[i]=mkf3(0,0,0);count[i]=0;
}
__global__ void k_ClusterDirectionAccumulate(const UnifiedState* state,const float3* direction,float3* sum,float* count){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;
    if(!(s.graspComponents[s.component[i]]&7))return;
    int r=s.safeCluster[i];
    if(materialCgFixed(s,i)){atomicAdd(count+r,1e6f);return;}
    add3(sum+r,direction[i]);atomicAdd(count+r,1.f);
}
__global__ void k_ClusterDirectionApply(const UnifiedState* state,float3* direction,const float3* sum,const float* count){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;
    if(!(s.graspComponents[s.component[i]]&7) || materialCgFixed(s,i))return;
    int r=s.safeCluster[i];float c=count[r];if(c<1.5f)return;
    direction[i]=c>=1e6f?mkf3(0,0,0):sum[r]*(1.f/c);
}
""", encoding="utf-8")
patch("xpbd/xpbd_solver.cu", """                xpbd_section_mark("cg: trust, projection");
""", """                xpbd_section_mark("cg: trust, projection");
                k_ClusterDirectionClear<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgA,d_coarseWeight);
                k_ClusterDirectionAccumulate<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgD,d_cgA,d_coarseWeight);
                k_ClusterDirectionApply<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_cgD,d_cgA,d_coarseWeight);
""")
print("applied")
