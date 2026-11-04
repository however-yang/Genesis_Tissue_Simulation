"""Add, for a debug build only, an external force on chosen nodes over a range of frames: LCS_DebugSetPush(mask, fx,
fy, fz, first, last) gives every node whose int mask is nonzero the force (fx, fy, fz) in native units (0.01 N) in
frames first..last-1, counted as in debug_crease_dump.py (apply that first). The force enters each substep's inertial
target, as gravity does.

Usage: python debug_push.py apply   (from endoscopic_tissue/native/vendor/cuda; undo with git checkout)
"""
import sys
from pathlib import Path


def patch(path, old, new):
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, old[:80])
    Path(path).write_text(text.replace(old, new), encoding="utf-8")


assert sys.argv[1] == "apply"
SOLVER = "xpbd/xpbd_solver.cu"
patch(SOLVER, "static int g_debugFrame = 0;\n", """static int g_debugFrame = 0;
static int* d_debugPushMask=nullptr;static float3 g_debugPush={0,0,0};static int g_debugPushFirst=0,g_debugPushLast=0;
__global__ void k_DebugPush(int n,const int* mask,float3 force,float h,const float* invMass,float3* target){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n && mask[i] && invMass[i]>0)target[i]=target[i]+force*(h*h*invMass[i]);
}
extern "C" __declspec(dllexport) int LCS_DebugSetPush(const int* mask,float fx,float fy,float fz,int first,int last);
""")
patch(SOLVER, """    cudaMemcpyAsync(d_unifiedTarget,d_pos,(size_t)g_N*sizeof(float3),cudaMemcpyDeviceToDevice);
""", """    cudaMemcpyAsync(d_unifiedTarget,d_pos,(size_t)g_N*sizeof(float3),cudaMemcpyDeviceToDevice);
    if(d_debugPushMask && g_debugFrame>=g_debugPushFirst && g_debugFrame<g_debugPushLast)
        k_DebugPush<<<Gr(g_N),64>>>(g_N,d_debugPushMask,g_debugPush,h,d_invMass,d_unifiedTarget);
""")
text = Path(SOLVER).read_text(encoding="utf-8")
Path(SOLVER).write_text(text + """
extern "C" __declspec(dllexport) int LCS_DebugSetPush(const int* mask,float fx,float fy,float fz,int first,int last){
    if(!d_debugPushMask && cudaMalloc(&d_debugPushMask,g_N*sizeof(int))!=cudaSuccess)return -1;
    if(cudaMemcpy(d_debugPushMask,mask,g_N*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess)return -2;
    g_debugPush=make_float3(fx,fy,fz);g_debugPushFirst=first;g_debugPushLast=last;return 0;
}
""", encoding="utf-8")
print("applied")
