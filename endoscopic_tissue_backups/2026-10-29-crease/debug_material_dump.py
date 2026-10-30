"""Extend the dump of debug_crease_dump.py (apply that first) with the material solve of the frame's last substep: the
PCG direction of every node before and after the volume-floor restriction, the line search of its component (chosen
step k, 0 when every trial is rejected, and the invalid-trial bits) with the energy change of the chosen trial, and
its component root. Appended to crease-<frame>.bin after the flags: float32 free direction (n, 3), float32
restricted direction (n, 3), int32 best | invalid << 4 (n), float32 energy change (n), int32 component (n).

Usage: python debug_material_dump.py apply   (from endoscopic_tissue/native/vendor/cuda; undo with git checkout)
"""
import sys
from pathlib import Path


def patch(path, old, new):
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, old[:80])
    Path(path).write_text(text.replace(old, new), encoding="utf-8")


assert sys.argv[1] == "apply"
SEARCH = "solver/material/material_line_search.cuh"
patch(SEARCH, "\n// Bound the *trial* strain before tangent projection;", """
__global__ void k_DebugLineSummary(const UnifiedState* state,const MaterialLineSearch* search,int* info,float* gain){
    const UnifiedState& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n)return;
    const auto& q=search[s.component[i]];int best=0;double lowest=0,tolerance=1e-10*fmax(1.,fabs(q.energy[0]));
    for(int k=1;k<5;k++)if(!(q.invalid&(1<<k))&&isfinite(q.energy[k])&&q.energy[k]<lowest-tolerance){best=k;lowest=q.energy[k];}
    info[i]=best|(q.invalid<<4);gain[i]=float(best?q.energy[best]:0.);
}
// Bound the *trial* strain before tangent projection;""")
SOLVER = "xpbd/xpbd_solver.cu"
patch(SOLVER, "static MaterialLineSearch* d_cgSearch=nullptr;\n", """static MaterialLineSearch* d_cgSearch=nullptr;
static float3 *d_debugFree=nullptr,*d_debugUsed=nullptr;static int* d_debugLine=nullptr;static float* d_debugGain=nullptr;
""")
patch(SOLVER, "    if(e==cudaSuccess && !d_cgSearch)e=cudaMalloc(&d_cgSearch,g_N*sizeof(MaterialLineSearch));\n",
      """    if(e==cudaSuccess && !d_cgSearch)e=cudaMalloc(&d_cgSearch,g_N*sizeof(MaterialLineSearch));
    if(e==cudaSuccess && !d_debugFree){cudaMalloc(&d_debugFree,g_N*sizeof(float3));cudaMalloc(&d_debugUsed,g_N*sizeof(float3));
        cudaMalloc(&d_debugLine,g_N*sizeof(int));cudaMalloc(&d_debugGain,g_N*sizeof(float));
        cudaMemset(d_debugFree,0,g_N*sizeof(float3));cudaMemset(d_debugUsed,0,g_N*sizeof(float3));
        cudaMemset(d_debugLine,0,g_N*sizeof(int));cudaMemset(d_debugGain,0,g_N*sizeof(float));}
""")
patch(SOLVER, """                xpbd_section_mark("cg: iterations");
""", """                xpbd_section_mark("cg: iterations");
                cudaMemcpyAsync(d_debugFree,d_cgD,g_N*sizeof(float3),cudaMemcpyDeviceToDevice,g_captureStream);
""")
patch(SOLVER, """                k_MaterialLineApply<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgD,d_cgSearch);
""", """                k_DebugLineSummary<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgSearch,d_debugLine,d_debugGain);
                cudaMemcpyAsync(d_debugUsed,d_cgD,g_N*sizeof(float3),cudaMemcpyDeviceToDevice,g_captureStream);
                k_MaterialLineApply<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgD,d_cgSearch);
""")
patch(SOLVER, """fwrite(hostFlags.data(),sizeof(int),hostFlags.size(),file);fclose(file);}""",
      """fwrite(hostFlags.data(),sizeof(int),hostFlags.size(),file);
            std::vector<float> extra(3*(size_t)g_N);std::vector<int> ints(g_N);std::vector<float> floats(g_N);
            cudaMemcpy(extra.data(),d_debugFree,3*sizeof(float)*g_N,cudaMemcpyDeviceToHost);fwrite(extra.data(),sizeof(float),extra.size(),file);
            cudaMemcpy(extra.data(),d_debugUsed,3*sizeof(float)*g_N,cudaMemcpyDeviceToHost);fwrite(extra.data(),sizeof(float),extra.size(),file);
            cudaMemcpy(ints.data(),d_debugLine,sizeof(int)*g_N,cudaMemcpyDeviceToHost);fwrite(ints.data(),sizeof(int),ints.size(),file);
            cudaMemcpy(floats.data(),d_debugGain,sizeof(float)*g_N,cudaMemcpyDeviceToHost);fwrite(floats.data(),sizeof(float),floats.size(),file);
            cudaMemcpy(ints.data(),s.component,sizeof(int)*g_N,cudaMemcpyDeviceToHost);fwrite(ints.data(),sizeof(int),ints.size(),file);
            fclose(file);}""")
print("applied")
