"""Add a debug export of the material PCG system to a debug build, for offline preconditioner prototypes.

Usage: python debug_material_dump.py apply|restore  (from endoscopic_tissue/native/vendor)

apply copies the touched files to <scratch>/debug-material-backup and adds: copies of the PCG right-hand side (after the
stencil assembly) and of its solution (after the iterations) inside the captured substep, and LCS_DebugDumpMaterial,
which writes the last substep's stencil, scales, masks, diagonal, right-hand side, solution and component flags to a
binary file. restore copies the files back.
"""
import shutil
import sys
from pathlib import Path

backup = Path(r"C:\WINDOWS\TEMP\claude\D--Simulation-genesis-world\6de6da1d-3b43-4fa3-91f1-fd2ef998be2d\scratchpad\debug-material-backup")
files = ["cuda/xpbd/xpbd_solver.cu", "src/plugin_api.cpp"]
if sys.argv[1] == "restore":
    for f in files:
        shutil.copyfile(backup / Path(f).name, f)
    print("restored")
    sys.exit(0)
backup.mkdir(parents=True, exist_ok=True)
for f in files:
    shutil.copyfile(f, backup / Path(f).name)


def patch(path, old, new):
    text = open(path, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in text else "\n"
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    assert text.count(old) == 1, (path, old[:60], text.count(old))
    open(path, "w", encoding="utf-8", newline="").write(text.replace(old, new))


solver = "cuda/xpbd/xpbd_solver.cu"
patch(solver, "static Symmetric3* d_cgH=nullptr;",
      "static Symmetric3* d_cgH=nullptr;\nstatic float3* d_debugRhs=nullptr;static float3* d_debugStep=nullptr;")
patch(solver, "    if(e==cudaSuccess && !d_cgR)e=cudaMalloc(&d_cgR,g_N*sizeof(float3));",
      "    if(e==cudaSuccess && !d_cgR)e=cudaMalloc(&d_cgR,g_N*sizeof(float3));\n"
      "    if(e==cudaSuccess && !d_debugRhs)e=cudaMalloc(&d_debugRhs,g_N*sizeof(float3));\n"
      "    if(e==cudaSuccess && !d_debugStep)e=cudaMalloc(&d_debugStep,g_N*sizeof(float3));")
patch(solver, "                k_MaterialStencilAssemble<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgR,d_cgH);",
      "                k_MaterialStencilAssemble<<<Gr(g_N),64,0,g_captureStream>>>(d_unified,d_cgMatrix,d_cgScale,d_cgR,d_cgH);\n"
      "                cudaMemcpyAsync(d_debugRhs,d_cgR,(size_t)g_N*sizeof(float3),cudaMemcpyDeviceToDevice,g_captureStream);")
patch(solver, '                xpbd_section_mark("cg: iterations");',
      '                xpbd_section_mark("cg: iterations");\n'
      "                cudaMemcpyAsync(d_debugStep,d_cgD,(size_t)g_N*sizeof(float3),cudaMemcpyDeviceToDevice,g_captureStream);")
patch(solver, 'extern "C" float xpbd_contact_repair_tolerance(){',
      '''extern "C" int xpbd_debug_dump_material(const char* path){
    if(!d_cgMatrix || !d_debugRhs)return -1;
    cudaDeviceSynchronize();int n=g_N,dims[3]={g_dimsX+1,g_dimsY+1,g_dimsZ+1};
    std::vector<__half> matrix((size_t)72*n);std::vector<float> scale((size_t)8*n);std::vector<unsigned short> mask(n);
    std::vector<float3> rhs(n),step(n);std::vector<Symmetric3> diagonal(n);std::vector<int> component(n),grasp(n);
    cudaMemcpy(matrix.data(),d_cgMatrix,matrix.size()*sizeof(__half),cudaMemcpyDeviceToHost);
    cudaMemcpy(scale.data(),d_cgScale,scale.size()*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(mask.data(),d_cgStencil,mask.size()*sizeof(unsigned short),cudaMemcpyDeviceToHost);
    cudaMemcpy(rhs.data(),d_debugRhs,n*sizeof(float3),cudaMemcpyDeviceToHost);
    cudaMemcpy(step.data(),d_debugStep,n*sizeof(float3),cudaMemcpyDeviceToHost);
    cudaMemcpy(diagonal.data(),d_cgH,n*sizeof(Symmetric3),cudaMemcpyDeviceToHost);
    cudaMemcpy(component.data(),d_unifiedComponents,n*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(grasp.data(),d_graspComponents,n*sizeof(int),cudaMemcpyDeviceToHost);
    FILE* f=fopen(path,"wb");if(!f)return -2;
    fwrite(&n,sizeof(int),1,f);fwrite(dims,sizeof(int),3,f);
    fwrite(matrix.data(),sizeof(__half),matrix.size(),f);fwrite(scale.data(),sizeof(float),scale.size(),f);
    fwrite(mask.data(),sizeof(unsigned short),mask.size(),f);fwrite(rhs.data(),sizeof(float3),n,f);
    fwrite(step.data(),sizeof(float3),n,f);fwrite(diagonal.data(),sizeof(Symmetric3),n,f);
    fwrite(component.data(),sizeof(int),n,f);fwrite(grasp.data(),sizeof(int),n,f);fclose(f);
    return (int)cudaGetLastError();
}
extern "C" float xpbd_contact_repair_tolerance(){''')
plugin = "src/plugin_api.cpp"
patch(plugin, 'extern "C" int xpbd_relax_reconstructed_contact();',
      'extern "C" int xpbd_relax_reconstructed_contact();\nextern "C" int xpbd_debug_dump_material(const char*);\n'
      'LCS_API int LCS_DebugDumpMaterial(const char* path){return xpbd_debug_dump_material(path);}')
print("applied")
