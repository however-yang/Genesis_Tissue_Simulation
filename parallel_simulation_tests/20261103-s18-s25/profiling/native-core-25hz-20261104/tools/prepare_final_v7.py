from pathlib import Path
import shutil
p=Path(__file__).resolve().parents[1];src=p/'native-final-v7/source';shutil.copytree(p/'native-tool-v2/source',src)
f=src/'vendor/cuda/contact/surface_contact.cuh';s=f.read_text()
a=s.index('__global__ void k_FindSurfaceLean(');b=s.index('static_assert(sizeof(SurfaceContact)',a);s=s[:a]+s[b:]
a=s.index('        if(g_coreOptimizations&4){');b=s.index('        k_AppendQueryKinds<<<',a)
old='''        k_FindSurfaceBatched<<<dim3((triangles+g_queryThreads-1)/g_queryThreads,kindCount),g_queryThreads,0,stream>>>(triangles,packed,bladeBatch,from,to,skin,
            vertices,masks,prev,pos,contacts,degree,current,previous,body,thickness,minGap,stationary,cacheSlots,cache,bladePhase,stride,groupBase,w.ids,w.counts);
'''
s=s[:a]+old+s[b:];f.write_text(s)
f=src/'vendor/cuda/xpbd/xpbd_solver.cu';s=f.read_text().replace('if(mode<0 || mode>7 || g_ready)return -10001;g_coreOptimizations=mode;','if((mode!=0 && mode!=2) || g_ready)return -10001;g_coreOptimizations=mode;');f.write_text(s)
shutil.copy2(p/'native-material-v6/source/vendor/cuda/gpu_body_check.cuh',src/'vendor/cuda/gpu_body_check.cuh')
f=src/'vendor/cuda/dc_recon.cu';s=f.read_text().replace('#include "topology_checks.cuh"','#include "topology_checks.cuh"\n#include "gpu_body_check.cuh"',1).replace('    topoShutdown();','    topoShutdown();\n    bodyCheckShutdown();',1);f.write_text(s)
f=src/'vendor/src/plugin_api.cpp';s=f.read_text()+'''\nint recon_current_body_depth(int,const double*,int,double*);\nint recon_test_body_depth(int,const float*,const double*,int,double*);\nLCS_API int LCS_CurrentBodyDepth(int count,const double* parameters,int flags,double* output){return recon_current_body_depth(count,parameters,flags,output);}\nLCS_API int LCS_TestBodyDepth(int count,const float* positions,const double* parameters,int flags,double* output){return recon_test_body_depth(count,positions,parameters,flags,output);}\n''';f.write_text(s)
s=(p/'tools/build_tool_v2.py').read_text().replace('native-tool-v2','native-final-v7').replace('build-tool-v2','build-final-v7');(p/'tools/build_final_v7.py').write_text(s)
# Rejected branches cannot be selected from the final comparison matrix.
import json
cases=[dict(name='reference',core=0,gpu_body=False,dll='native-final-v7/build/lib/libLiverCudaSim.so'),dict(name='tool',core=2,gpu_body=False,dll='native-final-v7/build/lib/libLiverCudaSim.so'),dict(name='tool-gpu-body',core=2,gpu_body=True,dll='native-final-v7/build/lib/libLiverCudaSim.so')]
(p/'results/final-initial-config.json').write_text(json.dumps(cases,indent=2));print(src)
