"""Apply reviewed opt-in training hooks to the isolated contact-groups source."""
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).parent
SOURCE=HERE/'native-baseline'
TARGET=HERE/'artifacts/training-state1/source'
if TARGET.exists() and sys.argv[1:]!=['--refresh']:raise FileExistsError(TARGET)
if sys.argv[1:] not in ([],['--refresh']):raise SystemExit('Only --refresh is supported')
assert TARGET.resolve().is_relative_to(HERE.resolve()/'artifacts')
shutil.copytree(SOURCE,TARGET,dirs_exist_ok=True)

def edit(path, old, new, count=1):
    file=TARGET/path
    text=file.read_text(encoding='utf-8-sig')
    if text.count(old)!=count:raise RuntimeError(f'{path}: expected {count} exact sites, found {text.count(old)}')
    file.write_text(text.replace(old,new),encoding='utf8')

xpbd='vendor/cuda/xpbd/xpbd_solver.cu'
cut='vendor/cuda/cut.cu'
edit(xpbd, '#include <utility>', '#include <utility>\n#include "../training/training_api.h"\n'
     'static int g_trainingFlags=0;\nstatic void trainingSubstep(float);\n'
     'static void trainingCapture();\nstatic void trainingShutdown();')
edit(xpbd, '        contactShape(stepTool,1,g_jawThickness,g_gp.jawGapClosed),contactShape(stepTool,2,g_jawThickness,g_gp.jawGapClosed));\n'
     '    k_update_velocity<<<Gr(g_N),64>>>(g_N,1.f/h,d_invMass,d_pos,d_prevPos,d_vel);',
     '        contactShape(stepTool,1,g_jawThickness,g_gp.jawGapClosed),contactShape(stepTool,2,g_jawThickness,g_gp.jawGapClosed));\n'
     '    // Report the forces actually applied in this substep before releasing an overloaded witness.\n'
     '    if(g_trainingFlags)trainingSubstep(h);\n'
     '    k_update_velocity<<<Gr(g_N),64>>>(g_N,1.f/h,d_invMass,d_pos,d_prevPos,d_vel);')
edit(xpbd, '    refreshGrip();g_grasping=true;if(!d_coupledTool)g_toolFrom=g_tool;',
     '    if(g_trainingFlags)trainingCapture();\n'
     '    refreshGrip();g_grasping=true;if(!d_coupledTool)g_toolFrom=g_tool;')
edit(xpbd, '    cudaFree(d_coreForceEMA); d_coreForceEMA = nullptr;',
     '    trainingShutdown();\n    cudaFree(d_coreForceEMA); d_coreForceEMA = nullptr;')
with (TARGET/xpbd).open('a',encoding='utf8') as f:f.write('\n#include "../training/xpbd_training.cuh"\n')
edit(cut, 'static float  g_tearLen2  = 0.f;', 'static unsigned char* d_trainingTearCause=nullptr;\n'
     'static int* d_trainingGraspNeighborhood=nullptr;\n'
     'static float  g_tearLen2  = 0.f;')
edit(cut, 'int* edgeWasExempt, int* edgeDecayTicks)\n{',
     'int* edgeWasExempt, int* edgeDecayTicks, unsigned char* trainingCause=nullptr,\n'
     'const int* trainingInitial=nullptr, const int* trainingInside=nullptr, const int* trainingNear=nullptr)\n{')
edit(cut, '    if (dbg) atomicAdd(&dbg[15], 1u);',
     '''    if(trainingCause){
        unsigned char cause=1;
        if(trainingInside[idA] && trainingInside[idB]){
            cause|=2;bool rim=false;
            for(int k=0;k<6;k++){
                int a=trainingInitial[6*idA+k],b=trainingInitial[6*idB+k];
                rim|=(a>=0 && trainingInside[a] && nbrIdx[6*idA+k]<0) ||
                     (b>=0 && trainingInside[b] && nbrIdx[6*idB+k]<0);
            }
            if(rim)cause|=4;if(trainingNear && (trainingNear[idA] || trainingNear[idB]))cause|=8;
        }trainingCause[e]=cause;
    }
    if (dbg) atomicAdd(&dbg[15], 1u);''')
edit(cut, 'c_particleRot, d_dbg, d_edgeWasExempt, d_edgeDecayTicks);',
     'c_particleRot, d_dbg, d_edgeWasExempt, d_edgeDecayTicks, d_trainingTearCause, d_nbrIdxAtInit, d_cornerInside, d_trainingGraspNeighborhood);')
edit(cut, '    g_toolSet = false; g_havePrevCorner = false; g_tearLen2 = 0.f;',
     '    cudaFree(d_trainingTearCause);d_trainingTearCause=nullptr;\n'
     '    cudaFree(d_trainingGraspNeighborhood);d_trainingGraspNeighborhood=nullptr;\n'
     '    g_toolSet = false; g_havePrevCorner = false; g_tearLen2 = 0.f;')
edit(cut, '    k_CarveRimBridge<<<Gc(cc), 64>>>(cc, c_nbrIdx, d_nbrIdxAtInit, d_graspRing);',
     '    if(d_trainingGraspNeighborhood){\n'
     '        cudaError_t e=cudaMemcpy(d_trainingGraspNeighborhood,d_graspRing,(size_t)cc*sizeof(int),cudaMemcpyDeviceToDevice);\n'
     '        if(e!=cudaSuccess)return -(int)e;\n'
     '    }\n'
     '    k_CarveRimBridge<<<Gc(cc), 64>>>(cc, c_nbrIdx, d_nbrIdxAtInit, d_graspRing);')
with (TARGET/cut).open('a',encoding='utf8') as f:f.write('\n#include "training/cut_training.cuh"\n')
api=TARGET/'vendor/src/plugin_api.cpp'
with api.open('a',encoding='utf8') as f:f.write('''
#include "training/training_api.h"
LCS_API int LCS_TrainingAbiVersion(){return 1;}
LCS_API int LCS_TrainingMetricsSize(){return sizeof(TrainingMetrics);}
LCS_API int LCS_EnableTrainingDiagnostics(int flags){return training_enable(flags);}
LCS_API int LCS_GetTrainingMetrics(TrainingMetrics* out){return training_read(out);}
LCS_API int LCS_GetTrainingCoreForces(int count,float* out){return training_core_forces(count,out);}
LCS_API int LCS_GetTrainingTearCauses(int count,unsigned char* out){return cut_training_causes(count,out);}
''')
(TARGET/'vendor/cuda/training').mkdir(exist_ok=True)
for p in (HERE/'training-native').glob('*'):
    shutil.copyfile(p,TARGET/'vendor/cuda/training'/p.name)
print('Prepared isolated training hooks:',TARGET)
