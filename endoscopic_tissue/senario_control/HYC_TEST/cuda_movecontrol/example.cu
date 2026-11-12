#include "../../../native/vendor/cuda/HYC_movecontrol/tool_coupling.cuh"
#include "../../../native/vendor/cuda/HYC_movecontrol/scene_input.cuh"
#include <cstdio>
#include <fstream>

namespace h=hyc_movecontrol;
static bool check(cudaError_t result){
    if(result==cudaSuccess)return true;std::fprintf(stderr,"%s\n",cudaGetErrorString(result));return false;
}
int main(int argc,char** argv){
    if(argc!=2){std::fprintf(stderr,"Pass the scene input file; no CUDA defaults are supplied.\n");return 1;}
    h::SceneInput input={};std::ifstream file(argv[1],std::ios::binary);
    if(!file.read(reinterpret_cast<char*>(&input),sizeof(input))){std::fprintf(stderr,"Cannot read scene input.\n");return 1;}
    float substep=0;
    if(h::sceneSubstep(input,substep)!=h::Ok){std::fprintf(stderr,"Invalid scene timing.\n");return 1;}
    h::InstrumentConfig configs[2]={};
    for(int i=0;i<2;i++)if(h::configFromSceneInput(configs[i],input.instruments[i])!=h::Ok){
        std::fprintf(stderr,"Invalid scene instrument configuration.\n");return 1;
    }
    h::InstrumentCommand commands[2]={h::initialCommand(configs[0]),h::initialCommand(configs[1])};
    // The scene supplies initial poses and targets. This example advances one scene frame.
    h::InstrumentConfig* deviceConfigs=nullptr;h::InstrumentCommand* deviceCommands=nullptr;
    h::InstrumentState* deviceStates=nullptr;int* deviceStatus=nullptr;
    int result=1;int status[2]={};h::InstrumentState states[2]={};
    if(!check(cudaMalloc(&deviceConfigs,sizeof(configs)))||!check(cudaMalloc(&deviceCommands,sizeof(commands)))||
       !check(cudaMalloc(&deviceStates,sizeof(states)))||!check(cudaMalloc(&deviceStatus,sizeof(status))))goto cleanup;
    if(!check(cudaMemcpy(deviceConfigs,configs,sizeof(configs),cudaMemcpyHostToDevice))||
       !check(cudaMemcpy(deviceCommands,commands,sizeof(commands),cudaMemcpyHostToDevice)))goto cleanup;
    h::k_InitializeInstruments<<<1,32>>>(2,deviceStates,deviceConfigs,deviceStatus);
    if(!check(cudaGetLastError())||!check(cudaMemcpy(status,deviceStatus,sizeof(status),cudaMemcpyDeviceToHost)))goto cleanup;
    if(status[0]||status[1])goto cleanup;
    for(int step=0;step<input.substeps;step++){
        h::k_PredictInstruments<<<1,32>>>(2,deviceStates,deviceConfigs,deviceCommands,substep,deviceStatus);
        if(!check(cudaGetLastError())||!check(cudaMemcpy(status,deviceStatus,sizeof(status),cudaMemcpyDeviceToHost)))goto cleanup;
        if(status[0]||status[1])goto cleanup;
        // An external tissue solver may accumulate tool.delta and tool.gapDelta here.
        h::k_ApplyInstruments<<<1,32>>>(2,deviceStates,deviceConfigs,deviceStatus);
        if(!check(cudaGetLastError())||!check(cudaMemcpy(status,deviceStatus,sizeof(status),cudaMemcpyDeviceToHost)))goto cleanup;
        if(status[0]||status[1])goto cleanup;
        h::k_EndInstruments<<<1,32>>>(2,deviceStates,deviceStatus);
        if(!check(cudaGetLastError())||!check(cudaMemcpy(status,deviceStatus,sizeof(status),cudaMemcpyDeviceToHost)))goto cleanup;
        if(status[0]||status[1])goto cleanup;
    }
    if(!check(cudaMemcpy(states,deviceStates,sizeof(states),cudaMemcpyDeviceToHost)))goto cleanup;
    for(int i=0;i<2;i++)std::printf("%s: roll=%.3f pitch=%.3f yaw=%.3f gap_cm=%.6f hinge_cm=(%.4f,%.4f,%.4f) jaw_stiffness=%.3f\n",
        i?"scissors":"grasper",states[i].joints[h::Roll].angle_deg,states[i].joints[h::Pitch].angle_deg,
        states[i].joints[h::Yaw].angle_deg,states[i].tool.pose.jawGap,
        states[i].actual.yaw.position.x,states[i].actual.yaw.position.y,states[i].actual.yaw.position.z,configs[i].opening.stiffness);
    result=0;
cleanup:
    if(status[0]||status[1])std::fprintf(stderr,"HYC status: %d %d\n",status[0],status[1]);
    if(deviceStatus)cudaFree(deviceStatus);if(deviceStates)cudaFree(deviceStates);
    if(deviceCommands)cudaFree(deviceCommands);if(deviceConfigs)cudaFree(deviceConfigs);return result;
}
