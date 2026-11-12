#include "../../../../native/vendor/cuda/HYC_movecontrol/tool_coupling.cuh"
#include "../../../../native/vendor/cuda/HYC_movecontrol/scene_input.cuh"
#include <fstream>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

// Compile the actual teacher headers alongside HYC, rather than a handwritten imitation.
namespace teacher_reference {
using hyc_movecontrol::mkf3;
using hyc_movecontrol::operator+;
using hyc_movecontrol::operator-;
using hyc_movecontrol::operator*;
using hyc_movecontrol::dot3;
using hyc_movecontrol::len3;
using hyc_movecontrol::cross3;
using hyc_movecontrol::compF3;
#include "../../../../native/vendor/cuda/contact/friction_contact.cuh"
#include "../../../../native/vendor/cuda/grasp/tool_coupling.cuh"
}

namespace h=hyc_movecontrol;
using h::operator+;
using h::operator-;
using h::operator*;
static_assert(sizeof(h::CoupledTool)==sizeof(teacher_reference::CoupledTool),"Teacher body layout changed");
static_assert(offsetof(h::CoupledTool,wristRadius)==offsetof(teacher_reference::CoupledTool,wristRadius),"Teacher body layout changed");

static int assertions=0;
static h::SceneInput sceneInput={};
static std::array<h::InstrumentConfig,2> sceneConfigs={};
static void require(bool condition,const char* message) {
    ++assertions;if(!condition)throw std::runtime_error(message);
}
static void gpu(cudaError_t result) {
    if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
}
template<class T> struct Device {
    T* data=nullptr;
    explicit Device(size_t count){gpu(cudaMalloc(&data,sizeof(T)*count));}
    ~Device(){if(data)cudaFree(data);}
    void upload(const T* source,size_t count){gpu(cudaMemcpy(data,source,sizeof(T)*count,cudaMemcpyHostToDevice));}
    void download(T* dest,size_t count){gpu(cudaMemcpy(dest,data,sizeof(T)*count,cudaMemcpyDeviceToHost));}
};
static void near(float a,float b,float tolerance,const char* message) {
    require(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tolerance,message);
}
static void near(float3 a,float3 b,float tolerance,const char* message) {
    near(a.x,b.x,tolerance,message);near(a.y,b.y,tolerance,message);near(a.z,b.z,tolerance,message);
}
static void compareBody(const h::CoupledTool& a,const teacher_reference::CoupledTool& b) {
    near(a.pose.jawHinge,b.pose.jawHinge,1e-6f,"teacher hinge mismatch");
    near(a.pose.shaftA,b.pose.shaftA,1e-6f,"teacher shaft mismatch");
    near(a.pose.jawAxis,b.pose.jawAxis,1e-6f,"teacher orientation mismatch");
    near(a.pose.jawGap,b.pose.jawGap,1e-6f,"teacher gap mismatch");
    near(a.velocity,b.velocity,1e-6f,"teacher velocity mismatch");
    near(a.gapVelocity,b.gapVelocity,1e-6f,"teacher gap velocity mismatch");
    near(a.inverseMass,b.inverseMass,1e-6f,"teacher effective mass mismatch");
    near(a.gapInverseMass,b.gapInverseMass,1e-6f,"teacher gap effective mass mismatch");
    near(a.drivePosition,b.drivePosition,1e-6f,"teacher retained drive mismatch");
    near(a.drivePrediction,b.drivePrediction,1e-6f,"teacher unconstrained prediction mismatch");
    near(a.driveGap,b.driveGap,1e-6f,"teacher drive gap mismatch");
    near(a.gapLo,b.gapLo,1e-6f,"teacher guide gap bound mismatch");
    near(a.gapHi,b.gapHi,1e-6f,"teacher guide gap bound mismatch");
}

// Independent double-precision 4x4 matrix reference for hierarchy multiplication.
struct Matrix {double a[4][4]={};};
static Matrix identity(){Matrix m;for(int i=0;i<4;i++)m.a[i][i]=1;return m;}
static Matrix multiply(const Matrix& a,const Matrix& b){
    Matrix m;for(int i=0;i<4;i++)for(int j=0;j<4;j++)for(int k=0;k<4;k++)m.a[i][j]+=a.a[i][k]*b.a[k][j];return m;
}
static Matrix referenceJoint(const h::JointConfig& config,double angle) {
    Matrix m=identity();double axis[3]={config.axis.x,config.axis.y,config.axis.z};
    double length=std::sqrt(axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2]);
    for(auto& coordinate:axis)coordinate/=length;
    double skew[3][3]={{0,-axis[2],axis[1]},{axis[2],0,-axis[0]},{-axis[1],axis[0],0}};
    double c=std::cos(angle*3.141592653589793/180),s=std::sin(angle*3.141592653589793/180);
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)m.a[i][j]=(i==j?c:0)+(1-c)*axis[i]*axis[j]+s*skew[i][j];
    m.a[0][3]=config.origin_cm.x;m.a[1][3]=config.origin_cm.y;m.a[2][3]=config.origin_cm.z;return m;
}
static Matrix asMatrix(const h::Transform& t){
    Matrix m=identity();float3 columns[4]={t.x,t.y,t.z,t.position};
    for(int j=0;j<4;j++){m.a[0][j]=columns[j].x;m.a[1][j]=columns[j].y;m.a[2][j]=columns[j].z;}return m;
}
static void compareFrame(const h::Transform& frame,const Matrix& reference){
    Matrix actual=asMatrix(frame);
    for(int i=0;i<4;i++)for(int j=0;j<4;j++)require(std::abs(actual.a[i][j]-reference.a[i][j])<4e-6,"hierarchy matrix mismatch");
}
static void testHierarchyAndValidation() {
    for(auto config:{sceneConfigs[0],sceneConfigs[1]}){
        require(h::validConfig(config),"preset rejected");
        h::InstrumentState state={};require(h::initializeInstrument(state,config)==h::Ok,"initialization rejected");
        for(int sample=0;sample<128;sample++){
            // Non-unit, oblique wrist axes check axis normalization and parent-local pivots.
            config.joints[h::Pitch].axis=h::mkf3(.4f,-2.f,.3f);
            config.joints[h::Yaw].origin_cm=h::mkf3(.91f,.2f,-.1f);
            state.joints[h::Roll].angle_deg=-160.f+2.5f*sample;
            state.joints[h::Pitch].angle_deg=-60.f+.8f*sample;
            state.joints[h::Yaw].angle_deg=45.f-.6f*sample;
            auto p=h::forwardKinematics(state.base,state.joints,config);
            Matrix expected=asMatrix(state.base);
            expected=multiply(expected,referenceJoint(config.joints[h::Roll],state.joints[h::Roll].angle_deg));compareFrame(p.roll,expected);
            expected=multiply(expected,referenceJoint(config.joints[h::Pitch],state.joints[h::Pitch].angle_deg));compareFrame(p.pitch,expected);
            expected=multiply(expected,referenceJoint(config.joints[h::Yaw],state.joints[h::Yaw].angle_deg));compareFrame(p.yaw,expected);
            compareFrame(p.upper_jaw,multiply(expected,referenceJoint(config.joints[h::Opening],state.joints[h::Opening].angle_deg*.5)));
            compareFrame(p.lower_jaw,multiply(expected,referenceJoint(config.joints[h::Opening],-state.joints[h::Opening].angle_deg*.5)));
            near(h::len3((p.upper_jaw.position+p.upper_jaw.x*config.jaw_length_cm)-
                        (p.lower_jaw.position+p.lower_jaw.x*config.jaw_length_cm)),p.gap_cm,2e-6f,"jaw tip separation mismatch");
        }
    }
    auto config=sceneConfigs[0];h::InstrumentState state={};h::initializeInstrument(state,config);
    auto before=state;
    auto command=h::initialCommand(config);
    command.target_deg[h::Pitch]=config.joints[h::Pitch].upper_deg+1;
    require(h::predictInstrument(state,config,command,.01f)==h::InvalidCommand,"out-of-range command accepted");
    require(std::memcmp(&state,&before,sizeof(state))==0,"invalid command partially applied");
    command=h::initialCommand(config);command.base_quaternion_xyzw=make_float4(0,0,0,0);
    require(h::predictInstrument(state,config,command,.01f)==h::InvalidCommand,"zero quaternion accepted");
    command=h::initialCommand(config);
    require(h::predictInstrument(state,config,command,0)==h::InvalidTimestep,"zero timestep accepted");
    require(h::predictInstrument(state,config,command,FLT_MAX)==h::InvalidTimestep,"overflowing timestep accepted");
    require(std::memcmp(&state,&before,sizeof(state))==0,"invalid input mutated state");
    config.joints[h::Opening].axis=h::mkf3(0,1,0);
    require(!h::validConfig(config),"unsupported jaw axis accepted");
    config=sceneConfigs[0];config.translation.mass_kg=0;
    require(!h::validConfig(config),"zero mass accepted");
}

__global__ void syntheticCorrection(h::CoupledTool* actual,teacher_reference::CoupledTool* reference) {
    actual->delta=reference->delta=h::mkf3(.0003f,-.0002f,.0001f);
    actual->gapDelta=reference->gapDelta=.0002f;
}
static void testTeacherOnGpu() {
    constexpr int count=16,steps=16;
    std::array<h::CoupledTool,count> actual={};std::array<teacher_reference::CoupledTool,count> reference={};
    Device<h::CoupledTool> deviceActual(count);Device<teacher_reference::CoupledTool> deviceReference(count);
    deviceActual.upload(actual.data(),count);deviceReference.upload(reference.data(),count);
    for(int step=0;step<steps;step++)for(int i=0;i<count;i++){
        auto config=sceneConfigs[0];h::InstrumentState initial={};h::initializeInstrument(initial,config);
        XpbdToolState first=initial.tool.pose,command=first;
        command.jawHinge=first.jawHinge+h::mkf3(.15f,-.05f,.08f);
        command.shaftA=first.shaftA+h::mkf3(.15f,-.05f,.08f);command.shaftB=first.shaftB+h::mkf3(.15f,-.05f,.08f);
        command.jawGap=i%2?.25f:.9f;
        float dt=(i%3+1)*.002f,mass=.01f+.002f*i,stiffness=300.f+50.f*i,damping=5.f;
        float jawStiffness=i%4?150.f:0,travel=.002f*(i%3+1);int guided=i%2;
        h::k_PredictTool<<<1,1>>>(deviceActual.data+i,command,first,dt,mass,stiffness,damping,travel,
            .005f,jawStiffness,2.f,1.2f,command.jawHinge,command.jawGap,guided);
        teacher_reference::k_PredictTool<<<1,1>>>(deviceReference.data+i,command,first,dt,mass,stiffness,damping,travel,
            .005f,jawStiffness,2.f,1.2f,command.jawHinge,command.jawGap,guided);
        syntheticCorrection<<<1,1>>>(deviceActual.data+i,deviceReference.data+i);
        h::k_ApplyTool<<<1,1>>>(deviceActual.data+i);teacher_reference::k_ApplyTool<<<1,1>>>(deviceReference.data+i);
        h::k_EndTool<<<1,1>>>(deviceActual.data+i);teacher_reference::k_EndTool<<<1,1>>>(deviceReference.data+i);
        gpu(cudaGetLastError());
        gpu(cudaMemcpy(actual.data()+i,deviceActual.data+i,sizeof(actual[i]),cudaMemcpyDeviceToHost));
        gpu(cudaMemcpy(reference.data()+i,deviceReference.data+i,sizeof(reference[i]),cudaMemcpyDeviceToHost));
        compareBody(actual[i],reference[i]);
    }
    // Shape/projection helpers must also agree for shaft, jaws, scissors blades, backs and wrist.
    auto config=sceneConfigs[1];h::InstrumentState state={};h::initializeInstrument(state,config);
    teacher_reference::CoupledTool original={};std::memcpy(&original,&state.tool,sizeof(original));
    for(int kind=0;kind<=5;kind++){
        auto shape=h::bodyContactShape(&state.tool,state.tool.pose,kind,.15f,0);
        auto expected=teacher_reference::bodyContactShape(&original,original.pose,kind,.15f,0);
        near(shape.a,expected.a,1e-6f,"contact capsule mismatch");near(shape.b,expected.b,1e-6f,"contact capsule mismatch");
        near(shape.center,expected.center,1e-6f,"contact box mismatch");near(shape.u,expected.u,1e-6f,"contact axis mismatch");
    }
    std::printf("Teacher comparison: 16 cases x 16 GPU substeps, including guided motion and synthetic corrections, passed.\n");
}

static void testInstrumentsOnGpu() {
    std::array<h::InstrumentConfig,2> configs={sceneConfigs[0],sceneConfigs[1]};
    std::array<h::InstrumentCommand,2> commands={h::initialCommand(configs[0]),h::initialCommand(configs[1])};
    std::array<h::InstrumentState,2> states={};std::array<int,2> status={};
    for(int i=0;i<2;i++){
        commands[i].base_position_cm=commands[i].base_position_cm+h::mkf3(.1f,-.1f,.05f);
        commands[i].target_deg[h::Roll]=15;commands[i].target_deg[h::Pitch]=-20;commands[i].target_deg[h::Yaw]=25;
        commands[i].target_deg[h::Opening]=i?50.f:10.f;
    }
    Device<h::InstrumentConfig> deviceConfigs(2);Device<h::InstrumentCommand> deviceCommands(2);
    Device<h::InstrumentState> deviceStates(2);Device<int> deviceStatus(2);
    Device<h::SceneInput> deviceInput(1);Device<float> deviceSubstep(1);
    deviceInput.upload(&sceneInput,1);
    h::k_LoadSceneInputs<<<1,32>>>(deviceInput.data,deviceConfigs.data,deviceSubstep.data,deviceStatus.data);
    gpu(cudaGetLastError());deviceStatus.download(status.data(),2);
    require(status[0]==h::Ok&&status[1]==h::Ok,"GPU scene parameter conversion failed");
    std::array<h::InstrumentConfig,2> decoded={};deviceConfigs.download(decoded.data(),2);
    for(int i=0;i<2;i++){
        near(decoded[i].opening.stiffness,configs[i].opening.stiffness,0,"GPU discarded scene stiffness");
        near(decoded[i].base_position_cm,configs[i].base_position_cm,0,"GPU discarded scene base pose");
        near(decoded[i].translation.max_travel_cm,configs[i].translation.max_travel_cm,0,"GPU discarded scene travel limit");
    }
    deviceCommands.upload(commands.data(),2);
    gpu(h::uploadJawBounds(configs[0]));
    h::k_InitializeInstruments<<<1,32>>>(2,deviceStates.data,deviceConfigs.data,deviceStatus.data);
    gpu(cudaGetLastError());deviceStatus.download(status.data(),2);
    require(status[0]==h::Ok&&status[1]==h::Ok,"GPU initialization failed");
    for(int step=0;step<120;step++){
        h::k_PredictInstruments<<<1,32>>>(2,deviceStates.data,deviceConfigs.data,deviceCommands.data,1.f/120,deviceStatus.data);
        gpu(cudaGetLastError());deviceStatus.download(status.data(),2);
        require(status[0]==h::Ok&&status[1]==h::Ok,"GPU prediction failed");
        h::k_ApplyInstruments<<<1,32>>>(2,deviceStates.data,deviceConfigs.data,deviceStatus.data);
        gpu(cudaGetLastError());deviceStatus.download(status.data(),2);
        require(status[0]==h::Ok&&status[1]==h::Ok,"GPU correction failed");
        h::k_EndInstruments<<<1,32>>>(2,deviceStates.data,deviceStatus.data);
        gpu(cudaGetLastError());deviceStatus.download(status.data(),2);
        require(status[0]==h::Ok&&status[1]==h::Ok,"GPU completion failed");
        deviceStates.download(states.data(),2);
        for(int i=0;i<2;i++){
            require(h::validLinks(states[i].actual),"non-finite or non-rigid actual link frame");
            for(int j=0;j<h::JointCount;j++)require(std::abs(states[i].joints[j].velocity_deg_s)<=configs[i].joints[j].max_speed_deg_s+1e-3f,"GPU joint speed exceeded");
            near(states[i].actual.yaw.position,states[i].tool.pose.jawHinge,2e-6f,"actual hinge differs from physical hinge");
            near(h::len3(states[i].actual.upper_jaw.x*configs[i].jaw_length_cm-states[i].actual.lower_jaw.x*configs[i].jaw_length_cm),
                 states[i].tool.pose.jawGap,3e-6f,"actual jaw frames ignore resisted gap");
        }
    }
    for(int i=0;i<2;i++)for(int j=0;j<h::JointCount;j++)near(states[i].joints[j].angle_deg,commands[i].target_deg[j],1e-4f,"GPU target not reached");
    auto before=states;
    commands[1].target_deg[h::Pitch]=100;deviceCommands.upload(commands.data(),2);
    h::k_PredictInstruments<<<1,32>>>(2,deviceStates.data,deviceConfigs.data,deviceCommands.data,1.f/120,deviceStatus.data);
    gpu(cudaGetLastError());deviceStatus.download(status.data(),2);deviceStates.download(states.data(),2);
    require(status[0]==h::Ok&&status[1]==h::InvalidCommand,"GPU per-instrument error isolation failed");
    require(std::memcmp(&before[1],&states[1],sizeof(states[1]))==0,"invalid GPU command changed instrument");
    std::printf("Instrument comparison: grasper + scissors, 120 GPU substeps, hierarchy, speed and actual jaw gap passed.\n");
}

int main(int argc,char** argv){
    try {
        require(argc==2,"Pass the scene input file prepared from the scene TOML");
        std::ifstream input(argv[1],std::ios::binary);
        require(bool(input.read(reinterpret_cast<char*>(&sceneInput),sizeof(sceneInput))),"Could not read scene input");
        float substep=0;require(h::sceneSubstep(sceneInput,substep)==h::Ok,"Invalid scene timing");
        for(int i=0;i<2;i++)require(h::configFromSceneInput(sceneConfigs[i],sceneInput.instruments[i])==h::Ok,"Invalid scene configuration");
        near(sceneConfigs[1].opening.stiffness,sceneInput.instruments[1].jaw_stiffness,0,"Scene jaw stiffness overridden");
        cudaDeviceProp device={};gpu(cudaGetDeviceProperties(&device,0));
        std::printf("GPU: %s (sm_%d%d)\n",device.name,device.major,device.minor);
        testHierarchyAndValidation();testTeacherOnGpu();testInstrumentsOnGpu();gpu(cudaDeviceSynchronize());
        std::printf("PASS: %d assertions\n",assertions);return 0;
    } catch(const std::exception& error){std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
