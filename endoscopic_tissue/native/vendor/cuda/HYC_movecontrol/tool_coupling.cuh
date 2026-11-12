#pragma once
#include "joints.cuh"
#include "translation.cuh"
#include "contact_geometry.cuh"

namespace hyc_movecontrol {

// 统一流程：场景配置→初始化→关节限速/父子变换→平移和开合预测→外部接触修正→速度回写。
// 这里只控制器械；组织求解器的接触、夹持和切割阶段仍由外部接入。
// 更新剪刀代理几何和剪切标记，不修改当前实际位姿。
__host__ __device__ inline void configureBody(CoupledTool& body,const InstrumentConfig& config,int cutting) {
    body.scissors=config.scissors;body.scissorsCutting=config.scissors?cutting:0;body.knifeBody=0;
    // 沿用老师的规则：剪刀间隙上限取模型限制，夹爪间隙上限随驱动目标扩展。
    if(config.scissors)body.maxGap=config.max_gap_cm;
    body.scissorsRadius=config.scissors_radius_cm;
    body.backContact=config.back_contact;
    for(int i=0;i<2;i++){body.edgeA[i]=config.edge_a[i];body.edgeB[i]=config.edge_b[i];body.backs[i]=config.backs[i];}
    body.wristFront=config.wrist_front_cm;body.wristLength=config.wrist_length_cm;body.wristRadius=config.wrist_radius_cm;
}
// 检查输出部件变换，避免 NaN、非正交方向或负间隙进入显示/接触适配。
__host__ __device__ inline bool validLinks(const LinkPoses& p) {
    return validTransform(p.shaft)&&validTransform(p.roll)&&validTransform(p.pitch)&&validTransform(p.yaw)&&
           validTransform(p.upper_jaw)&&validTransform(p.lower_jaw)&&finite(p.shaft_start_cm)&&
           finite(p.shaft_end_cm)&&finite(p.gap_cm)&&p.gap_cm>=0;
}
// 检查主要工具状态；成功输出才允许提交给调用方。
__host__ __device__ inline bool validToolState(const CoupledTool& body) {
    const auto& p=body.pose;
    return finite(p.jawHinge)&&finite(p.jawAxis)&&finite(p.jawNormal)&&finite(p.jawMid)&&
           finite(p.shaftA)&&finite(p.shaftB)&&finite(p.jawGap)&&p.jawGap>=0&&
           finite(body.velocity)&&finite(body.gapVelocity)&&finite(body.inverseMass)&&
           finite(body.gapInverseMass)&&finite(body.drivePosition)&&finite(body.drivePrediction)&&finite(body.driveGap);
}
// 根据场景的初始底座、初始角和尺寸建立零速度状态，不隐式执行一个运动子步。
__host__ __device__ inline Status initializeInstrument(InstrumentState& state,const InstrumentConfig& config) {
    if(!validConfig(config))return InvalidConfig;
    InstrumentState next={};
    next.base=quaternionTransform(config.base_position_cm,config.base_quaternion_xyzw);
    for(int i=0;i<JointCount;i++)next.joints[i]={config.joints[i].initial_deg,config.joints[i].initial_deg,0};
    next.commanded=forwardKinematics(next.base,next.joints,config);
    if(!validLinks(next.commanded))return InvalidConfig;
    next.actual=next.commanded;next.tool.pose=toolPose(next.commanded,config);next.tool.previous=next.tool.pose;
    next.tool.drivePosition=next.tool.drivePrediction=next.tool.pose.jawHinge;next.tool.driveGap=next.tool.pose.jawGap;
    next.tool.maxGap=next.tool.pose.jawGap;
    next.tool.initialized=1;next.initialized=1;configureBody(next.tool,config,0);
    state=next;return Ok;
}
// 一个 h 子步只推进一次。先在 next 副本中计算，全部合法后再更新 state。
// 因此错误配置、越界目标或溢出不会留下只更新了一部分关节的状态。
__host__ __device__ inline Status predictInstrument(InstrumentState& state,const InstrumentConfig& config,
                                                    const InstrumentCommand& command,float h) {
    if(!validConfig(config))return InvalidConfig;
    if(!validCommand(command,config))return InvalidCommand;
    if(!finite(h)||h<=0)return InvalidTimestep;
    if(!state.initialized)return NotInitialized;
    const auto& t=config.translation;const auto& o=config.opening;
    float mass=t.mass_kg+h*t.damping+h*h*t.stiffness;
    float jawMass=o.mass_kg+h*o.damping+h*h*o.stiffness;
    if(!finite(mass)||!finite(jawMass))return InvalidTimestep;
    for(int i=0;i<JointCount;i++)if(!finite(config.joints[i].max_speed_deg_s*h))return InvalidTimestep;
    InstrumentState next=state;
    next.base=quaternionTransform(command.base_position_cm,command.base_quaternion_xyzw);
    for(int i=0;i<JointCount;i++){
        next.joints[i].target_deg=command.target_deg[i];advanceJoint(next.joints[i],config.joints[i],h);
    }
    // 先计算运动学目标，再将末端目标交给老师的平移/开合弹簧公式。
    next.commanded=forwardKinematics(next.base,next.joints,config);
    if(!validLinks(next.commanded))return InvalidCommand;
    XpbdToolState target=toolPose(next.commanded,config);
    predictTool(&next.tool,target,state.tool.pose,h,t.mass_kg,t.stiffness,t.damping,t.max_travel_cm,
                o.mass_kg,o.stiffness,o.damping,o.closing_speed_cm_s,target.jawHinge,target.jawGap,
                t.guided,o.opening_speed_multiplier);
    configureBody(next.tool,config,command.cutting);
    synchronizeLinks(next,config);
    if(!validToolState(next.tool)||!validLinks(next.actual))return InvalidCommand;
    state=next;return Ok;
}
// 外部接触求解结束后应用 tool.delta/tool.gapDelta，并同步实际部件位姿。
__host__ __device__ inline Status applyInstrument(InstrumentState& state,const InstrumentConfig& config) {
    if(!validConfig(config))return InvalidConfig;
    if(!state.initialized)return NotInitialized;
    if(!finite(state.tool.delta)||!finite(state.tool.gapDelta))return InvalidCommand;
    InstrumentState next=state;applyTool(&next.tool);synchronizeLinks(next,config);
    if(!validToolState(next.tool)||!validLinks(next.actual))return InvalidCommand;
    state=next;return Ok;
}
// 应在本子步所有位置修正完成后调用，用最终实际位移回写速度。
__host__ __device__ inline Status endInstrument(InstrumentState& state) {
    if(!state.initialized)return NotInitialized;
    if(!finite(state.tool.h)||state.tool.h<=0)return InvalidTimestep;
    InstrumentState next=state;endTool(&next.tool);
    if(!validToolState(next.tool))return InvalidCommand;
    state=next;return Ok;
}

// 批量 CUDA 入口：一个线程处理一把器械，count 是数组长度。
// 初始化调用一次；每子步按 Predict→外部接触→Apply→End 顺序调用。
// 每次调用后检查 CUDA 错误与 status；本模块没有内部时钟或自动循环。
__global__ void k_InitializeInstruments(int count,InstrumentState* states,const InstrumentConfig* configs,int* status) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    Status result=initializeInstrument(states[i],configs[i]);if(status)status[i]=result;
}
__global__ void k_PredictInstruments(int count,InstrumentState* states,const InstrumentConfig* configs,
                                   const InstrumentCommand* commands,float h,int* status) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    Status result=predictInstrument(states[i],configs[i],commands[i],h);if(status)status[i]=result;
}
__global__ void k_ApplyInstruments(int count,InstrumentState* states,const InstrumentConfig* configs,int* status) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    Status result=applyInstrument(states[i],configs[i]);if(status)status[i]=result;
}
__global__ void k_EndInstruments(int count,InstrumentState* states,int* status) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    Status result=endInstrument(states[i]);if(status)status[i]=result;
}

} // namespace hyc_movecontrol
