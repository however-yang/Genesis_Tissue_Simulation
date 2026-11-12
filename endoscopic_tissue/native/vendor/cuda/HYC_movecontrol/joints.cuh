#pragma once
#include "state.cuh"

namespace hyc_movecontrol {

// 先检查整条输入，避免一个关节更新后才发现另一目标越界。
__host__ __device__ inline bool validCommand(const InstrumentCommand& command,const InstrumentConfig& config) {
    if(!finite(command.base_position_cm)||!validQuaternion(command.base_quaternion_xyzw)||
       (command.cutting!=0&&command.cutting!=1))return false;
    for(int i=0;i<JointCount;i++)
        if(!finite(command.target_deg[i])||command.target_deg[i]<config.joints[i].lower_deg||
           command.target_deg[i]>config.joints[i].upper_deg)return false;
    return true;
}
// 初始命令完全取自场景参数；构造命令不推进时间、不执行运动。
__host__ __device__ inline InstrumentCommand initialCommand(const InstrumentConfig& config) {
    InstrumentCommand command={};command.base_position_cm=config.base_position_cm;
    command.base_quaternion_xyzw=config.base_quaternion_xyzw;
    for(int i=0;i<JointCount;i++)command.target_deg[i]=config.joints[i].initial_deg;
    return command;
}
__host__ __device__ inline void advanceJoint(JointState& state,const JointConfig& config,float h) {
    // 沿用 RevoluteJoint.advance()：本步最大角度变化=角速度上限×h。
    // 这里推进的是运动学控制角，没有计算关节惯性或关节力矩。
    float d=clamp(state.target_deg-state.angle_deg,-config.max_speed_deg_s*h,config.max_speed_deg_s*h);
    state.angle_deg+=d;state.velocity_deg_s=d/h;
}
// 父子链：底座→Roll→Pitch→Yaw→上下瓣，Yaw 与 Opening 共享最后一个铰点。
// gap_cm<0 时按控制角生成目标开度；传入非负值时按实际间隙重建上下瓣。
__host__ __device__ inline LinkPoses forwardKinematics(const Transform& base,const JointState* joints,
                                                       const InstrumentConfig& config,float gap_cm=-1.f) {
    LinkPoses p={};p.shaft=base;
    p.roll=compose(base,jointTransform(config.joints[Roll].axis,config.joints[Roll].origin_cm,joints[Roll].angle_deg));
    p.pitch=compose(p.roll,jointTransform(config.joints[Pitch].axis,config.joints[Pitch].origin_cm,joints[Pitch].angle_deg));
    p.yaw=compose(p.pitch,jointTransform(config.joints[Yaw].axis,config.joints[Yaw].origin_cm,joints[Yaw].angle_deg));
    // opening 是两瓣总夹角，每瓣绕局部开合轴旋转相反的半角。
    float opening=joints[Opening].angle_deg;
    p.gap_cm=gap_cm<0?2*config.jaw_length_cm*sinf(radians(opening)*.5f):gap_cm;
    if(gap_cm>=0)opening=degrees(2*asinf(clamp(gap_cm/(2*config.jaw_length_cm),0,.95f)));
    p.upper_jaw=compose(p.yaw,jointTransform(config.joints[Opening].axis,config.joints[Opening].origin_cm,opening*.5f));
    p.lower_jaw=compose(p.yaw,jointTransform(config.joints[Opening].axis,config.joints[Opening].origin_cm,-opening*.5f));
    // 沿用原运动模块的杆体长度计算：总杆长扣除腕部标称偏移，得到外露杆段。
    float wrist_span=len3(config.joints[Pitch].origin_cm+config.joints[Yaw].origin_cm);
    float exposed_length=fmaxf(0,config.shaft_length_cm-wrist_span);
    p.shaft_start_cm=p.roll.position-base.x*exposed_length;p.shaft_end_cm=p.roll.position;
    return p;
}
// 将层级位姿转为老师求解器使用的末端结构：铰点、方向、杆端点、尺寸和间隙。
__host__ __device__ inline XpbdToolState toolPose(const LinkPoses& p,const InstrumentConfig& config) {
    XpbdToolState t={};t.shaftA=p.shaft_start_cm;t.shaftB=p.shaft_end_cm;t.shaftR=config.shaft_radius_cm;
    t.jawHinge=p.yaw.position;t.jawAxis=p.yaw.x;t.jawNormal=p.yaw.y;
    t.jawMid=t.jawHinge+t.jawAxis*(config.jaw_length_cm*.5f);
    t.jawHalfWidth=config.jaw_half_width_cm;t.jawLen=config.jaw_length_cm;t.jawGap=p.gap_cm;
    t.grasp01=clamp(1-p.gap_cm/config.max_gap_cm,0,1);return t;
}
// 平移修正对全部部件施加相同位移，保留腕部父子关系和所有旋转方向。
__host__ __device__ inline void translateLinks(LinkPoses& p,float3 delta) {
    p.shaft.position=p.shaft.position+delta;p.roll.position=p.roll.position+delta;
    p.pitch.position=p.pitch.position+delta;p.yaw.position=p.yaw.position+delta;
    p.upper_jaw.position=p.upper_jaw.position+delta;p.lower_jaw.position=p.lower_jaw.position+delta;
    p.shaft_start_cm=p.shaft_start_cm+delta;p.shaft_end_cm=p.shaft_end_cm+delta;
}
__host__ __device__ inline void synchronizeLinks(InstrumentState& state,const InstrumentConfig& config) {
    // 用实际间隙重建两瓣，再把整条链移至实际铰点；腕部角度仍由控制指令指定。
    // 显示时读取 actual，控制目标仍保留在 commanded 和 joints 中。
    state.actual=forwardKinematics(state.base,state.joints,config,state.tool.pose.jawGap);
    translateLinks(state.actual,state.tool.pose.jawHinge-state.actual.yaw.position);
}

} // namespace hyc_movecontrol
