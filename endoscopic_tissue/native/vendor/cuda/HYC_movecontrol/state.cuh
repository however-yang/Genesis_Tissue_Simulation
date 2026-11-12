#pragma once
#include "config.cuh"

namespace hyc_movecontrol {

// 保留老师 grasp/tool_coupling.cuh 的字段顺序，用独立命名空间区分类型。
// 字段布局相同不代表能直接替换主求解器指针；接入时仍需显式传递状态。
struct CoupledTool {
    // pose 是当前实际状态，previous 是本子步开始前的实际状态。
    XpbdToolState pose,previous;
    float3 velocity,delta;
    // inverseMass 是隐式弹簧的有效逆质量；h 是当前子步时长。
    float inverseMass,h;
    int initialized;
    // 与整体平移相对应的开合速度、间隙修正、有效逆质量和最大间隙。
    float gapVelocity,gapDelta,gapInverseMass,maxGap;
    const float3* feasibleNormal;
    // 驱动目标跨子步保留，避免接触受阻时丢失弹簧预载。
    float3 drivePosition,drivePrediction;float driveGap;
    // guided 的移动线段及允许的开合区间。
    int guided;float3 guideAxis,guideOrigin;float guideLength,gapLo,gapHi;
    int scissors,scissorsCutting,knifeBody;
    // 接触代理数据：双刃、刀背和腕部；这些字段本身不执行组织切割。
    float3 edgeA[2],edgeB[2];
    float scissorsRadius;
    int backContact;
    XpbdScissorsBackBox backs[2];
    float wristFront,wristLength,wristRadius;
};

// angle_deg 是当前控制角，target_deg 是目标角；两者不等同于受力求解结果。
struct JointState { float angle_deg,target_deg,velocity_deg_s; };
// 六个部件的世界变换，供外部显示或接触适配使用。
struct LinkPoses {
    Transform shaft,roll,pitch,yaw,upper_jaw,lower_jaw;
    float3 shaft_start_cm,shaft_end_cm;
    float gap_cm;
};
// 一把器械独立拥有工具、关节和底座状态，数组中的两把器械互不覆盖。
struct InstrumentState {
    CoupledTool tool;
    JointState joints[JointCount];
    Transform base;
    // commanded 是运动学目标；actual 跟随实际工具铰点和实际开度。
    LinkPoses commanded,actual;
    int initialized;
};
struct InstrumentCommand {
    // 输入绝对底座世界位姿、四个关节目标角；时间步长由外部调用方提供。
    float3 base_position_cm;
    float4 base_quaternion_xyzw;
    float target_deg[JointCount];
    // cutting 只记录剪切许可，实际断裂判定仍由组织求解器完成。
    int cutting=0;
};

} // namespace hyc_movecontrol
