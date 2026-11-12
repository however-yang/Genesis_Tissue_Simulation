#pragma once
#include "math.cuh"
#include "../xpbd/xpbd_solver.h"

namespace hyc_movecontrol {

// 本文件只描述参数结构和合法范围；数值由场景通过 scene_input.cuh 传入。
// 位置/间隙使用厘米，角度使用度，时间使用秒，质量使用千克。
// stiffness、damping 沿用老师求解器的原生单位，不在这里重复进行单位换算。
// 状态码由调用方检查：0 成功，负数分别表示配置、指令、步长或初始化错误。
enum Status { Ok=0, InvalidConfig=-1, InvalidCommand=-2, InvalidTimestep=-3, NotInitialized=-4 };
// 数组顺序固定，必须与 Python 打包关节数据的顺序一致。
enum JointIndex { Roll=0, Pitch=1, Yaw=2, Opening=3, JointCount=4 };

// 关节转轴和铰点偏移均相对父节点定义；initial_deg 是场景的初始控制角。
struct JointConfig {
    float3 axis;
    float3 origin_cm;
    float lower_deg,upper_deg,initial_deg,max_speed_deg_s;
};
// 整体平移弹簧：质量、刚度、阻尼，以及驱动目标每个子步允许移动的最大距离。
// guided=1 时，工具仅沿请求的移动方向求解；guided=0 时允许三维平移响应。
struct TranslationConfig {
    float mass_kg,stiffness,damping,max_travel_cm;
    int guided;
};
// 开合弹簧以两瓣尖端间隙为坐标；张开速度=闭合速度×opening_speed_multiplier。
struct OpeningConfig {
    float mass_kg,stiffness,damping,closing_speed_cm_s,opening_speed_multiplier;
};
// 单把器械的完整运行参数。没有独立 CUDA 默认值，也不负责读取场景文件。
struct InstrumentConfig {
    // 底座世界位姿与 Roll→Pitch→Yaw→Opening 关节配置。
    JointConfig joints[JointCount];
    float3 base_position_cm;
    float4 base_quaternion_xyzw;
    TranslationConfig translation;
    OpeningConfig opening;
    // 夹口尺寸及轴杆胶囊尺寸；max_gap_cm 是场景允许的最大尖端间隙。
    float jaw_length_cm,jaw_half_width_cm,max_gap_cm;
    float shaft_length_cm,shaft_radius_cm;
    int use_jaw_bounds;
    // 每瓣钳口在闭合铰点坐标系中的包围盒中心与半尺寸，供接触代理使用。
    float3 jaw_bounds_center[2],jaw_bounds_half[2];
    int scissors;
    // 剪刀两刃的有限线段端点；局部坐标轴依次是尖端方向、开合方向、宽度方向。
    float3 edge_a[2],edge_b[2];
    float scissors_radius_cm;
    int back_contact;
    XpbdScissorsBackBox backs[2];
    // 腕部胶囊：从铰点向前 front、向后 length；radius=0 表示不启用该代理。
    float wrist_front_cm,wrist_length_cm,wrist_radius_cm;
};

// 仅校验，不修正或补齐输入。参数不合法时由上层返回 InvalidConfig。
__host__ __device__ inline bool validConfig(const InstrumentConfig& c) {
    if(!finite(c.base_position_cm)||!validQuaternion(c.base_quaternion_xyzw))return false;
    if(!finite(c.jaw_length_cm)||c.jaw_length_cm<=0 || !finite(c.jaw_half_width_cm)||c.jaw_half_width_cm<=0 ||
       !finite(c.max_gap_cm)||c.max_gap_cm<=0||c.max_gap_cm>1.9f*c.jaw_length_cm ||
       !finite(c.shaft_length_cm)||c.shaft_length_cm<=0||!finite(c.shaft_radius_cm)||c.shaft_radius_cm<0)return false;
    for(int i=0;i<JointCount;i++){
        const auto& j=c.joints[i];float length=len3(j.axis);
        if(!finite(j.axis)||!finite(length)||length<1e-8f||!finite(j.origin_cm)||
           !finite(j.lower_deg)||!finite(j.upper_deg)||!finite(j.initial_deg)||
           j.lower_deg>j.initial_deg||j.initial_deg>j.upper_deg||
           !finite(j.max_speed_deg_s)||j.max_speed_deg_s<=0)return false;
    }
    // 当前模型要求两瓣绕 Yaw 的同一铰点、本地 +Z 轴对称开合。
    // 最大开合角对应的尖端距离还必须落在场景设定的最大间隙内。
    const auto& jaw=c.joints[Opening];
    if(jaw.lower_deg<0||jaw.upper_deg>=180||len3(jaw.origin_cm)>1e-8f||
       len3(unit(jaw.axis)-mkf3(0,0,1))>1e-5f ||
       2*c.jaw_length_cm*sinf(radians(jaw.upper_deg)*0.5f)>c.max_gap_cm+1e-6f)return false;
    // 正质量和正刚度保证弹簧有效质量可求逆；阻尼允许为零。
    const auto& t=c.translation;const auto& o=c.opening;
    if(!finite(t.mass_kg)||t.mass_kg<=0||!finite(t.stiffness)||t.stiffness<=0||
       !finite(t.damping)||t.damping<0||!finite(t.max_travel_cm)||t.max_travel_cm<0||
       (t.guided!=0&&t.guided!=1)||!finite(o.mass_kg)||o.mass_kg<=0||
       !finite(o.stiffness)||o.stiffness<=0||!finite(o.damping)||o.damping<0||
       !finite(o.closing_speed_cm_s)||o.closing_speed_cm_s<=0||
       !finite(o.opening_speed_multiplier)||o.opening_speed_multiplier<=0)return false;
    if((c.scissors!=0&&c.scissors!=1)||(c.back_contact!=0&&c.back_contact!=1))return false;
    if(c.use_jaw_bounds!=0&&c.use_jaw_bounds!=1)return false;
    if(c.use_jaw_bounds)for(int i=0;i<2;i++){
        const auto& half=c.jaw_bounds_half[i];
        if(!finite(c.jaw_bounds_center[i])||!finite(half)||half.x<=0||half.y<=0||half.z<=0)return false;
    }
    if(!finite(c.scissors_radius_cm)||c.scissors_radius_cm<0 ||
       !finite(c.wrist_front_cm)||!finite(c.wrist_length_cm)||!finite(c.wrist_radius_cm)||c.wrist_radius_cm<0||
       (c.wrist_radius_cm>0&&c.wrist_front_cm+c.wrist_length_cm<=0))return false;
    // 剪刀刀背盒的方向必须正交，双刃线段不能退化为一个点。
    if(c.scissors){
        for(int i=0;i<2;i++){
            if(!finite(c.edge_a[i])||!finite(c.edge_b[i])||len3(c.edge_b[i]-c.edge_a[i])<1e-8f)return false;
            if(c.back_contact){const auto& b=c.backs[i];
                if(!finite(b.center)||!finite(b.halfSize)||!finite(b.axisX)||!finite(b.axisY)||
                   b.halfSize.x<=0||b.halfSize.y<=0||b.halfSize.z<=0||
                   fabsf(len3(b.axisX)-1)>1e-4f||fabsf(len3(b.axisY)-1)>1e-4f||fabsf(dot3(b.axisX,b.axisY))>1e-4f)return false;
            }
        }
    }
    return true;
}

} // namespace hyc_movecontrol
