#pragma once
#include "config.cuh"
#include <cstdint>

namespace hyc_movecontrol {

// Python→CUDA 的输入结构。使用普通 float 数组，避免 CUDA float4 的 16 字节对齐差异。
// 字段顺序与 senario_control/hyc_movecontrol_config.py 的 ctypes 结构一一对应。
// 这里只传递场景数值，没有另一套默认参数；修改字段布局时必须同步两端。
struct SceneJointInput {
    float axis[3],origin_cm[3];
    float lower_deg,upper_deg,initial_deg,max_speed_deg_s;
};
struct SceneBackInput {float center[3],half_size[3],axis_x[3],axis_y[3];};
struct SceneInstrumentInput {
    SceneJointInput joints[JointCount];
    float base_position_cm[3],base_quaternion_xyzw[4];
    float mass_kg,stiffness,damping,max_travel_cm;
    std::int32_t guided;
    float jaw_mass_kg,jaw_stiffness,jaw_damping,jaw_speed_cm_s,opening_speed_multiplier;
    float jaw_length_cm,jaw_half_width_cm,max_gap_cm,shaft_length_cm,shaft_radius_cm;
    std::int32_t use_jaw_bounds;
    float jaw_bounds_center[2][3],jaw_bounds_half[2][3];
    std::int32_t scissors;
    float edge_a[2][3],edge_b[2][3],scissors_radius_cm;
    std::int32_t back_contact;
    SceneBackInput backs[2];
    float wrist_front_cm,wrist_length_cm,wrist_radius_cm;
};
struct SceneInput {
    // abi_version 校验传输版本；frame_dt_s/substeps 决定每个 CUDA 子步的时长。
    std::uint32_t abi_version;
    float frame_dt_s;
    std::int32_t substeps;
    // 固定顺序：instruments[0] 为夹爪，instruments[1] 为剪刀。
    SceneInstrumentInput instruments[2];
};
// 在编译期检查传输字节数，防止字段或内存对齐改变后悄悄读取错位。
static_assert(sizeof(SceneJointInput)==40,"Scene joint transport ABI");
static_assert(sizeof(SceneInstrumentInput)==468,"Scene instrument transport ABI");
static_assert(sizeof(SceneInput)==948,"Scene transport ABI");

__host__ __device__ inline float3 sceneVector(const float* p) {return mkf3(p[0],p[1],p[2]);}
// 将传输数组转换为内部 float3/float4 参数；检查成功才提交，不补齐缺失默认值。
__host__ __device__ inline Status configFromSceneInput(InstrumentConfig& config,const SceneInstrumentInput& input) {
    InstrumentConfig next={};
    for(int i=0;i<JointCount;i++){
        const auto& j=input.joints[i];
        next.joints[i]={sceneVector(j.axis),sceneVector(j.origin_cm),j.lower_deg,j.upper_deg,j.initial_deg,j.max_speed_deg_s};
    }
    next.base_position_cm=sceneVector(input.base_position_cm);
    const float* q=input.base_quaternion_xyzw;next.base_quaternion_xyzw=make_float4(q[0],q[1],q[2],q[3]);
    next.translation={input.mass_kg,input.stiffness,input.damping,input.max_travel_cm,input.guided};
    next.opening={input.jaw_mass_kg,input.jaw_stiffness,input.jaw_damping,input.jaw_speed_cm_s,input.opening_speed_multiplier};
    next.jaw_length_cm=input.jaw_length_cm;next.jaw_half_width_cm=input.jaw_half_width_cm;next.max_gap_cm=input.max_gap_cm;
    next.shaft_length_cm=input.shaft_length_cm;next.shaft_radius_cm=input.shaft_radius_cm;
    next.use_jaw_bounds=input.use_jaw_bounds;next.scissors=input.scissors;
    for(int i=0;i<2;i++){
        next.jaw_bounds_center[i]=sceneVector(input.jaw_bounds_center[i]);next.jaw_bounds_half[i]=sceneVector(input.jaw_bounds_half[i]);
        next.edge_a[i]=sceneVector(input.edge_a[i]);next.edge_b[i]=sceneVector(input.edge_b[i]);
        const auto& b=input.backs[i];next.backs[i]={sceneVector(b.center),sceneVector(b.half_size),sceneVector(b.axis_x),sceneVector(b.axis_y)};
    }
    next.scissors_radius_cm=input.scissors_radius_cm;next.back_contact=input.back_contact;
    next.wrist_front_cm=input.wrist_front_cm;next.wrist_length_cm=input.wrist_length_cm;next.wrist_radius_cm=input.wrist_radius_cm;
    if(!validConfig(next))return InvalidConfig;
    config=next;return Ok;
}
// 场景一帧被分成 substeps 个固定子步：h=frame_dt_s/substeps。
__host__ __device__ inline Status sceneSubstep(const SceneInput& input,float& h) {
    if(input.abi_version!=1)return InvalidCommand;
    if(!finite(input.frame_dt_s)||input.frame_dt_s<=0||input.substeps<=0)return InvalidTimestep;
    float next=input.frame_dt_s/input.substeps;
    if(!finite(next)||next<=0)return InvalidTimestep;
    h=next;return Ok;
}
// 上传 SceneInput 后调用本核函数，再用 configs 调用 k_InitializeInstruments。
// 两个线程分别转换两把器械；线程 0 写出共用子步时长，status 分别记录转换结果。
__global__ void k_LoadSceneInputs(const SceneInput* input,InstrumentConfig* configs,float* h,int* status) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=2)return;
    float step=0;Status result=sceneSubstep(*input,step);
    if(result==Ok)result=configFromSceneInput(configs[i],input->instruments[i]);
    if(result==Ok&&i==0&&h)*h=step;
    if(status)status[i]=result;
}

} // namespace hyc_movecontrol
