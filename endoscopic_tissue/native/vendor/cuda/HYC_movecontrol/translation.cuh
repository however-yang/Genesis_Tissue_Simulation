#pragma once
#include "state.cuh"

namespace hyc_movecontrol {

// 平移与开合逻辑来自老师的 grasp/tool_coupling.cuh，保留弹簧、导向和修正公式。
// 张开速度倍率可由配置传入；底层兼容接口保留老师的默认倍率 4。
// h 是子步时长；driveTarget/gapTarget 是驱动目标，b->pose 是实际工具状态。
__host__ __device__ inline XpbdToolState shiftTool(XpbdToolState t,float3 d){
    t.jawHinge=t.jawHinge+d;t.jawMid=t.jawMid+d;
    t.shaftA=t.shaftA+d;t.shaftB=t.shaftB+d;return t;
}
__host__ __device__ inline void predictTool(CoupledTool* b,XpbdToolState command,XpbdToolState initial,
    float h,float mass,float stiffness,float damping,float maxTravel,
    float jawMass,float jawStiffness,float jawDamping,float jawSpeed,float3 driveTarget,float gapTarget,int guided=0,float openingSpeedMultiplier=4.f){
    // 首次调用建立初始位置、间隙和零速度；后续调用延续上一子步的驱动状态。
    if(!b->initialized){b->pose=initial;b->drivePosition=initial.jawHinge;b->driveGap=initial.jawGap;b->velocity=mkf3(0,0,0);b->gapVelocity=0;b->maxGap=initial.jawGap;b->initialized=1;}
    b->previous=b->pose;b->h=h;b->feasibleNormal=nullptr;b->guided=guided;
    // 阻尼弹簧的隐式欧拉有效质量：m_eff=m+h*c+h²*k。
    // 接触求解使用同一个有效逆质量，使组织与工具在本子步内响应彼此的修正。
    float effectiveMass=mass+h*damping+h*h*stiffness;
    b->inverseMass=1.f/effectiveMass;
    // 限制驱动目标每步推进的距离，不截断接触受力产生的实际位移。
    // drivePosition 跨子步保存；不能用受阻位置覆盖它，否则会清除累积预载。
    float3 driveMove=driveTarget-b->drivePosition;float travel=len3(driveMove);
    if(travel>maxTravel)driveMove=driveMove*(maxTravel/travel);
    b->drivePosition=guided?driveTarget:b->drivePosition+driveMove;
    float3 request=driveTarget-b->previous.jawHinge;float distance=len3(request);
    b->guideOrigin=b->previous.jawHinge;b->guideAxis=distance>1e-6f?request*(1.f/distance):mkf3(0,0,0);
    b->guideLength=distance>1e-6f?fminf(distance,maxTravel):0;
    // v_next=(m*v_prev+h*k*(驱动目标-实际铰点))/m_eff，随后用 h*v_next 预测位移。
    float3 v=(b->velocity*mass+(b->drivePosition-b->pose.jawHinge)*(h*stiffness))*b->inverseMass;
    float3 move=v*h;
    // 保存未受导向限制的弹簧预测位置；guided 只限制当前几何迭代的位置。
    b->drivePrediction=b->previous.jawHinge+move;
    if(guided)move=b->guideAxis*fminf(b->guideLength,fmaxf(0.f,dot3(move,b->guideAxis)));
    // 使用命令中的方向和几何，将铰点平移到预测位置；旋转方向仍为指定运动学方向。
    b->pose=shiftTool(command,b->previous.jawHinge+move-command.jawHinge);
    b->delta=mkf3(0,0,0);
    b->gapDelta=0;b->gapInverseMass=0;
    // 将两瓣尖端间隙视为一个弹簧自由度，和整体平移使用同样的隐式更新规则。
    if(jawStiffness>0){
        b->maxGap=fmaxf(b->maxGap,gapTarget);
        b->gapInverseMass=1.f/(jawMass+h*jawDamping+h*h*jawStiffness);
        float openingSpeed=openingSpeedMultiplier*jawSpeed;
        b->driveGap+=fminf(openingSpeed*h,fmaxf(-jawSpeed*h,gapTarget-b->driveGap));
        float v=(jawMass*b->gapVelocity+h*jawStiffness*(b->driveGap-b->previous.jawGap))*b->gapInverseMass;
        b->pose.jawGap=fminf(b->maxGap,fmaxf(0.f,b->previous.jawGap+v*h));
    }
    b->gapLo=fminf(b->previous.jawGap,b->driveGap);
    // 闭合时允许组织压力将夹口重新撑开，不能把上一间隙当作只闭不张的锁扣。
    // 张开阶段的上界同时覆盖上一间隙和当前驱动间隙；guided 再应用导向范围。
    b->gapHi=b->driveGap<b->previous.jawGap?b->maxGap:fmaxf(b->previous.jawGap,b->driveGap);
    if(guided)b->pose.jawGap=fminf(b->gapHi,fmaxf(b->gapLo,b->pose.jawGap));
}
// 应用外部接触求解器累计的位移/间隙修正，然后清空本轮修正量。
__host__ __device__ inline void applyTool(CoupledTool* b){
    b->pose=shiftTool(b->pose,b->delta);b->delta=mkf3(0,0,0);
    if(b->gapInverseMass>0)b->pose.jawGap=fminf(b->maxGap,fmaxf(0.f,b->pose.jawGap+b->gapDelta));b->gapDelta=0;
}
// 子步结束时，以修正后的实际位置差计算速度，而不是直接采用驱动目标速度。
__host__ __device__ inline void endTool(CoupledTool* b){
    b->velocity=(b->pose.jawHinge-b->previous.jawHinge)*(1.f/b->h);
    b->gapVelocity=(b->pose.jawGap-b->previous.jawGap)/b->h;
    b->feasibleNormal=nullptr;
}
// 下面的核函数保留老师的工具修正接口：分别管理位置、速度及可移动性限制。
__global__ void k_ClearContactMobility(CoupledTool* b){b->feasibleNormal=nullptr;}
__global__ void k_ClearToolVelocityDelta(CoupledTool* b){b->delta=mkf3(0,0,0);b->gapDelta=0;}
__global__ void k_ApplyToolVelocity(CoupledTool* b){
    b->velocity=b->velocity+b->delta;b->gapVelocity+=b->gapDelta;
    b->delta=mkf3(0,0,0);b->gapDelta=0;
}
__global__ void k_ContactMobility(CoupledTool* b,int n,const float* invMass,const int* needed,
    const float3* start,const float3* target,const float3* accepted,float3* blocked){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i==0)b->feasibleNormal=blocked;if(i>=n)return;
    float3 normal=mkf3(0,0,0);
    {float3 rejected=target[i]-accepted[i];float length=len3(rejected);
        if(length>1e-6f)normal=rejected*(1.f/length);}
    // 用“目标位置-已接受位置”记录本轮被拒绝的移动方向。
    // 在该轮协调中暂时禁止这些节点移动；下一子步会恢复材料移动能力。
    blocked[i]=normal;
}
// 接触修正的节点移动能力：受本轮几何保护限制的节点返回零，否则按逆质量响应。
__device__ inline float3 contactMobility(const CoupledTool* b,int id,float inverseMass,float3 impulse){
    if(b && b->feasibleNormal && dot3(b->feasibleNormal[id],b->feasibleNormal[id])>.5f)return mkf3(0,0,0);
    return impulse*inverseMass;
}

// 底层接口保留老师的参数形式；场景控制优先使用 tool_coupling.cuh 的组合入口。
__global__ void k_PredictTool(CoupledTool* b,XpbdToolState command,XpbdToolState initial,
    float h,float mass,float stiffness,float damping,float maxTravel,
    float jawMass,float jawStiffness,float jawDamping,float jawSpeed,float3 driveTarget,float gapTarget,int guided=0){
    predictTool(b,command,initial,h,mass,stiffness,damping,maxTravel,jawMass,jawStiffness,jawDamping,jawSpeed,driveTarget,gapTarget,guided);
}
__global__ void k_ApplyTool(CoupledTool* b){applyTool(b);}
__global__ void k_EndTool(CoupledTool* b){endTool(b);}

} // namespace hyc_movecontrol
