#pragma once
#include "translation.cuh"

namespace hyc_movecontrol {
// 在独立命名空间中复用老师的接触几何，保留其辅助函数依赖顺序。
#include "../contact/friction_contact.cuh"

// kind：0 杆体，1/2 两瓣或双刃，3/4 两片刀背，5 剪刀腕部。
// 夹爪使用有向包围盒，剪刀刃口使用有限胶囊；thickness/minGap 沿用原接触参数。
__host__ __device__ inline ContactShape bodyContactShape(const CoupledTool* body,XpbdToolState t,int kind,float thickness,float minGap){
    if(!body || (!body->scissors && !body->knifeBody) || kind==0)return contactShape(t,kind,thickness,minGap);
    ContactShape s={};s.capsule=1;s.radius=body->scissorsRadius;
    float3 u=contactUnit(t.jawAxis,mkf3(0,0,1)),n=contactUnit(t.jawNormal,mkf3(0,1,0));
    float sn=body->scissors?fminf(.95f,fmaxf(0.f,t.jawGap)/(2.f*fmaxf(t.jawLen,1e-6f))):0;
    float cs=sqrtf(1-sn*sn),side=(kind==1||kind==3)?1.f:-1.f;
    // 将闭合模型坐标旋转到当前瓣片方向，再把局部代理放到世界铰点位置。
    float3 along=u*cs+n*(side*sn),opening=n*cs-u*(side*sn),width=cross3(u,n);
    if(kind==5){
        s={};if(!body->scissors || !(body->wristRadius>0))return s;
        s.capsule=1;s.radius=body->wristRadius;
        s.a=t.jawHinge+u*body->wristFront;s.b=t.jawHinge-u*body->wristLength;return s;
    }
    if(kind>=3){
        s={};if(!body->backContact)return s;
        const auto& box=body->backs[kind-3];s.half=box.halfSize;
        s.center=t.jawHinge+along*box.center.x+opening*box.center.y+width*box.center.z;
        s.u=along*box.axisX.x+opening*box.axisX.y+width*box.axisX.z;
        s.n=along*box.axisY.x+opening*box.axisY.y+width*box.axisY.z;s.w=cross3(s.u,s.n);return s;
    }
    float3 a=body->edgeA[kind-1],b=body->edgeB[kind-1];
    s.a=t.jawHinge+along*a.x+opening*a.y+width*a.z;
    s.b=t.jawHinge+along*b.x+opening*b.y+width*b.z;
    return s;
}
// 用间隙正/负微扰计算接触见证点对尖端间隙的梯度，供开合接触修正使用。
// 见证点始终取自同一套有限代理几何，使开合梯度与接触检测保持一致。
__device__ inline float3 jawGapGradient(XpbdToolState t,int kind,float thickness,float minGap,float3 point,const CoupledTool* body=nullptr){
    if(kind==0 || (body && body->knifeBody))return mkf3(0,0,0);
    float eps=fmaxf(1e-5f,1e-3f*t.jawLen);XpbdToolState plus=t,minus=t;
    plus.jawGap+=eps;minus.jawGap=fmaxf(0.f,t.jawGap-eps);
    ContactShape from=bodyContactShape(body,t,kind,thickness,minGap);
    return (witnessAt(from,bodyContactShape(body,plus,kind,thickness,minGap),point)-
            witnessAt(from,bodyContactShape(body,minus,kind,thickness,minGap),point))*(1.f/(plus.jawGap-minus.jawGap));
}

// 将场景提供的钳口包围盒上传 CUDA 常量区；CPU 端也保存相同数据。
// 与老师实现一样，每个 CUDA 编译单元共享一套夹爪几何；不是逐器械独立常量。
inline cudaError_t uploadJawBounds(const InstrumentConfig& config){
    if(!validConfig(config))return cudaErrorInvalidValue;
    for(int i=0;i<2;i++)h_jawMeshBounds[i]={config.jaw_bounds_center[i],config.jaw_bounds_half[i],config.use_jaw_bounds};
    return cudaMemcpyToSymbol(c_jawMeshBounds,h_jawMeshBounds,sizeof(h_jawMeshBounds));
}

} // namespace hyc_movecontrol
