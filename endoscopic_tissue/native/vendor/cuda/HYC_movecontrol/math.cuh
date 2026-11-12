#pragma once
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace hyc_movecontrol {

// 使用列向量坐标约定：x/y/z 是旋转矩阵的三列，position 是平移列，单位厘米。
// __host__ __device__ 表示同一辅助函数既可在 CPU 使用，也可被 CUDA 核函数调用。
struct Transform {
    float3 x, y, z, position;
};

__host__ __device__ inline float3 mkf3(float x, float y, float z) { return make_float3(x, y, z); }
__host__ __device__ inline float3 operator+(float3 a, float3 b) { return mkf3(a.x+b.x,a.y+b.y,a.z+b.z); }
__host__ __device__ inline float3 operator-(float3 a, float3 b) { return mkf3(a.x-b.x,a.y-b.y,a.z-b.z); }
__host__ __device__ inline float3 operator*(float3 a, float s) { return mkf3(a.x*s,a.y*s,a.z*s); }
__host__ __device__ inline float dot3(float3 a, float3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__host__ __device__ inline float len3(float3 a) { return sqrtf(dot3(a,a)); }
__host__ __device__ inline float3 cross3(float3 a, float3 b) {
    return mkf3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
}
__host__ __device__ inline float compF3(float3 v, int k) { return k==0?v.x:(k==1?v.y:v.z); }
__host__ __device__ inline bool finite(float v) { return v==v && fabsf(v)<=FLT_MAX; }
__host__ __device__ inline bool finite(float3 v) { return finite(v.x)&&finite(v.y)&&finite(v.z); }
__host__ __device__ inline float clamp(float x, float lo, float hi) { return fminf(hi,fmaxf(lo,x)); }
__host__ __device__ inline float radians(float degrees) { return degrees*0.01745329251994329577f; }
__host__ __device__ inline float degrees(float radians) { return radians*57.29577951308232088f; }
__host__ __device__ inline float3 unit(float3 axis) { return axis*(1.f/len3(axis)); }
__host__ __device__ inline Transform identityTransform() {
    return {mkf3(1,0,0),mkf3(0,1,0),mkf3(0,0,1),mkf3(0,0,0)};
}
__host__ __device__ inline float3 transformVector(const Transform& t, float3 v) {
    return t.x*v.x+t.y*v.y+t.z*v.z;
}
// 世界变换=父节点世界变换×子节点局部变换；子节点铰点偏移随父节点方向旋转。
__host__ __device__ inline Transform compose(const Transform& parent, const Transform& child) {
    return {transformVector(parent,child.x),transformVector(parent,child.y),
            transformVector(parent,child.z),parent.position+transformVector(parent,child.position)};
}
// Rodrigues 公式实现任意轴旋转；转轴先归一化，输入角度从度转换为弧度。
__host__ __device__ inline float3 rotateVector(float3 vector, float3 axis, float angle_deg) {
    axis=unit(axis);
    float c=cosf(radians(angle_deg)),s=sinf(radians(angle_deg));
    return vector*c+cross3(axis,vector)*s+axis*(dot3(axis,vector)*(1-c));
}
// 构造父→子的局部变换：平移至铰点 origin_cm，再表达子节点旋转后的方向。
__host__ __device__ inline Transform jointTransform(float3 axis, float3 origin_cm, float angle_deg) {
    Transform t=identityTransform();
    t.x=rotateVector(t.x,axis,angle_deg);t.y=rotateVector(t.y,axis,angle_deg);
    t.z=rotateVector(t.z,axis,angle_deg);t.position=origin_cm;return t;
}
__host__ __device__ inline bool validQuaternion(float4 q) {
    return finite(q.x)&&finite(q.y)&&finite(q.z)&&finite(q.w) &&
        fmaxf(fmaxf(fabsf(q.x),fabsf(q.y)),fmaxf(fabsf(q.z),fabsf(q.w)))>0;
}
__host__ __device__ inline Transform quaternionTransform(float3 position_cm, float4 q) {
    // 四元数顺序为 (x,y,z,w)。先缩放再归一化，避免大数值平方导致溢出。
    float scale=fmaxf(fmaxf(fabsf(q.x),fabsf(q.y)),fmaxf(fabsf(q.z),fabsf(q.w)));
    q.x/=scale;q.y/=scale;q.z/=scale;q.w/=scale;
    float inv=1.f/sqrtf(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);
    q.x*=inv;q.y*=inv;q.z*=inv;q.w*=inv;
    return {mkf3(1-2*(q.y*q.y+q.z*q.z),2*(q.x*q.y+q.z*q.w),2*(q.x*q.z-q.y*q.w)),
            mkf3(2*(q.x*q.y-q.z*q.w),1-2*(q.x*q.x+q.z*q.z),2*(q.y*q.z+q.x*q.w)),
            mkf3(2*(q.x*q.z+q.y*q.w),2*(q.y*q.z-q.x*q.w),1-2*(q.x*q.x+q.y*q.y)),position_cm};
}
// 检查有限值、单位轴、轴间正交及右手坐标系，防止无效变换进入后续计算。
__host__ __device__ inline bool validTransform(const Transform& t) {
    return finite(t.position)&&finite(t.x)&&finite(t.y)&&finite(t.z) &&
        fabsf(dot3(t.x,t.x)-1)<1e-4f && fabsf(dot3(t.y,t.y)-1)<1e-4f &&
        fabsf(dot3(t.z,t.z)-1)<1e-4f && fabsf(dot3(t.x,t.y))<1e-4f &&
        fabsf(dot3(t.x,t.z))<1e-4f && fabsf(dot3(t.y,t.z))<1e-4f &&
        dot3(cross3(t.x,t.y),t.z)>0.999f;
}

} // namespace hyc_movecontrol
