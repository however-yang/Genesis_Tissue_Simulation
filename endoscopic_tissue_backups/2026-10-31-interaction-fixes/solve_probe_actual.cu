#include <cuda_runtime.h>
#include <cmath>
__device__ float3 mkf3(float a,float b,float c){return make_float3(a,b,c);}
__device__ float3 operator*(float3 v,float a){return make_float3(v.x*a,v.y*a,v.z*a);}
struct Symmetric3 {
    float xx,yy,zz,xy,xz,yz;
    __device__ void diagonal(float s){xx+=s;yy+=s;zz+=s;}
    __device__ void outer(float3 v,float s){xx+=s*v.x*v.x;yy+=s*v.y*v.y;zz+=s*v.z*v.z;xy+=s*v.x*v.y;xz+=s*v.x*v.z;yz+=s*v.y*v.z;}
    __device__ float3 solve(float3 r)const{
        float a=yy*zz-yz*yz,b=xz*yz-xy*zz,c=xy*yz-xz*yy;
        float d=xx*zz-xz*xz,e=xy*xz-xx*yz,f=xx*yy-xy*xy;
        float det=xx*a+xy*b+xz*c;
        if(!(det>1e-25f))return mkf3(0,0,0);
        return mkf3(a*r.x+b*r.y+c*r.z,b*r.x+d*r.y+e*r.z,c*r.x+e*r.y+f*r.z)*(1.f/det);
    }
    // The original float adjugate loses accuracy when a stiff contact direction
    // dominates the inertial/elastic directions at a cut rim. Keep the same H
    // and gradient; only refactor badly cancelling vertex systems in double.
    __device__ __noinline__ float3 solvePrecise(float3 r)const{
        double X=xx,Y=yy,Z=zz,A=xy,B=xz,C=yz;
        if(!(X>0))return mkf3(0,0,0);
        double a=A/X,b=B/X,d=Y-a*A,e=C-a*B;
        if(!(d>0))return mkf3(0,0,0);
        double c=e/d,f=Z-b*B-c*e;
        if(!(f>0) || !(X*d*f>1e-25))return mkf3(0,0,0);
        double y=(double)r.y-a*r.x,z=((double)r.z-b*r.x-c*y)/f;
        y=y/d-c*z;
        return mkf3((float)((double)r.x/X-a*y-b*z),(float)y,(float)z);
    }
    __device__ float3 solveVertex(float3 r)const{
        float a=yy*zz-yz*yz,b=xz*yz-xy*zz,c=xy*yz-xz*yy;
        float det=xx*a+xy*b+xz*c;
        // Expanded absolute determinant terms bound float cancellation. The
        // 64-epsilon trigger is numerical, independent of material stiffness,
        // contact force, mesh size, or an imposed displacement limit.
        float terms=fabsf(xx*yy*zz)+fabsf(xx*yz*yz)+2.f*fabsf(xy*xz*yz)+fabsf(xy*xy*zz)+fabsf(xz*xz*yy);
        if(!(det>64.f*1.1920928955078125e-7f*terms))return solvePrecise(r);
        return solve(r);
    }

};
struct S3 {
 float xx,yy,zz,xy,xz,yz;
 __device__ float3 legacy(float3 r) const {
  float a=yy*zz-yz*yz,b=xz*yz-xy*zz,c=xy*yz-xz*yy;
  float d=xx*zz-xz*xz,e=xy*xz-xx*yz,f=xx*yy-xy*xy;
  float det=xx*a+xy*b+xz*c;
  if(!(det>1e-25f))return make_float3(0,0,0);
  float k=1.f/det;return make_float3((a*r.x+b*r.y+c*r.z)*k,(b*r.x+d*r.y+e*r.z)*k,(c*r.x+e*r.y+f*r.z)*k);
 }
 __device__ float3 ldlt(float3 r) const {
  if(!(xx>0))return make_float3(0,0,0);
  float a=xy/xx,b=xz/xx,d=yy-a*xy,e=yz-a*xz;
  if(!(d>0))return make_float3(0,0,0);
  float c=e/d,f=zz-b*xz-c*e;
  if(!(f>0) || !(xx*d*f>1e-25f))return make_float3(0,0,0);
  float y=r.y-a*r.x,z=(r.z-b*r.x-c*y)/f;
  y=y/d-c*z;return make_float3(r.x/xx-a*y-b*z,y,z);
 }
 __device__ __noinline__ float3 precise(float3 r) const {
  double X=xx,Y=yy,Z=zz,A=xy,B=xz,C=yz;
  if(!(X>0))return make_float3(0,0,0);
  double a=A/X,b=B/X,d=Y-a*A,e=C-a*B;
  if(!(d>0))return make_float3(0,0,0);
  double c=e/d,f=Z-b*B-c*e;
  if(!(f>0) || !(X*d*f>1e-25))return make_float3(0,0,0);
  double y=(double)r.y-a*r.x,z=((double)r.z-b*r.x-c*y)/f;
  y=y/d-c*z;return make_float3((float)((double)r.x/X-a*y-b*z),(float)y,(float)z);
 }
 __device__ float3 guarded(float3 r) const {
  float a=yy*zz-yz*yz,b=xz*yz-xy*zz,c=xy*yz-xz*yy;
  float d=xx*zz-xz*xz,e=xy*xz-xx*yz,f=xx*yy-xy*xy;
  float det=xx*a+xy*b+xz*c;
  // Conservative cancellation bound for the unexpanded determinant.
  float terms=fabsf(xx*yy*zz)+fabsf(xx*yz*yz)+2.f*fabsf(xy*xz*yz)+fabsf(xy*xy*zz)+fabsf(xz*xz*yy);
  if(!(det>64.f*1.1920928955078125e-7f*terms))return precise(r);
  if(!(det>1e-25f))return make_float3(0,0,0);
  float k=1.f/det;return make_float3((a*r.x+b*r.y+c*r.z)*k,(b*r.x+d*r.y+e*r.z)*k,(c*r.x+e*r.y+f*r.z)*k);
 }
};
__global__ void probe(const float* input,float* output,int n,int mode){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const float* p=input+9*i;S3 h={p[0],p[1],p[2],p[3],p[4],p[5]};float3 r=make_float3(p[6],p[7],p[8]);
 Symmetric3 actual={p[0],p[1],p[2],p[3],p[4],p[5]};
 float3 x=mode==4?actual.solveVertex(r):mode==0?h.legacy(r):mode==1?h.ldlt(r):mode==2?h.precise(r):h.guarded(r);
 output[3*i]=x.x;output[3*i+1]=x.y;output[3*i+2]=x.z;
}
extern "C" __declspec(dllexport) int RunSolveProbe(const float* input,float* output,int n,int mode){
 float *a=nullptr,*b=nullptr;cudaError_t e=cudaMalloc(&a,9*n*sizeof(float));if(e!=cudaSuccess)return (int)e;
 e=cudaMalloc(&b,3*n*sizeof(float));if(e==cudaSuccess)e=cudaMemcpy(a,input,9*n*sizeof(float),cudaMemcpyHostToDevice);
 if(e==cudaSuccess){probe<<<(n+255)/256,256>>>(a,b,n,mode);e=cudaGetLastError();}
 if(e==cudaSuccess)e=cudaMemcpy(output,b,3*n*sizeof(float),cudaMemcpyDeviceToHost);
 cudaFree(a);cudaFree(b);return (int)e;
}
