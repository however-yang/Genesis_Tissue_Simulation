
#include <cuda_runtime.h>
#include <math.h>
#define REPAIR_CONE_ALGEBRA_ONLY
#include "repair_cone.cuh"
__global__ void probe(const double* inputs,double* outputs,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const double* p=inputs+42*i;repair_cone::Matrix h={p[0],p[1],p[2],p[3],p[4],p[5]};
 repair_cone::Vector g={p[6],p[7],p[8]},x;int count=int(p[9]);repair_cone::Plane planes[8];
 for(int j=0;j<count;j++)planes[j]={{p[10+4*j],p[11+4*j],p[12+4*j]},p[13+4*j]};
 bool ok=repair_cone::solve(h,g,planes,count,x);double* q=outputs+4*i;
 q[0]=ok;q[1]=x.x;q[2]=x.y;q[3]=x.z;
}
extern "C" int run_cone(int n,const double* inputs,double* outputs){
 double *a=nullptr,*b=nullptr;cudaError_t e=cudaMalloc(&a,n*42*sizeof(double));
 if(e==cudaSuccess)e=cudaMalloc(&b,n*4*sizeof(double));
 if(e==cudaSuccess)e=cudaMemcpy(a,inputs,n*42*sizeof(double),cudaMemcpyHostToDevice);
 if(e==cudaSuccess){probe<<<(n+63)/64,64>>>(a,b,n);e=cudaDeviceSynchronize();}
 if(e==cudaSuccess)e=cudaMemcpy(outputs,b,n*4*sizeof(double),cudaMemcpyDeviceToHost);
 cudaFree(a);cudaFree(b);return int(e);
}
