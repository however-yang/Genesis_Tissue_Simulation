// Micro-benchmark: cost of warp-reduced float atomics into one shared block, as the coarse modes issue them.
// Each warp of the launch adds 9 values to the same 9 words (one component root), against the same launch writing
// its partial sums to per-warp slots (no contention), and against one atomic per block of 256 threads.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k_Shared(int n,float* block){
    int i=blockIdx.x*blockDim.x+threadIdx.x;float v[9];
    for(int j=0;j<9;j++)v[j]=i<n?1e-3f*(j+1):0.f;
    for(int d=16;d;d>>=1)for(int j=0;j<9;j++)v[j]+=__shfl_down_sync(0xffffffffu,v[j],d);
    if((threadIdx.x&31)==0)for(int j=0;j<9;j++)atomicAdd(block+j,v[j]);
}
__global__ void k_Private(int n,float* partial){
    int i=blockIdx.x*blockDim.x+threadIdx.x;float v[9];
    for(int j=0;j<9;j++)v[j]=i<n?1e-3f*(j+1):0.f;
    for(int d=16;d;d>>=1)for(int j=0;j<9;j++)v[j]+=__shfl_down_sync(0xffffffffu,v[j],d);
    if((threadIdx.x&31)==0)for(int j=0;j<9;j++)partial[9*(i/32)+j]=v[j];
}
__global__ void k_BlockShared(int n,float* block){
    __shared__ float sum[9];
    int i=blockIdx.x*blockDim.x+threadIdx.x;float v[9];
    if(threadIdx.x<9)sum[threadIdx.x]=0;
    __syncthreads();
    for(int j=0;j<9;j++)v[j]=i<n?1e-3f*(j+1):0.f;
    for(int d=16;d;d>>=1)for(int j=0;j<9;j++)v[j]+=__shfl_down_sync(0xffffffffu,v[j],d);
    if((threadIdx.x&31)==0)for(int j=0;j<9;j++)atomicAdd(sum+j,v[j]);
    __syncthreads();
    if(threadIdx.x<9)atomicAdd(block+threadIdx.x,sum[threadIdx.x]);
}
int main(){
    const int n=600000;float *block,*partial;cudaMalloc(&block,64*sizeof(float));cudaMalloc(&partial,9*(n/32+1)*sizeof(float));
    cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);
    for(int round=0;round<3;round++){
        float t[3];
        cudaEventRecord(a);for(int r=0;r<20;r++)k_Shared<<<(n+63)/64,64>>>(n,block);cudaEventRecord(b);cudaEventSynchronize(b);cudaEventElapsedTime(&t[0],a,b);
        cudaEventRecord(a);for(int r=0;r<20;r++)k_Private<<<(n+63)/64,64>>>(n,partial);cudaEventRecord(b);cudaEventSynchronize(b);cudaEventElapsedTime(&t[1],a,b);
        cudaEventRecord(a);for(int r=0;r<20;r++)k_BlockShared<<<(n+255)/256,256>>>(n,block);cudaEventRecord(b);cudaEventSynchronize(b);cudaEventElapsedTime(&t[2],a,b);
        printf("per launch: shared root %.1f us, private slots %.1f us, block-reduced root %.1f us\n",t[0]*50,t[1]*50,t[2]*50);
    }
    return 0;
}
