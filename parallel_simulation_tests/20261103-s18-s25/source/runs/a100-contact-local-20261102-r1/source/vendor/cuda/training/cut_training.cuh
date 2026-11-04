#include "training_api.h"
#include <algorithm>
extern "C" int cut_training_enable(){
    if(!g_cut_ready || d_trainingTearCause)return -1;
    cudaError_t e=cudaMalloc(&d_trainingTearCause,(size_t)std::max(g_cdesc.gridEdgeCount,1));
    if(e==cudaSuccess)e=cudaMemset(d_trainingTearCause,0,(size_t)std::max(g_cdesc.gridEdgeCount,1));
    if(e==cudaSuccess)e=cudaMalloc(&d_trainingGraspNeighborhood,(size_t)g_cdesc.cornerCount*sizeof(int));
    if(e==cudaSuccess)e=cudaMemset(d_trainingGraspNeighborhood,0,(size_t)g_cdesc.cornerCount*sizeof(int));
    return -(int)e;
}
__device__ bool trainingCutRim(int i,const int* nbr,const int* initial,const int* material){
    if(!material[i])return false;
    for(int k=0;k<6;k++){int j=initial[6*i+k];if(j>=0 && material[j] && nbr[6*i+k]<0)return true;}
    return false;
}
__device__ void trainingMax(float* target,float value){
    unsigned mask=__activemask(),maximum=__reduce_max_sync(mask,__float_as_uint(value));
    if(maximum && (threadIdx.x&31)==__ffs(mask)-1)atomicMax((unsigned*)target,maximum);
}
__global__ void k_TrainingCutMetrics(int n,const GridEdgeGpu2* edges,int3 dims,const float3* pos,const float3* rest,
    const int* nbr,const int* initial,const int* inside,const int* ring,const unsigned char* causes,TrainingMetrics* m){
    int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=n)return;
    auto ge=edges[e];int3 c=make_int3(ge.bcX,ge.bcY,ge.bcZ),dir=make_int3(ge.axis==0,ge.axis==1,ge.axis==2);
    int a=CorIdCuda(c,dims),b=CorIdCuda(c+dir,dims);
    if(causes[e]&1)atomicAdd(&m->tearEdges,1u);
    if(!inside[a] || !inside[b])return;
    bool rim=trainingCutRim(a,nbr,initial,inside)||trainingCutRim(b,nbr,initial,inside);
    bool near=ring[a] || ring[b];
    if(causes[e]&2){
        atomicAdd(&m->materialTearEdges,1u);
        if(causes[e]&4)atomicAdd(&m->rimTearEdges,1u);else atomicAdd(&m->interiorTearEdges,1u);
        if(causes[e]&8)atomicAdd(&m->nearGraspTearEdges,1u);
    }
    if(nbr[6*a+2*ge.axis+1]<0)return;
    float3 d=pos[b]-pos[a],r=rest[b]-rest[a];float stretch=sqrtf(dot3(d,d)/fmaxf(dot3(r,r),1e-20f));
    trainingMax(&m->maxStretch,stretch);
    trainingMax(&m->rimStretch,rim?stretch:0.f);
    trainingMax(&m->graspRingStretch,near?stretch:0.f);
}
extern "C" int cut_training_reduce(TrainingMetrics* deviceOut){
    if(!g_cut_ready || !d_trainingTearCause || !deviceOut)return -1;
    k_TrainingCutMetrics<<<Gc(g_cdesc.gridEdgeCount),64>>>(g_cdesc.gridEdgeCount,d_gridEdges,g_cdims,c_cornerPos,c_cornerRest,
        c_nbrIdx,d_nbrIdxAtInit,d_cornerInside,d_trainingGraspNeighborhood,d_trainingTearCause,deviceOut);
    return -(int)cudaGetLastError();
}
extern "C" int cut_training_causes(int count,unsigned char* out){
    if(!out || count!=g_cdesc.gridEdgeCount || !d_trainingTearCause)return -1;
    return -(int)cudaMemcpy(out,d_trainingTearCause,(size_t)count,cudaMemcpyDeviceToHost);
}
