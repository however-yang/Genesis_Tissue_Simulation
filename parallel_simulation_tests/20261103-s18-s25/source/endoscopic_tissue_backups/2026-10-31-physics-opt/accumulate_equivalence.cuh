// Diagnostic-only stress test of the actual blockAccumulate implementation.
// Dyadic/integer terms fit exactly in float, so no rounding tolerance is needed.
__host__ __device__ inline int probeDestination(unsigned i,int pattern){
    if(pattern==0)return 0;
    if(pattern==1)return i%3==0?-1:(i/8)%5;
    if(pattern==2)return i%32==31?(i/64)%7:-1;
    if(pattern==3)return -1;
    if(pattern==4)return (i/32)%7;
    return i%11==0?-1:(i*2654435761u)%7;
}
__global__ void k_ProbeAccumulate(CoarseBlock* blocks,int pattern){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    int destination=probeDestination(i,pattern);
    float values[9];for(int j=0;j<9;j++)values[j]=(int(i%3)-1)*(j+1);
    blockAccumulate(destination<0?nullptr:blocks+destination,values,
        [](const void* target,const float* values){coarseAdd(target,values);});
}
extern "C" __declspec(dllexport) int LCS_ProbeBlockAccumulate(int* output){
    if(!output)return -1;
    CoarseBlock* blocks=nullptr;cudaError_t error=cudaMalloc(&blocks,7*sizeof(CoarseBlock));
    if(error!=cudaSuccess)return -(int)error;
    int different=0;constexpr int count=4096*ACCUMULATE_THREADS;
    for(int pattern=0;pattern<6;pattern++){
        error=cudaMemset(blocks,0,7*sizeof(CoarseBlock));if(error!=cudaSuccess)break;
        k_ProbeAccumulate<<<4096,ACCUMULATE_THREADS>>>(blocks,pattern);
        CoarseBlock host[7]={};error=cudaMemcpy(host,blocks,sizeof(host),cudaMemcpyDeviceToHost);
        if(error!=cudaSuccess)break;
        long long sum[7]={};for(unsigned i=0;i<count;i++){
            int destination=probeDestination(i,pattern);
            if(destination>=0)sum[destination]+=int(i%3)-1;
        }
        for(int k=0;k<7;k++){
            float values[9]={host[k].gradient.x,host[k].gradient.y,host[k].gradient.z,
                host[k].xx,host[k].yy,host[k].zz,host[k].xy,host[k].xz,host[k].yz};
            for(int j=0;j<9;j++)if(values[j]!=float(sum[k]*(j+1)))different++;
        }
    }
    cudaFree(blocks);output[0]=different;output[1]=6;output[2]=count*6;
    return error==cudaSuccess?0:-(int)error;
}
