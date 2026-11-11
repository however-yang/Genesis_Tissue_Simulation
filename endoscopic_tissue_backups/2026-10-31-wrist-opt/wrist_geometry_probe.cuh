// Diagnostic-only entry point exercising the production wrist helper.
__global__ void k_ProbeWristGeometry(int count,const float* input,float* output){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const float* f=input+16*i;float* out=output+12*i;
    float3 q[3]={mkf3(f[0],f[1],f[2]),mkf3(f[3],f[4],f[5]),mkf3(f[6],f[7],f[8])};
    ContactShape shape={};shape.capsule=1;shape.a=mkf3(f[9],f[10],f[11]);
    shape.b=mkf3(f[12],f[13],f[14]);shape.radius=f[15];
    float3 bary[3]={triangleWitness(shape,q[0],q[1],q[2]),wristTriangleWitness(shape,q),bladeTriangleWitness(shape,q)};
    for(int k=0;k<3;k++){
        out[3*k]=bary[k].x;out[3*k+1]=bary[k].y;out[3*k+2]=bary[k].z;
        float3 n,s;out[9+k]=shapeDistance(shape,q[0]*bary[k].x+q[1]*bary[k].y+q[2]*bary[k].z,n,s);
    }
}
extern "C" __declspec(dllexport) int LCS_ProbeWristGeometry(int count,const float* input,float* output){
    if(count<=0 || !input || !output)return -1;
    float *deviceInput=nullptr,*deviceOutput=nullptr;cudaError_t error;
    error=cudaMalloc(&deviceInput,count*16*sizeof(float));if(error!=cudaSuccess)return -(int)error;
    error=cudaMalloc(&deviceOutput,count*12*sizeof(float));
    if(error==cudaSuccess)error=cudaMemcpy(deviceInput,input,count*16*sizeof(float),cudaMemcpyHostToDevice);
    if(error==cudaSuccess){k_ProbeWristGeometry<<<(count+63)/64,64>>>(count,deviceInput,deviceOutput);error=cudaGetLastError();}
    if(error==cudaSuccess)error=cudaMemcpy(output,deviceOutput,count*12*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);cudaFree(deviceOutput);return error==cudaSuccess?0:-(int)error;
}
