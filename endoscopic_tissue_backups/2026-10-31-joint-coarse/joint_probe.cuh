// Synthetic oracle entry point. Only included in a separate probe DLL.
__global__ void k_ProbeJointAlgebra(int count,const float* inputs,float* outputs){
 int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count)return;const float* in=inputs+43*index;float* out=outputs+126*index;
 float3 p[4],inverse[3];float4 weights[4];int component[4]={},live=1,base=0;
 for(int i=0;i<4;i++)p[i]=mkf3(in[3*i],in[3*i+1],in[3*i+2]);
 for(int i=0;i<3;i++)inverse[i]=mkf3(in[12+3*i],in[13+3*i],in[14+3*i]);
 for(int i=0;i<4;i++)weights[i]=make_float4(in[21+4*i],in[22+4*i],in[23+4*i],in[24+4*i]);
 KuhnTet rest={};rest.b0=mkf3(1,0,0);rest.b1=mkf3(0,1,0);rest.b2=mkf3(0,0,1);rest.metric[6]=1;rest.muV=in[37];rest.bulkV=in[38];
 for(int i=0;i<4;i++)rest.offset[i]=i;rest.metric[0]=rest.metric[1]=rest.metric[2]=1;rest.metric[3]=rest.metric[4]=rest.metric[5]=sqrtf(2.f);
 UnifiedState s={};s.n=4;s.nt=1;s.cells=1;s.kuhn=&rest;s.cellBase=&base;s.component=component;s.live=&live;s.pos=p;s.previousInverse=inverse;
 s.minimum=in[39];s.damping=in[40];s.h=in[41];s.hardening=in[42];s.coarseMode=1;
 float v[JointTerms]={};JointBlock block; jointTetTerms(s,0,weights,&block,v);for(int i=0;i<90;i++)out[i]=v[i];
 for(int mode=0;mode<4;mode++){float w[4];for(int j=0;j<4;j++)w[j]=jointW(weights[j],mode);s.coarseWeight=w;
  float3 gradient=mkf3(0,0,0);Symmetric3 H={};float bound=1;CoarseBlock old={};coarseTetTerms(s,0,&old,false,gradient,H,bound);
  float values[9]={gradient.x,gradient.y,gradient.z,H.xx,H.yy,H.zz,H.xy,H.xz,H.yz};for(int j=0;j<9;j++)out[90+9*mode+j]=values[j];}
}
extern "C" __declspec(dllexport) int LCS_ProbeJointAlgebra(int count,const float* inputs,float* outputs){
 if(count<=0||!inputs||!outputs)return -1;float *in=nullptr,*out=nullptr;cudaError_t e=cudaMalloc(&in,size_t(count)*43*sizeof(float));
 if(e==cudaSuccess)e=cudaMalloc(&out,size_t(count)*126*sizeof(float));
 if(e==cudaSuccess)e=cudaMemcpy(in,inputs,size_t(count)*43*sizeof(float),cudaMemcpyHostToDevice);
 if(e==cudaSuccess){k_ProbeJointAlgebra<<<(count+31)/32,32>>>(count,in,out);e=cudaGetLastError();}
 if(e==cudaSuccess)e=cudaMemcpy(outputs,out,size_t(count)*126*sizeof(float),cudaMemcpyDeviceToHost);cudaFree(in);cudaFree(out);return e==cudaSuccess?0:-(int)e;
}
extern "C" __declspec(dllexport) int LCS_ReadJointDiagnostics(unsigned long long* output,float* margin){
 if(!output||!margin)return -1;JointStats value={};cudaError_t e=cudaMemcpyFromSymbol(&value,g_jointStats,sizeof(value));if(e!=cudaSuccess)return -(int)e;
 output[0]=value.roots;output[1]=value.accepted;output[2]=value.zero;output[3]=value.fallback;output[4]=value.disabled;output[5]=value.negativeMargin;output[6]=value.volumeViolation;output[7]=value.closingViolation;output[8]=value.energyViolation;*margin=value.minimumNegativeMargin;
 JointStats zero={};e=cudaMemcpyToSymbol(g_jointStats,&zero,sizeof(zero));return e==cudaSuccess?0:-(int)e;
}
__global__ void k_ProbeJointCubic(int count,const float* input,int* output){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){const float* x=input+6*i;output[i]=materialLineFeasible(x[0],x[1],x[2],x[3],x[4],x[5]);}}
extern "C" __declspec(dllexport) int LCS_ProbeJointCubic(int count,const float* input,int* output){
 if(count<=0||!input||!output)return -1;float* in=nullptr;int* out=nullptr;cudaError_t e=cudaMalloc(&in,size_t(count)*6*sizeof(float));if(e==cudaSuccess)e=cudaMalloc(&out,size_t(count)*sizeof(int));
 if(e==cudaSuccess)e=cudaMemcpy(in,input,size_t(count)*6*sizeof(float),cudaMemcpyHostToDevice);
 if(e==cudaSuccess){k_ProbeJointCubic<<<(count+63)/64,64>>>(count,in,out);e=cudaGetLastError();}if(e==cudaSuccess)e=cudaMemcpy(output,out,size_t(count)*sizeof(int),cudaMemcpyDeviceToHost);
 cudaFree(in);cudaFree(out);return e==cudaSuccess?0:-(int)e;
}

__global__ void k_ProbeJointRounding(int count,const float* input,float* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;const float* v=input+27*i;float* o=output+5*i;
 float3 p[4],trial[4],d[4];for(int j=0;j<4;j++){p[j]=mkf3(v[3*j],v[3*j+1],v[3*j+2]);float3 step=mkf3(v[12+3*j],v[13+3*j],v[14+3*j]);trial[j]=jointTrialPoint(p[j],step,v[24]);d[j]=trial[j]-p[j];}
 TetVolumePath J=unifiedVolumePath(p[1]-p[0],p[2]-p[0],p[3]-p[0],d[1]-d[0],d[2]-d[0],d[3]-d[0],v[25]);
 float initial=jointStoredDet(p[0],p[1],p[2],p[3],v[25]),actual=jointStoredDet(trial[0],trial[1],trial[2],trial[3],v[25]),floor=fminf(v[26],initial);
 bool line=materialLineFeasible(1.f,fminf(v[26],J.j0),J.j0,J.j1,J.j2,J.j3);
 o[0]=line;o[1]=line && actual>=floor;o[2]=materialLineJ(1.f,J.j0,J.j1,J.j2,J.j3);o[3]=actual;o[4]=floor;
}
extern "C" __declspec(dllexport) int LCS_ProbeJointRounding(int count,const float* input,float* output){
 if(count<=0||!input||!output)return -1;float *in=nullptr,*out=nullptr;cudaError_t e=cudaMalloc(&in,size_t(count)*27*sizeof(float));if(e==cudaSuccess)e=cudaMalloc(&out,size_t(count)*5*sizeof(float));
 if(e==cudaSuccess)e=cudaMemcpy(in,input,size_t(count)*27*sizeof(float),cudaMemcpyHostToDevice);
 if(e==cudaSuccess){k_ProbeJointRounding<<<(count+63)/64,64>>>(count,in,out);e=cudaGetLastError();}if(e==cudaSuccess)e=cudaMemcpy(output,out,size_t(count)*5*sizeof(float),cudaMemcpyDeviceToHost);
 cudaFree(in);cudaFree(out);return e==cudaSuccess?0:-(int)e;
}
