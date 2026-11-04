"""Frozen hierarchy equality and CUDA-event ABBA timing; no live solver edits."""
from pathlib import Path
import argparse,ctypes as ct,hashlib,json,os,subprocess
import numpy as np
SOURCE=r'''
#include <cuda_runtime.h>
#include <math.h>
inline __device__ float3 mkf3(float x,float y,float z){return make_float3(x,y,z);}
inline __device__ float3 operator+(float3 a,float3 b){return mkf3(a.x+b.x,a.y+b.y,a.z+b.z);}
struct FragmentTreeNode{float3 low,high,currentLow,currentHigh,areaVector;float area;int root;};
inline __device__ float3 fragmentMin3(float3 a,float3 b){return mkf3(fminf(a.x,b.x),fminf(a.y,b.y),fminf(a.z,b.z));}
inline __device__ float3 fragmentMax3(float3 a,float3 b){return mkf3(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z));}
inline __device__ int fragmentMergeRoot(int a,int b){return a==-1?b:b==-1?a:a==b?a:-2;}
__global__ void original(int first,int count,FragmentTreeNode* tree,const int* queryCount){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count||!*queryCount)return;i+=first;
 FragmentTreeNode a=tree[2*i],b=tree[2*i+1],c;
 c.low=fragmentMin3(a.low,b.low);c.high=fragmentMax3(a.high,b.high);c.currentLow=fragmentMin3(a.currentLow,b.currentLow);c.currentHigh=fragmentMax3(a.currentHigh,b.currentHigh);
 c.areaVector=a.areaVector+b.areaVector;c.area=a.area+b.area;c.root=fragmentMergeRoot(a.root,b.root);tree[i]=c;
}
#include "fragment_tree_fused.cuh"
extern "C" int run(int base,int queried,int fused,const void* input,void* output,float* timings){
 FragmentTreeNode* d=nullptr;int* q=nullptr;cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;cudaEvent_t a=nullptr,b=nullptr;
 cudaError_t e=cudaSuccess;
 #define CHECK(call) if((e=(call))!=cudaSuccess)goto done
 CHECK(cudaMalloc(&d,2*base*sizeof(FragmentTreeNode)));CHECK(cudaMalloc(&q,sizeof(int)));
 CHECK(cudaMemcpy(d,input,2*base*sizeof(FragmentTreeNode),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(q,&queried,sizeof(int),cudaMemcpyHostToDevice));
 CHECK(cudaStreamCreate(&stream));CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
 if(fused){int level=base/2;for(;level>128;level/=2)original<<<(level+63)/64,64,0,stream>>>(level,level,d,q);if(level)k_FragmentTreeTop<<<1,128,0,stream>>>(level,d,q);}
 else for(int level=base/2;level;level/=2)original<<<(level+63)/64,64,0,stream>>>(level,level,d,q);
 CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,0));
 for(int j=0;j<10;j++)CHECK(cudaGraphLaunch(exec,stream));
 CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));
 for(int r=0;r<20;r++){CHECK(cudaEventRecord(a,stream));for(int j=0;j<100;j++)CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaEventRecord(b,stream));CHECK(cudaEventSynchronize(b));CHECK(cudaEventElapsedTime(timings+r,a,b));timings[r]/=100;}
 CHECK(cudaMemcpy(output,d,2*base*sizeof(FragmentTreeNode),cudaMemcpyDeviceToHost));
 done:
 if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);if(stream)cudaStreamDestroy(stream);if(a)cudaEventDestroy(a);if(b)cudaEventDestroy(b);cudaFree(d);cudaFree(q);return (int)e;
}
'''
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
 h=Path(__file__).resolve().parent;p=h.parents[2];src=out/'probe.cu';src.write_text(SOURCE);lib=out/'probe.so'
 cmd=[str(p/'envs/genesis-py311-cu125-v1/bin/nvcc'),'-std=c++17','-O3','-arch=sm_80','-shared','-Xcompiler=-fPIC','-ccbin=/usr/bin/g++-12','-I'+str(h),str(src),'-o',str(lib)]
 with (out/'compile.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
 dll=ct.CDLL(str(lib));dll.run.argtypes=[ct.c_int,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p];dll.run.restype=ct.c_int
 dtype=np.dtype([('floats','<f4',16),('root','<i4')]);assert dtype.itemsize==68
 rng=np.random.default_rng(20261103);rows=[]
 for base in [1,2,32,128,256,1024,2048,8192]:
  data=np.zeros(2*base,dtype=dtype);data['floats']=rng.normal(size=(2*base,16));data['root']=rng.integers(-2,7,2*base)
  for queried in [0,1]:
   outputs=[];timings={0:[],1:[]}
   for mode in [0,1,1,0]:
    dst=np.empty_like(data);times=np.empty(20,np.float32)
    code=dll.run(base,queried,mode,data.ctypes.data,dst.ctypes.data,times.ctypes.data);assert code==0,code
    outputs.append(dst.copy());timings[mode].extend(times.tolist())
   exact=all(np.array_equal(outputs[0].view(np.uint8),o.view(np.uint8)) for o in outputs[1:]);assert exact
   row=dict(base=base,queries=queried,all_nodes_bitwise_exact=exact,original_ms=float(np.median(timings[0])),fused_ms=float(np.median(timings[1])),original_samples=timings[0],fused_samples=timings[1]);rows.append(row)
   print('TREE',base,queried,exact,row['original_ms'],row['fused_ms'],flush=True)
 r=dict(completed=True,passed=True,tests=len(rows),rows=rows,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],header_sha256=hashlib.sha256((h/'fragment_tree_fused.cuh').read_bytes()).hexdigest(),scope='Frozen random finite hierarchy nodes, original child order, graph event timings with ABBA. No live collision, full frame or physics acceptance; not integrated.')
 (out/'result.json').write_text(json.dumps(r,indent=2))
if __name__=='__main__':main()
