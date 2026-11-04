// Derived diagnostic workspace is recomputed on every call. It must not
// enter the physical allocation archive or alter its allocation-ID sequence.
// No existing physical allocation or snapshot field is excluded or changed.
template<class T> static cudaError_t topoRawAllocate(T** p,size_t n){return (::cudaMalloc)(reinterpret_cast<void**>(p),n);}
static cudaError_t topoRawFree(void* p){return (::cudaFree)(p);}
// Read-only exact diagnostics over the CURRENT reconstructed triangle soup.
// Independent scratch: no contact cache, solver, reconstruction or topology is mutated.
struct TopoPoint { long long x,y,z; };
static TopoPoint* d_topoRounded=nullptr;
static int *d_topoTable=nullptr,*d_topoInverse=nullptr,*d_topoCounts=nullptr,*d_topoDirection=nullptr,*d_topoStats=nullptr;
static unsigned long long *d_topoEdges=nullptr,*d_topoFaces=nullptr;
static float* d_topoBounds=nullptr;
static int g_topoCapacity=0,g_topoSlots=0,g_topoFaceSlots=0;
static void topoShutdown(){
    topoRawFree(d_topoRounded);topoRawFree(d_topoTable);topoRawFree(d_topoInverse);topoRawFree(d_topoCounts);topoRawFree(d_topoDirection);topoRawFree(d_topoStats);topoRawFree(d_topoEdges);topoRawFree(d_topoFaces);topoRawFree(d_topoBounds);
    d_topoRounded=nullptr;d_topoTable=d_topoInverse=d_topoCounts=d_topoDirection=d_topoStats=nullptr;d_topoEdges=d_topoFaces=nullptr;d_topoBounds=nullptr;g_topoCapacity=g_topoSlots=g_topoFaceSlots=0;
}
__device__ inline unsigned long long topoMix(unsigned long long x){
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ull;x=(x^(x>>27))*0x94d049bb133111ebull;return x^(x>>31);
}
__global__ void k_TopoInit(float* bounds){if(threadIdx.x==0){for(int k=0;k<3;k++){bounds[k]=INFINITY;bounds[k+3]=-INFINITY;}}}
__global__ void k_TopoRound(int n,const float3* pos,TopoPoint* rounded,int* stats){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float3 p=pos[i];
    double x=__dmul_rn((double)p.x,100000.),y=__dmul_rn((double)p.y,100000.),z=__dmul_rn((double)p.z,100000.);
    if(!isfinite(x)||!isfinite(y)||!isfinite(z)||fabs(x)>9e18||fabs(y)>9e18||fabs(z)>9e18){atomicExch(stats+6,1);rounded[i]={0,0,0};return;}
    rounded[i]={__double2ll_rn(x),__double2ll_rn(y),__double2ll_rn(z)};
}
__global__ void k_TopoWeld(int n,int slots,const TopoPoint* rounded,int* table,int* inverse,const int* stats){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||stats[6])return;TopoPoint p=rounded[i];
    unsigned long long h=topoMix((unsigned long long)p.x)^topoMix((unsigned long long)p.y+0x9e3779b97f4a7c15ull)^topoMix((unsigned long long)p.z+0x517cc1b727220a95ull);
    int slot=(int)(h&(slots-1));
    for(int probe=0;probe<slots;probe++){
        int j=atomicCAS(table+slot,-1,i);
        if(j<0){inverse[i]=i;return;}TopoPoint q=rounded[j];
        if(q.x==p.x&&q.y==p.y&&q.z==p.z){inverse[i]=j;return;}slot=(slot+1)&(slots-1);
    }
}
__device__ inline int topoSlot(unsigned long long key,unsigned long long* table,int slots,bool& duplicate){
    int slot=(int)(topoMix(key)&(slots-1));
    for(int probe=0;probe<slots;probe++){
        unsigned long long previous=atomicCAS(table+slot,~0ull,key);
        if(previous==~0ull||previous==key){duplicate=previous==key;return slot;}slot=(slot+1)&(slots-1);
    }return -1;
}
__global__ void k_TopoFaces(int n,int slots,int faceSlots,const float3* pos,const float4* aux,const int* inverse,
    unsigned long long* edges,int* counts,int* direction,unsigned long long* faces,int* stats,float* bounds){
    int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=n/3||stats[6])return;int i=3*t;
    int ids[3]={inverse[i],inverse[i+1],inverse[i+2]};
    for(int k=0;k<3;k++){
        int a=ids[k],b=ids[(k+1)%3],lo=min(a,b),hi=max(a,b);bool duplicate;
        int slot=topoSlot((unsigned long long)lo*n+hi,edges,slots,duplicate);
        if(slot<0){atomicExch(stats+7,1);continue;}atomicAdd(counts+slot,1);atomicAdd(direction+slot,a<b?1:(a>b?-1:0));
    }
    int lo=min(ids[0],min(ids[1],ids[2])),hi=max(ids[0],max(ids[1],ids[2])),mid=ids[0]+ids[1]+ids[2]-lo-hi;bool duplicate;
    int slot=topoSlot(((unsigned long long)lo*n+mid)*n+hi,faces,faceSlots,duplicate);
    if(slot<0)atomicExch(stats+7,1);else if(duplicate)atomicAdd(stats+4,1);
    // Explicit IEEE-rounded operations reproduce numpy float32 cross/norm;
    // the simulation's own compilation and arithmetic remain unchanged.
    float3 a=pos[i],b=pos[i+1],c=pos[i+2];
    float x0=__fsub_rn(b.x,a.x),x1=__fsub_rn(b.y,a.y),x2=__fsub_rn(b.z,a.z);
    float y0=__fsub_rn(c.x,a.x),y1=__fsub_rn(c.y,a.y),y2=__fsub_rn(c.z,a.z);
    float z0=__fsub_rn(__fmul_rn(x1,y2),__fmul_rn(x2,y1)),z1=__fsub_rn(__fmul_rn(x2,y0),__fmul_rn(x0,y2)),z2=__fsub_rn(__fmul_rn(x0,y1),__fmul_rn(x1,y0));
    float area=__fsqrt_rn(__fadd_rn(__fadd_rn(__fmul_rn(z0,z0),__fmul_rn(z1,z1)),__fmul_rn(z2,z2)));
    if(area<1e-8f)atomicAdd(stats+1,1);
    if(aux[i].w>.5f){
        atomicAdd(stats,1);
        for(int k=0;k<3;k++){float3 p=pos[i+k];renderMin(bounds,p.x);renderMin(bounds+1,p.y);renderMin(bounds+2,p.z);renderMax(bounds+3,p.x);renderMax(bounds+4,p.y);renderMax(bounds+5,p.z);}
    }
}
__global__ void k_TopoCount(int slots,const int* counts,const int* direction,int* stats){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=slots)return;int n=counts[i];
    if(n==1)atomicAdd(stats+2,1);else if(n>2)atomicAdd(stats+3,1);else if(n==2&&direction[i]!=0)atomicExch(stats+5,1);
}
static int topoPower(int n){int value=8;while(value<n*2)value*=2;return value;}
static cudaError_t topoAllocate(int n){
    if(g_topoCapacity>=n)return cudaSuccess;topoShutdown();
    int capacity=1;while(capacity<n)capacity*=2;
    int slots=topoPower(capacity),faceSlots=topoPower(capacity/3);cudaError_t e=cudaSuccess;
#define TOPO_ALLOC(pointer,bytes) if(e==cudaSuccess)e=topoRawAllocate(&pointer,bytes)
    TOPO_ALLOC(d_topoRounded,capacity*sizeof(TopoPoint));TOPO_ALLOC(d_topoInverse,capacity*sizeof(int));TOPO_ALLOC(d_topoTable,slots*sizeof(int));
    TOPO_ALLOC(d_topoEdges,slots*sizeof(unsigned long long));TOPO_ALLOC(d_topoCounts,slots*sizeof(int));TOPO_ALLOC(d_topoDirection,slots*sizeof(int));
    TOPO_ALLOC(d_topoFaces,faceSlots*sizeof(unsigned long long));TOPO_ALLOC(d_topoStats,8*sizeof(int));TOPO_ALLOC(d_topoBounds,6*sizeof(float));
#undef TOPO_ALLOC
    if(e!=cudaSuccess){topoShutdown();return e;}g_topoCapacity=capacity;g_topoSlots=slots;g_topoFaceSlots=faceSlots;return cudaSuccess;
}
static int topoCompute(int n,const float3* pos,const float4* aux,int* output,float* bounds){
    if(n<=0||n%3||n>1<<20||!pos||!aux||!output||!bounds)return -10001;
    cudaError_t e=topoAllocate(n);if(e!=cudaSuccess)return -(int)e;
#define TOPO_CALL(call) do{e=call;if(e!=cudaSuccess)return -(int)e;}while(0)
    TOPO_CALL(cudaMemsetAsync(d_topoTable,0xff,g_topoSlots*sizeof(int)));
    TOPO_CALL(cudaMemsetAsync(d_topoEdges,0xff,g_topoSlots*sizeof(unsigned long long)));
    TOPO_CALL(cudaMemsetAsync(d_topoFaces,0xff,g_topoFaceSlots*sizeof(unsigned long long)));
    TOPO_CALL(cudaMemsetAsync(d_topoCounts,0,g_topoSlots*sizeof(int)));TOPO_CALL(cudaMemsetAsync(d_topoDirection,0,g_topoSlots*sizeof(int)));TOPO_CALL(cudaMemsetAsync(d_topoStats,0,8*sizeof(int)));
    k_TopoInit<<<1,1>>>(d_topoBounds);k_TopoRound<<<(n+127)/128,128>>>(n,pos,d_topoRounded,d_topoStats);
    k_TopoWeld<<<(n+127)/128,128>>>(n,g_topoSlots,d_topoRounded,d_topoTable,d_topoInverse,d_topoStats);
    k_TopoFaces<<<(n/3+127)/128,128>>>(n,g_topoSlots,g_topoFaceSlots,pos,aux,d_topoInverse,d_topoEdges,d_topoCounts,d_topoDirection,d_topoFaces,d_topoStats,d_topoBounds);
    k_TopoCount<<<(g_topoSlots+127)/128,128>>>(g_topoSlots,d_topoCounts,d_topoDirection,d_topoStats);
    TOPO_CALL(cudaGetLastError());int stats[8];TOPO_CALL(cudaMemcpy(stats,d_topoStats,sizeof stats,cudaMemcpyDeviceToHost));TOPO_CALL(cudaMemcpy(bounds,d_topoBounds,6*sizeof(float),cudaMemcpyDeviceToHost));
    output[0]=n/3;for(int k=0;k<5;k++)output[k+1]=stats[k];output[6]=stats[5]==0;output[7]=stats[6];output[8]=stats[7];return 0;
#undef TOPO_CALL
}
int recon_exact_topology(int expected,int* output,float* bounds){
    int count=0;int result=recon_get_counts(&count,nullptr);if(result)return result;
    if(expected!=count)return -10002;return topoCompute(count,d_expandPos,d_expandAux,output,bounds);
}
int recon_test_exact_topology(int n,const float* positions,const float* anchors,int* output,float* bounds){
    if(n<=0||n%3||n>1<<20||!positions||!anchors||!output||!bounds)return -10001;
    float3* pos=nullptr;float4* aux=nullptr;cudaError_t e=topoRawAllocate(&pos,n*sizeof(float3));
    if(e==cudaSuccess)e=topoRawAllocate(&aux,n*sizeof(float4));
    if(e==cudaSuccess)e=cudaMemcpy(pos,positions,n*sizeof(float3),cudaMemcpyHostToDevice);
    if(e==cudaSuccess)e=cudaMemcpy(aux,anchors,n*sizeof(float4),cudaMemcpyHostToDevice);
    int result=e==cudaSuccess?topoCompute(n,pos,aux,output,bounds):-(int)e;topoRawFree(pos);topoRawFree(aux);return result;
}
