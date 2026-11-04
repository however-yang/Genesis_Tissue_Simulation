// CUDA prepares Unity's interleaved vertex buffer. Topology/positions are unchanged.
// Smooth skin normals/tangents use component identity; cut walls keep flat normals.
struct RenderFour {
    float x,y,z,w;
    __device__ void operator=(float4 a){x=a.x;y=a.y;z=a.z;w=a.w;}
    __device__ operator float4() const{return make_float4(x,y,z,w);}
};
struct RenderVertex { float3 p,n; RenderFour tangent; float2 uv; RenderFour aux; };
static_assert(sizeof(RenderVertex)==64,"Unity vertex layout");
static RenderVertex* d_render=nullptr;
static float3 *d_renderNormal=nullptr,*d_renderTangent=nullptr;
static float* d_renderBounds=nullptr;
static int* d_renderCounts=nullptr;
static int g_renderCapacity=0;
__device__ float renderAxis(float3 p,int axis){return axis==0?p.x:axis==1?p.y:p.z;}
__device__ float3 renderUnit(float3 v,float3 fallback){float s=dot3(v,v);return s>1e-12f?v/sqrtf(s):fallback;}
__device__ void renderAdd(float3* p,float3 v){atomicAdd(&p->x,v.x);atomicAdd(&p->y,v.y);atomicAdd(&p->z,v.z);}
__device__ void renderMin(float* p,float v){int old=*(int*)p,assumed;while(v<__int_as_float(old)){assumed=old;old=atomicCAS((int*)p,assumed,__float_as_int(v));if(old==assumed)break;}}
__device__ void renderMax(float* p,float v){int old=*(int*)p,assumed;while(v>__int_as_float(old)){assumed=old;old=atomicCAS((int*)p,assumed,__float_as_int(v));if(old==assumed)break;}}
__device__ int renderKey(int raw,int voxels){unsigned t=(unsigned)raw;return (t&0x80000000u)?voxels+(t&0x3fffffffu):(int)t;}
__device__ float2 renderUv(float4 a,float3 origin,float3 size,int u,int v){float3 p=make_float3(a.x,a.y,a.z);return make_float2((renderAxis(p,u)-renderAxis(origin,u))/fmaxf(renderAxis(size,u),1e-6f),(renderAxis(p,v)-renderAxis(origin,v))/fmaxf(renderAxis(size,v),1e-6f));}
__global__ void k_RenderInit(float* bounds,int* counts){if(threadIdx.x==0){for(int k=0;k<3;k++){bounds[k]=1e30f;bounds[k+3]=-1e30f;}counts[0]=counts[1]=0;}}
__global__ void k_RenderAccumulate(int count,int voxels,const int* tags,const float3* p,const float4* aux,
    float3 origin,float3 size,int u,int v,float3* normals,float3* tangents){
    int i=3*(blockIdx.x*blockDim.x+threadIdx.x);if(i>=count)return;
    if(aux[i].w>=.5f || aux[i+1].w>=.5f || aux[i+2].w>=.5f)return;
    float3 e1=p[i+1]-p[i],e2=p[i+2]-p[i],n=cross3(e1,e2),t=make_float3(0,0,0);
    float2 a=renderUv(aux[i],origin,size,u,v),b=renderUv(aux[i+1],origin,size,u,v),c=renderUv(aux[i+2],origin,size,u,v);
    float du=b.x-a.x,dv=b.y-a.y,eu=c.x-a.x,ev=c.y-a.y,det=du*ev-dv*eu;
    if(fabsf(det)>1e-8f)t=(e1*ev-e2*dv)/det;
    for(int k=0;k<3;k++){int key=renderKey(tags[i+k],voxels);renderAdd(normals+key,n);renderAdd(tangents+key,t);}
}
__global__ void k_RenderPack(int count,int voxels,const int* tags,const float3* p,const float4* aux,
    float3 origin,float3 size,int u,int v,const float3* normals,const float3* tangents,
    RenderVertex* out,int* counters,float* bounds){
    int i=3*(blockIdx.x*blockDim.x+threadIdx.x);if(i>=count)return;
    bool wall=aux[i].w>=.5f || aux[i+1].w>=.5f || aux[i+2].w>=.5f;
    int offset=wall?count-3*(atomicAdd(counters+1,1)+1):3*atomicAdd(counters,1);
    float3 face=renderUnit(cross3(p[i+1]-p[i],p[i+2]-p[i]),make_float3(0,1,0));
    for(int k=0;k<3;k++){
        RenderVertex r={};r.p=p[i+k];r.aux=aux[i+k];r.uv=renderUv(r.aux,origin,size,u,v);
        int key=renderKey(tags[i+k],voxels);r.n=wall?face:renderUnit(normals[key],make_float3(0,1,0));
        float3 axis=fabsf(r.n.y)<.9f?make_float3(0,1,0):make_float3(1,0,0);
        float3 t=renderUnit(tangents[key]-r.n*dot3(r.n,tangents[key]),renderUnit(cross3(axis,r.n),make_float3(1,0,0)));
        r.tangent=make_float4(t.x,t.y,t.z,1);out[offset+k]=r;
        for(int a=0;a<3;a++){renderMin(bounds+a,renderAxis(r.p,a));renderMax(bounds+a+3,renderAxis(r.p,a));}
    }
}
static void renderShutdown(){cudaFree(d_render);cudaFree(d_renderNormal);cudaFree(d_renderTangent);cudaFree(d_renderBounds);cudaFree(d_renderCounts);d_render=nullptr;d_renderNormal=d_renderTangent=nullptr;d_renderBounds=nullptr;d_renderCounts=nullptr;g_renderCapacity=0;}
int recon_render_buffer(int count,void* output,int* counts,float* bounds,const float* uvMin,const float* uvSize){
    if(!g_ready || count<0 || count>3*g_desc.triCapacity || count%3 || !output || !counts || !bounds || !uvMin || !uvSize)return -1;
    cudaError_t e=cudaSuccess;
    if(!d_renderNormal){
        e=cudaMalloc(&d_renderNormal,(size_t)9*g_desc.voxelCount*sizeof(float3));
        if(e==cudaSuccess)e=cudaMalloc(&d_renderTangent,(size_t)9*g_desc.voxelCount*sizeof(float3));
        if(e==cudaSuccess)e=cudaMalloc(&d_renderBounds,6*sizeof(float));
        if(e==cudaSuccess)e=cudaMalloc(&d_renderCounts,2*sizeof(int));
    }
    if(e==cudaSuccess && count>g_renderCapacity){cudaFree(d_render);d_render=nullptr;g_renderCapacity=0;e=cudaMalloc(&d_render,(size_t)count*sizeof(RenderVertex));if(e==cudaSuccess)g_renderCapacity=count;}
    if(e!=cudaSuccess){renderShutdown();return -(int)e;}
    float3 origin=make_float3(uvMin[0],uvMin[1],uvMin[2]),size=make_float3(uvSize[0],uvSize[1],uvSize[2]);
    int u=0,v=1;
    if(size.x<=size.y && size.x<=size.z){u=1;v=2;}
    else if(size.y<=size.x && size.y<=size.z){u=0;v=2;}
    cudaMemsetAsync(d_renderNormal,0,(size_t)9*g_desc.voxelCount*sizeof(float3));
    cudaMemsetAsync(d_renderTangent,0,(size_t)9*g_desc.voxelCount*sizeof(float3));
    k_RenderInit<<<1,1>>>(d_renderBounds,d_renderCounts);
    if(count){
        k_RenderAccumulate<<<(count/3+127)/128,128>>>(count,g_desc.voxelCount,d_tri,d_expandPos,d_expandAux,origin,size,u,v,d_renderNormal,d_renderTangent);
        k_RenderPack<<<(count/3+127)/128,128>>>(count,g_desc.voxelCount,d_tri,d_expandPos,d_expandAux,origin,size,u,v,d_renderNormal,d_renderTangent,d_render,d_renderCounts,d_renderBounds);
        e=cudaMemcpy(output,d_render,(size_t)count*sizeof(RenderVertex),cudaMemcpyDeviceToHost);
    }
    if(e==cudaSuccess)e=cudaMemcpy(counts,d_renderCounts,2*sizeof(int),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(bounds,d_renderBounds,6*sizeof(float),cudaMemcpyDeviceToHost);
    return -(int)e;
}
