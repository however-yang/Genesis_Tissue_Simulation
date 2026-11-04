#include <math_constants.h>
// Isolated experiment: conservative groups of 64 surface triangles.
// Fine witnesses, friction, preload, and the narrow query remain unchanged.
struct ContactGroupBounds { float3 lo, hi; };
static inline int contactGroupCount(int triangles){return (triangles+63)/64;}
__device__ inline float3 groupMin(float3 a,float3 b){return mkf3(fminf(a.x,b.x),fminf(a.y,b.y),fminf(a.z,b.z));}
__device__ inline float3 groupMax(float3 a,float3 b){return mkf3(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z));}
__device__ inline bool groupFinite(float3 v){return isfinite(v.x)&&isfinite(v.y)&&isfinite(v.z);}
__global__ void k_BuildContactGroups(int triangles,const float4* current,const float3* previous,ContactGroupBounds* groups){
    __shared__ float3 lo[64],hi[64];
    int lane=threadIdx.x,t=blockIdx.x*64+lane;
    float3 mn=mkf3(CUDART_INF_F,CUDART_INF_F,CUDART_INF_F),mx=mn*-1.f;
    if(t<triangles){
        float4 q[3]={current[3*t],current[3*t+1],current[3*t+2]};
        if(q[0].w!=0 && q[1].w!=0 && q[2].w!=0){
            bool finite=true;
            for(int k=0;k<3;k++){
                float3 a=mkf3(q[k].x,q[k].y,q[k].z),b=previous[3*t+k];
                finite=finite && groupFinite(a) && groupFinite(b);
                mn=groupMin(mn,groupMin(a,b));mx=groupMax(mx,groupMax(a,b));
            }
            // Unknown geometry may never be excluded by this optimization.
            if(!finite){mn=mkf3(-CUDART_INF_F,-CUDART_INF_F,-CUDART_INF_F);mx=mn*-1.f;}
        }
    }
    lo[lane]=mn;hi[lane]=mx;__syncthreads();
    for(int stride=32;stride;stride>>=1){
        if(lane<stride){lo[lane]=groupMin(lo[lane],lo[lane+stride]);hi[lane]=groupMax(hi[lane],hi[lane+stride]);}
        __syncthreads();
    }
    if(lane==0){groups[blockIdx.x].lo=lo[0];groups[blockIdx.x].hi=hi[0];}
}
__global__ void k_ClassifyContactGroups(int count,int kinds,const ContactGroupBounds* groups,unsigned char* rejected,
    const CoupledTool* body,float thickness,float minGap,float radius,bool stationary,bool blade){
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count*kinds)return;
    int kind=index/count,group=index%count;rejected[index]=0;if(!body)return;
    ContactShape from=bodyContactShape(body,stationary?body->pose:body->previous,kind,thickness,minGap);
    ContactShape to=bodyContactShape(body,body->pose,kind,thickness,minGap);
    float skin=blade && kind<3?0:radius;
    float3 lo,hi,l0,h0;shapeBounds(to,skin,lo,hi);shapeBounds(from,skin,l0,h0);
    // Include endpoint boxes AND both bounding spheres for every box query.
    // This is deliberately broader than the original rotational-branch test.
    if(!to.capsule){
        float r=fmaxf(len3(to.half),len3(from.half))+skin;
        float3 extent=mkf3(r,r,r);
        lo=groupMin(lo,to.center-extent);hi=groupMax(hi,to.center+extent);
        l0=groupMin(l0,from.center-extent);h0=groupMax(h0,from.center+extent);
    }
    lo=groupMin(lo,l0);hi=groupMax(hi,h0);
    ContactGroupBounds b=groups[group];
    if(!groupFinite(lo)||!groupFinite(hi))return;
    // Empty groups consist only of invalid triangles; the original also clears them.
    if(b.lo.x==CUDART_INF_F && b.hi.x==-CUDART_INF_F){rejected[index]=1;return;}
    if(!groupFinite(b.lo)||!groupFinite(b.hi))return;
    for(int ax=0;ax<3;ax++){
        float lower=compF3(lo,ax),upper=compF3(hi,ax);
        float scale=fmaxf(1.f,fmaxf(fabsf(lower),fabsf(upper)));
        float margin=32.f*1.1920928955078125e-7f*scale;
        lower=nextafterf(lower-margin,-CUDART_INF_F);upper=nextafterf(upper+margin,CUDART_INF_F);
        if(compF3(b.hi,ax)<lower || compF3(b.lo,ax)>upper){rejected[index]=1;return;}
    }
}
static void buildContactGroups(int triangles,const float4* current,const float3* previous,ContactGroupBounds* groups,cudaStream_t stream=0){
    if(triangles>0)k_BuildContactGroups<<<contactGroupCount(triangles),64,0,stream>>>(triangles,current,previous,groups);
}
static void classifyContactGroups(int triangles,int kinds,const ContactGroupBounds* groups,unsigned char* rejected,
    const CoupledTool* body,float thickness,float minGap,float radius,bool stationary,bool blade,cudaStream_t stream=0){
    int count=contactGroupCount(triangles);if(count>0)k_ClassifyContactGroups<<<(count*kinds+63)/64,64,0,stream>>>(count,kinds,groups,rejected,body,thickness,minGap,radius,stationary,blade);
}
