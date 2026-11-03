// Isolated experiment: resolve a four-face cut-wall edge without moving a
// feature point or changing material connectivity. Unsupported stars remain
// unchanged and must fail the independent surface validator.
struct PinchEdge {
    int owner, count, halfedge[4], reserved[2];
};
static PinchEdge* d_pinchEdges=nullptr;
static unsigned* d_pinchStats=nullptr;
static int g_pinchCapacity=0;

__device__ inline unsigned pinchKey(int raw){return (unsigned)raw&0xbfffffffu;}
__device__ inline void pinchEdgeKey(const int* tris,int token,unsigned& a,unsigned& b){
    int face=token/3,k=token%3;
    a=pinchKey(tris[3*face+k]);b=pinchKey(tris[3*face+(k+1)%3]);
    if(a>b){unsigned t=a;a=b;b=t;}
}
__device__ inline unsigned pinchHash(unsigned a,unsigned b,int cap){
    return ((a*73856093u)^(b*19349663u))&(cap-1);
}
__device__ inline int pinchFind(unsigned a,unsigned b,const int* tris,
                                const PinchEdge* table,int cap){
    if(a>b){unsigned t=a;a=b;b=t;}
    unsigned h=pinchHash(a,b,cap);
    for(int k=0;k<cap;k++){
        int at=(h+k)&(cap-1),owner=table[at].owner;
        if(owner<0)return -1;
        unsigned x,y;pinchEdgeKey(tris,owner,x,y);
        if(a==x && b==y)return at;
    }
    return -1;
}
__global__ void k_ClearPinchEdges(int cap,PinchEdge* table){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=cap)return;
    table[i].owner=-1;table[i].count=0;
    for(int k=0;k<4;k++)table[i].halfedge[k]=-1;
    table[i].reserved[0]=table[i].reserved[1]=0;
}
__global__ void k_IndexPinchEdges(int bound,const unsigned* count,const int* tris,
                                  PinchEdge* table,int cap){
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=bound || (unsigned)t>=*count)return;
    unsigned a,b;pinchEdgeKey(tris,t,a,b);unsigned h=pinchHash(a,b,cap);
    for(int k=0;k<cap;k++){
        PinchEdge* entry=table+((h+k)&(cap-1));int old=atomicCAS(&entry->owner,-1,t);
        unsigned x=0,y=0;if(old>=0)pinchEdgeKey(tris,old,x,y);
        if(old<0 || (x==a && y==b)){
            int slot=atomicAdd(&entry->count,1);
            if(slot<4)entry->halfedge[slot]=t;
            return;
        }
    }
}
__device__ inline int3 pinchVoxel(unsigned key,int3 dims){
    unsigned v=(key&0x3fffffffu)/8u;
    return make_int3(v%dims.x,(v/dims.x)%dims.y,v/(dims.x*dims.y));
}
__device__ inline float3 pinchPoint(unsigned key,const float4* fp){
    float4 p=fp[key&0x3fffffffu];return make_float3(p.x,p.y,p.z);
}
struct PinchPlan{int faces[4],triangles[12];};
__device__ inline bool makePinchPlan(int slot,const PinchEdge* table,int cap,
                                    const int* tris,const float4* fp,int3 dims,PinchPlan& out){
    const PinchEdge& edge=table[slot];if(edge.count!=4)return false;
    unsigned a,b;pinchEdgeKey(tris,edge.owner,a,b);
    if(a==b || !(a&0x80000000u) || !(b&0x80000000u))return false;
    int3 va=pinchVoxel(a,dims),vb=pinchVoxel(b,dims);
    if(abs(va.x-vb.x)+abs(va.y-vb.y)+abs(va.z-vb.z)!=1)return false;
    int positive[2],negative[2],np=0,nn=0;
    unsigned third[4];int faces[4];
    for(int i=0;i<4;i++){
        int token=edge.halfedge[i];if(token<0)return false;
        int face=token/3,k=token%3;faces[i]=face;
        for(int j=0;j<3;j++)if(((unsigned)tris[3*face+j]&0xc0000000u)!=0xc0000000u)return false;
        unsigned start=pinchKey(tris[3*face+k]);third[i]=pinchKey(tris[3*face+(k+2)%3]);
        if(third[i]==a || third[i]==b)return false;
        for(int j=0;j<i;j++)if(faces[j]==face || third[j]==third[i])return false;
        if(start==a){if(np==2)return false;positive[np++]=i;}
        else{if(nn==2)return false;negative[nn++]=i;}
        // A face touching two ambiguous edges needs a larger reconstruction.
        // Reject it instead of allowing two concurrent edits of that face.
        for(int j=1;j<3;j++){
            unsigned x,y;pinchEdgeKey(tris,3*face+(k+j)%3,x,y);
            int found=pinchFind(x,y,tris,table,cap);
            if(found<0 || table[found].count!=2)return false;
        }
    }
    if(np!=2 || nn!=2)return false;
    if(third[positive[0]]>third[positive[1]]){int t=positive[0];positive[0]=positive[1];positive[1]=t;}
    // The retained counterexample has two opposing fronts, each with two
    // components in its outer cell. Pair DIFFERENT outer cells along each
    // consistently oriented wall; capping each front alone pinches vertices.
    int match=-1;
    for(int j=0;j<2;j++){
        unsigned c0=third[positive[0]]&0x3fffffffu,c1=third[positive[1]]&0x3fffffffu;
        unsigned d0=third[negative[j]]&0x3fffffffu,d1=third[negative[1-j]]&0x3fffffffu;
        if(c0/8!=d0/8 && c0/8==d1/8 && c1/8==d0/8){match=j;break;}
    }
    if(match<0)return false;
    float3 pa=pinchPoint(a,fp),pb=pinchPoint(b,fp);
    for(int pair=0;pair<2;pair++){
        int ip=positive[pair],im=negative[pair?1-match:match];
        unsigned c=third[ip],d=third[im];
        if(pinchFind(c,d,tris,table,cap)>=0)return false;
        float3 pc=pinchPoint(c,fp),pd=pinchPoint(d,fp);
        float3 old0=cross3(pb-pa,pc-pa),old1=cross3(pa-pb,pd-pb);
        float3 next0=cross3(pd-pa,pc-pa),next1=cross3(pc-pb,pd-pb);
        if(dot3(old0,old1)<=0.f || dot3(next0,next0)<1e-16f || dot3(next1,next1)<1e-16f)return false;
        float3 outward=old0+old1;
        if(dot3(next0,outward)<=0.f || dot3(next1,outward)<=0.f)return false;
        out.faces[2*pair]=faces[ip];out.faces[2*pair+1]=faces[im];
        unsigned keys[6]={a,d,c,b,c,d};
        for(int j=0;j<6;j++)out.triangles[6*pair+j]=(int)(keys[j]|0x40000000u);
    }
    return true;
}
__global__ void k_SelectPinch(int cap,const PinchEdge* table,const int* tris,
                              const float4* fp,int3 dims,unsigned* stats){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=cap || table[i].count<=2)return;
    PinchPlan plan;
    if(makePinchPlan(i,table,cap,tris,fp,dims,plan)){
        atomicMin(stats,(unsigned)i);atomicAdd(stats+1,1u);
    }else atomicAdd(stats+3,1u);
}
__global__ void k_ApplyPinch(const PinchEdge* table,int cap,const int* tris,int* output,
                             const float4* fp,int3 dims,unsigned* stats){
    if(*stats==~0u)return;
    PinchPlan plan;
    if(!makePinchPlan((int)*stats,table,cap,tris,fp,dims,plan))return;
    // One patch per reconstruction prevents indirect cross-patch conflicts.
    // A second defect is retained for validation, not silently disregarded.
    for(int i=0;i<4;i++)for(int j=0;j<3;j++)output[3*plan.faces[i]+j]=plan.triangles[3*i+j];
    stats[2]=1;
}
static cudaError_t runPinchRepair(int indices,const unsigned* count,const int* tris,int* output,
                                  const float4* fp,int3 dims,PinchEdge*& table,int& capacity,unsigned*& stats){
    int needed=1;while(needed<2*indices)needed*=2;
    cudaError_t e=cudaSuccess;
    if(needed>capacity){
        cudaFree(table);table=nullptr;capacity=0;
        e=cudaMalloc(&table,(size_t)needed*sizeof(PinchEdge));if(e!=cudaSuccess)return e;capacity=needed;
    }
    if(!stats){e=cudaMalloc(&stats,4*sizeof(unsigned));if(e!=cudaSuccess)return e;}
    e=cudaMemsetAsync(stats,0,4*sizeof(unsigned));if(e!=cudaSuccess)return e;
    e=cudaMemsetAsync(stats,0xff,sizeof(unsigned));if(e!=cudaSuccess)return e;
    e=cudaMemcpyAsync(output,tris,(size_t)indices*sizeof(int),cudaMemcpyDeviceToDevice);if(e!=cudaSuccess)return e;
    k_ClearPinchEdges<<<(capacity+63)/64,64>>>(capacity,table);
    k_IndexPinchEdges<<<(indices+63)/64,64>>>(indices,count,tris,table,capacity);
    k_SelectPinch<<<(capacity+63)/64,64>>>(capacity,table,tris,fp,dims,stats);
    k_ApplyPinch<<<1,1>>>(table,capacity,tris,output,fp,dims,stats);
    return cudaGetLastError();
}
static cudaError_t repairWallPinches(){
    unsigned count=0;cudaError_t e=cudaMemcpy(&count,d_triCounter,sizeof(count),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess || count==0)return e;
    if(count>(unsigned)g_cleanCapacity || count>(unsigned)(3*g_desc.triCapacity))return cudaErrorInvalidValue;
    e=runPinchRepair((int)count,d_triCounter,d_tri,d_cleanTriangles,d_cut_cutFP,g_dims,
        d_pinchEdges,g_pinchCapacity,d_pinchStats);
    if(e!=cudaSuccess)return e;
    return cudaMemcpyAsync(d_tri,d_cleanTriangles,(size_t)count*sizeof(int),cudaMemcpyDeviceToDevice);
}
extern "C" LCS_HANDOFF_EXPORT int LCS_ReadPinchStats(unsigned* out){
    if(!out || !d_pinchStats)return -1;
    return -(int)cudaMemcpy(out,d_pinchStats,4*sizeof(unsigned),cudaMemcpyDeviceToHost);
}
// Offline GPU mesh probe. It cannot be called on an initialized simulation.
extern "C" LCS_HANDOFF_EXPORT int LCS_ProbePinchSurface(int indices,const int* input,
    int fpCount,const float* points,const int* dims,int* output,unsigned* statistics){
    if(g_ready || indices<=0 || indices%3 || fpCount<=0 || !input || !points || !dims || !output || !statistics)return -1;
    for(int i=0;i<3;i++)if(dims[i]<=0)return -1;
    for(int i=0;i<indices;i++)if(((unsigned)input[i]&0x80000000u) && ((unsigned)input[i]&0x3fffffffu)>=(unsigned)fpCount)return -1;
    int *src=nullptr,*dst=nullptr;unsigned *count=nullptr,*stats=nullptr;float4* fp=nullptr;
    PinchEdge* table=nullptr;int capacity=0;cudaError_t e=cudaSuccess;
    auto execute=[&]()->cudaError_t{
        cudaError_t q=cudaMalloc(&src,(size_t)indices*sizeof(int));if(q!=cudaSuccess)return q;
        q=cudaMalloc(&dst,(size_t)indices*sizeof(int));if(q!=cudaSuccess)return q;
        q=cudaMalloc(&count,sizeof(unsigned));if(q!=cudaSuccess)return q;
        q=cudaMalloc(&fp,(size_t)fpCount*sizeof(float4));if(q!=cudaSuccess)return q;
        q=cudaMemcpy(src,input,(size_t)indices*sizeof(int),cudaMemcpyHostToDevice);if(q!=cudaSuccess)return q;
        q=cudaMemcpy(count,&indices,sizeof(unsigned),cudaMemcpyHostToDevice);if(q!=cudaSuccess)return q;
        q=cudaMemcpy(fp,points,(size_t)fpCount*sizeof(float4),cudaMemcpyHostToDevice);if(q!=cudaSuccess)return q;
        q=runPinchRepair(indices,count,src,dst,fp,make_int3(dims[0],dims[1],dims[2]),table,capacity,stats);if(q!=cudaSuccess)return q;
        q=cudaMemcpy(output,dst,(size_t)indices*sizeof(int),cudaMemcpyDeviceToHost);if(q!=cudaSuccess)return q;
        return cudaMemcpy(statistics,stats,4*sizeof(unsigned),cudaMemcpyDeviceToHost);
    };
    e=execute();cudaFree(src);cudaFree(dst);cudaFree(count);cudaFree(fp);cudaFree(stats);cudaFree(table);
    return -(int)e;
}
