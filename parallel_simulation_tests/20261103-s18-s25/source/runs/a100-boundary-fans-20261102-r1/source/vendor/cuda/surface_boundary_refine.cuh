// Experimental boundary-fan refinement. A volume component can have multiple
// boundary patches. Do not split the material graph to make its boundary fit.
#include "boundary_inputs.cuh"
struct BoundaryEdge {int owner,count,halfedge[4],reserved[2];};
static BoundaryEdge* d_boundaryEdges=nullptr;
static unsigned* d_boundaryStats=nullptr;
static int* d_featureBindings=nullptr;
static int g_boundaryCapacity=0,g_featureBindingCapacity=0;
__device__ inline unsigned boundaryKey(int raw){return (unsigned)raw&0xbfffffffu;}
__device__ inline void boundaryEdgeKey(const int* tris,int token,unsigned& a,unsigned& b){
    int f=token/3,k=token%3;a=boundaryKey(tris[3*f+k]);b=boundaryKey(tris[3*f+(k+1)%3]);
    if(a>b){unsigned t=a;a=b;b=t;}
}
__device__ inline unsigned boundaryHash(unsigned a,unsigned b,int cap){return ((a*73856093u)^(b*19349663u))&(cap-1);}
__device__ inline int boundaryFind(unsigned a,unsigned b,const int* tris,const BoundaryEdge* table,int cap){
    if(a>b){unsigned t=a;a=b;b=t;}unsigned hash=boundaryHash(a,b,cap);
    for(int k=0;k<cap;k++){
        int at=(hash+k)&(cap-1),owner=table[at].owner;if(owner<0)return -1;
        unsigned x,y;boundaryEdgeKey(tris,owner,x,y);if(x==a && y==b)return at;
    }
    return -1;
}
__global__ void k_ClearBoundaryEdges(int cap,BoundaryEdge* table){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=cap)return;
    table[i].owner=-1;table[i].count=0;
    for(int k=0;k<4;k++)table[i].halfedge[k]=-1;
    table[i].reserved[0]=table[i].reserved[1]=0;
}
__global__ void k_IdentityFeatureBindings(int count,int* bindings){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)bindings[i]=i%8;
}
__global__ void k_IndexBoundaryEdges(int indices,const int* tris,BoundaryEdge* table,int cap){
    int token=blockIdx.x*blockDim.x+threadIdx.x;if(token>=indices)return;
    unsigned a,b;boundaryEdgeKey(tris,token,a,b);unsigned hash=boundaryHash(a,b,cap);
    for(int k=0;k<cap;k++){
        BoundaryEdge* entry=table+((hash+k)&(cap-1));int old=atomicCAS(&entry->owner,-1,token);
        unsigned x=0,y=0;if(old>=0)boundaryEdgeKey(tris,old,x,y);
        if(old<0 || (x==a && y==b)){
            int slot=atomicAdd(&entry->count,1);if(slot<4)entry->halfedge[slot]=token;return;
        }
    }
}
__device__ inline int3 boundaryVoxel(unsigned key,int3 dims){
    unsigned v=(key&0x3fffffffu)/8;return make_int3(v%dims.x,(v/dims.x)%dims.y,v/(dims.x*dims.y));
}
__device__ inline float3 boundaryPoint(unsigned key,const float4* fp){
    float4 p=fp[key&0x3fffffffu];return make_float3(p.x,p.y,p.z);
}
__device__ inline bool interiorSingleComponent(unsigned key,const BoundaryInputs& input,const float4* fp){
    if(!(key&0x80000000u) || (key&7u)!=0)return false;
    int v=(key&0x3fffffffu)/8;
    if(v>=input.voxelCount || input.isectCounts[v]!=0 || fp[8*v+1].w>.5f)return false;
    if(input.connectivity[input.boundaryMasks[v]&4095u].compCount!=1)return false;
    for(int k=0;k<8;k++)if(!input.inside[input.voxelCorners[8*v+k]])return false;
    return true;
}
__global__ void k_SelectBoundaryEdge(int cap,const BoundaryEdge* table,const int* tris,
                                    const float4* fp,BoundaryInputs input,unsigned* stats){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=cap || table[i].count<=2)return;
    const BoundaryEdge& edge=table[i];bool valid=edge.count==4;
    unsigned a,b;boundaryEdgeKey(tris,edge.owner,a,b);
    valid=valid && a!=b && interiorSingleComponent(a,input,fp) && interiorSingleComponent(b,input,fp);
    if(valid){
        int3 va=boundaryVoxel(a,input.dims),vb=boundaryVoxel(b,input.dims);
        valid=abs(va.x-vb.x)+abs(va.y-vb.y)+abs(va.z-vb.z)==1;
    }
    if(valid)for(int k=0;k<4;k++){
        int token=edge.halfedge[k];valid=valid && token>=0;
        if(token>=0)for(int j=0;j<3;j++)valid=valid && (((unsigned)tris[3*(token/3)+j]&0xc0000000u)==0xc0000000u);
    }
    if(valid){atomicMin(stats,(unsigned)i);atomicAdd(stats+1,1u);}else atomicAdd(stats+3,1u);
}
struct BoundaryFan {
    unsigned center;
    int voxel,count,faces[64],group[64];
    unsigned sources[2];
    float3 point[2];
};
__device__ inline int fanGroup(const BoundaryFan& fan,int face){
    for(int k=0;k<fan.count;k++)if(fan.faces[k]==face)return fan.group[k];
    return -1;
}
__device__ inline int fanFaceEdge(const int* triangle,int voxel,int3 dims){
    int3 low=boundaryVoxel(boundaryKey(triangle[0]),dims),high=low;
    for(int k=1;k<3;k++){
        int3 q=boundaryVoxel(boundaryKey(triangle[k]),dims);
        low=make_int3(min(low.x,q.x),min(low.y,q.y),min(low.z,q.z));
        high=make_int3(max(high.x,q.x),max(high.y,q.y),max(high.z,q.z));
    }
    int3 width=high-low;int axis=-1;
    if(width.x==0 && width.y==1 && width.z==1)axis=0;
    if(width.x==1 && width.y==0 && width.z==1)axis=1;
    if(width.x==1 && width.y==1 && width.z==0)axis=2;
    if(axis<0)return -1;
    for(int k=0;k<4;k++)if(VoxId(high+ES_VoxelOffset(axis,k),dims)==voxel)return ES_LocalEdge(axis,k);
    return -1;
}
__device__ inline bool collectBoundaryFan(unsigned center,unsigned opposite,int count,const int* tris,
    const BoundaryEdge* edges,int capacity,BoundaryInputs input,BoundaryFan& fan){
    fan.center=center;fan.voxel=(center&0x3fffffffu)/8;fan.count=0;fan.sources[0]=fan.sources[1]=0;
    for(int f=0;f<count;f++){
        int occurrences=0;for(int k=0;k<3;k++)occurrences+=boundaryKey(tris[3*f+k])==center;
        if(!occurrences)continue;if(occurrences!=1 || fan.count==64)return false;
        for(int k=0;k<3;k++)if(((unsigned)tris[3*f+k]&0xc0000000u)!=0xc0000000u)return false;
        fan.faces[fan.count]=f;fan.group[fan.count++]=-1;
    }
    int groups=0;
    for(int seed=0;seed<fan.count;seed++)if(fan.group[seed]<0){
        if(groups==2)return false;fan.group[seed]=groups;
        bool changed=true;
        for(int pass=0;pass<fan.count && changed;pass++){
            changed=false;
            for(int i=0;i<fan.count;i++)if(fan.group[i]==groups)for(int j=0;j<fan.count;j++)if(fan.group[j]<0){
                bool touch=false;
                for(int x=0;x<3;x++){
                    unsigned key=boundaryKey(tris[3*fan.faces[i]+x]);if(key==center || key==opposite)continue;
                    for(int y=0;y<3;y++)touch=touch || key==boundaryKey(tris[3*fan.faces[j]+y]);
                }
                if(touch){fan.group[j]=groups;changed=true;}
            }
        }
        groups++;
    }
    if(groups!=2)return false;
    int frontCount[2]={0,0};
    for(int i=0;i<fan.count;i++){
        int face=fan.faces[i],g=fan.group[i];
        for(int k=0;k<3;k++){
            unsigned other=boundaryKey(tris[3*face+k]);if(other==center)continue;
            if(other==opposite){frontCount[g]++;continue;}
            int slot=boundaryFind(center,other,tris,edges,capacity);
            if(slot<0 || edges[slot].count!=2)return false;
        }
        int local=fanFaceEdge(tris+3*face,fan.voxel,input.dims);if(local<0)return false;
        fan.sources[g]|=1u<<local;
    }
    if(frontCount[0]!=2 || frontCount[1]!=2 || !fan.sources[0] || !fan.sources[1] || (fan.sources[0]&fan.sources[1]))return false;
    if(fan.sources[0]>fan.sources[1]){
        unsigned tmp=fan.sources[0];fan.sources[0]=fan.sources[1];fan.sources[1]=tmp;
        for(int k=0;k<fan.count;k++)fan.group[k]=1-fan.group[k];
    }
    return true;
}
__device__ inline bool fitBoundaryFan(BoundaryFan& fan,BoundaryInputs input){
    int v=fan.voxel;float3 origin=input.positions[input.voxelCorners[8*v]],center=make_float3(0,0,0);
    for(int k=0;k<8;k++)center=center+input.positions[input.voxelCorners[8*v+k]];
    center=center*.125f;
    float3 extent=input.rest[input.voxelCorners[8*v+6]]-input.rest[input.voxelCorners[8*v]];
    float3 half=make_float3(fabsf(extent.x)*.5f,fabsf(extent.y)*.5f,fabsf(extent.z)*.5f);
    int3 vc=boundaryVoxel(fan.center,input.dims);
    for(int group=0;group<2;group++){
        float3 sum=make_float3(0,0,0);int3 normals=make_int3(0,0,0);int count=0;
        int available=min(input.cutPointCount,(int)*input.cutPointCounter);
        for(int i=0;i<available;i++){
            CutPointGpu cp=input.cutPoints[i];if(!input.active[cp.ownerParticle])continue;
            int lc=LocalCornerOf(CornerCoordCuda(cp.ownerParticle,input.dims),vc);if(lc<0)continue;
            GridEdgeGpu2 edge=input.gridEdges[cp.edgeId];int3 base=make_int3(edge.bcX,edge.bcY,edge.bcZ);int local=-1;
            for(int k=0;k<4;k++)if(VoxId(base+ES_VoxelOffset(edge.axis,k),input.dims)==v)local=ES_LocalEdge(edge.axis,k);
            if(local<0 || !(fan.sources[group]&(1u<<local)))continue;
            RotGpu rotation=input.rotations[cp.ownerParticle];
            sum=sum+(input.positions[cp.ownerParticle]-origin)+mulRv(rotation,cp.localOffset);
            float3 normal=mulRv(rotation,cp.matNrm);
            normals.x+=(int)rintf(normal.x*POS_SCALE);normals.y+=(int)rintf(normal.y*POS_SCALE);normals.z+=(int)rintf(normal.z*POS_SCALE);count++;
        }
        if(count<2)return false;
        float3 centroid=sum/(float)count+origin;
        float3 x=make_float3(fminf(fmaxf(centroid.x,center.x-half.x),center.x+half.x),
            fminf(fmaxf(centroid.y,center.y-half.y),center.y+half.y),fminf(fmaxf(centroid.z,center.z-half.z),center.z+half.z));
        float3 normal=make_float3((float)normals.x,(float)normals.y,(float)normals.z)/POS_SCALE;float magnitude=length3(normal);
        if(magnitude>1e-6f && magnitude>.25f*count){normal=normal/magnitude;x=x-dot3(normal,x-centroid)*normal;}
        if(!isfinite(x.x)||!isfinite(x.y)||!isfinite(x.z))return false;
        fan.point[group]=x;
    }
    return true;
}
__device__ inline bool boundaryGeometryValid(const BoundaryFan& a,const BoundaryFan& b,const int* tris,const float4* fp){
    for(int side=0;side<2;side++){
        const BoundaryFan& fan=side?b:a;
        for(int i=0;i<fan.count;i++){
            int face=fan.faces[i];float3 old[3],next[3];
            for(int k=0;k<3;k++){
                unsigned key=boundaryKey(tris[3*face+k]);old[k]=boundaryPoint(key,fp);next[k]=old[k];
                if(key==a.center){int g=fanGroup(a,face);if(g<0)return false;next[k]=a.point[g];}
                if(key==b.center){int g=fanGroup(b,face);if(g<0)return false;next[k]=b.point[g];}
            }
            float3 before=cross3(old[1]-old[0],old[2]-old[0]),after=cross3(next[1]-next[0],next[2]-next[0]);
            if(dot3(after,after)<1e-16f || dot3(before,after)<=0.f)return false;
        }
    }
    return true;
}
__global__ void k_RefineBoundaryFan(const BoundaryEdge* edges,int cap,int count,const int* tris,int* output,
                                   float4* fp,int* bindings,BoundaryInputs input,unsigned* stats){
    if(*stats==~0u)return;
    unsigned ka,kb;const BoundaryEdge& edge=edges[*stats];boundaryEdgeKey(tris,edge.owner,ka,kb);
    BoundaryFan a,b;
    bool valid=collectBoundaryFan(ka,kb,count,tris,edges,cap,input,a) && collectBoundaryFan(kb,ka,count,tris,edges,cap,input,b);
    if(valid){
        int pairs[4]={0,0,0,0};
        for(int k=0;k<4;k++){int face=edge.halfedge[k]/3,ga=fanGroup(a,face),gb=fanGroup(b,face);if(ga<0||gb<0){valid=false;break;}pairs[2*ga+gb]++;}
        valid=valid && ((pairs[0]==2 && pairs[3]==2 && !pairs[1] && !pairs[2]) || (pairs[1]==2 && pairs[2]==2 && !pairs[0] && !pairs[3]));
    }
    valid=valid && fitBoundaryFan(a,input) && fitBoundaryFan(b,input);
    valid=valid && boundaryGeometryValid(a,b,tris,fp);
    if(!valid){stats[3]++;return;}
    for(int side=0;side<2;side++){
        BoundaryFan& fan=side?b:a;
        for(int g=0;g<2;g++){
            float3 p=fan.point[g];fp[8*fan.voxel+g]=make_float4(p.x,p.y,p.z,1);
            // Low 3 bits select the unchanged volume component. Higher bits
            // identify this boundary patch for contact/grasp cache invalidation.
            bindings[8*fan.voxel+g]=(int)((fan.sources[g]+1u)<<3);
        }
        for(int i=0;i<fan.count;i++)for(int k=0;k<3;k++)if(boundaryKey(tris[3*fan.faces[i]+k])==fan.center)
            output[3*fan.faces[i]+k]=(int)(0xc0000000u|(unsigned)(8*fan.voxel+fan.group[i]));
    }
    stats[2]=1;
}
static cudaError_t runBoundaryRefine(int indices,const int* tris,int* output,float4* fp,int* bindings,
    BoundaryInputs input,BoundaryEdge*& edges,int& capacity,unsigned*& stats){
    int needed=1;while(needed<2*indices)needed*=2;cudaError_t e=cudaSuccess;
    if(needed>capacity){cudaFree(edges);edges=nullptr;capacity=0;e=cudaMalloc(&edges,(size_t)needed*sizeof(BoundaryEdge));if(e!=cudaSuccess)return e;capacity=needed;}
    if(!stats){e=cudaMalloc(&stats,4*sizeof(unsigned));if(e!=cudaSuccess)return e;}
    e=cudaMemsetAsync(stats,0,4*sizeof(unsigned));if(e!=cudaSuccess)return e;
    e=cudaMemsetAsync(stats,0xff,sizeof(unsigned));if(e!=cudaSuccess)return e;
    e=cudaMemcpyAsync(output,tris,(size_t)indices*sizeof(int),cudaMemcpyDeviceToDevice);if(e!=cudaSuccess)return e;
    k_IdentityFeatureBindings<<<(8*input.voxelCount+63)/64,64>>>(8*input.voxelCount,bindings);
    k_ClearBoundaryEdges<<<(capacity+63)/64,64>>>(capacity,edges);
    k_IndexBoundaryEdges<<<(indices+63)/64,64>>>(indices,tris,edges,capacity);
    k_SelectBoundaryEdge<<<(capacity+63)/64,64>>>(capacity,edges,tris,fp,input,stats);
    k_RefineBoundaryFan<<<1,1>>>(edges,capacity,indices/3,tris,output,fp,bindings,input,stats);
    return cudaGetLastError();
}
static cudaError_t refineBoundaryFans(){
    unsigned indices=0;cudaError_t e=cudaMemcpy(&indices,d_triCounter,sizeof(unsigned),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess || indices==0)return e;
    if(indices>(unsigned)g_cleanCapacity || indices>(unsigned)(3*g_desc.triCapacity))return cudaErrorInvalidValue;
    if(g_featureBindingCapacity<8*g_desc.voxelCount){
        cudaFree(d_featureBindings);d_featureBindings=nullptr;g_featureBindingCapacity=0;
        e=cudaMalloc(&d_featureBindings,(size_t)8*g_desc.voxelCount*sizeof(int));if(e!=cudaSuccess)return e;g_featureBindingCapacity=8*g_desc.voxelCount;
    }
    e=runBoundaryRefine(indices,d_tri,d_cleanTriangles,d_cut_cutFP,d_featureBindings,cut_boundary_inputs(),d_boundaryEdges,g_boundaryCapacity,d_boundaryStats);
    if(e!=cudaSuccess)return e;
    return cudaMemcpyAsync(d_tri,d_cleanTriangles,(size_t)indices*sizeof(int),cudaMemcpyDeviceToDevice);
}
extern "C" LCS_HANDOFF_EXPORT int LCS_ReadBoundaryRefineStats(unsigned* out){
    if(!out || !d_boundaryStats)return -1;return -(int)cudaMemcpy(out,d_boundaryStats,4*sizeof(unsigned),cudaMemcpyDeviceToHost);
}
// Standalone frozen-mesh probe, unavailable after simulation initialization.
// Host arrays are copied into temporary device buffers; no state archive is
// restored across library builds. The buffer order is documented by the types.
extern "C" LCS_HANDOFF_EXPORT int LCS_ProbeBoundarySurface(int indices,const int* triangles,
    const float* featurePoints,const int* counts,const void* const* buffers,int* output,
    float* outputFeatures,int* outputBindings,unsigned* outputStats){
    if(g_ready || indices<=0 || indices%3 || !triangles || !featurePoints || !counts || !buffers ||
        !output || !outputFeatures || !outputBindings || !outputStats)return -1;
    for(int k=0;k<7;k++)if(counts[k]<=0)return -1;
    long long nv=(long long)counts[0]*counts[1]*counts[2],nc=(long long)(counts[0]+1)*(counts[1]+1)*(counts[2]+1);
    if(nv!=counts[3] || nc!=counts[4] || nv>16000000)return -1;
    for(int k=0;k<11;k++)if(!buffers[k])return -1;
    for(int k=0;k<indices;k++)if(((unsigned)triangles[k]&0x80000000u) && ((unsigned)triangles[k]&0x3fffffffu)>=8u*(unsigned)nv)return -1;
    size_t sizes[11]={(size_t)counts[6]*sizeof(CutPointGpu),(size_t)counts[5]*sizeof(GridEdgeGpu2),
        (size_t)nc*sizeof(RotGpu),(size_t)nc*sizeof(float3),(size_t)nc*sizeof(float3),(size_t)nv*8*sizeof(int),
        4096*sizeof(Conn4096Gpu),(size_t)nv*sizeof(unsigned),(size_t)nv*sizeof(int),(size_t)nc*sizeof(int),(size_t)nc*sizeof(int)};
    void* data[11]={};int *src=nullptr,*dst=nullptr,*binding=nullptr;float4* fp=nullptr;
    unsigned *counter=nullptr,*stats=nullptr;BoundaryEdge* edges=nullptr;int capacity=0;
    auto execute=[&]()->cudaError_t{
        cudaError_t e=cudaSuccess;
        for(int k=0;k<11;k++){
            e=cudaMalloc(data+k,sizes[k]);if(e!=cudaSuccess)return e;
            e=cudaMemcpy(data[k],buffers[k],sizes[k],cudaMemcpyHostToDevice);if(e!=cudaSuccess)return e;
        }
        e=cudaMalloc(&src,(size_t)indices*sizeof(int));if(e!=cudaSuccess)return e;
        e=cudaMalloc(&dst,(size_t)indices*sizeof(int));if(e!=cudaSuccess)return e;
        e=cudaMalloc(&fp,(size_t)nv*8*sizeof(float4));if(e!=cudaSuccess)return e;
        e=cudaMalloc(&binding,(size_t)nv*8*sizeof(int));if(e!=cudaSuccess)return e;
        e=cudaMalloc(&counter,sizeof(unsigned));if(e!=cudaSuccess)return e;
        e=cudaMemcpy(src,triangles,(size_t)indices*sizeof(int),cudaMemcpyHostToDevice);if(e!=cudaSuccess)return e;
        e=cudaMemcpy(fp,featurePoints,(size_t)nv*8*sizeof(float4),cudaMemcpyHostToDevice);if(e!=cudaSuccess)return e;
        e=cudaMemcpy(counter,counts+6,sizeof(unsigned),cudaMemcpyHostToDevice);if(e!=cudaSuccess)return e;
        BoundaryInputs input={};input.dims=make_int3(counts[0],counts[1],counts[2]);input.voxelCount=(int)nv;
        input.cornerCount=(int)nc;input.gridEdgeCount=counts[5];input.cutPointCount=counts[6];input.cutPointCounter=counter;
        input.cutPoints=(CutPointGpu*)data[0];input.gridEdges=(GridEdgeGpu2*)data[1];input.rotations=(RotGpu*)data[2];
        input.positions=(float3*)data[3];input.rest=(float3*)data[4];input.voxelCorners=(int*)data[5];
        input.connectivity=(Conn4096Gpu*)data[6];input.boundaryMasks=(unsigned*)data[7];input.isectCounts=(int*)data[8];
        input.active=(int*)data[9];input.inside=(int*)data[10];
        e=runBoundaryRefine(indices,src,dst,fp,binding,input,edges,capacity,stats);if(e!=cudaSuccess)return e;
        e=cudaMemcpy(output,dst,(size_t)indices*sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return e;
        e=cudaMemcpy(outputFeatures,fp,(size_t)nv*8*sizeof(float4),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return e;
        e=cudaMemcpy(outputBindings,binding,(size_t)nv*8*sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return e;
        return cudaMemcpy(outputStats,stats,4*sizeof(unsigned),cudaMemcpyDeviceToHost);
    };
    cudaError_t error=execute();for(void* p:data)cudaFree(p);
    cudaFree(src);cudaFree(dst);cudaFree(fp);cudaFree(binding);cudaFree(counter);cudaFree(edges);cudaFree(stats);
    return -(int)error;
}
