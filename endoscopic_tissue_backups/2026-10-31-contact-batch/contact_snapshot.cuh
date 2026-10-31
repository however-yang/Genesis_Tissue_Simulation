// Diagnostic-only exact failing-input capture and replay; never a production include.
#include <cstdio>
#include <cstring>
struct ContactSnapshotHeader {int magic,n,kind,stride,stationary,tool,hasCache,hasPhase;float skin,thickness,minGap;};
static char contactSnapshotPath[1024]={};static bool contactSnapshotWritten=false;
extern "C" __declspec(dllexport) int LCS_SetContactProbeDump(const char* path){
    if(!path || strlen(path)>=sizeof(contactSnapshotPath))return -1;
    strcpy(contactSnapshotPath,path);contactSnapshotWritten=false;return 0;
}
static bool writeContactSnapshot(int t,int kind,int stride,int stationary,int tool,const SurfaceContact* source,const float4* current,const float3* previous){
    if(!contactSnapshotPath[0])return true;
    ContactSnapshotHeader header={0x43535031,g_N,kind,stride,stationary,tool,!tool || kind>=3,tool && kind<3,
        tool && kind<3?0:g_contactRadius,tool?0:g_jawThickness,tool?0:g_gp.jawGapClosed};
    FILE* file=fopen(contactSnapshotPath,"wb");if(!file)return false;
    bool ok=fwrite(&header,sizeof(header),1,file)==1;
    auto save=[&](const void* pointer,size_t bytes){
        std::vector<unsigned char> data(bytes);
        if(cudaMemcpy(data.data(),pointer,bytes,cudaMemcpyDeviceToHost)!=cudaSuccess)return false;
        return fwrite(data.data(),1,bytes,file)==bytes;
    };
    JawMeshBounds jaws[2];ok=ok && cudaMemcpyFromSymbol(jaws,c_jawMeshBounds,sizeof(jaws))==cudaSuccess;
    ok=ok && fwrite(jaws,sizeof(jaws),1,file)==1;
    ok=ok && save(tool?d_bladeBody:d_coupledTool,sizeof(CoupledTool));
    int phase=0;if(header.hasPhase)ok=ok && cudaMemcpy(&phase,d_bladePhase,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess;
    ok=ok && fwrite(&phase,sizeof(int),1,file)==1;
    ok=ok && save(d_surfaceVertices+3*t,3*sizeof(SurfaceVertex));
    ok=ok && save(current+3*t,3*sizeof(float4)) && save(previous+3*t,3*sizeof(float3));
    ok=ok && save(d_pos,g_N*sizeof(float3)) && save(d_prevPos,g_N*sizeof(float3));
    ok=ok && save(source+stride*t,stride*sizeof(SurfaceContact));
    if(header.hasCache)ok=ok && save(tool?d_bladeBackCacheSlots:d_contactCacheSlots,CONTACT_CACHE_SIZE*sizeof(int)) &&
        save(tool?d_bladeBackCache:d_contactCache,CONTACT_CACHE_SIZE*sizeof(SurfaceContact));
    if(fclose(file)!=0)ok=false;
    printf("[CONTACT-SNAPSHOT] tool=%d kind=%d stationary=%d triangle=%d saved=%d\n",tool,kind,stationary,t,int(ok));
    return ok;
}
struct ContactReplayMemory {std::vector<void*> pointers;~ContactReplayMemory(){for(void* p:pointers)cudaFree(p);}};
struct ContactReplayFile {FILE* file;~ContactReplayFile(){if(file)fclose(file);}};
extern "C" __declspec(dllexport) int LCS_ReplayContactSnapshot(const char* path,int* output){
    if(!path || !output)return -1;
    ContactReplayFile file={fopen(path,"rb")};if(!file.file)return -2;
    ContactSnapshotHeader h={};if(fread(&h,sizeof(h),1,file.file)!=1 || h.magic!=0x43535031 || h.n<=0 || h.stride<1 || h.stride>6)return -3;
    JawMeshBounds jaws[2];if(fread(jaws,sizeof(jaws),1,file.file)!=1)return -4;
    if(cudaMemcpyToSymbol(c_jawMeshBounds,jaws,sizeof(jaws))!=cudaSuccess)return -5;
    ContactReplayMemory memory;cudaError_t error=cudaSuccess;
#define REPLAY_ALLOC(name,type,count) type* name=nullptr;do{error=cudaMalloc(&name,(size_t)(count)*sizeof(type));if(error!=cudaSuccess)return -(int)error;memory.pointers.push_back(name);}while(false)
#define REPLAY_LOAD(name,type,count) REPLAY_ALLOC(name,type,count);do{std::vector<type> values(count);if(fread(values.data(),sizeof(type),count,file.file)!=(size_t)(count))return -6;error=cudaMemcpy(name,values.data(),(size_t)(count)*sizeof(type),cudaMemcpyHostToDevice);if(error!=cudaSuccess)return -(int)error;}while(false)
    REPLAY_LOAD(body,CoupledTool,1);REPLAY_LOAD(phase,int,1);
    REPLAY_LOAD(vertices,SurfaceVertex,3);REPLAY_LOAD(current,float4,3);REPLAY_LOAD(previous,float3,3);
    REPLAY_LOAD(pos,float3,h.n);REPLAY_LOAD(prev,float3,h.n);REPLAY_LOAD(source,SurfaceContact,h.stride);
    REPLAY_ALLOC(slots,int,CONTACT_CACHE_SIZE);REPLAY_ALLOC(cache,SurfaceContact,CONTACT_CACHE_SIZE);
    if(h.hasCache){
        std::vector<int> values(CONTACT_CACHE_SIZE);std::vector<SurfaceContact> records(CONTACT_CACHE_SIZE);
        if(fread(values.data(),sizeof(int),values.size(),file.file)!=values.size() || fread(records.data(),sizeof(SurfaceContact),records.size(),file.file)!=records.size())return -6;
        error=cudaMemcpy(slots,values.data(),values.size()*sizeof(int),cudaMemcpyHostToDevice);if(error!=cudaSuccess)return -(int)error;
        error=cudaMemcpy(cache,records.data(),records.size()*sizeof(SurfaceContact),cudaMemcpyHostToDevice);if(error!=cudaSuccess)return -(int)error;
    }
    REPLAY_ALLOC(reference,SurfaceContact,h.stride);REPLAY_ALLOC(candidate,SurfaceContact,h.stride);
    REPLAY_ALLOC(degree,int,h.n);REPLAY_ALLOC(active,int,6);REPLAY_ALLOC(count,int,1);
    REPLAY_ALLOC(statistics,int,9);REPLAY_ALLOC(candidates,int,12);REPLAY_ALLOC(queries,BladeQueryBounds,6);
    const int* cs=h.hasCache?slots:nullptr;const SurfaceContact* cr=h.hasCache?cache:nullptr;const int* bp=h.hasPhase?phase:nullptr;
    cudaMemcpy(reference,source,h.stride*sizeof(SurfaceContact),cudaMemcpyDeviceToDevice);cudaMemset(degree,0,h.n*sizeof(int));cudaMemset(count,0,sizeof(int));
    k_FindSurfaceReference<<<1,64>>>(1,h.kind,ContactShape{},ContactShape{},h.skin,vertices,nullptr,prev,pos,reference,degree,active,count,current,previous,body,h.thickness,h.minGap,h.stationary,cs,cr,bp,h.stride,h.tool && h.kind>=3);
    for(int mode=0;mode<3;mode++){
        cudaMemcpy(candidate,source,h.stride*sizeof(SurfaceContact),cudaMemcpyDeviceToDevice);cudaMemset(degree,0,h.n*sizeof(int));cudaMemset(count,0,sizeof(int));cudaMemset(statistics,0,9*sizeof(int));
        int queue[12]={};queue[h.kind]=1;cudaMemcpy(candidates,queue,sizeof(queue),cudaMemcpyHostToDevice);
        if(mode==2 && h.tool){
            cudaMemset(candidates,0,6*sizeof(int));
            k_PrepareBladeQueries<<<1,32>>>(1u<<h.kind,body,h.skin,h.stationary,bp,queries);
            k_BladeSurfaceCandidates<<<1,128>>>(1,1u<<h.kind,1,current,previous,candidate,queries,candidates);
        }
        if(mode==0)k_FindSurface<<<1,64>>>(1,h.kind,ContactShape{},ContactShape{},h.skin,vertices,nullptr,prev,pos,candidate,degree,active,count,current,previous,body,h.thickness,h.minGap,h.stationary,cs,cr,bp,h.stride,h.tool && h.kind>=3);
        else k_FindSurface<true><<<1,64>>>(1,h.kind,ContactShape{},ContactShape{},h.skin,vertices,nullptr,prev,pos,candidate,degree,active,count,current,previous,body,h.thickness,h.minGap,h.stationary,cs,cr,bp,h.stride,h.tool && h.kind>=3,candidates+6+h.kind,candidates+h.kind);
        k_ProbeContactRecords<<<1,64>>>(h.stride,reference,candidate,statistics);
        error=cudaMemcpy(output+9*mode,statistics,9*sizeof(int),cudaMemcpyDeviceToHost);if(error!=cudaSuccess)return -(int)error;
        SurfaceContact a,b;cudaMemcpy(&a,reference+h.kind,sizeof(a),cudaMemcpyDeviceToHost);cudaMemcpy(&b,candidate+h.kind,sizeof(b),cudaMemcpyDeviceToHost);
        printf("[CONTACT-REPLAY] mode=%d kind=%d count=%d/%d weight0=%.9g/%.9g\n",mode,h.kind,a.count,b.count,a.weight[0],b.weight[0]);
    }
#undef REPLAY_LOAD
#undef REPLAY_ALLOC
    return 0;
}
