// Diagnostic-only failing query and group inputs; never a production include.
#include <cstdio>
#include <cstring>
struct ContactSnapshotHeader {int magic,n,kind,stride,stationary,tool,hasCache,hasPhase;float skin,thickness,minGap;};
static char contactSnapshotPath[1024]={};static bool contactSnapshotWritten=false;
extern "C" __declspec(dllexport) int LCS_SetContactProbeDump(const char* path){
    if(!path || strlen(path)>=sizeof(contactSnapshotPath))return -1;
    strcpy(contactSnapshotPath,path);contactSnapshotWritten=false;return 0;
}
static bool writeContactSnapshot(int t,int kind,int stride,int stationary,int tool,const SurfaceContact* source,const float4* current,const float3* previous,const ContactGroupBounds* groups,const unsigned char* rejected){
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
    // Appended GRP1 footer stores the complete group geometry and coarse decision.
    int start=64*(t/64),count=std::min(64,g_surfaceCount-start);
    int footer[5]={0x47525031,g_surfaceCount,t,start,count};
    ok=ok && fwrite(footer,sizeof(footer),1,file)==1;
    ok=ok && save(groups+t/64,sizeof(ContactGroupBounds)) && save(rejected+kind*contactGroupCount(g_surfaceCount)+t/64,1);
    ok=ok && save(current+3*start,3*count*sizeof(float4)) && save(previous+3*start,3*count*sizeof(float3));
    if(fclose(file)!=0)ok=false;
    printf("[CONTACT-SNAPSHOT] tool=%d kind=%d stationary=%d triangle=%d saved=%d\n",tool,kind,stationary,t,int(ok));
    return ok;
}
