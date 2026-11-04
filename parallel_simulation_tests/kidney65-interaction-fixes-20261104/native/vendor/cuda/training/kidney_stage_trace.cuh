// Diagnostic only: compare the first replayed step after every existing mark.
// Both runs use the same direct-kernel path. Never use these timings as FPS.
static int g_kidneyTraceMode=0;
static size_t g_kidneyTraceIndex=0;
static std::vector<std::vector<float3>> g_kidneyTraceReference;
static std::vector<std::string> g_kidneyTraceNames;
static std::string g_kidneyTraceReport;
static void kidneyTraceMark(const char* name){
    if(!g_kidneyTraceMode || !d_pos || g_N<=0)return;
    std::vector<float3> nodes(g_N);
    cudaError_t e=cudaMemcpy(nodes.data(),d_pos,g_N*sizeof(float3),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess){g_kidneyTraceReport+="CUDA_ERROR\n";return;}
    if(g_kidneyTraceMode==1){g_kidneyTraceReference.push_back(std::move(nodes));g_kidneyTraceNames.push_back(name);}
    else if(g_kidneyTraceIndex>=g_kidneyTraceReference.size() || g_kidneyTraceNames[g_kidneyTraceIndex]!=name){g_kidneyTraceReport+="STAGE_MISMATCH\n";}
    else{
        const auto& ref=g_kidneyTraceReference[g_kidneyTraceIndex];double maximum=0;int worst=-1,changed=0;
        for(int i=0;i<g_N;i++){
            double x=(double)nodes[i].x-ref[i].x,y=(double)nodes[i].y-ref[i].y,z=(double)nodes[i].z-ref[i].z;
            double squared=x*x+y*y+z*z;
            if(squared>0)changed++;
            if(squared>maximum){maximum=squared;worst=i;}
        }
        char line[256];snprintf(line,sizeof line,"%zu\t%s\t%.12g\t%d\t%d\n",g_kidneyTraceIndex,name,sqrt(maximum)*10,changed,worst);
        g_kidneyTraceReport+=line;
    }
    g_kidneyTraceIndex++;
}
extern "C" LCS_COMBINED_EXPORT int LCS_KidneyTraceBegin(int reference){
    if(!g_ready || g_kidneyTraceMode)return -10101;
    if(reference){g_kidneyTraceReference.clear();g_kidneyTraceNames.clear();}
    else if(g_kidneyTraceReference.empty())return -10102;
    g_kidneyTraceMode=reference?1:2;g_kidneyTraceIndex=0;g_kidneyTraceReport.clear();
    xpbd_profile_sections(1);kidneyTraceMark("replay initial");return 0;
}
extern "C" LCS_COMBINED_EXPORT int LCS_KidneyTraceEnd(char* text,int capacity){
    if(g_kidneyTraceMode){kidneyTraceMark("replay final");g_kidneyTraceMode=0;xpbd_profile_sections(0);}
    if(!text || capacity<=(int)g_kidneyTraceReport.size())return (int)g_kidneyTraceReport.size()+1;
    memcpy(text,g_kidneyTraceReport.c_str(),g_kidneyTraceReport.size()+1);return 0;
}
