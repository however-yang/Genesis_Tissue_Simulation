// Diagnostic fixture calling the real contact law, with no live scene state.
#ifdef ENDO_KIDNEY_CONTACT_TRACE
__global__ void k_KidneyContactProbe(float* out){
    int i=threadIdx.x;if(i>=6)return;
    CoupledTool tool={};tool.scissors=1;tool.backContact=1;
    tool.pose.jawAxis=mkf3(1,0,0);tool.pose.jawNormal=mkf3(0,1,0);tool.pose.jawLen=1;
    tool.wristFront=.5f;tool.wristLength=.5f;tool.wristRadius=.1f;
    auto& back=tool.backs[0];back.center=mkf3(.5f,0,0);back.halfSize=mkf3(.5f,.1f,.2f);
    back.axisX=mkf3(1,0,0);back.axisY=mkf3(0,1,0);
    int phase=BladePressing;UnifiedState s={};s.tool=&tool;s.cutPhase=&phase;s.h=.01f;
    SurfaceContact c={};c.kind=i<3?5:3;c.plane.normal=mkf3(0,1,0);c.plane.target=mkf3(0,.1f,0);
    // Separated lateral exit, ordinary penetration, and deep CCD crossing.
    float3 x=i%3==0?mkf3(1.2f,0,0):(i%3==1?mkf3(.25f,.09f,0):mkf3(.25f,-.2f,0));
    c.previous=x; // Stationary witness, not an unloading impact from the box centre.
    float3 n,force;Symmetric3 H={};out[4*i]=unifiedGap(s,c,x,n);
    unifiedContactAt(s,c,x,1.f,force,H);out[4*i+1]=len3(force);out[4*i+2]=n.y;
    float3 nn,ss;out[4*i+3]=shapeDistance(bodyContactShape(&tool,tool.pose,c.kind,0,0),x,nn,ss);
}
extern "C" LCS_COMBINED_EXPORT int LCS_KidneyContactProbe(float* output,int capacity){
    if(!output||capacity<24)return -1;
    float* device=nullptr;cudaError_t err=queryPrivateAlloc(&device,24*sizeof(float));
    if(err!=cudaSuccess)return -(int)err;
    k_KidneyContactProbe<<<1,32>>>(device);
    err=cudaMemcpy(output,device,24*sizeof(float),cudaMemcpyDeviceToHost);
    queryPrivateFree(device);return -(int)err;
}

// Exercise the production manifold, including its finite box clipping and
// capsule quadrature. One patch versus four triangles sharing its witness
// must not multiply the wrist load. This fixture does not alter a live scene.
__global__ void k_KidneyManifoldProbe(float* output){
    __shared__ SurfaceContact source[24],contacts[64];
    __shared__ SurfaceVertex vertices[12];
    __shared__ float3 pos[9];
    __shared__ int ids[64],owners[64],degree[9],count;
    __shared__ float quadrature[64];
    int test=blockIdx.x,lane=threadIdx.x,triangles=test==1?4:1;
    CoupledTool tool={};tool.scissors=1;tool.backContact=1;
    tool.pose.jawAxis=mkf3(1,0,0);tool.pose.jawNormal=mkf3(0,1,0);tool.pose.jawLen=1;
    tool.wristFront=.5f;tool.wristLength=.5f;tool.wristRadius=.1f;
    tool.backs[0].center=mkf3(.5f,0,0);tool.backs[0].halfSize=mkf3(.5f,.1f,.2f);
    tool.backs[0].axisX=mkf3(1,0,0);tool.backs[0].axisY=mkf3(0,1,0);
    int kind=test<2?5:3;
    if(lane==0){
        count=0;for(int i=0;i<9;i++)degree[i]=0;
        float x=test==2?1.25f:.25f;
        pos[0]=mkf3(x,.09f,0);
        for(int j=0;j<=triangles;j++)pos[j+1]=mkf3(x+.05f-.1f*j/triangles,.09f,.02f);
        for(int t=0;t<triangles;t++){
            for(int j=0;j<3;j++){
                SurfaceVertex v={};v.count=1;v.id[0]=j==0?0:t+j;v.weight[0]=1;
                vertices[3*t+j]=v;
            }
            SurfaceContact c={};c.count=1;c.id[0]=0;c.weight[0]=1;c.kind=kind;c.previous=pos[0];
            c.plane.normal=mkf3(0,1,0);c.plane.target=mkf3(x,.1f,0);
            c.referenceShape=bodyContactShape(&tool,tool.pose,kind,0,0);
            source[6*t+kind]=c;
        }
    }
    __syncthreads();
    for(int t=0;t<triangles;t++)unifiedManifoldContact(6*t+kind,lane,source,vertices,pos,pos,&tool,
        0,0,0,contacts,ids,&count,owners,quadrature,degree,6);
    __syncthreads();
    if(lane==0){
        int phase=BladePressing;UnifiedState s={};s.tool=&tool;s.cutPhase=&phase;s.h=.01f;
        s.contacts=contacts;s.contactQuadrature=quadrature;s.pos=pos;
        float area=0,load=0;int loaded=0;
        for(int i=0;i<count;i++){
            float3 force;Symmetric3 H={};unifiedContactAt(s,contacts[i],unifiedPoint(contacts[i],pos),1,force,H);
            area+=quadrature[i];load+=len3(force);loaded+=len3(force)>0;
        }
        output[4*test]=count;output[4*test+1]=area;output[4*test+2]=load;output[4*test+3]=loaded;
    }
}
extern "C" LCS_COMBINED_EXPORT int LCS_KidneyManifoldProbe(float* output,int capacity){
    if(!output||capacity<16)return -1;
    float* device=nullptr;cudaError_t err=queryPrivateAlloc(&device,16*sizeof(float));
    if(err!=cudaSuccess)return -(int)err;
    k_KidneyManifoldProbe<<<4,32>>>(device);
    err=cudaMemcpy(output,device,16*sizeof(float),cudaMemcpyDeviceToHost);
    queryPrivateFree(device);return -(int)err;
}
#endif
