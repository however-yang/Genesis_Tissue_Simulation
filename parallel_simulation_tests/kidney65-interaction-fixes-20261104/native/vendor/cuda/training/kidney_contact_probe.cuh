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
#endif
