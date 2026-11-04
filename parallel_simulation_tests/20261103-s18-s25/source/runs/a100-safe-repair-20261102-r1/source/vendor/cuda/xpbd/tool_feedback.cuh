// Included after pressure_blade.cuh in the production solver translation unit.
// One device-to-host transfer per solve/reconstruction observation boundary.
#include "tool_feedback.h"
struct ToolFeedbackPacket {
    XpbdToolState pose;
    int initialized;
    XpbdToolFeedback feedback;
};
static ToolFeedbackPacket* d_toolFeedback=nullptr;

__global__ void k_PackToolFeedback(ToolFeedbackPacket* packet, const CoupledTool* tool,
    XpbdToolState fallback, bool armed, const float* error, const int* live,
    bool embedded, bool grasping, bool fallbackLock, const float4* jawLoad,
    const float4* reaction, int substeps, const CoupledTool* blade,
    const int* bladePhase, const float* bladeResult)
{
    packet->initialized=armed && tool->initialized;
    packet->pose=packet->initialized ? tool->pose : fallback;
    auto& f=packet->feedback;
    f.hinge=packet->pose.jawHinge;f.jawGap=packet->pose.jawGap;
    f.gripError=error[0];f.live=(int)error[1];
    f.load[0]=jawLoad->x;f.load[1]=jawLoad->y;f.load[2]=jawLoad->z;f.load[3]=jawLoad->w;
    f.graspLocked=embedded ? (error[1]>0 && grasping ? 1 : 0) : (fallbackLock ? 1 : 0);
    f.reaction=mkf3(reaction->x,reaction->y,reaction->z);
    f.contacts=(int)ceilf(reaction->w/fmaxf(1.f,(float)substeps));
    f.blade[0]=blade->pose.jawHinge.x;f.blade[1]=blade->pose.jawHinge.y;f.blade[2]=blade->pose.jawHinge.z;
    for(int k=0;k<4;k++)f.blade[3+k]=bladeResult[k];
    f.blade[7]=(float)*bladePhase;f.blade[8]=bladeResult[4];f.blade[9]=bladeResult[5];
}

extern "C" int xpbd_get_tool_feedback(XpbdToolFeedback* result)
{
    if(!result || !g_bound || !d_coupledTool || !d_jawLoad || !d_contactReaction ||
        !d_gripError || !d_bladeBody || !d_bladePhase || !d_bladeResult || g_scissorsMode)
        return XPBD_ERR_BAD_ARG;
    cudaError_t e=cudaSuccess;
    if(!d_toolFeedback)e=cudaMalloc(&d_toolFeedback,sizeof(ToolFeedbackPacket));
    if(e!=cudaSuccess)return -(int)e;
    e=observeGrip();if(e!=cudaSuccess)return -(int)e;
    k_PackToolFeedback<<<1,1>>>(d_toolFeedback,d_coupledTool,g_tool,g_toolArmed,d_gripError,d_gripLive,
        g_embeddedGrip,g_grasping,g_anchorAssist && g_captureWedge && d_captureInside && g_grasping,
        d_jawLoad,d_contactReaction,g_p.numSubsteps,d_bladeBody,d_bladePhase,d_bladeResult);
    e=cudaGetLastError();if(e!=cudaSuccess)return -(int)e;
    ToolFeedbackPacket packet;
    e=cudaMemcpy(&packet,d_toolFeedback,sizeof(packet),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return -(int)e;
    if(packet.initialized)g_tool=g_toolFrom=packet.pose;
    *result=packet.feedback;return 0;
}
