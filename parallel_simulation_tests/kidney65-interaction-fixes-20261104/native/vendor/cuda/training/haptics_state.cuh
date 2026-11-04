// Generated from the reviewed inventory. Do not hand edit.
#include "state_archive.h"
#include <cstddef>
namespace lcs_state {
void collect_haptics(Fields& f){
    f.pod("haptics.d_triVms",d_triVms);
    f.pod("haptics.d_vmsKey",d_vmsKey);
    f.pod("haptics.d_vmsTri",d_vmsTri);
    f.pod("haptics.d_contactDisp",d_contactDisp);
    f.pod("haptics.d_penCount",d_penCount);
    f.pod("haptics.d_depthSum",d_depthSum);
    f.pod("haptics.g_triCap",g_triCap);
    f.pod("haptics.d_toolTri",d_toolTri);
    f.pod("haptics.g_toolTriCap",g_toolTriCap);
    f.pod("haptics.g_toolTriCount",g_toolTriCount);
    f.pod("haptics.g_toolCenter",g_toolCenter);
    f.pod("haptics.g_toolBoundR",g_toolBoundR);
    f.pod("haptics.g_hp",g_hp);
    f.pod("haptics.g_cornerCount",g_cornerCount);
    f.pod("haptics.g_physS",g_physS);
    f.pod("haptics.g_physE",g_physE);
    f.pod("haptics.g_physPos",g_physPos);
    f.pod("haptics.g_physQuat",g_physQuat);
    f.pod("haptics.g_linVel",g_linVel);
    f.pod("haptics.g_angVel",g_angVel);
    f.pod("haptics.g_proxyS",g_proxyS);
    f.pod("haptics.g_proxyE",g_proxyE);
    f.pod("haptics.g_proxyPos_dev",g_proxyPos_dev);
    f.pod("haptics.g_proxyQuat",g_proxyQuat);
    f.pod("haptics.g_hasProxy",g_hasProxy);
    f.pod("haptics.g_prevPhysPos",g_prevPhysPos);
    f.pod("haptics.g_havePrev",g_havePrev);
    f.pod("haptics.g_stickAnchor",g_stickAnchor);
    f.pod("haptics.g_slipping",g_slipping);
    f.pod("haptics.g_lastNormal",g_lastNormal);
    f.pod("haptics.g_graspMu",g_graspMu);
    f.pod("haptics.g_toolRadius",g_toolRadius);
    f.pod("haptics.g_force",g_force);
    f.pod("haptics.g_torque",g_torque);
    f.pod("haptics.g_inContact",g_inContact);
    f.pod("haptics.g_minTri",g_minTri);
    f.pod("haptics.c_cornerPos",c_cornerPos);
    f.pod("haptics.c_hapticForce",c_hapticForce);
    f.pod("haptics.c_active",c_active);
    f.pod("haptics.c_pinned",c_pinned);
    f.pod("haptics.c_vel",c_vel);
    f.pod("haptics.g_ready",g_ready);
}
}
