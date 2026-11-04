// Generated from the reviewed inventory. Do not hand edit.
#include "state_archive.h"
#include <cstddef>
namespace lcs_state {
void collect_physics(Fields& f){
    f.pod("physics.d_vel",d_vel);
    f.pod("physics.d_mass",d_mass);
    f.pod("physics.d_pinned",d_pinned);
    f.pod("physics.d_active",d_active);
    f.pod("physics.d_extForce",d_extForce);
    f.pod("physics.d_hapticForce",d_hapticForce);
    f.pod("physics.d_nbrIdx",d_nbrIdx);
    f.pod("physics.d_restNbr",d_restNbr);
    f.pod("physics.d_bendPairs",d_bendPairs);
    f.pod("physics.d_particleRot",d_particleRot);
    f.pod("physics.d_particleRotPrev",d_particleRotPrev);
    f.pod("physics.g_pdesc",g_pdesc);
    f.pod("physics.g_phys_ready",g_phys_ready);
    f.pod("physics.g_graspNExempt",g_graspNExempt);
}
}
