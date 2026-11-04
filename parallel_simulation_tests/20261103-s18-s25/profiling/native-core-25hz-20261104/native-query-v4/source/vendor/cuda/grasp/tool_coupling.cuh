// Translational rigid instrument DOF, solved with the tissue constraints.
// The command is a spring endpoint, never a second collision/render pose.
struct CoupledTool {
    XpbdToolState pose, previous;
    float3 velocity, delta;
    float inverseMass, h;
    int initialized;
    float gapVelocity,gapDelta,gapInverseMass,maxGap;
    const float3* feasibleNormal;
    float3 drivePosition,drivePrediction;float driveGap;
    int guided;float3 guideAxis,guideOrigin;float guideLength,gapLo,gapHi;
    int scissors, scissorsCutting, knifeBody;
    float3 edgeA[2],edgeB[2];
    float scissorsRadius;
    int backContact;
    XpbdScissorsBackBox backs[2];
    // Scissors wrist capsule along the hinge axis, from wristFront ahead of the
    // hinge to wristLength behind it; wristRadius 0 leaves it out of contact.
    float wristFront, wristLength, wristRadius;
};
// The grasper retains its own OBB geometry; scissors have two finite capsules.
__host__ __device__ inline ContactShape bodyContactShape(const CoupledTool* body,XpbdToolState t,int kind,float thickness,float minGap){
    if(!body || (!body->scissors && !body->knifeBody) || kind==0)return contactShape(t,kind,thickness,minGap);
    ContactShape s={};s.capsule=1;s.radius=body->scissorsRadius;
    float3 u=contactUnit(t.jawAxis,mkf3(0,0,1)),n=contactUnit(t.jawNormal,mkf3(0,1,0));
    float sn=body->scissors?fminf(.95f,fmaxf(0.f,t.jawGap)/(2.f*fmaxf(t.jawLen,1e-6f))):0;
    float cs=sqrtf(1-sn*sn),side=(kind==1||kind==3)?1.f:-1.f;
    float3 along=u*cs+n*(side*sn),opening=n*cs-u*(side*sn),width=cross3(u,n);
    if(kind==5){
        s={};if(!body->scissors || !(body->wristRadius>0))return s;
        s.capsule=1;s.radius=body->wristRadius;
        s.a=t.jawHinge+u*body->wristFront;s.b=t.jawHinge-u*body->wristLength;return s;
    }
    if(kind>=3){
        s={};if(!body->backContact)return s;
        const auto& box=body->backs[kind-3];s.half=box.halfSize;
        s.center=t.jawHinge+along*box.center.x+opening*box.center.y+width*box.center.z;
        s.u=along*box.axisX.x+opening*box.axisX.y+width*box.axisX.z;
        s.n=along*box.axisY.x+opening*box.axisY.y+width*box.axisY.z;s.w=cross3(s.u,s.n);return s;
    }
    float3 a=body->edgeA[kind-1],b=body->edgeB[kind-1];
    s.a=t.jawHinge+along*a.x+opening*a.y+width*a.z;
    s.b=t.jawHinge+along*b.x+opening*b.y+width*b.z;
    return s;
}
__host__ __device__ inline XpbdToolState shiftTool(XpbdToolState t,float3 d){
    t.jawHinge=t.jawHinge+d;t.jawMid=t.jawMid+d;
    t.shaftA=t.shaftA+d;t.shaftB=t.shaftB+d;return t;
}
__global__ void k_PredictTool(CoupledTool* b,XpbdToolState command,XpbdToolState initial,
    float h,float mass,float stiffness,float damping,float maxTravel,
    float jawMass,float jawStiffness,float jawDamping,float jawSpeed,float3 driveTarget,float gapTarget,int guided=0){
    if(!b->initialized){b->pose=initial;b->drivePosition=initial.jawHinge;b->driveGap=initial.jawGap;b->velocity=mkf3(0,0,0);b->gapVelocity=0;b->maxGap=initial.jawGap;b->initialized=1;}
    b->previous=b->pose;b->h=h;b->feasibleNormal=nullptr;b->guided=guided;
    // Backward Euler for a damped spring. Its effective mass enters BOTH sides
    // of contact/attachment constraints in this same substep (no lagged force).
    float effectiveMass=mass+h*damping+h*h*stiffness;
    b->inverseMass=1.f/effectiveMass;
    // Slew-limit the COMMAND, never the force-generated displacement. Capping
    // the latter silently removes spring impulse under load and can jam a held
    // instrument regardless of its requested drive stiffness.
    // Do not feed the pose interpolated from last frame's resisted tool into
    // the drive endpoint: that erases accumulated preload every render frame.
    float3 driveMove=driveTarget-b->drivePosition;float travel=len3(driveMove);
    if(travel>maxTravel)driveMove=driveMove*(maxTravel/travel);
    b->drivePosition=guided?driveTarget:b->drivePosition+driveMove;
    float3 request=driveTarget-b->previous.jawHinge;float distance=len3(request);
    b->guideOrigin=b->previous.jawHinge;b->guideAxis=distance>1e-6f?request*(1.f/distance):mkf3(0,0,0);
    b->guideLength=distance>1e-6f?fminf(distance,maxTravel):0;
    float3 v=(b->velocity*mass+(b->drivePosition-b->pose.jawHinge)*(h*stiffness))*b->inverseMass;
    float3 move=v*h;
    // Keep the unconstrained backward-Euler minimizer for the spring energy.
    // The geometric guide clips the initial ITERATE, not the applied work.
    b->drivePrediction=b->previous.jawHinge+move;
    if(guided)move=b->guideAxis*fminf(b->guideLength,fmaxf(0.f,dot3(move,b->guideAxis)));
    b->pose=shiftTool(command,b->previous.jawHinge+move-command.jawHinge);
    b->delta=mkf3(0,0,0);
    b->gapDelta=0;b->gapInverseMass=0;
    if(jawStiffness>0){
        b->maxGap=fmaxf(b->maxGap,gapTarget);
        b->gapInverseMass=1.f/(jawMass+h*jawDamping+h*h*jawStiffness);
        float openingSpeed=4.f*jawSpeed;
        b->driveGap+=fminf(openingSpeed*h,fmaxf(-jawSpeed*h,gapTarget-b->driveGap));
        float v=(jawMass*b->gapVelocity+h*jawStiffness*(b->driveGap-b->previous.jawGap))*b->gapInverseMass;
        b->pose.jawGap=fminf(b->maxGap,fmaxf(0.f,b->previous.jawGap+v*h));
    }
    b->gapLo=fminf(b->previous.jawGap,b->driveGap);
    // A closing actuator is force driven: tissue pressure may reopen it while
    // solving the coupled objective. Clamping to last step's gap is a ratchet
    // that accumulates overlap. During opening/no G, forbid unsolicited closing.
    b->gapHi=b->driveGap<b->previous.jawGap?b->maxGap:fmaxf(b->previous.jawGap,b->driveGap);
    if(guided)b->pose.jawGap=fminf(b->gapHi,fmaxf(b->gapLo,b->pose.jawGap));
}
__global__ void k_ApplyTool(CoupledTool* b){
    b->pose=shiftTool(b->pose,b->delta);b->delta=mkf3(0,0,0);
    if(b->gapInverseMass>0)b->pose.jawGap=fminf(b->maxGap,fmaxf(0.f,b->pose.jawGap+b->gapDelta));b->gapDelta=0;
}
__global__ void k_EndTool(CoupledTool* b){
    b->velocity=(b->pose.jawHinge-b->previous.jawHinge)*(1.f/b->h);
    b->gapVelocity=(b->pose.jawGap-b->previous.jawGap)/b->h;
    b->feasibleNormal=nullptr;
}
__global__ void k_ClearContactMobility(CoupledTool* b){b->feasibleNormal=nullptr;}
__global__ void k_ClearToolVelocityDelta(CoupledTool* b){b->delta=mkf3(0,0,0);b->gapDelta=0;}
__global__ void k_ApplyToolVelocity(CoupledTool* b){
    b->velocity=b->velocity+b->delta;b->gapVelocity+=b->gapDelta;
    b->delta=mkf3(0,0,0);b->gapDelta=0;
}
__global__ void k_ContactMobility(CoupledTool* b,int n,const float* invMass,const int* needed,
    const float3* start,const float3* target,const float3* accepted,float3* blocked){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i==0)b->feasibleNormal=blocked;if(i>=n)return;
    float3 normal=mkf3(0,0,0);
    {float3 rejected=target[i]-accepted[i];float length=len3(rejected);
        if(length>1e-6f)normal=rejected*(1.f/length);}
    // Only nodes corrected by the determinant safeguard are inactive during
    // this reconciliation. All nodes regain their material mobility next substep.
    blocked[i]=normal;
}
__device__ inline float3 contactMobility(const CoupledTool* b,int id,float inverseMass,float3 impulse){
    if(b && b->feasibleNormal && dot3(b->feasibleNormal[id],b->feasibleNormal[id])>.5f)return mkf3(0,0,0);
    return impulse*inverseMass;
}
// Jacobian of a jaw witness with respect to tip separation, on the same finite
// jaw geometry used by CCD (including hinge rotation and thickness).
__device__ inline float3 jawGapGradient(XpbdToolState t,int kind,float thickness,float minGap,float3 point,const CoupledTool* body=nullptr){
    if(kind==0 || (body && body->knifeBody))return mkf3(0,0,0);
    float eps=fmaxf(1e-5f,1e-3f*t.jawLen);XpbdToolState plus=t,minus=t;
    plus.jawGap+=eps;minus.jawGap=fmaxf(0.f,t.jawGap-eps);
    ContactShape from=bodyContactShape(body,t,kind,thickness,minGap);
    return (witnessAt(from,bodyContactShape(body,plus,kind,thickness,minGap),point)-
            witnessAt(from,bodyContactShape(body,minus,kind,thickness,minGap),point))*(1.f/(plus.jawGap-minus.jawGap));
}
