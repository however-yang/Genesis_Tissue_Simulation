// Read-only diagnostics. A query must not rebuild the solver's patch, invalidate
// its points, or change the floating-point reduction order of a later solve.
// Validity is checked against current cut ownership even before the next substep.
__device__ inline bool observedGripValid(const GripPoint& p,const unsigned* masks,const int* active){
    if(!p.valid)return false;
    for(int k=0;k<3;k++)if(masks && (masks[p.voxel[k]]&4095u)!=p.mask[k])return false;
    for(int k=0;k<p.count;k++)if(active && !active[p.id[k]])return false;
    return true;
}

// One ordered pass over the captured patch (not the whole grid). Double sums
// avoid making repeated feedback depend on the scheduling of float atomics.
// The centroid and tangent-error definitions are the same as PatchError and
// GripTangentError; the diagnostic never writes any physical state.
__global__ void k_ReadGripObservation(int count,const GripPoint* points,
    const unsigned* masks,const int* active,const float3* pos,
    const CoupledTool* body,XpbdToolState fallback,float minGap,float* result){
    XpbdToolState pose=body?body->pose:fallback;
    double weight=0,lx=0,ly=0,lz=0,px=0,py=0,pz=0;
    int live=0;float tangent=0;
    for(int i=0;i<count;i++){
        const GripPoint& p=points[i];if(!observedGripValid(p,masks,active))continue;
        ++live;double w=1.0/p.complianceScale;
        float3 x=p.offset;
        for(int j=0;j<p.count;j++)x=x+pos[p.id[j]]*p.weight[j];
        weight+=w;lx+=w*p.local.x;ly+=w*p.local.y;lz+=w*p.local.z;
        px+=w*x.x;py+=w*x.y;pz+=w*x.z;
        float3 a,n,b;gripFrame(pose,a,n,b);
        float3 r=x-gripTarget(p,pose,minGap);r=r-n*dot3(r,n);
        tangent=fmaxf(tangent,len3(r));
    }
    float error=0;
    if(weight>0){
        GripPatch patch={};patch.local=mkf3(float(lx/weight),float(ly/weight),float(lz/weight));
        float3 centroid=mkf3(float(px/weight),float(py/weight),float(pz/weight));
        error=len3(centroid-patchTarget(patch,pose,minGap));
    }
    result[0]=error;result[1]=(float)live;result[2]=tangent;
}
