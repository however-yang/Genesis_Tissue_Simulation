// Persistent material-point attachments. Capture a clipped surface witness, not
// every lattice corner supporting a tiny triangle. Stencils are never recaptured
// during a hold; a changed cut owner invalidates the affected attachment.
struct GripPoint {
    int id[24], count, voxel[3], valid;
    unsigned mask[3];
    float weight[24], area, complianceScale;
    float3 offset, local, lambda;
};

// Anchor targets and contact share the measured inner faces of the visible pads.
// The legacy gap applies only to procedural fallback jaws.
__host__ __device__ inline float gripInnerOffset(float minGap){
#ifdef __CUDA_ARCH__
    JawMeshBounds up=c_jawMeshBounds[0],down=c_jawMeshBounds[1];
#else
    JawMeshBounds up=h_jawMeshBounds[0],down=h_jawMeshBounds[1];
#endif
    return up.enabled&&down.enabled?fmaxf(0.f,fminf(up.center.y-up.half.y,-down.center.y-down.half.y)):.5f*minGap;
}
__host__ __device__ inline float gripAperture(XpbdToolState t,float minGap,float u){
    float sn=fminf(.95f,t.jawGap/(2.f*fmaxf(t.jawLen,1e-6f)));
    float cs=sqrtf(1.f-sn*sn);
    return fmaxf(1e-6f,u*sn/cs+gripInnerOffset(minGap)/cs);
}
__device__ inline void gripFrame(XpbdToolState t,float3& axis,float3& normal,float3& width){
    axis=contactUnit(t.jawAxis,mkf3(0,0,1));
    normal=t.jawNormal-axis*dot3(t.jawNormal,axis);
    normal=contactUnit(normal,mkf3(0,1,0));
    width=contactUnit(cross3(axis,normal),mkf3(1,0,0));
}
__device__ inline float3 gripTarget(const GripPoint& p,XpbdToolState t,float minGap){
    float3 a,n,w;gripFrame(t,a,n,w);
    // Closing work acts on the coupled aperture DOF as well as the tissue.
    return t.jawHinge+a*p.local.x+w*p.local.y+n*(p.local.z*gripAperture(t,minGap,p.local.x));
}

__global__ void k_CaptureGrip(int triangles,const SurfaceVertex* vertices,const unsigned* masks,
    const float3* pos,const int* active,const int* pinned,XpbdToolState tool,float minGap,
    GripPoint* points,int* count,float* moments){
    int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=triangles)return;
    SurfaceVertex v[3]={vertices[3*t],vertices[3*t+1],vertices[3*t+2]};
    float3 axis,normal,width;gripFrame(tool,axis,normal,width);
    float3 p[12],bary[12],next[12],nextBary[12];
    for(int j=0;j<3;j++){
        if(!v[j].count || (masks && (masks[v[j].voxel]&4095u)!=v[j].mask))return;
        float3 r=surfacePoint(v[j],pos)-tool.jawHinge;
        p[j]=mkf3(dot3(r,axis),dot3(r,width),dot3(r,normal));
        bary[j]=mkf3(j==0,j==1,j==2);
    }
    float sn=fminf(.95f,tool.jawGap/(2.f*fmaxf(tool.jawLen,1e-6f))),cs=sqrtf(1.f-sn*sn);
    float3 planes[6]={mkf3(-1,0,0),mkf3(1,0,0),mkf3(0,-1,0),mkf3(0,1,0),mkf3(-sn/cs,0,1),mkf3(-sn/cs,0,-1)};
    float bound[6]={-.1f*tool.jawLen,tool.jawLen*cs,tool.jawHalfWidth,tool.jawHalfWidth,gripInnerOffset(minGap)/cs,gripInnerOffset(minGap)/cs};
    int n=3;
    for(int plane=0;plane<6 && n;plane++){
        int m=0;float3 a=p[n-1],ba=bary[n-1];float da=dot3(planes[plane],a)-bound[plane];
        for(int j=0;j<n;j++){
            float3 b=p[j],bb=bary[j];float db=dot3(planes[plane],b)-bound[plane];
            if((da<=0)!=(db<=0)){
                float q=da/(da-db);if(m>=12)return;
                next[m]=a+(b-a)*q;nextBary[m++]=ba+(bb-ba)*q;
            }
            if(db<=0){if(m>=12)return;next[m]=b;nextBary[m++]=bb;}
            a=b;ba=bb;da=db;
        }
        n=m;for(int j=0;j<n;j++){p[j]=next[j];bary[j]=nextBary[j];}
    }
    if(n<3)return;
    float area=0;float3 local=mkf3(0,0,0),bc=local;
    for(int j=1;j<n-1;j++){
        float a=.5f*len3(cross3(p[j]-p[0],p[j+1]-p[0]));
        area+=a;local=local+(p[0]+p[j]+p[j+1])*(a/3.f);
        bc=bc+(bary[0]+bary[j]+bary[j+1])*(a/3.f);
    }
    if(area<=1e-12f)return;
    local=local*(1.f/area);bc=bc*(1.f/area);
    GripPoint out={};out.area=area;out.valid=1;out.local=local;
    out.local.z=local.z/gripAperture(tool,minGap,local.x);
    float mobile=0;
    for(int j=0;j<3;j++){
        out.voxel[j]=v[j].voxel;out.mask[j]=v[j].mask;
        float b=compF3(bc,j);out.offset=out.offset+v[j].offset*b;
        for(int k=0;k<v[j].count;k++){
            float weight=b*v[j].weight[k];if(weight<=1e-8f)continue;
            int id=v[j].id[k];if(!active[id])return;
            int slot=0;while(slot<out.count && out.id[slot]!=id)slot++;
            if(slot==out.count){out.id[slot]=id;out.count++;}
            out.weight[slot]+=weight;if(!pinned[id])mobile+=weight;
        }
    }
    if(out.count==0 || mobile<.01f)return;
    int index=atomicAdd(count,1);points[index]=out; // at most one point per triangle
    atomicAdd(moments,area);atomicAdd(moments+1,area*local.x);atomicAdd(moments+2,area*local.y);
}

__global__ void k_GripWeights(int count,GripPoint* p,float sigma,float* moments){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    float area=fmaxf(moments[0],1e-20f);
    float u=p[i].local.x-moments[1]/area,w=p[i].local.y-moments[2]/area;
    // A circular Gaussian in the jaw tangent plane, integrated over bite area.
    float weight=p[i].area*expf(-.5f*(u*u+w*w)/(sigma*sigma));
    p[i].complianceScale=weight;atomicAdd(moments+3,weight);
}
__global__ void k_NormalizeGrip(int count,GripPoint* p,float sigma,const float* moments){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    float w=p[i].complianceScale;
    if(w<=1e-12f){p[i].valid=0;return;}
    // This is an explicit anchor joint, with one finite configured patch
    // stiffness. Bite area distributes that stiffness; it must not silently
    // turn a small but valid captured patch into a weak adhesive spring.
    // Zero-area witnesses are rejected above and cut ownership still releases
    // invalid points. Frictional pushing remains in the contact law.
    p[i].complianceScale=moments[3]/w;
}
__global__ void k_RefreshGrip(int count,GripPoint* p,const unsigned* masks,const int* active,
    int* degree,int* grasped,int* core,int* live){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !p[i].valid)return;
    GripPoint& a=p[i];
    for(int k=0;k<3;k++)if(masks && (masks[a.voxel[k]]&4095u)!=a.mask[k]){a.valid=0;return;}
    for(int k=0;k<a.count;k++)if(!active[a.id[k]]){a.valid=0;return;}
    atomicAdd(live,1);
    for(int k=0;k<a.count;k++){
        int id=a.id[k];atomicAdd(degree+id,1);atomicExch(grasped+id,1);atomicExch(core+id,1);
    }
}
struct GripSolveStep {
    XpbdToolState tool,from;
    float minGap,alpha,damping,maxStep;
};
// One Gaussian material-region attachment. It transports the captured region
// without prescribing a flattened target for every surface triangle inside it.
// Contact still determines the squeeze and transmits Coulomb friction.
struct GripPatch {
    float3 local,offset,lambda,point,old,dl;float strength,mobility;
    float3 refU,refV,offsetU,offsetV,pointU,pointV,oldU,oldV,lambdaU,lambdaV,dlU,dlV;
    float massUU,massUV,massVV,mass0U,mass0V;
};
__global__ void k_ClearPatch(GripPatch* p){p->local=p->offset=mkf3(0,0,0);p->strength=0;p->refU=p->refV=p->offsetU=p->offsetV=mkf3(0,0,0);}
__global__ void k_BuildPatch(int count,const GripPoint* points,float* weights,GripPatch* patch){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !points[i].valid)return;
    const GripPoint& p=points[i];float w=1.f/p.complianceScale;
    add3(&patch->local,p.local*w);add3(&patch->offset,p.offset*w);atomicAdd(&patch->strength,w);
    for(int j=0;j<p.count;j++)atomicAdd(weights+p.id[j],p.weight[j]*w);
}
__global__ void k_NormalizePatch(int n,float* weights,GripPatch* p){
    int i=blockIdx.x*blockDim.x+threadIdx.x;float inv=1.f/fmaxf(p->strength,1e-20f);
    if(i<n)weights[i]*=inv;if(i==0){p->local=p->local*inv;p->offset=p->offset*inv;}
}
// Four affine patch constraints complement the three centroid constraints.
// Their gradients have zero net translation. Unlike one constraint per triangle,
// the solve size is independent of triangulation and shares one exact 3x3 mass
// matrix. Normal squeeze is left to contact and material mechanics.
__global__ void k_BuildPatchShape(int count,int n,const GripPoint* points,float sigma,float* weights,GripPatch* patch){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !points[i].valid)return;
    const GripPoint& p=points[i];float weight=1.f/(p.complianceScale*fmaxf(patch->strength,1e-20f));
    float u=weight*(p.local.x-patch->local.x)/sigma,v=weight*(p.local.y-patch->local.y)/sigma;
    add3(&patch->refU,p.local*u);add3(&patch->refV,p.local*v);
    add3(&patch->offsetU,p.offset*u);add3(&patch->offsetV,p.offset*v);
    for(int j=0;j<p.count;j++){atomicAdd(weights+n+p.id[j],p.weight[j]*u);atomicAdd(weights+2*n+p.id[j],p.weight[j]*v);}
}
__device__ inline float3 patchTarget(const GripPatch& p,XpbdToolState t,float minGap){
    float3 a,n,w;gripFrame(t,a,n,w);return t.jawHinge+a*p.local.x+w*p.local.y+n*(p.local.z*gripAperture(t,minGap,p.local.x));
}
__global__ void k_BeginPatchEval(GripPatch* p){p->point=p->old=p->offset;p->mobility=0;p->pointU=p->oldU=p->offsetU;p->pointV=p->oldV=p->offsetV;p->massUU=p->massUV=p->massVV=p->mass0U=p->mass0V=0;}
__global__ void k_EvalPatch(int n,const float* weights,const float3* pos,const float3* previous,
    const float* invMass,const CoupledTool* body,GripPatch* p){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n || weights[i]<=0)return;
    float w=weights[i],mass=contactMobility(body,i,invMass[i],mkf3(1,0,0)).x;
    add3(&p->point,pos[i]*w);add3(&p->old,previous[i]*w);
    atomicAdd(&p->mobility,w*w*mass);
    float u=weights[n+i],v=weights[2*n+i];
    add3(&p->pointU,pos[i]*u);add3(&p->pointV,pos[i]*v);add3(&p->oldU,previous[i]*u);add3(&p->oldV,previous[i]*v);
    atomicAdd(&p->mass0U,mass*w*u);atomicAdd(&p->mass0V,mass*w*v);
    atomicAdd(&p->massUU,mass*u*u);atomicAdd(&p->massUV,mass*u*v);atomicAdd(&p->massVV,mass*v*v);
}
__global__ void k_SolvePatch(GripPatch* p,const GripSolveStep* frame,CoupledTool* body,float sigma){
    p->dl=p->dlU=p->dlV=mkf3(0,0,0);if(p->strength<=0 || !body)return;
    float3 a,n,w,oa,on,ow;gripFrame(body->pose,a,n,w);gripFrame(body->previous,oa,on,ow);
    float3 target=patchTarget(*p,body->pose,frame->minGap),old=patchTarget(*p,body->previous,frame->minGap);
    XpbdToolState plus=body->pose,minus=plus;float eps=fmaxf(1e-5f,1e-3f*plus.jawLen);
    plus.jawGap+=eps;minus.jawGap=fmaxf(0.f,minus.jawGap-eps);
    float3 g=(patchTarget(*p,plus,frame->minGap)-patchTarget(*p,minus,frame->minGap))*(1.f/(plus.jawGap-minus.jawGap));
    float alpha=frame->alpha/fmaxf(p->strength,1e-6f),damp=frame->damping,scale=1+damp;
    float3 r0=(p->point-target)+p->lambda*alpha+((p->point-p->old)-(target-old))*damp;
    float3 tu=a*p->refU.x+w*p->refU.y,tv=a*p->refV.x+w*p->refV.y;
    float3 otu=oa*p->refU.x+ow*p->refU.y,otv=oa*p->refV.x+ow*p->refV.y;
    // Galerkin compliance uses the captured Gaussian covariance. Identity
    // compliance silently softens narrow bites by sigma^2 / biteVariance.
    float au=alpha*fmaxf(p->refU.x/sigma,1e-8f);
    float av=alpha*fmaxf(p->refV.y/sigma,1e-8f);
    float auv=alpha*.5f*(p->refU.y+p->refV.x)/sigma;
    float3 ru=(p->pointU-tu)+(p->pointU-p->oldU-tu+otu)*damp+p->lambdaU*au+p->lambdaV*auv;
    float3 rv=(p->pointV-tv)+(p->pointV-p->oldV-tv+otv)*damp+p->lambdaV*av+p->lambdaU*auv;
    float m00=scale*(p->mobility+body->inverseMass)+alpha,m01=scale*p->mass0U,m02=scale*p->mass0V;
    float m11=scale*p->massUU+au,m12=scale*p->massUV+auv,m22=scale*p->massVV+av;
    float c00=m11*m22-m12*m12,c01=m02*m12-m01*m22,c02=m01*m12-m02*m11;
    float c11=m00*m22-m02*m02,c12=m01*m02-m00*m12,c22=m00*m11-m01*m01;
    float det=m00*c00+m01*c01+m02*c02;if(!(det>1e-20f))return;
    float normalRhs=dot3(r0,n),normalGap=dot3(g,n);
    r0=r0-n*normalRhs;ru=ru-n*dot3(ru,n);rv=rv-n*dot3(rv,n);
    p->dl=(r0*c00+ru*c01+rv*c02)*(-1.f/det);
    p->dlU=(r0*c01+ru*c11+rv*c12)*(-1.f/det);
    p->dlV=(r0*c02+ru*c12+rv*c22)*(-1.f/det);
    p->dl=p->dl+n*(-normalRhs/(m00+scale*body->gapInverseMass*normalGap*normalGap));
    p->lambda=p->lambda+p->dl;p->lambdaU=p->lambdaU+p->dlU;p->lambdaV=p->lambdaV+p->dlV;
    body->pose=shiftTool(body->pose,p->dl*(-body->inverseMass));
    body->pose.jawGap=fminf(body->maxGap,fmaxf(0.f,body->pose.jawGap-body->gapInverseMass*dot3(g,p->dl)));
}
__global__ void k_ApplyPatch(int n,const float* weights,const float* invMass,float3* pos,
    const CoupledTool* body,const GripPatch* p){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n && weights[i]>0)
        pos[i]=pos[i]+contactMobility(body,i,invMass[i],p->dl*weights[i]+p->dlU*weights[n+i]+p->dlV*weights[2*n+i]);
}
__global__ void k_PatchError(const GripPatch* p,const CoupledTool* body,const int* live,float minGap,float* result){
    result[0]=p->strength>0?len3(p->point-patchTarget(*p,body->pose,minGap)):0;result[1]=(float)*live;
}
__global__ void k_PatchReaction(const GripPatch* p,const CoupledTool* body,float scale,float4* result,float4* loads){
    if(p->strength>0){float3 f=p->lambda*(-scale);atomicAdd(&result->x,f.x);atomicAdd(&result->y,f.y);atomicAdd(&result->z,f.z);
        float3 a,n,w;gripFrame(body->pose,a,n,w);float load=fmaxf(0.f,dot3(f,n)*p->local.z);
        if(p->local.z>=0)atomicAdd(&loads->x,load);else atomicAdd(&loads->y,load);
    }
}
__global__ void k_ResetGrip(int count,GripPoint* points,GripSolveStep* frame,GripSolveStep next){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i==0)*frame=next;
    if(i<count)points[i].lambda=mkf3(0,0,0);
}
// Preserve the finite captured patch in the two jaw tangential directions.
// The opening-normal coordinate is deliberately free: only unilateral contact
// squeezes tissue. A centroid constraint alone has a large null space and can
// satisfy its error bound while the visible patch collapses to a spike.
__global__ void k_GripTangentError(int count,const GripPoint* points,const float3* pos,
    const CoupledTool* body,float minGap,float* result){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !points[i].valid)return;
    const GripPoint& p=points[i];float3 x=p.offset,a,n,w;gripFrame(body->pose,a,n,w);
    for(int j=0;j<p.count;j++)x=x+pos[p.id[j]]*p.weight[j];
    float3 r=x-gripTarget(p,body->pose,minGap);r=r-n*dot3(r,n);
    atomicMax((unsigned*)result,__float_as_uint(len3(r)));
}
__global__ void k_SolveGrip(int count,GripPoint* points,const float3* pos,const float3* previous,
    const float* invMass,const int* degree,const GripSolveStep* frame,float3* delta,const int* live,CoupledTool* body=nullptr){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !points[i].valid)return;
    GripPoint& a=points[i];float3 x=a.offset,old=a.offset;
    float3 ax=mkf3(0,0,0),ay=ax,az=ax;
    for(int k=0;k<a.count;k++){
        int id=a.id[k];float w=a.weight[k];x=x+pos[id]*w;old=old+previous[id]*w;
        float scale=w*w*fmaxf(1.f,(float)degree[id]);
        ax=ax+contactMobility(body,id,invMass[id],mkf3(1,0,0))*scale;
        ay=ay+contactMobility(body,id,invMass[id],mkf3(0,1,0))*scale;
        az=az+contactMobility(body,id,invMass[id],mkf3(0,0,1))*scale;
    }
    float3 gapGradient=mkf3(0,0,0);
    if(body){
        XpbdToolState plus=body->pose,minus=plus;float eps=fmaxf(1e-5f,1e-3f*plus.jawLen);
        plus.jawGap+=eps;minus.jawGap=fmaxf(0.f,minus.jawGap-eps);
        gapGradient=(gripTarget(a,plus,frame->minGap)-gripTarget(a,minus,frame->minGap))*(1.f/(plus.jawGap-minus.jawGap));
        float count=fmaxf(1.f,(float)*live),w=body->inverseMass*count,wg=body->gapInverseMass*count;
        ax=ax+mkf3(w,0,0)+gapGradient*(gapGradient.x*wg);
        ay=ay+mkf3(0,w,0)+gapGradient*(gapGradient.y*wg);
        az=az+mkf3(0,0,w)+gapGradient*(gapGradient.z*wg);
    }
    float3 target=gripTarget(a,body?body->pose:frame->tool,frame->minGap),oldTarget=gripTarget(a,body?body->previous:frame->from,frame->minGap);
    float compliance=frame->alpha*a.complianceScale,damping=frame->damping;
    float regularizer=compliance/(1.f+damping);
    ax.x+=regularizer;ay.y+=regularizer;az.z+=regularizer;
    float det=dot3(ax,cross3(ay,az));if(!(det>1e-20f))return;
    float3 rhs=((x-target)+a.lambda*compliance+((x-old)-(target-oldTarget))*damping)*(-1.f/(1.f+damping));
    float3 dl=mkf3(dot3(rhs,cross3(ay,az)),dot3(rhs,cross3(az,ax)),dot3(rhs,cross3(ax,ay)))*(1.f/det);
    float largest=body?len3(dl)*body->inverseMass:0;
    for(int k=0;k<a.count;k++)largest=fmaxf(largest,len3(dl)*invMass[a.id[k]]*fabsf(a.weight[k]));
    if(largest>frame->maxStep)dl=dl*(frame->maxStep/largest);
    a.lambda=a.lambda+dl;
    if(body){add3(&body->delta,dl*(-body->inverseMass));atomicAdd(&body->gapDelta,-body->gapInverseMass*dot3(gapGradient,dl));}
    for(int k=0;k<a.count;k++)add3(delta+a.id[k],contactMobility(body,a.id[k],invMass[a.id[k]],dl)*a.weight[k]);
}
// End-of-tick diagnostics use the SAME material point and tool pose as the
// constraint. A held flag alone does not imply a converged attachment.
__global__ void k_GripError(int count,const GripPoint* points,const float3* pos,
    const CoupledTool* body,float minGap,float* result){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !points[i].valid)return;
    const GripPoint& p=points[i];float3 x=p.offset;
    for(int j=0;j<p.count;j++)x=x+pos[p.id[j]]*p.weight[j];
    float error=len3(x-gripTarget(p,body->pose,minGap));
    atomicMax((unsigned*)result,__float_as_uint(error));atomicAdd(result+1,1.f);
}
__global__ void k_GripReaction(int count,const GripPoint* points,XpbdToolState tool,float scale,float4* reaction,float4* loads){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !points[i].valid)return;
    const GripPoint& a=points[i];float3 f=a.lambda*(-scale),axis,normal,width;gripFrame(tool,axis,normal,width);
    atomicAdd(&reaction->x,f.x);atomicAdd(&reaction->y,f.y);atomicAdd(&reaction->z,f.z);
    // The normal anchor follows the actual face separation, so its work must
    // also oppose the closing actuator. No unaccounted compression by anchors.
    float load=fmaxf(0.f,dot3(f,normal)*a.local.z);
    if(a.local.z>=0)atomicAdd(&loads->x,load);else atomicAdd(&loads->y,load);
}
