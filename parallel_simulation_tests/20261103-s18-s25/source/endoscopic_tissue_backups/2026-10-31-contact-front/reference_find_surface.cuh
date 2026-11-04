// Frozen original inline helpers, independent of the candidate.
__device__ inline float3 triangleWitnessReference(ContactShape shape,float3 a,float3 b,float3 c){
    float3 q[3]={a,b,c},bar=mkf3(1.f/3,1.f/3,1.f/3),best=bar,n,s;
    float bestD=shapeDistance(shape,(a+b+c)*(1.f/3),n,s);
    for(int k=0;k<3;k++){float d=shapeDistance(shape,q[k],n,s);if(d<bestD){bestD=d;best=mkf3(k==0,k==1,k==2);}}
    for(int it=0;it<12;it++){
        float3 p=a*bar.x+b*bar.y+c*bar.z;shapeDistance(shape,p,n,s);
        int k=dot3(n,b)<dot3(n,a)?1:0;if(dot3(n,c)<dot3(n,q[k]))k=2;
        float3 target=mkf3(k==0,k==1,k==2),dir=q[k]-p;
        // Reuse the surviving golden-section sample. Same eight interval
        // reductions; nine SDF evaluations instead of sixteen per line search.
        float lo=0,hi=1,l=.381966f,r=.618034f;
        float dl=shapeDistance(shape,p+dir*l,n,s),dr=shapeDistance(shape,p+dir*r,n,s);
        for(int j=0;j<8;j++){
            if(dl<dr){
                hi=r;r=l;dr=dl;l=lo+(hi-lo)*.381966f;
                if(j<7)dl=shapeDistance(shape,p+dir*l,n,s);
            }else{
                lo=l;l=r;dl=dr;r=lo+(hi-lo)*.618034f;
                if(j<7)dr=shapeDistance(shape,p+dir*r,n,s);
            }
        }
        float t=.5f*(lo+hi);bar=bar*(1-t)+target*t;
        float d=shapeDistance(shape,a*bar.x+b*bar.y+c*bar.z,n,s);
        if(d<bestD){bestD=d;best=bar;}
    }
    return best;
}
__device__ inline ContactRecord traceTriangleBoxReference(ContactShape from,ContactShape to,float skin,const float3* p,const float3* q,float3& bary){
    // ACCD's relative-motion bound: common translation cannot close the gap.
    // Retain the existing conservative rotation bound for the finite jaw.
    ContactRecord out={};float3 translation=to.center-from.center;
    float3 motion=q[0]-p[0];
    if(len3(to.u-from.u)+len3(to.n-from.n)<1e-7f){
        float3 axes[3]={to.u,to.n,to.w};
        for(int ax=0;ax<3;ax++){
            float lo=1e30f,hi=-1e30f;
            for(int k=0;k<3;k++){
                float a=dot3(p[k]-from.center,axes[ax]),b=dot3(q[k]-to.center,axes[ax]);
                lo=fminf(lo,fminf(a,b));hi=fmaxf(hi,fmaxf(a,b));
            }
            float half=compF3(to.half,ax)+skin;if(lo>half || hi<-half)return out;
        }
    }
    if(len3(to.u-from.u)+len3(to.n-from.n)<1e-7f && len3(q[1]-p[1]-motion)+len3(q[2]-p[2]-motion)<1e-7f){
        float3 a[3],axes[3]={mkf3(1,0,0),mkf3(0,1,0),mkf3(0,0,1)};
        for(int k=0;k<3;k++){float3 x=p[k]-from.center;a[k]=mkf3(dot3(x,to.u),dot3(x,to.n),dot3(x,to.w));}
        float3 v=motion-translation;v=mkf3(dot3(v,to.u),dot3(v,to.n),dot3(v,to.w));
        float3 half=to.half+mkf3(skin,skin,skin);float enter=0,leave=1;
        for(int k=0;k<3;k++){
            if(!sweptAxisInterval(a[0],a[1],a[2],v,axes[k],half,enter,leave))return out;
            float3 edge=a[(k+1)%3]-a[k];
            for(int j=0;j<3;j++)if(!sweptAxisInterval(a[0],a[1],a[2],v,cross3(edge,axes[j]),half,enter,leave))return out;
        }
        if(!sweptAxisInterval(a[0],a[1],a[2],v,cross3(a[1]-a[0],a[2]-a[0]),half,enter,leave))return out;
        ContactShape hit=shapeLerp(from,to,enter);float3 x[3];for(int k=0;k<3;k++)x[k]=p[k]+motion*enter;
        if(!clippedBoxWitness(hit,x,skin+2e-6f,bary))bary=triangleWitnessReference(hit,x[0],x[1],x[2]);float3 n,s;
        shapeDistance(hit,x[0]*bary.x+x[1]*bary.y+x[2]*bary.z,n,s);
        float3 end=witnessAt(hit,to,s),start=witnessAt(hit,from,s);
        out.normal=n;out.target=end+n*skin;out.motion=end-start;out.valid=1;return out;
    }
    float speed=4.f*len3(to.half)*(len3(to.u-from.u)+len3(to.n-from.n));
    float vertexSpeed=0;for(int k=0;k<3;k++)vertexSpeed=fmaxf(vertexSpeed,len3(q[k]-p[k]-translation));speed+=vertexSpeed;
    double time=0;float tol=fmaxf(1e-6f,1e-4f*len3(to.half));ContactShape hit=from;float3 x[3];bool found=false;
    while(time<=1.){
        hit=shapeLerp(from,to,time);for(int k=0;k<3;k++)x[k]=p[k]*(1-time)+q[k]*time;
        float gap=triangleBoxGap(hit,x[0],x[1],x[2])-skin;
        if(gap<=tol){found=true;break;}
        if(speed<=1e-12f || time>=1.f)return out;
        time=fmin(1.,time+.8*(double)gap/speed);
    }
    if(!found)return out;
    if(!clippedBoxWitness(hit,x,skin+tol,bary))return out;float3 n,s;
    shapeDistance(hit,x[0]*bary.x+x[1]*bary.y+x[2]*bary.z,n,s);
    float3 end=witnessAt(hit,to,s),start=witnessAt(hit,from,s);
    n=contactUnit(to.u*dot3(n,hit.u)+to.n*dot3(n,hit.n)+to.w*dot3(n,hit.w),n);
    out.normal=n;out.target=end+n*skin;out.motion=end-start;out.valid=1;return out;
}
__global__ void k_FindSurfaceReference(int triangles,int kind,ContactShape from,ContactShape to,float skin,
    const SurfaceVertex* vertices,const unsigned* masks,const float3* prev,const float3* pos,
    SurfaceContact* contacts,int* degree,int* active,int* activeCount,const float4* current,const float3* previous,
    const CoupledTool* body=nullptr,float thickness=0,float minGap=0,bool stationary=false,
    const int* cacheSlots=nullptr,const SurfaceContact* cache=nullptr,const int* bladePhase=nullptr,int stride=3,bool cacheComponents=false){
    int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=triangles)return;
    int slot=stride*t+kind;
    if(body){from=bodyContactShape(body,stationary?body->pose:body->previous,kind,thickness,minGap);to=bodyContactShape(body,body->pose,kind,thickness,minGap);}
    // Reject distant triangles before reading/probing the material-contact
    // cache or copying a 24-node witness. This is the dominant no-contact path.
    if(to.capsule?to.radius<=0:to.half.x<=0||to.half.z<=0){contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;return;}
    float3 q[3],p[3];
    for(int k=0;k<3;k++){
        float4 point=current[3*t+k];if(point.w==0){contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;return;}
        q[k]=mkf3(point.x,point.y,point.z);p[k]=previous[3*t+k];
    }
    // An unfractured surface is a unilateral barrier on its incident side.
    // Re-querying the closest side of a thin cylinder every substep caps the
    // preload at K*radius, then reverses the normal and lets the blade through.
    // Carry this material witness until fracture or separation, including cut
    // components. Rebind its interpolation below to the current reconstruction.
    bool preload=body && to.capsule && bladePhase && (*bladePhase==0 || *bladePhase==4) && contacts[slot].count>0;
    for(int k=0;k<3 && preload;k++)preload=contacts[slot].owner[k]==vertices[3*t+k].voxel &&
        contacts[slot].mask[k]==vertices[3*t+k].mask && contacts[slot].component[k]==vertices[3*t+k].component;
    ContactRecord preloadPlane={};float3 preloadBary=mkf3(0,0,0);
    if(preload){
        const SurfaceContact& old=contacts[slot];preloadBary=bladeTriangleWitness(to,q);
        float3 normal=rotateShaftVector(old.referenceShape,to,old.plane.normal);
        float3 end=witnessAt(old.referenceShape,to,old.plane.target-old.plane.normal*skin);
        float3 x=q[0]*preloadBary.x+q[1]*preloadBary.y+q[2]*preloadBary.z;
        // Project onto the incident plane and require the finite capsule's
        // footprint. This is not an infinite plane or an enlarged collision box.
        float gap=dot3(x-end,normal);float3 nn,ss;
        float lateral=shapeDistance(to,x-normal*(gap+to.radius),nn,ss);
        preload=gap<=0 && lateral<=fmaxf(1e-6f,1e-3f*to.radius);
        if(preload){preloadPlane.normal=normal;preloadPlane.target=end+normal*skin;
            preloadPlane.motion=end-witnessAt(to,from,end);preloadPlane.valid=1;}
    }
    float3 lo,hi,l0,h0;shapeBounds(to,skin,lo,hi);shapeBounds(from,skin,l0,h0);
    // Endpoint OBB bounds alone miss intermediate rotational sweeps.
    if(!to.capsule && (len3(to.u-from.u)+len3(to.n-from.n)>1e-5f)){
        float r=len3(to.half)+skin;float3 extent=mkf3(r,r,r);lo=to.center-extent;hi=to.center+extent;l0=from.center-extent;h0=from.center+extent;
    }
    for(int ax=0;ax<3;ax++){
        float mn=1e30f,mx=-1e30f;for(int k=0;k<3;k++){mn=fminf(mn,fminf(compF3(q[k],ax),compF3(p[k],ax)));mx=fmaxf(mx,fmaxf(compF3(q[k],ax),compF3(p[k],ax)));}
        if(!preload && (mx<fminf(compF3(lo,ax),compF3(l0,ax)) || mn>fmaxf(compF3(hi,ax),compF3(h0,ax)))){contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;return;}
    }
    // Persist a material witness while its triangle owners and cut masks match.
    // A nearest-face query inside a thin jaw can otherwise flip its normal and
    // push tissue through the opposite face on the next substep.
    // Retain the incident side only for the same voxel, cut mask AND component.
    // The actual material witness is rebuilt below from current vertex weights.
    SurfaceContact retained;bool keep=body && contacts[slot].count>0;
    for(int k=0;k<3 && keep;k++)keep=vertices[3*t+k].count>0 &&
        contacts[slot].component[k]==vertices[3*t+k].component &&
        contacts[slot].owner[k]==vertices[3*t+k].voxel && contacts[slot].mask[k]==vertices[3*t+k].mask;
    if(keep)retained=contacts[slot];
    if(!keep && body && cacheSlots && cache && cacheComponents){
        uint3 key[3];bool valid=true;
        for(int k=0;k<3;k++){const SurfaceVertex& v=vertices[3*t+k];valid=valid && v.count>0;key[k]=make_uint3(v.voxel,v.mask,v.component+1);}
        if(valid){
            sortMaterialKey(key);unsigned hash=materialContactHash(key,kind);
            for(int probe=0;probe<CONTACT_CACHE_SIZE;probe++){
                unsigned slot=(hash+probe)&(CONTACT_CACHE_SIZE-1);if(cacheSlots[slot]<0)break;
                if(sameMaterialContactKey(cache[slot],key,kind)){retained=cache[slot];keep=true;break;}
            }
        }
    }
    if(!keep && body && cacheSlots && cache && !cacheComponents){
        int key[3];bool intact=true;
        for(int k=0;k<3;k++){const SurfaceVertex& v=vertices[3*t+k];key[k]=v.voxel;intact=intact && v.count>0 && v.component<0 && v.mask==0;}
        if(intact){
            contactKey(key);unsigned h=contactHash(key,kind);
            for(int probe=0;probe<CONTACT_CACHE_SIZE;probe++){
                unsigned s=(h+probe)&(CONTACT_CACHE_SIZE-1);if(cacheSlots[s]<0)break;
                if(sameContactKey(cache[s],key,kind)){
                    retained=cache[s];keep=true;
                    for(int k=0;k<3;k++){retained.owner[k]=vertices[3*t+k].voxel;retained.mask[k]=vertices[3*t+k].mask;}
                    break;
                }
            }
        }
    }
    // Rejected triangles have no live contact. Do not clear the full 24-node
    // stencil for every triangle, jaw and substep: only the validity fields.
    contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;
    ContactRecord retainedPlane={};
    if(keep){
        float3 x=retained.offset,old=x;
        for(int j=0;j<retained.count;j++){x=x+pos[retained.id[j]]*retained.weight[j];old=old+prev[retained.id[j]]*retained.weight[j];}
        float3 normal=retained.plane.normal;
        float3 start=witnessAt(retained.referenceShape,from,retained.plane.target-normal*skin);
        if(!to.capsule)normal=contactUnit(from.u*dot3(normal,retained.referenceShape.u)+from.n*dot3(normal,retained.referenceShape.n)+from.w*dot3(normal,retained.referenceShape.w),normal);
        float3 fromNormal=normal;
        float3 end=witnessAt(from,to,start),nn,ss;
        float distance=shapeDistance(to,x,nn,ss);
        if(to.capsule)normal=rotateShaftVector(from,to,normal);
        else normal=contactUnit(to.u*dot3(normal,from.u)+to.n*dot3(normal,from.n)+to.w*dot3(normal,from.w),normal);
        // Only retain while within the current finite tool. No infinite-plane
        // adhesion when the witness slides off a jaw edge or withdraws.
        // A material witness may slide past the finite face while another part
        // of its triangle is still inside the jaw. Dropping the incident side
        // based on that one witness lets a nearest-face query flip through a
        // thin blade. Retain a box contact using the whole finite triangle;
        // release still occurs when its finite overlap disappears.
        float finiteDistance=to.capsule?distance:triangleBoxGap(to,q[0],q[1],q[2]);
        // The inertial predictor may leave the finite face, then be rejected by
        // the material solve. Keep its entry side for the complete swept pair;
        // inspecting only the predicted endpoint discards the accepted contact
        // before CCD starts from that same touching configuration. At a corner
        // the fresh nearest-face query can then select opposite sides on
        // triangles sharing one node. The finite manifold still drops samples
        // outside the CURRENT footprint and positive gaps exert no force.
        float signedDistance=dot3(x-end,normal);
        if(!to.capsule)signedDistance=fminf(dot3(q[0]-end,normal),fminf(dot3(q[1]-end,normal),dot3(q[2]-end,normal)));
        bool startContact=false;
        if(!to.capsule){
            float startDistance=fminf(dot3(p[0]-start,fromNormal),fminf(dot3(p[1]-start,fromNormal),dot3(p[2]-start,fromNormal)));
            startContact=triangleBoxGap(from,p[0],p[1],p[2])<=2.f*skin && startDistance<=2.f*skin;
        }
        if(startContact || (finiteDistance<=2.f*skin && signedDistance<=2.f*skin)){
            // Keep the incident SIDE, not a stale barycentric witness. A triangle
            // can rotate about its first contact while another portion penetrates.
            retainedPlane.normal=normal;retainedPlane.valid=1;
            retainedPlane.target=end+normal*skin;retainedPlane.motion=end-start;
        }
    }
    SurfaceContact out={};out.kind=kind;
    float3 bary=mkf3(0,0,0);
    if(preload){bary=preloadBary;out.plane=preloadPlane;}
    else if(retainedPlane.valid && !to.capsule){
        bary=triangleWitnessReference(to,q[0],q[1],q[2]);
        out.plane=retainedPlane;out.stick=retained.stick;
        float3 normal=out.plane.normal;
        // Supporting plane on the retained incident side, on the finite box.
        // Previous positions use the same refreshed material interpolation.
        float support=fabsf(dot3(normal,to.u))*to.half.x+
            fabsf(dot3(normal,to.n))*to.half.y+fabsf(dot3(normal,to.w))*to.half.z;
        float3 x=q[0]*bary.x+q[1]*bary.y+q[2]*bary.z;
        float3 surface=x+normal*(support-dot3(x-to.center,normal));
        out.plane.target=surface+normal*skin;
        out.plane.motion=surface-witnessAt(to,from,surface);
    }else if(to.capsule){
        bary=bladePhase?bladeTriangleWitness(to,q):triangleWitnessReference(to,q[0],q[1],q[2]);
        out.plane=traceToolContact(from,to,skin,p[0]*bary.x+p[1]*bary.y+p[2]*bary.z,q[0]*bary.x+q[1]*bary.y+q[2]*bary.z);
        // The swept endpoint witness is not the first point on a face hit by
        // a grazing capsule. Its radial TOI normal can be almost tangential,
        // and persisting that normal artificially caps the preload. On the
        // interior of a triangle use its incident face normal; keep edge/vertex
        // contacts radial and retain the finite capsule footprint checks.
        if(bladePhase && (*bladePhase==0 || *bladePhase==2 || *bladePhase==4) && out.plane.valid &&
           fminf(bary.x,fminf(bary.y,bary.z))>1e-4f){
            float3 normal=cross3(q[1]-q[0],q[2]-q[0]);
            if(dot3(normal,normal)>1e-16f){
                normal=contactUnit(normal,out.plane.normal);
                float3 oldPoint=p[0]*bary.x+p[1]*bary.y+p[2]*bary.z;
                float3 oldAxis=from.b-from.a;
                float oldT=fminf(1.f,fmaxf(0.f,dot3(oldPoint-from.a,oldAxis)/fmaxf(dot3(oldAxis,oldAxis),1e-20f)));
                if(dot3(normal,oldPoint-(from.a+oldAxis*oldT))<0)normal=normal*(-1.f);
                float3 point=q[0]*bary.x+q[1]*bary.y+q[2]*bary.z,axis=to.b-to.a;
                float tLine=fminf(1.f,fmaxf(0.f,dot3(point-to.a,axis)/fmaxf(dot3(axis,axis),1e-20f)));
                float3 surface=to.a+axis*tLine+normal*to.radius;
                out.plane.normal=normal;out.plane.target=surface+normal*skin;
                out.plane.motion=surface-witnessAt(to,from,surface);
            }
        }
    }else out.plane=traceTriangleBoxReference(from,to,skin,p,q,bary);
    float3 x=q[0]*bary.x+q[1]*bary.y+q[2]*bary.z;
    float3 old=p[0]*bary.x+p[1]*bary.y+p[2]*bary.z;
    if(!out.plane.valid)return;
    if(body && !to.capsule){
        // A finite face manifold must use an actual OBB face, not a corner SDF
        // diagonal extended into an artificial slanted supporting half-space.
        float3 axes[3]={to.u,to.n,to.w};int face=0;
        for(int a=1;a<3;a++)if(fabsf(dot3(out.plane.normal,axes[a]))>fabsf(dot3(out.plane.normal,axes[face])))face=a;
        float sign=dot3(out.plane.normal,axes[face])>=0?1.f:-1.f;
        out.plane.normal=axes[face]*sign;
        float3 surface=x+out.plane.normal*(compF3(to.half,face)-dot3(x-to.center,out.plane.normal));
        out.plane.target=surface+out.plane.normal*skin;
        out.plane.motion=surface-witnessAt(to,from,surface);
    }
    if(body && stationary){
        float3 surface=out.plane.target-out.plane.normal*skin;
        out.plane.motion=surface-witnessAt(to,bodyContactShape(body,body->previous,kind,thickness,minGap),surface);
    }
    SurfaceVertex v[3]={vertices[3*t],vertices[3*t+1],vertices[3*t+2]};
    out.previous=old;
    out.toolOrigin=body?body->pose.jawHinge:mkf3(0,0,0);
    out.referenceShape=to;out.toolGap=body?body->pose.jawGap:0;
    out.gapGradient=body?jawGapGradient(body->pose,kind,thickness,minGap,out.plane.target-out.plane.normal*skin,body):mkf3(0,0,0);
    out.incoming=dot3(x-old-out.plane.motion,out.plane.normal)<0;
    for(int k=0;k<3;k++){
        out.owner[k]=v[k].voxel;out.mask[k]=v[k].mask;out.component[k]=v[k].component;
        out.offset=out.offset+v[k].offset*compF3(bary,k);
        for(int j=0;j<v[k].count;j++){
            float w=compF3(bary,k)*v[k].weight[j];if(w<1e-8f)continue;
            int id=v[k].id[j],s=0;while(s<out.count && out.id[s]!=id)s++;
            if(s==out.count){out.id[s]=id;out.count++;}
            out.weight[s]+=w;
        }
    }
    for(int j=0;j<out.count;j++)atomicAdd(&degree[out.id[j]],1);
    contacts[slot]=out;
    if(out.count)active[atomicAdd(activeCount,1)]=slot;
}
