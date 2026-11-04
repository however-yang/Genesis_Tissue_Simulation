// Detached-component contact on the same embedded surface used for rendering.
// Unilateral, mass-weighted Jacobi constraints; no debris is frozen or discarded.
struct FragmentVertex { int root; float3 current,previous,recovery; };
__device__ inline float3 fragmentBroadcast(float3 value,int lane){
    return mkf3(__shfl_sync(0xffffffff,value.x,lane),__shfl_sync(0xffffffff,value.y,lane),__shfl_sync(0xffffffff,value.z,lane));
}
__device__ inline int fragmentBestLane(float value,bool minimum){
    float best=value;
    for(int offset=16;offset;offset/=2){
        float other=__shfl_down_sync(0xffffffff,best,offset);
        best=minimum?fminf(best,other):fmaxf(best,other);
    }
    best=__shfl_sync(0xffffffff,best,0);
    return __ffs(__ballot_sync(0xffffffff,value==best))-1;
}
#ifdef ENDO_FRAGMENT_TRACE
__global__ void k_FragmentTrace(int phase,int n,const int* component,const int* support,const int* grasp,
    const float* inverseMass,const float3* pos,const float3* previous,const float3* target,const float3* velocity){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && inverseMass[i]>0 && !support[component[i]] && !(grasp[component[i]]&1))
        printf("FRAG %d %d pos %.7g %.7g %.7g prev %.7g target %.7g vel %.7g\n",phase,i,pos[i].x,pos[i].y,pos[i].z,previous[i].y,target[i].y,velocity[i].y);
}
#endif
__global__ void k_FragmentSupports(int n,const int* component,const float* inverseMass,const int* active,int* supported){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && (!active || active[i]) && inverseMass[i]==0)atomicExch(supported+component[i],1);
}
__device__ inline int fragmentRoot(const SurfaceVertex& v,const int* component){
    int root=v.count?component[v.id[0]]:-1;
    for(int k=1;k<v.count;k++)if(component[v.id[k]]!=root)root=-1;
    return root;
}
// Contact queries (surface points, then nodes) of detached, unheld components.
// Supported tissue has no query, so the passes never visit it.
__global__ void k_FragmentQueries(int triangles,int nodes,const int* component,const int* active,const SurfaceVertex* vertices,
    const int* supported,const int* grasp,const float* inverseMass,int* queries,int* count){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=3*triangles+nodes)return;
    int root;
    if(i<3*triangles)root=fragmentRoot(vertices[i],component);
    else{int id=i-3*triangles;if((active && !active[id]) || inverseMass[id]<=0)return;root=component[id];}
    if(root<0 || (supported[root]&1) || (grasp[root]&1))return;
    queries[atomicAdd(count,1)]=i;
}
__global__ void k_FragmentVertices(int count,const SurfaceVertex* vertices,const int* component,
    const float3* pos,const float3* previous,const float3* start,FragmentVertex* output,const int* queryCount){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !*queryCount)return;
    const SurfaceVertex& v=vertices[i];FragmentVertex r={};r.root=fragmentRoot(v,component);
    r.current=surfacePoint(v,pos);r.previous=surfacePoint(v,previous);
    r.recovery=contactUnit(r.previous-surfacePoint(v,start),mkf3(.169481f,.97966f,.114622f));output[i]=r;
}
__device__ inline void fragmentWeight(int id,float weight,int* ids,float* weights,int& count){
    for(int k=0;k<count;k++)if(ids[k]==id){weights[k]+=weight;return;}
    ids[count]=id;weights[count++]=weight;
}
__device__ inline bool fragmentRay(float3 point,float3 direction,const float3* q,float lower,float& time,float3& bary){
    float3 e0=q[1]-q[0],e1=q[2]-q[0],h=cross3(direction,e1);float determinant=dot3(e0,h);
    if(fabsf(determinant)<1e-12f)return false;
    float inverse=1.f/determinant;float3 r=point-q[0];float v=dot3(r,h)*inverse;
    float3 k=cross3(r,e0);float w=dot3(direction,k)*inverse;time=dot3(e1,k)*inverse;
    bary=mkf3(1-v-w,v,w);return time>lower && v>=0 && w>=0 && v+w<=1;
}
// Follow oriented crossings to a genuine outside interval. Folded cut lips can
// have an outward nearest normal while the point remains inside another layer.
__device__ inline bool fragmentExit(int triangles,const FragmentVertex* points,FragmentVertex p,int winding,float skin,
    int& selected,float3& bary,float3& direction,float& distance){
    // Freeze the incident direction for this solve. Recomputing it from the
    // corrected position reverses the ray after a partial repair and can eject
    // the fragment through the opposite side of the entire organ.
    direction=p.recovery;
    float lower=1e-6f,candidate=-1;int candidateTri=-1;float3 candidateBary={};
    for(int event=0;event<=triangles;event++){
        float next=1e30f;int hit=-1;float3 hitBary={},hitNormal={};
        for(int t=threadIdx.x%32;t<triangles;t+=32){
            const FragmentVertex& a=points[3*t];const FragmentVertex& b=points[3*t+1];const FragmentVertex& c=points[3*t+2];
            if(a.root<0 || a.root==p.root || a.root!=b.root || a.root!=c.root)continue;
            float3 q[3]={a.current,b.current,c.current},weights;float time;
            if(fragmentRay(p.current,direction,q,lower,time,weights) && time<next){
                next=time;hit=t;hitBary=weights;hitNormal=cross3(q[1]-q[0],q[2]-q[0]);
            }
        }
        int lane=fragmentBestLane(next,true);
        next=__shfl_sync(0xffffffff,next,lane);hit=__shfl_sync(0xffffffff,hit,lane);
        hitBary=fragmentBroadcast(hitBary,lane);hitNormal=fragmentBroadcast(hitNormal,lane);
        if(candidate>=0 && (hit<0 || next-candidate>2*skin)){
            selected=candidateTri;bary=candidateBary;distance=candidate+skin;return true;
        }
        if(hit<0)return false;
        winding-=dot3(hitNormal,direction)>0?1:-1;
        candidate=winding==0?next:-1;candidateTri=hit;candidateBary=hitBary;lower=next+1e-6f;
    }
    return false;
}
// One query from k_FragmentQueries, solved by the whole warp.
__device__ void fragmentQuery(int i,int lane,int triangles,const int* component,const float3* pos,const float3* previous,const float3* start,
    const SurfaceVertex* vertices,const FragmentVertex* points,const float* inverseMass,float3* delta,int* degree){
    SurfaceVertex query={};FragmentVertex p;
    if(i<3*triangles){query=vertices[i];p=points[i];}
    else{
        int id=i-3*triangles;
        query.count=1;query.id[0]=id;query.weight[0]=1;query.offset=mkf3(0,0,0);
        p.root=component[id];p.current=pos[id];p.previous=previous[id];
        p.recovery=contactUnit(previous[id]-start[id],mkf3(.169481f,.97966f,.114622f));
    }
    const float skin=.001f; // 0.01 mm, well below the 0.2 mm instrument tolerance.
    float best=0;int selected=-1;float3 normal=mkf3(0,0,0),bary=normal;
    float solidAngle=0;
    float nearest=1e30f,nearestSign=1;int nearestTri=-1;float3 nearestBary=normal,nearestDirection=normal;
    for(int t=lane;t<triangles;t+=32){
        const FragmentVertex& a=points[3*t];const FragmentVertex& b=points[3*t+1];const FragmentVertex& c=points[3*t+2];
        if(a.root<0 || a.root==p.root || b.root!=a.root || c.root!=a.root)continue;
        float3 q[3]={a.current,b.current,c.current},old[3]={a.previous,b.previous,c.previous};
        float3 va=q[0]-p.current,vb=q[1]-p.current,vc=q[2]-p.current;
        float la=len3(va),lb=len3(vb),lc=len3(vc);
        solidAngle+=2.f*atan2f(dot3(va,cross3(vb,vc)),la*lb*lc+dot3(va,vb)*lc+dot3(vb,vc)*la+dot3(vc,va)*lb);
        // Swept AABB broad phase includes triangle motion and the contact skin.
        float3 lo=mkf3(fminf(q[0].x,fminf(q[1].x,q[2].x)),fminf(q[0].y,fminf(q[1].y,q[2].y)),fminf(q[0].z,fminf(q[1].z,q[2].z)));
        float3 hi=mkf3(fmaxf(q[0].x,fmaxf(q[1].x,q[2].x)),fmaxf(q[0].y,fmaxf(q[1].y,q[2].y)),fmaxf(q[0].z,fmaxf(q[1].z,q[2].z)));
        for(int j=0;j<3;j++){lo.x=fminf(lo.x,old[j].x);lo.y=fminf(lo.y,old[j].y);lo.z=fminf(lo.z,old[j].z);hi.x=fmaxf(hi.x,old[j].x);hi.y=fmaxf(hi.y,old[j].y);hi.z=fmaxf(hi.z,old[j].z);}
        float3 distance=mkf3(fmaxf(0.f,fmaxf(lo.x-p.current.x,p.current.x-hi.x)),fmaxf(0.f,fmaxf(lo.y-p.current.y,p.current.y-hi.y)),fmaxf(0.f,fmaxf(lo.z-p.current.z,p.current.z-hi.z)));
        if(dot3(distance,distance)<nearest){
            ContactShape point={};point.a=point.b=p.current;
            float3 w=bladeTriangleWitness(point,q),r=p.current-(q[0]*w.x+q[1]*w.y+q[2]*w.z);
            float d=dot3(r,r);float3 n=cross3(q[1]-q[0],q[2]-q[0]);float len=len3(n);
            if(d<nearest && len>1e-10f){nearest=d;nearestTri=t;nearestBary=w;nearestSign=dot3(r,n);
                nearestDirection=d>1e-20f?r*(-1.f/sqrtf(d)):n*(1.f/len);}
        }
        if(fminf(p.current.x,p.previous.x)>hi.x+skin || fmaxf(p.current.x,p.previous.x)<lo.x-skin ||
           fminf(p.current.y,p.previous.y)>hi.y+skin || fmaxf(p.current.y,p.previous.y)<lo.y-skin ||
           fminf(p.current.z,p.previous.z)>hi.z+skin || fmaxf(p.current.z,p.previous.z)<lo.z-skin)continue;
        float3 n=cross3(q[1]-q[0],q[2]-q[0]),pn=cross3(old[1]-old[0],old[2]-old[0]);
        float length=len3(n),pl=len3(pn);if(length<1e-10f || pl<1e-10f)continue;n=n*(1.f/length);pn=pn*(1.f/pl);
        float now=dot3(p.current-q[0],n),before=dot3(p.previous-old[0],pn);
        if(now>=skin || before<-2*skin)continue;
        float time=before>skin?fminf(1.f,fmaxf(0.f,(before-skin)/fmaxf(before-now,1e-12f))):0.f;
        float3 hit=p.previous*(1-time)+p.current*time,qt[3];
        for(int j=0;j<3;j++)qt[j]=old[j]*(1-time)+q[j]*time;
        ContactShape point={};point.a=point.b=hit;
        float3 w=bladeTriangleWitness(point,qt),closest=qt[0]*w.x+qt[1]*w.y+qt[2]*w.z;
        if(len3(hit-closest)>3*skin)continue;
        float penetration=skin-dot3(p.current-(q[0]*w.x+q[1]*w.y+q[2]*w.z),n);
        if(penetration>best){best=penetration;selected=t;normal=n;bary=w;}
    }
    for(int offset=16;offset;offset/=2)solidAngle+=__shfl_down_sync(0xffffffff,solidAngle,offset);
    solidAngle=__shfl_sync(0xffffffff,solidAngle,0);
    int contactLane=fragmentBestLane(best,false),nearestLane=fragmentBestLane(nearest,true);
    best=__shfl_sync(0xffffffff,best,contactLane);selected=__shfl_sync(0xffffffff,selected,contactLane);
    normal=fragmentBroadcast(normal,contactLane);bary=fragmentBroadcast(bary,contactLane);
    nearest=__shfl_sync(0xffffffff,nearest,nearestLane);nearestSign=__shfl_sync(0xffffffff,nearestSign,nearestLane);
    nearestTri=__shfl_sync(0xffffffff,nearestTri,nearestLane);
    nearestBary=fragmentBroadcast(nearestBary,nearestLane);nearestDirection=fragmentBroadcast(nearestDirection,nearestLane);
    // Topology changes may create an initially overlapping point. CCD alone
    // cannot repair an invalid starting configuration. The nearest oriented
    // boundary supplies a local minimum-translation exit in this case.
    int winding=(int)roundf(solidAngle/(4.f*3.14159265358979323846f));
    if(winding!=0){
        int exitTriangle;float3 exitBary,exitDirection;float exitDistance;
        if(fragmentExit(triangles,points,p,winding,skin,exitTriangle,exitBary,exitDirection,exitDistance)){
            selected=exitTriangle;best=exitDistance;normal=exitDirection;bary=exitBary;
        }
    }
    if(selected<0 && nearestTri>=0 && nearestSign<0){
        selected=nearestTri;best=sqrtf(nearest)+skin;normal=nearestDirection;bary=nearestBary;
    }
#ifdef ENDO_FRAGMENT_TRACE
    if(lane==0 && i>=3*triangles)printf("HIT %d at %.7g old %.7g wind %d near %.7g sign %.7g tri %d move %.7g normal %.7g %.7g %.7g\n",i-3*triangles,p.current.y,p.previous.y,winding,sqrtf(nearest),nearestSign,selected,best,normal.x,normal.y,normal.z);
#endif
    if(selected<0 || lane!=0)return;
    int ids[32],count=0;float weights[32];const SurfaceVertex& v=query;
    for(int k=0;k<v.count;k++)fragmentWeight(v.id[k],v.weight[k],ids,weights,count);
    for(int j=0;j<3;j++){
        const SurfaceVertex& other=vertices[3*selected+j];float w=compF3(bary,j);
        for(int k=0;k<other.count;k++)fragmentWeight(other.id[k],-w*other.weight[k],ids,weights,count);
    }
    float denominator=0;for(int k=0;k<count;k++)denominator+=inverseMass[ids[k]]*weights[k]*weights[k];
    if(denominator<1e-20f)return;
    for(int k=0;k<count;k++)if(inverseMass[ids[k]]>0 && fabsf(weights[k])>1e-10f){
        add3(delta+ids[k],normal*(best*inverseMass[ids[k]]*weights[k]/denominator));atomicAdd(degree+ids[k],1);
    }
}
__global__ void k_FragmentContact(int triangles,const int* queries,const int* queryCount,const int* component,const float3* pos,
    const float3* previous,const float3* start,const SurfaceVertex* vertices,const FragmentVertex* points,const float* inverseMass,
    float3* delta,int* degree){
    // A warp owns one query. Cooperate over ALL target triangles; reducing
    // the same winding, nearest witness and CCD candidates avoids a serial
    // whole-organ scan on one GPU lane for every fragment and solver pass.
    int warps=gridDim.x*blockDim.x/32;
    for(int q=(blockIdx.x*blockDim.x+threadIdx.x)/32;q<*queryCount;q+=warps)
        fragmentQuery(queries[q],threadIdx.x%32,triangles,component,pos,previous,start,vertices,points,inverseMass,delta,degree);
}
// Applies and clears the corrections of one pass, leaving delta and degree zero for the next one. Every node the
// passes move belongs to a detached component with a query (k_FragmentQueries): without one they are all zero.
__global__ void k_ApplyFragmentContact(int n,float3* pos,float3* delta,int* degree,const int* queryCount){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n || !*queryCount || !degree[i])return;
    pos[i]=pos[i]+delta[i]*(.8f/degree[i]);delta[i]=mkf3(0,0,0);degree[i]=0;
}

// Components without any live volume element use inertial particles and
// their remaining elastic edges. They have no volumetric fracture resistance.
__global__ void k_UnsupportedFlight(int n,const int* support,const int* components,const int* grasp,
    const float* inverseMass,const float3* target,float3* pos){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n || inverseMass[i]<=0 || support[components[i]] || (grasp[components[i]]&1))return;
    pos[i]=target[i];
}
// Its nodes are nodes with a query (k_FragmentQueries), so none moves without one.
__global__ void k_UnsupportedTools(int n,const int* support,const int* components,const int* grasp,
    const float* inverseMass,const float3* previous,float3* pos,const CoupledTool* grasper,const CoupledTool* blade,
    float thickness,float minGap,const int* queryCount){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || !*queryCount || inverseMass[i]<=0 || support[components[i]] || (grasp[components[i]]&1))return;
    float3 x=pos[i];
    // Prescribed instrument guides provide the boundary for these unsupported
    // point remnants. The deformable tissue keeps its coupled force solve.
    for(int tool=0;tool<2;tool++){
        const CoupledTool* b=tool?blade:grasper;if(!b)continue;
        for(int shape=0;shape<3;shape++){
            int kind=tool?(shape==0?0:shape+2):shape;
            if(tool && kind>=3 && !b->backContact)continue;
            ContactShape from=bodyContactShape(b,b->previous,kind,tool?0:thickness,tool?0:minGap);
            ContactShape to=bodyContactShape(b,b->pose,kind,tool?0:thickness,tool?0:minGap);
            ContactRecord c=traceToolContact(from,to,.001f,previous[i],x);
            if(c.valid){float depth=dot3(x-c.target,c.normal);if(depth<0)x=x-c.normal*depth;}
            float3 normal,surface;float distance=shapeDistance(to,x,normal,surface);
            if(distance<.001f)x=x+normal*(.001f-distance);
        }
    }
    pos[i]=x;
}

__global__ void k_FragmentVolumes(int count,const int* ids,const int* live,const int* component,int* support){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !live[i])return;
    // All tets of a component set the same flag: once it is set, skip the contended atomic.
    int* flag=support+component[ids[4*i]];if(!(*flag&2))atomicOr(flag,2);
}
// Its nodes are nodes with a query (k_FragmentQueries), so none moves without one.
__global__ void k_FragmentEdges(int n,const int* neighbors,const float3* rest,const int* component,const int* support,
    const int* grasp,const float* inverseMass,const float3* pos,float young,float h,float* lambda,float3* delta,int* degree,
    const int* queryCount){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || !*queryCount || inverseMass[i]<=0 || support[component[i]] || (grasp[component[i]]&1))return;
    float cell[3];for(int k=0;k<3;k++)cell[k]=fmaxf(len3(rest[6*i+2*k]),len3(rest[6*i+2*k+1]));
    for(int a=0;a<3;a++){
        int slot=6*i+2*a+1,j=neighbors[slot];if(j<0 || support[component[j]])continue;
        float3 d=pos[j]-pos[i];float length=len3(d),restLength=len3(rest[slot]);if(length<1e-10f || restLength<1e-10f)continue;
        float stiffness=young*cell[(a+1)%3]*cell[(a+2)%3]/restLength;
        if(stiffness<=0)continue;
        float alpha=1.f/(stiffness*h*h),dl=(-(length-restLength)-alpha*lambda[slot])/(inverseMass[i]+inverseMass[j]+alpha);
        lambda[slot]+=dl;float3 normal=d*(1.f/length);
        add3(delta+i,normal*(-inverseMass[i]*dl));add3(delta+j,normal*(inverseMass[j]*dl));
        atomicAdd(degree+i,1);atomicAdd(degree+j,1);
    }
}

__global__ void k_FragmentIncomingVelocity(int n,const int* component,const int* support,const int* grasp,
    const float3* before,const float3* pos,float3* velocity){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || support[component[i]] || (grasp[component[i]]&1))return;
    float3 correction=pos[i]-before[i];float length=len3(correction);if(length<1e-7f)return;
    float3 normal=correction*(1.f/length);float incoming=dot3(velocity[i],normal);
    if(incoming<0)velocity[i]=velocity[i]-normal*incoming;
}
