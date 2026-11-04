"""Scalar fused versions of the existing rigid-envelope distance diagnostics.
No fastmath, altered sampling, geometric approximation, or changed thresholds.
"""
import numpy as np
from numba import njit

@njit(nogil=True,cache=False)
def dot3(a,b):return (a[0]*b[0]+a[1]*b[1])+a[2]*b[2]

@njit(nogil=True,cache=False)
def vertices(center,frame,half):
    result=np.empty((8,3),np.float64)
    for i in range(8):
        for k in range(3):
            value=0.
            for j in range(3):value+=((1. if i&(1<<j) else -1.)*half[j])*frame[j,k]
            result[i,k]=center[k]+value
    return result

@njit(nogil=True,cache=False)
def box_distance(center_a,frame_a,half_a,center_b,frame_b,half_b):
    maximum=-np.inf
    for i in range(15):
        if i<3:axis=frame_a[i].copy()
        elif i<6:axis=frame_b[i-3].copy()
        else:
            a=frame_a[(i-6)//3];b=frame_b[(i-6)%3]
            axis=np.array((a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]))
        length=np.sqrt(dot3(axis,axis))
        if length<=1e-10:continue
        axis/=length
        sep=abs(dot3(axis,center_b-center_a))
        support_a=0.;support_b=0.
        for k in range(3):
            support_a+=abs(dot3(axis,frame_a[k]))*half_a[k]
            support_b+=abs(dot3(axis,frame_b[k]))*half_b[k]
        maximum=max(maximum,(sep-support_a)-support_b)
    if maximum<=0:return maximum
    av=vertices(center_a,frame_a,half_a);bv=vertices(center_b,frame_b,half_b);best=np.inf
    for direction in range(2):
        points=av if direction==0 else bv
        frame=frame_b if direction==0 else frame_a
        center=center_b if direction==0 else center_a
        half=half_b if direction==0 else half_a
        for i in range(8):
            rel=points[i]-center;distance=0.
            for k in range(3):
                q=max(abs(dot3(rel,frame[k]))-half[k],0.)
                distance+=q*q
            best=min(best,distance)
    # All 12x12 segment pairs with the same interior and four endpoint cases.
    for ai in range(8):
        for ak in range(3):
            if ai&(1<<ak):continue
            a=av[ai];u=av[ai^(1<<ak)]-a;aa=dot3(u,u)
            for bi in range(8):
                for bk in range(3):
                    if bi&(1<<bk):continue
                    c=bv[bi];v=bv[bi^(1<<bk)]-c;w=a-c
                    cc=dot3(v,v);bb=dot3(u,v);dd=dot3(u,w);ee=dot3(v,w);det=aa*cc-bb*bb
                    for case in range(5):
                        if case==0:
                            if det<=1e-24:continue
                            s=(bb*ee-cc*dd)/det;t=(aa*ee-bb*dd)/det
                            if s<0 or s>1 or t<0 or t>1:continue
                        elif case==1 or case==3:
                            s=0. if case==1 else 1.;t=min(1.,max(0.,(ee+s*bb)/max(cc,1e-30)))
                        else:
                            t=0. if case==2 else 1.;s=min(1.,max(0.,(t*bb-dd)/max(aa,1e-30)))
                        q=(w+s*u)-t*v;best=min(best,dot3(q,q))
    return np.sqrt(max(0.,best))

@njit(nogil=True,cache=False)
def capsule_box_distance(start,end,radius,center,frame,half):
    a=np.empty(3,np.float64);b=np.empty(3,np.float64)
    for k in range(3):a[k]=dot3(start-center,frame[k]);b[k]=dot3(end-center,frame[k])
    u=b-a;cuts=np.empty(8,np.float64);cuts[0]=0.;cuts[1]=1.;n=2
    for k in range(3):
        if abs(u[k])>1e-15:
            for side in (-1.,1.):
                t=(side*half[k]-a[k])/u[k]
                if 0<t<1:cuts[n]=t;n+=1
    cuts=np.sort(cuts[:n]);samples=np.empty(15,np.float64);samples[:n]=cuts;count=n
    for i in range(n-1):
        lo,hi=cuts[i],cuts[i+1]
        if lo==hi:continue
        middle=a+(lo+hi)*.5*u;v=np.zeros(3,np.float64);c=np.zeros(3,np.float64)
        for k in range(3):
            if abs(middle[k])>half[k]:v[k]=u[k];c[k]=a[k]-np.sign(middle[k])*half[k]
        vv=dot3(v,v)
        if vv>1e-30:samples[count]=min(hi,max(lo,-dot3(v,c)/vv));count+=1
    best=np.inf
    for i in range(count):
        point=a+samples[i]*u;distance=0.
        for k in range(3):
            q=max(abs(point[k])-half[k],0.);distance+=q*q
        best=min(best,distance)
    return np.sqrt(best)-radius

@njit(nogil=True,cache=False)
def between_pads(positions,hinge,frame,bounds,sn,cs):
    result=np.ones(len(positions),np.bool_)
    for i in range(len(positions)):
        dx=positions[i,0]-hinge[0];dy=positions[i,1]-hinge[1];dz=positions[i,2]-hinge[2]
        x=(dx*frame[0,0]+dy*frame[0,1])+dz*frame[0,2]
        y=(dx*frame[1,0]+dy*frame[1,1])+dz*frame[1,2]
        z=(dx*frame[2,0]+dy*frame[2,1])+dz*frame[2,2]
        for jaw in range(2):
            side=1 if jaw==0 else -1
            along=x*cs+side*y*sn;across=y*cs-side*x*sn
            if not (abs(along-bounds[jaw,0,0])<=bounds[jaw,1,0] and side*(across-bounds[jaw,0,1])<=-bounds[jaw,1,1] and abs(z-bounds[jaw,0,2])<=bounds[jaw,1,2]):result[i]=False
    return result
