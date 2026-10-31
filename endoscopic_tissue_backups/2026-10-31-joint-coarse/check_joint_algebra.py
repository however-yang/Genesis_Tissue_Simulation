"""Independent float64 energy/Jacobian oracle for the actual CUDA 12-DOF tet terms."""
import ctypes as ct
from pathlib import Path
import json
import sys
import numpy as np
from threadpoolctl import threadpool_limits

rng=np.random.default_rng(20261031)

def cases():
    result=[]
    for i in range(180):
        F=np.eye(3)+rng.normal(0,.18,(3,3))
        if np.linalg.det(F)<.15: F=np.eye(3)
        if i<16: F=np.diag([.1501+.02*i,1.,1.])
        origin=rng.uniform(-2,2,3)
        p=np.vstack([origin,origin+F.T])
        inverse=np.linalg.inv(np.eye(3)+rng.normal(0,.03,(3,3)))
        weights=rng.normal(0,.5,(4,4))
        if i==16: weights[:]=weights[0]
        if i==17: weights[:,3]=weights[:,1]
        if i%3==0: weights[0]=0
        result.append(np.r_[p.ravel(),inverse.ravel(),weights.ravel(),2.3,5.1,.15,.04 if i%2 else 0,1/90,.2 if i%4 else 0])
    return np.array(result,dtype=np.float32)

def model(row):
    p=row[:12].reshape(4,3);inverse=row[12:21].reshape(3,3);w=row[21:37].reshape(4,4)
    mu,bulk,floor,tau,dt,hardening=row[37:]
    lam=bulk+mu/3;F=(p[1:]-p[0]).T;J=np.linalg.det(F);q=(w[1:]-w[0]).T
    jacF=np.array([np.outer(np.eye(3)[a],q[m]) for m in range(4) for a in range(3)])
    cof=J*np.linalg.inv(F).T
    jacJ=np.einsum('ab,iab->i',cof,jacF)
    barrier=0
    if J<2*floor:
        z=J;d=z-2*floor;barrier=-lam*(2*np.log(z/(2*floor))+4*d/z-d*d/(z*z))
    H=mu*np.einsum('iab,jab->ij',jacF,jacF)+(lam+barrier)*np.outer(jacJ,jacJ)
    G=F@inverse;dG=jacF@inverse
    dS=np.array([G.T@v+v.T@G for v in dG])
    H+=mu*tau/(4*dt)*np.einsum('iab,jab->ij',dS,dS)
    rest=np.vstack([np.zeros(3),np.eye(3)])
    for j in range(4):
        for k in range(j+1,4):
            e=p[j]-p[k];l=np.linalg.norm(e);l0=np.linalg.norm(rest[j]-rest[k]);strain=max(0,l/l0-1)
            if strain>0:
                force=3*mu*hardening*strain**5/l0;curve=15*mu*hardening*strain**4/l0**2;n=e/l
                A=force/l*np.eye(3)+(curve-force/l)*np.outer(n,n)
                B=np.concatenate([(w[j,m]-w[k,m])*np.eye(3) for m in range(4)],axis=1)
                H+=B.T@A@B
    def energy(z):
        points=p+w@z.reshape(4,3);f=(points[1:]-points[0]).T;j=np.linalg.det(f)
        e=.5*mu*(np.sum(f*f)-3)-mu*(j-1)+.5*lam*(j-1)**2
        if j<2*floor: e-=lam*(j-2*floor)**2*np.log(j/(2*floor))
        g=f@inverse;S=g.T@g-np.eye(3);e+=mu*tau/(8*dt)*np.sum(S*S)
        for a in range(4):
            for b in range(a+1,4):
                strain=max(0,np.linalg.norm(points[a]-points[b])/np.linalg.norm(rest[a]-rest[b])-1)
                e+=.5*mu*hardening*strain**6
        return e
    eps=1e-6
    grad=np.array([(energy(eps*np.eye(12)[i])-energy(-eps*np.eye(12)[i]))/(2*eps) for i in range(12)])
    return grad,H

def main():
    inputs=cases();out=np.zeros((len(inputs),126),np.float32);dll=ct.CDLL(str(Path(sys.argv[1]).resolve()))
    ptr=ct.POINTER(ct.c_float)
    dll.LCS_ProbeJointAlgebra.argtypes=(ct.c_int,ptr,ptr);dll.LCS_ProbeJointAlgebra.restype=ct.c_int
    assert dll.LCS_ProbeJointAlgebra(len(inputs),inputs.ctypes.data_as(ptr),out.ctypes.data_as(ptr))==0
    assert np.isfinite(out).all()
    maxima=dict(gradient=0.,curvature=0.,baseline_diagonal=0.,baseline_gradient=0.,negative_eigenvalue=0.)
    for index,row in enumerate(inputs.astype(np.float64)):
        expected_g,expected_H=model(row);H=np.zeros((12,12));at=12
        for i in range(12):
            for j in range(i,12): H[i,j]=H[j,i]=out[index,at];at+=1
        g=out[index,:12]
        maxima['gradient']=max(maxima['gradient'],np.max(np.abs(g-expected_g))/max(1,np.max(np.abs(expected_g))))
        maxima['curvature']=max(maxima['curvature'],np.max(np.abs(H-expected_H))/max(1,np.max(np.abs(expected_H))))
        eig=np.linalg.eigvalsh(H);maxima['negative_eigenvalue']=max(maxima['negative_eigenvalue'],max(0,-eig[0])/max(1,eig[-1]))
        for m in range(4):
            old=out[index,90+9*m:99+9*m];block=H[3*m:3*m+3,3*m:3*m+3]
            packed=block[[0,1,2,0,0,1],[0,1,2,1,2,2]]
            maxima['baseline_diagonal']=max(maxima['baseline_diagonal'],np.max(np.abs(packed-old[3:]))/max(1,np.max(np.abs(old[3:]))))
            maxima['baseline_gradient']=max(maxima['baseline_gradient'],np.max(np.abs(g[3*m:3*m+3]-old[:3]))/max(1,np.max(np.abs(old[:3]))))
    assert maxima['gradient']<2e-4,maxima
    assert maxima['curvature']<1e-4,maxima
    assert maxima['baseline_diagonal']<1e-4 and maxima['baseline_gradient']<1e-4,maxima
    assert maxima['negative_eigenvalue']<1e-6,maxima
    curves=[[1,.2,1,-4,4,0],[1,.2,1,-3,3,0],[1,.2,1,-5,8,-3],[1,.2,1,0,0,0]]
    for i in range(2000): curves.append([rng.uniform(.05,1.5),.2,rng.uniform(.3,2),*rng.uniform(-8,8,3)])
    expected=[];selected=[]
    for row in np.array(curves,dtype=np.float32):
        a,floor,j0,j1,j2,j3=map(float,row);points=[0,a]
        roots=np.roots([3*j3,2*j2,j1])
        points.extend(r.real for r in roots if abs(r.imag)<1e-10 and 0<r.real<a)
        low=min(((j3*x+j2)*x+j1)*x+j0 for x in points)
        if abs(low-floor)>1e-4: selected.append(row);expected.append(low>=floor)
    curves=np.array(selected,dtype=np.float32);answers=np.empty(len(curves),np.int32)
    dll.LCS_ProbeJointCubic.argtypes=(ct.c_int,ptr,ct.POINTER(ct.c_int));dll.LCS_ProbeJointCubic.restype=ct.c_int
    assert dll.LCS_ProbeJointCubic(len(curves),curves.ctypes.data_as(ptr),answers.ctypes.data_as(ct.POINTER(ct.c_int)))==0
    np.testing.assert_array_equal(answers,np.array(expected,dtype=np.int32))
    result=dict(tet_cases=len(inputs),normalized_max_errors={k:float(v) for k,v in maxima.items()},cubic_cases=len(curves),cubic_disagreements=0,endpoint_only_counterexample_rejected=bool(answers[0]==0),note='GN curvature includes cross modes; agreement is with an independent Jacobian oracle, not the exact indefinite Hessian.')
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
