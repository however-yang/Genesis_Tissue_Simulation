"""Compiled CPU hash counting for the unchanged positional topology predicates."""
import numpy as np
from numba import njit

@njit(cache=False)
def mix(x):
    x=np.uint64(x)
    x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    return x^(x>>np.uint64(31))

@njit(cache=False)
def capacity(n):
    size=8
    while size<n*2:size*=2
    return size

@njit(cache=False)
def count_topology(rounded,triangles):
    table=np.full(capacity(len(rounded)),-1,np.int64)
    inverse=np.empty(len(rounded),np.int64);n=0
    for i in range(len(rounded)):
        x,y,z=rounded[i]
        h=mix(np.uint64(x))^mix(np.uint64(y)+np.uint64(0x9e3779b97f4a7c15))^mix(np.uint64(z)+np.uint64(0x517cc1b727220a95))
        slot=np.int64(h&np.uint64(len(table)-1))
        while table[slot]>=0:
            j=table[slot]
            if rounded[j,0]==x and rounded[j,1]==y and rounded[j,2]==z:break
            slot=(slot+1)&(len(table)-1)
        if table[slot]<0:table[slot]=i;inverse[i]=n;n+=1
        else:inverse[i]=inverse[table[slot]]
    if n>=2**21:raise ValueError('Packed face keys exceed signed 64-bit range')
    size=capacity(3*len(triangles))
    keys=np.full(size,-1,np.int64);counts=np.zeros(size,np.int32);direction=np.zeros(size,np.int32)
    face_keys=np.full(capacity(len(triangles)),-1,np.int64);duplicates=0
    for f in triangles:
        a,b,c=inverse[f[0]],inverse[f[1]],inverse[f[2]]
        for first,second in ((a,b),(b,c),(c,a)):
            lo,hi=min(first,second),max(first,second);key=lo*n+hi
            slot=np.int64(mix(key)&np.uint64(size-1))
            while keys[slot]>=0 and keys[slot]!=key:slot=(slot+1)&(size-1)
            keys[slot]=key;counts[slot]+=1
            direction[slot]+=1 if first<second else (-1 if first>second else 0)
        lo=min(a,b,c);hi=max(a,b,c);mid=a+b+c-lo-hi
        key=(lo*n+mid)*n+hi;slot=np.int64(mix(key)&np.uint64(len(face_keys)-1))
        while face_keys[slot]>=0 and face_keys[slot]!=key:slot=(slot+1)&(len(face_keys)-1)
        if face_keys[slot]>=0:duplicates+=1
        face_keys[slot]=key
    boundary=0;nonmanifold=0;winding=True
    for i in range(size):
        if counts[i]==1:boundary+=1
        elif counts[i]>2:nonmanifold+=1
        elif counts[i]==2 and direction[i]!=0:winding=False
    return boundary,nonmanifold,duplicates,winding
