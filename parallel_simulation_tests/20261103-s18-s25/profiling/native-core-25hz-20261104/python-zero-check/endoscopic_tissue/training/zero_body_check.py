"""Exact-zero shortcut for diagnostic envelopes; no contact forces are changed.

Finite samples certified farther outside than a conservative float32 arithmetic
error bound return the exact reference zero. Any possible contact, ambiguous
rounding, nonfinite or unsupported input uses the unchanged NumPy calculations.
The existing sample selection/centroids/midpoints remain unchanged.
"""
import numpy as np
from numba import njit
enabled=False
validate=False
calls=clear_returns=fallbacks=validated_clear=mismatches=0

@njit(cache=False,nogil=True)
def definitely_clear(relative,axis,frame0,frame1,box0,box1,orientation0,orientation1,shaft_length,front,radius):
    if len(relative)==0:return False
    axis_norm=0.;frame_norm=0.;orientation_norm=0.;box_norm=0.;relative_norm=0.
    for i in range(3):
        if not np.isfinite(axis[i]):return False
        axis_norm+=abs(np.float64(axis[i]))
        for j in range(3):
            for value in (np.float64(frame0[i,j]),np.float64(frame1[i,j]),np.float64(orientation0[i,j]),np.float64(orientation1[i,j])):
                if not np.isfinite(value):return False
            frame_norm=max(frame_norm,abs(np.float64(frame0[i,j])),abs(np.float64(frame1[i,j])))
            orientation_norm=max(orientation_norm,abs(np.float64(orientation0[i,j])),abs(np.float64(orientation1[i,j])))
        for value in (box0[i],box1[i],box0[i+3],box1[i+3]):
            if not np.isfinite(value):return False
            box_norm=max(box_norm,abs(np.float64(value)))
    if axis_norm>4 or frame_norm>4 or orientation_norm>4:return False
    if not np.isfinite(shaft_length+front+radius) or min(shaft_length,front,radius)<0:return False
    body=-np.inf;wrist=-np.inf
    for i in range(len(relative)):
        x=np.float64(relative[i,0]);y=np.float64(relative[i,1]);z=np.float64(relative[i,2])
        if not np.isfinite(x+y+z):return False
        relative_norm=max(relative_norm,abs(x),abs(y),abs(z))
        if relative_norm>1e5:return False
        for which in range(2):
            frame=frame0 if which==0 else frame1;box=box0 if which==0 else box1;orientation=orientation0 if which==0 else orientation1
            # float64 working values only certify a strict outside margin.
            lx=(x*np.float64(frame[0,0])+y*np.float64(frame[0,1])+z*np.float64(frame[0,2]))-np.float64(box[0])
            ly=(x*np.float64(frame[1,0])+y*np.float64(frame[1,1])+z*np.float64(frame[1,2]))-np.float64(box[1])
            lz=(x*np.float64(frame[2,0])+y*np.float64(frame[2,1])+z*np.float64(frame[2,2]))-np.float64(box[2])
            depth=np.inf
            for k in range(3):
                local=lx*np.float64(orientation[k,0])+ly*np.float64(orientation[k,1])+lz*np.float64(orientation[k,2])
                depth=min(depth,np.float64(box[k+3])-abs(local))
            body=max(body,depth)
        along=x*np.float64(axis[0])+y*np.float64(axis[1])+z*np.float64(axis[2]);along=min(front,max(-shaft_length,along))
        dx=x-along*np.float64(axis[0]);dy=y-along*np.float64(axis[1]);dz=z-along*np.float64(axis[2])
        wrist=max(wrist,radius-np.sqrt(dx*dx+dy*dy+dz*dz))
    # Upper bound covers both three-term dot products, translations, clipping,
    # norm evaluation and any legal float32 dot-product summation/FMA ordering.
    # It only decides fallback, never changes an acceptance/contact threshold.
    scale=1.+(3.*relative_norm*frame_norm+box_norm)*3.*orientation_norm+box_norm
    scale=max(scale,1.+relative_norm*(1.+axis_norm*axis_norm)+abs(shaft_length)+abs(front)+abs(radius))
    error=128.*np.finfo(np.float32).eps*scale
    return body < -error and wrist < -error
