"""Signed clearance of explicit rigid instrument envelopes, in metres.

Blade/jaw envelopes enclose the original visible meshes; the wrist capsule is
the same one used by the tissue-contact diagnostic. This does not add contact
forces. Negative clearance must pass nominal-trajectory regression before use.
"""
from dataclasses import dataclass
import numpy as np
from ..instrument_geometry import SCISSORS_WRIST_FRONT

SIGNS = ((np.arange(8)[:, None] >> np.arange(3)) & 1) * 2 - 1
EDGE_IDS = np.array([(i, i ^ (1 << k)) for i in range(8) for k in range(3) if not i & (1 << k)])


@dataclass(frozen=True)
class Box:
    center: np.ndarray
    frame: np.ndarray
    half: np.ndarray

    def vertices(self):
        return self.center + (SIGNS*self.half) @ self.frame


def _segment_pairs(a, b, c, d):
    # All segment pairs, with endpoint candidates for parallel/degenerate cases.
    u, v = b-a, d-c
    w = a[:, None, :]-c[None, :, :]
    aa, cc = np.sum(u*u, axis=1)[:, None], np.sum(v*v, axis=1)[None, :]
    bb = u @ v.T
    dd = np.einsum('ik,ijk->ij', u, w)
    ee = np.einsum('jk,ijk->ij', v, w)
    det = aa*cc-bb*bb
    s = np.divide(bb*ee-cc*dd, det, out=np.zeros_like(det), where=det > 1e-24)
    t = np.divide(aa*ee-bb*dd, det, out=np.zeros_like(det), where=det > 1e-24)
    candidates = [(s, t, (det > 1e-24)&(s >= 0)&(s <= 1)&(t >= 0)&(t <= 1))]
    for fixed in (0., 1.):
        candidates.append((np.full_like(det, fixed), np.clip((ee+fixed*bb)/np.maximum(cc,1e-30),0,1), True))
        candidates.append((np.clip((fixed*bb-dd)/np.maximum(aa,1e-30),0,1), np.full_like(det,fixed), True))
    best = np.inf
    for s,t,mask in candidates:
        delta = w+s[...,None]*u[:,None,:]-t[...,None]*v[None,:,:]
        best = min(best, float(np.min(np.where(mask, np.sum(delta*delta,axis=-1), np.inf))))
    return np.sqrt(max(0.,best))


def box_distance(a, b):
    axes = np.concatenate((a.frame,b.frame,np.cross(a.frame[:,None,:],b.frame[None,:,:]).reshape(-1,3)))
    length = np.linalg.norm(axes,axis=1)
    axes = axes[length > 1e-10]/length[length > 1e-10,None]
    separation = np.abs(axes@(b.center-a.center))-np.abs(axes@a.frame.T)@a.half-np.abs(axes@b.frame.T)@b.half
    maximum = float(separation.max())
    if maximum <= 0: return maximum  # exact translational SAT depth for boxes
    av, bv = a.vertices(), b.vertices()
    distances = []
    for points, box in ((av,b),(bv,a)):
        local = (points-box.center)@box.frame.T
        distances.append(float(np.linalg.norm(np.maximum(np.abs(local)-box.half,0),axis=1).min()))
    distances.append(_segment_pairs(av[EDGE_IDS[:,0]],av[EDGE_IDS[:,1]],bv[EDGE_IDS[:,0]],bv[EDGE_IDS[:,1]]))
    return min(distances)


def capsule_box_distance(start, end, radius, box):
    # Exact distance from a line segment to an AABB: each interval between
    # face crossings has a quadratic squared distance with a known minimum.
    a, b = (start-box.center)@box.frame.T, (end-box.center)@box.frame.T
    u = b-a
    cuts = [0.,1.]
    for k in range(3):
        if abs(u[k]) > 1e-15:
            cuts.extend(float(t) for t in ((-box.half[k]-a[k])/u[k],(box.half[k]-a[k])/u[k]) if 0 < t < 1)
    cuts = np.unique(cuts)
    samples = list(cuts)
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        middle = a+(lo+hi)*.5*u
        outside = np.abs(middle) > box.half
        v = np.where(outside,u,0)
        c = np.where(outside,a-np.sign(middle)*box.half,0)
        if v@v > 1e-30: samples.append(float(np.clip(-(v@c)/(v@v),lo,hi)))
    points = a+np.asarray(samples)[:,None]*u
    distance = float(np.linalg.norm(np.maximum(np.abs(points)-box.half,0),axis=1).min())
    return distance-radius  # when centerline is inside, -radius is a depth lower bound


def jaw_boxes(geometry, hinge, axis, normal, gap):
    axis, normal, hinge = np.asarray(axis),np.asarray(normal),np.asarray(hinge)
    if not np.isfinite(np.r_[hinge,axis,normal,gap]).all(): raise ValueError('Nonfinite instrument pose')
    if not np.allclose(np.stack((axis,normal))@np.stack((axis,normal)).T,np.eye(2),atol=1e-5):
        raise ValueError('Instrument axes must be orthonormal')
    sn = np.clip(gap/(2*geometry.length),0,.95); cs = np.sqrt(1-sn*sn)
    boxes=[]
    for side, vertices in zip((1,-1),geometry.vertices):
        low,high = vertices.min(axis=0),vertices.max(axis=0)
        frame = np.stack((axis*cs+side*normal*sn,normal*cs-side*axis*sn,np.cross(axis,normal)))
        boxes.append(Box((hinge+(low+high)*.5@frame)*.01,frame,(high-low)*.005))
    return boxes


def instrument_distance(grasper, scissors, *, grasper_hinge, grasper_axis,
                        grasper_normal, grasper_gap, scissors_hinge, scissors_axis,
                        scissors_normal, scissors_gap):
    a = jaw_boxes(grasper,grasper_hinge,grasper_axis,grasper_normal,grasper_gap)
    b = jaw_boxes(scissors.jaws,scissors_hinge,scissors_axis,scissors_normal,scissors_gap)
    start = (np.asarray(scissors_hinge)-scissors.jaws.shaft_length*np.asarray(scissors_axis))*.01
    end = (np.asarray(scissors_hinge)+SCISSORS_WRIST_FRONT*np.asarray(scissors_axis))*.01
    return min(*(box_distance(x,y) for x in a for y in b),
               *(capsule_box_distance(start,end,scissors.jaws.shaft_radius*.01,x) for x in a))
