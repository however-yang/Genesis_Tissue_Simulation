"""Read-only frame bracket for the two exported cutting edges and new material cuts.

This never straightens an incision or changes native fracture decisions. Endpoint
samples bracket a whole native step; they are not substep contact proof.
"""
import numpy as np


def closest_edge_points(edges):
    a,b = np.asarray(edges[0],float)
    c,d = np.asarray(edges[1],float)
    u,v,w = b-a,d-c,a-c
    uu,vv,uv,uw,vw = np.dot(u,u),np.dot(v,v),np.dot(u,v),np.dot(u,w),np.dot(v,w)
    if min(uu,vv) <= 0 or not np.isfinite((uu,vv,uv,uw,vw)).all():
        raise ValueError('Invalid exported cutting edges')
    candidates=[]
    for s in (0.,1.): candidates.append((s,float(np.clip((vw+s*uv)/vv,0,1))))
    for t in (0.,1.): candidates.append((float(np.clip((t*uv-uw)/uu,0,1)),t))
    det = uu*vv-uv*uv
    if det > 1e-12*uu*vv:
        s,t = (uv*vw-vv*uw)/det,(uu*vw-uv*uw)/det
        if 0 <= s <= 1 and 0 <= t <= 1: candidates.append((s,t))
    s,t = min(candidates,key=lambda st:np.linalg.norm(w+st[0]*u-st[1]*v))
    first,second = a+s*u,c+t*v
    return first,second


def cut_origin_bracket(before, after, material_mask):
    prior, current = before.topology['neighbors'],after.topology['neighbors']
    owner, dim = np.where((prior[:,1::2] >= 0) & (current[:,1::2] < 0))
    slot = 1+2*dim
    target = prior[owner,slot]
    material = np.asarray(material_mask,bool)
    keep = material[owner] & material[target]
    owner,target,slot = owner[keep],target[keep],slot[keep]
    if not len(owner): return None
    fraction = np.asarray(after.topology['cut_fractions'])[owner,slot]
    known = np.isfinite(fraction) & (fraction >= 0) & (fraction <= 1)
    fractions = np.where(known,fraction,.5)
    rows=[]
    for label,state in [('before',before),('after',after)]:
        first,second = closest_edge_points(state.metadata['tools']['scissors_edges_m'])
        crossing = .5*(first+second)
        points = (1-fractions[:,None])*state.arrays['positions_m'][owner]+fractions[:,None]*state.arrays['positions_m'][target]
        distances = np.linalg.norm(points-crossing,axis=1)
        rows.append({'sample':label,'closest_edge_points_m':[first.tolist(),second.tolist()],
                     'edge_separation_m':float(np.linalg.norm(first-second)),
                     'min_new_cut_distance_to_edge_midpoint_m':float(distances.min()),
                     'max_new_cut_distance_to_edge_midpoint_m':float(distances.max())})
    return {'new_material_cut_edges':len(owner),'material_edge_ids':np.stack((owner,target),axis=1).tolist(),
            'cut_fraction_known':known.tolist(),'samples':rows,
            'scope':'exported straight edge segments and frame endpoints; not true curved-edge/substep certification',
            'criterion':'cut initiation near the two edges meeting current tissue; incision need not be straight',
            'automatic_acceptance':False}
