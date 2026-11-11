"""Classify absolute-envelope exits without hiding deformation or attached-node motion."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def intact_material_cell(grid,obs,members):
    # Conservative volume witness: an interior lattice cell with all eight
    # material corners and all twelve reciprocal edges still present. Four
    # material-labelled points alone can be a planar/unsupported remnant.
    material=np.asarray(grid.inside,dtype=bool)
    seeds=np.flatnonzero(members&material)
    if not len(seeds):return None
    neighbors=np.asarray(obs.neighbors)
    corners=np.full((len(seeds),8),-1,np.int64);corners[:,0]=seeds
    for target,source,slot in [(1,0,1),(2,0,3),(4,0,5),(3,1,3),(5,1,5),(6,2,5),(7,3,5)]:
        valid=corners[:,source]>=0
        corners[valid,target]=neighbors[corners[valid,source],slot]
    corners=corners[np.all(corners>=0,axis=1)]
    if not len(corners):return None
    keep=np.all(members[corners]&material[corners],axis=1)
    for axis in range(3):
        bit=1<<axis
        for local in range(8):
            if local&bit:continue
            lower=corners[:,local];upper=corners[:,local|bit]
            keep&=(neighbors[lower,2*axis+1]==upper)&(neighbors[upper,2*axis]==lower)
    corners=corners[keep]
    if not len(corners):return None
    rest=np.asarray(grid.positions,dtype=np.float64)
    edges=rest[corners[:,[1,2,4]]]-rest[corners[:,0,None]]
    valid=np.flatnonzero(np.abs(np.linalg.det(edges))>1e-12)
    return corners[valid[0]].tolist() if len(valid) else None

def audit_detached_flight(grid,obs,displacement):
    active=np.asarray(grid.active,dtype=bool);material=np.asarray(grid.inside,dtype=bool)
    ids=np.flatnonzero(active);rows=np.repeat(ids,6);cols=obs.neighbors[ids].reshape(-1)
    keep=(cols>=0)&active[np.maximum(cols,0)]
    graph=coo_matrix((np.ones(keep.sum(),np.int8),(rows[keep],cols[keep])),shape=(len(active),)*2)
    labels=connected_components(graph,directed=False)[1];events=[]
    for root in np.unique(labels[active&(displacement>60)]):
        members=active&(labels==root);mat=members&material
        result=dict(root=int(root),active_nodes=int(members.sum()),material_nodes=int(mat.sum()),absolute_max_mm=float(displacement[members].max()))
        if np.any(grid.pinned[members]) or np.any(obs.grip_flags[members]):
            result.update(allowed=False,reason='A pinned or captured component left the absolute envelope');events.append(result);continue
        if mat.sum()<4:
            result.update(allowed=False,reason='Unsupported isolated remnants retain the absolute guard');events.append(result);continue
        cell=intact_material_cell(grid,obs,members)
        if cell is None:
            result.update(allowed=False,reason='No intact interior material cell; retain the absolute guard');events.append(result);continue
        result['intact_material_cell']=cell
        reference=grid.positions[members].astype(np.float64);current=obs.nodes[members].astype(np.float64)
        x=reference-reference.mean(axis=0);y=current-current.mean(axis=0)
        U,S,Vt=np.linalg.svd(x.T@y);rotation=U@Vt
        if np.linalg.det(rotation)<0:U[:,-1]*=-1;rotation=U@Vt
        residual=np.linalg.norm(x@rotation-y,axis=1)*10
        result.update(allowed=bool(residual.max()<=60),reason='Detached volume: absolute flight and intrinsic deformation reported separately',
            intrinsic_max_mm=float(residual.max()),intrinsic_rms_mm=float(np.sqrt(np.mean(residual**2))),com_mm=(current.mean(axis=0)*10).tolist())
        events.append(result)
    return bool(events) and all(e['allowed'] for e in events),events
