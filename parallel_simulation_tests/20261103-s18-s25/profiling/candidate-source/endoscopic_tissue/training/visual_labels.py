"""Privileged geometry labels, never deployment observations.

The surface ABI returns native contact-affine bindings including an offset.
The pixel composition below requires an exact visible triangle/barycentric pass;
texture rest anchors and nearest nodes are deliberately not substitutes.
"""
import ctypes as ct
import numpy as np
import cv2


def surface_supports(native, surface):
    try:
        version=native.dll['LCS_SurfaceBindingsAbiVersion']
        export=native.dll['LCS_GetSurfaceBindings']
    except AttributeError as error:
        raise RuntimeError('Explicit surface-binding ABI v2 library required') from error
    version.argtypes=[];version.restype=ct.c_int
    if version()!=2:raise ValueError('Surface-binding ABI mismatch')
    n=len(surface.vertices)
    ids=np.full((n,8),-1,np.int32);weights=np.zeros((n,8),np.float32)
    offsets=np.zeros((n,3),np.float32);identities=np.zeros((n,4),np.int32);valid=np.zeros(n,np.int32)
    export.argtypes=[ct.c_int,ct.POINTER(ct.c_int)]+[ct.c_void_p]*5;export.restype=ct.c_int
    written=ct.c_int()
    rc=export(n,ct.byref(written),*(a.ctypes.data_as(ct.c_void_p) for a in (ids,weights,offsets,identities,valid)))
    if rc or written.value!=n:raise RuntimeError(f'Surface-binding readback failed: {rc}, vertices {written.value}/{n}')
    return dict(node_ids=ids,weights=weights,offset_m=offsets*.01,identities=identities,
                valid=valid.astype(bool),triangles=surface.triangles.copy())


def compose_pixel_supports(triangle_index, barycentric, bindings):
    index=np.asarray(triangle_index)
    bary=np.asarray(barycentric,dtype=np.float64)
    if index.dtype.kind not in 'iu' or bary.shape!=index.shape+(3,):
        raise ValueError('Expected integer visible triangle IDs and perspective-correct barycentrics')
    flat=index.reshape(-1);bary=bary.reshape(-1,3)
    triangles=np.asarray(bindings['triangles'])
    from .pixel_compose_fast import compose
    ids,weights,offset,valid=compose(flat,bary,triangles,
        np.asarray(bindings['node_ids']),np.asarray(bindings['weights']),
        np.asarray(bindings['offset_m']),np.asarray(bindings['valid']))
    return dict(node_ids=ids.reshape(index.shape+(24,)),weights=weights.reshape(index.shape+(24,)),
                offset_m=offset.reshape(index.shape+(3,)),valid=valid.reshape(index.shape))


def segmentation(view):
    context=view.scene.visualizer.context
    old=context.seg_node_map.copy()
    try:
        # Background also participates in depth occlusion with class zero.
        context.seg_node_map.clear()
        context.seg_node_map.update({node:np.array([0,0,0],np.int32) for node in context._scene.mesh_nodes})
        for mesh in view.tissue_meshes:
            if mesh is not None:context.seg_node_map[context.external_nodes[mesh.name]]=np.array([0,0,1],np.int32)
        for node in (*view.grasper_jaws,*view.grasper_body):
            context.seg_node_map[node]=np.array([0,0,2],np.int32)
        for node in (*getattr(view,'scissors_blades',()),*getattr(view,'scissors_body',())):
            context.seg_node_map[node]=np.array([0,0,3],np.int32)
        _,_,labels,_=view.camera.render(rgb=False,depth=False,segmentation=True,colorize_seg=False,force_render=True)
    finally:
        context.seg_node_map.clear();context.seg_node_map.update(old)
    xy,_=cv2.convertMaps(*view.lens_map,cv2.CV_32FC2)
    mapped=cv2.remap(np.asarray(labels,dtype=np.float32)[::-1],xy,None,cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return np.where(view.vignette[...,0]>0,mapped,0).astype(np.uint8)
