"""Same positional topology criteria using a read-only current-soup CUDA ABI."""
import ctypes
import numpy as np


def decode(metrics,bounds):
    wall=int(metrics[1]);box=bounds.reshape(2,3)
    return dict(triangles=int(metrics[0]),wall_triangles=wall,degenerate_triangles=int(metrics[2]),
                positional_boundary_edges=int(metrics[3]),positional_nonmanifold_edges=int(metrics[4]),
                duplicate_triangles=int(metrics[5]),winding_consistent=bool(metrics[6]),
                wall_bounds_native_mm=(box*np.float32(10)).tolist() if wall else None,
                wall_extent_mm_not_incision_depth=((box[1]-box[0])*np.float32(10)).tolist() if wall else None)


def topology_statistics(native,surface):
    # The environment passes the unmodified result of native.surface from this
    # same completed tick; the ABI additionally checks the current vertex count.
    n=len(surface.vertices)
    if n<=0 or n>1<<20 or surface.triangles.size!=n or not np.array_equal(surface.triangles.ravel(),np.arange(n,dtype=np.int32)):
        from .topology_fast import topology_statistics_fast
        return topology_statistics_fast(surface)
    function=native.dll.LCS_GetExactTopology
    function.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];function.restype=ctypes.c_int
    metrics=np.empty(9,np.int32);bounds=np.empty(6,np.float32)
    status=function(n,metrics.ctypes.data,bounds.ctypes.data)
    if status:raise RuntimeError('Current reconstructed topology readback failed: '+str(status))
    if metrics[7] or metrics[8]:
        from .topology_fast import topology_statistics_fast
        return topology_statistics_fast(surface)
    return decode(metrics,bounds)
