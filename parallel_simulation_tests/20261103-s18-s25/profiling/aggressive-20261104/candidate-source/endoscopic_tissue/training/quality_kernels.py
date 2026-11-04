"""Fused CPU diagnostics over arrays already required by the training record.

No fast-math, changed thresholds, or temporal caching. Float32 arithmetic matches
the NumPy reference's subtraction, squared norm, sqrt, and unit conversion.
"""
import numpy as np
from numba import njit

@njit(cache=False,nogil=True)
def node_extrema(nodes, previous, rest, active, speed_scale):
    finite = True
    displacement = np.float32(0)
    speed = np.float32(0)
    displacement_nan = False
    speed_nan = False
    for i in range(len(nodes)):
        for k in range(3):
            if not np.isfinite(nodes[i, k]):
                finite = False
        if active[i] == 0:
            continue
        d0 = nodes[i, 0] - rest[i, 0]
        d1 = nodes[i, 1] - rest[i, 1]
        d2 = nodes[i, 2] - rest[i, 2]
        distance = np.sqrt((d0*d0 + d1*d1) + d2*d2) * np.float32(.01)
        displacement_nan |= np.isnan(distance)
        displacement = max(displacement, distance)
        if len(previous):
            d0 = nodes[i, 0] - previous[i, 0]
            d1 = nodes[i, 1] - previous[i, 1]
            d2 = nodes[i, 2] - previous[i, 2]
            value = np.sqrt((d0*d0 + d1*d1) + d2*d2) * speed_scale
            speed_nan |= np.isnan(value)
            speed = max(speed, value)
    if displacement_nan:
        displacement = np.float32(np.nan)
    if speed_nan:
        speed = np.float32(np.nan)
    return finite, displacement, speed

@njit(cache=False,nogil=True)
def connected_material(neighbors, material):
    # Same undirected positive-axis material graph as scipy.connected_components.
    parent = np.full(len(material), -1, np.int32)
    size = np.ones(len(material), np.int32)
    count = 0
    for i in range(len(material)):
        if material[i]:
            parent[i] = i
            count += 1
    for i in range(len(material)):
        if parent[i] < 0:
            continue
        for slot in (1, 3, 5):
            j = neighbors[i, slot]
            if j < 0 or not material[j]:
                continue
            a, b = i, j
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            if a != b:
                if size[a] < size[b]:
                    a, b = b, a
                parent[b] = a
                size[a] += size[b]
                count -= 1
    return count

@njit(cache=False,nogil=True)
def nearby_surface(vertices, triangles, hinge, frame, lows, highs):
    """Conservative broad phase only; original narrow-phase samples stay intact."""
    coords = np.empty((len(vertices),3), np.float64)
    near_vertices = np.zeros(len(vertices), np.bool_)
    for i in range(len(vertices)):
        x=vertices[i,0]-hinge[0];y=vertices[i,1]-hinge[1];z=vertices[i,2]-hinge[2]
        for k in range(3):
            coords[i,k]=(frame[k,0]*x+frame[k,1]*y)+frame[k,2]*z
        for box in range(len(lows)):
            inside=True
            for k in range(3):
                inside &= coords[i,k]>=lows[box,k]-1e-12 and coords[i,k]<=highs[box,k]+1e-12
            near_vertices[i] |= inside
    near_vertices[:min(2,len(vertices))]=True
    near_triangles = np.zeros(len(triangles), np.bool_)
    for i in range(len(triangles)):
        a,b,c=triangles[i]
        for box in range(len(lows)):
            inside=True
            for k in range(3):
                low=min(coords[a,k],coords[b,k],coords[c,k]);high=max(coords[a,k],coords[b,k],coords[c,k])
                inside &= high>=lows[box,k]-1e-12 and low<=highs[box,k]+1e-12
            near_triangles[i] |= inside
    return near_vertices, near_triangles
