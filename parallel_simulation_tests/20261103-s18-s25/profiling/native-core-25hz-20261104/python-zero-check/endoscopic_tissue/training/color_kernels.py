"""One-pass packing with NumPy-owned, independent contiguous HWC RGB output."""
import numpy as np
from numba import njit

@njit(nogil=True,cache=False)
def _pack_rgb_into(image,result):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            result[y,x,0]=image[y,x,0]
            result[y,x,1]=image[y,x,1]
            result[y,x,2]=image[y,x,2]

def pack_rgb(image):
    result=np.empty((image.shape[0],image.shape[1],3),dtype=np.uint8)
    _pack_rgb_into(image,result)
    return result
