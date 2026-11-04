"""One-pass uint8 packing; preserves the public independent contiguous HWC RGB."""
import numpy as np
from numba import njit
@njit(nogil=True,cache=False)
def pack_rgb(image):
    result=np.empty((image.shape[0],image.shape[1],3),np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            result[y,x,0]=image[y,x,0]
            result[y,x,1]=image[y,x,1]
            result[y,x,2]=image[y,x,2]
    return result
