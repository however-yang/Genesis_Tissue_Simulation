"""Bounded worker-local image transport; public observations still own CPU arrays.

One outstanding request per worker is enforced by ParallelNativePool. The worker
publishes a generation only after copying all images. The parent copies them into
ordinary independent arrays before returning (and before another action can be
sent). No shared view escapes the reader, so retaining observations is safe.
"""
from multiprocessing.shared_memory import SharedMemory
import struct
import time
import numpy as np

KEYS={'rgb':np.dtype('uint8'),'depth_m':np.dtype('float32'),'depth_valid':np.dtype('bool')}
HEADER=64

class ImageBuffer:
    def __init__(self,size):
        if type(size) is not int or not HEADER<size<=64*1024**2:raise ValueError('Image buffer budget must be 65 bytes..64 MiB')
        self.memory=SharedMemory(create=True,size=size)
        self.size=size
        self.memory.buf[:HEADER]=bytes(HEADER)
    @property
    def spec(self):return dict(name=self.memory.name,size=self.size)
    def close(self):
        try:self.memory.close()
        finally:
            try:self.memory.unlink()
            except FileNotFoundError:pass

class ImageWriter:
    def __init__(self,connection,spec):
        self.connection=connection;self.memory=SharedMemory(name=spec['name']);self.size=spec['size']
        if self.memory.size!=self.size:
            self.memory.close();raise ValueError('Shared image size mismatch')
    def recv(self):return self.connection.recv()
    def send(self,message):
        obs=message.get('observation')
        if not isinstance(obs,dict) or not any(k in obs for k in KEYS):return self.connection.send(message)
        started=time.perf_counter();entries=[];offset=HEADER
        for key,dtype in KEYS.items():
            if key not in obs:continue
            value=obs[key]
            if not isinstance(value,np.ndarray) or value.dtype!=dtype or value.ndim not in (2,3):raise ValueError('Unexpected image array: '+key)
            if key=='rgb' and (value.ndim!=3 or value.shape[-1]!=3):raise ValueError('Expected HWC RGB')
            if key!='rgb' and value.ndim!=2:raise ValueError('Expected 2D depth/mask')
            offset=(offset+63)//64*64
            entries.append(dict(key=key,offset=offset,shape=list(value.shape),dtype=dtype.str,nbytes=value.nbytes));offset+=value.nbytes
        if offset>self.size:raise MemoryError('Observation exceeds explicit shared image budget; frame not dropped')
        generation=message['request']
        if type(generation) is not int or not 0<generation<2**64:raise ValueError('Invalid image generation')
        struct.pack_into('<Q',self.memory.buf,0,0)
        for item in entries:
            target=np.ndarray(item['shape'],dtype=np.dtype(item['dtype']),buffer=self.memory.buf,offset=item['offset'])
            np.copyto(target,obs[item['key']]);del target
        struct.pack_into('<Q',self.memory.buf,0,generation)
        forwarded=dict(message);forwarded['observation']={k:v for k,v in obs.items() if k not in KEYS}
        forwarded['_shared_images']=dict(schema=1,generation=generation,entries=entries)
        forwarded['image_transport']=dict(mode='shared_memory_owned_copy',bytes=sum(x['nbytes'] for x in entries),worker_copy_ms=(time.perf_counter()-started)*1000)
        return self.connection.send(forwarded)
    def close(self):
        try:self.connection.close()
        finally:self.memory.close()

class ImageReader:
    def __init__(self,connection,owner):self.connection=connection;self.owner=owner
    def fileno(self):return self.connection.fileno()
    def poll(self,*args):return self.connection.poll(*args)
    def send(self,*args):return self.connection.send(*args)
    def recv(self):
        message=self.connection.recv();descriptor=message.pop('_shared_images',None)
        if descriptor is None:return message
        started=time.perf_counter();memory=self.owner.memory
        if descriptor.get('schema')!=1 or descriptor.get('generation')!=message.get('request'):raise ValueError('Wrong image message generation')
        generation=descriptor['generation']
        if struct.unpack_from('<Q',memory.buf,0)[0]!=generation:raise ValueError('Stale image generation before copy')
        entries=descriptor.get('entries',[])
        if len(entries)>len(KEYS):raise ValueError('Too many image arrays')
        seen=set();previous_end=HEADER;obs=message['observation']
        for item in entries:
            key=item['key'];shape=item['shape'];dtype=np.dtype(item['dtype']);offset=item['offset']
            if key not in KEYS or key in seen or key in obs or dtype!=KEYS[key]:raise ValueError('Invalid image key/dtype')
            if (not isinstance(shape,list) or len(shape)!=(3 if key=='rgb' else 2)
                    or any(type(n) is not int or not 0<n<=16384 for n in shape)
                    or (key=='rgb' and shape[-1]!=3)):raise ValueError('Invalid image shape')
            count=1
            for n in shape:count*=n
            size=count*dtype.itemsize
            if type(offset) is not int or offset<previous_end or offset%64 or size!=item['nbytes'] or offset+size>self.owner.size:raise ValueError('Image extent exceeds buffer')
            obs[key]=np.ndarray(shape,dtype=dtype,buffer=memory.buf,offset=offset).copy()
            previous_end=offset+size;seen.add(key)
        if struct.unpack_from('<Q',memory.buf,0)[0]!=generation:raise ValueError('Image overwritten during copy')
        message['image_transport']['parent_copy_ms']=(time.perf_counter()-started)*1000
        return message
    def close(self):self.connection.close()
