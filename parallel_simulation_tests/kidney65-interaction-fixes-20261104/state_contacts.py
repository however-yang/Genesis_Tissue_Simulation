"""Read-only LCSTATE1/contact inspection, validated against live observations.

This decoder never changes an archive or passes a modified buffer to the DLL.
Only the pinned ABI1 numeric layout is supported. The native loader and public
snapshot codec perform their own integrity checks before this inspection.
"""
import struct
import numpy as np


def name_hash(name):
    value=name.encode();h=1469598103934665603 ^ len(value)
    for i in range(0,len(value)//8*8,8):
        h=((h ^ int.from_bytes(value[i:i+8],'little'))*1099511628211)&((1<<64)-1)
    tail=value[len(value)//8*8:]
    return ((h ^ int.from_bytes(tail,'little'))*1099511628211)&((1<<64)-1)


class Archive:
    def __init__(self,raw):
        self.raw=memoryview(raw);self.allocations={};self.fields={}
        magic,total,_,_,abi,allocations,fields,reserved=struct.unpack_from('<QQQQIIII',raw)
        assert magic==0x314554415453434c and total==len(raw) and abi==1 and reserved==0
        assert 0<allocations<4096 and 0<fields<4096
        offset=48
        for _ in range(allocations):
            key,_,size,stride,pointers,reserved=struct.unpack_from('<QQQQII',raw,offset)
            assert reserved==0 and pointers<512 and key not in self.allocations
            offset+=40+4*pointers
            assert size<=len(raw)-offset
            self.allocations[key]=self.raw[offset:offset+size];offset+=size
        for _ in range(fields):
            name,size,stride,pointers,item_bytes=struct.unpack_from('<QQQII',raw,offset)
            assert pointers<512 and name not in self.fields
            offset+=32+4*pointers
            assert size<=len(raw)-offset
            self.fields[name]=self.raw[offset:offset+size];offset+=size
        assert offset==len(raw)

    def field(self,name):return self.fields[name_hash(name)]

    def pointer(self,name):
        data=self.field(name);assert len(data)==8
        token,=struct.unpack('<Q',data);assert token
        allocation=self.allocations[token>>32];offset=token&0xffffffff
        assert offset<len(allocation)
        return allocation[offset:]


# cuda float3 has 4-byte alignment. Assert the full 468-byte contact stride
# independently from the allocation capacity, then verify exported positions
# exactly and every active ID/count/kind before interpreting any contact.
CONTACT=np.dtype(dict(names=['ids','count','kind','weights','offset','normal','target',
    'owner','component','mask','tool_origin','gap_gradient','tool_gap'],
    formats=[('<i4',24),'<i4','<i4',('<f4',24),('<f4',3),('<f4',3),('<f4',3),
             ('<i4',3),('<i4',3),('<u4',3),('<f4',3),('<f4',3),'<f4'],
    offsets=[0,96,100,104,200,224,236,280,292,304,316,328,340],itemsize=468))


def blade_contacts(raw,observed):
    archive=Archive(raw)
    n,=struct.unpack('<i',archive.field('xpbd.g_N'))
    capacity,=struct.unpack('<i',archive.field('xpbd.g_bladeCapacity'))
    positions=np.frombuffer(archive.pointer('xpbd.d_pos'),dtype='<f4').reshape(n,3)
    np.testing.assert_array_equal(positions,observed.nodes)
    assert len(archive.pointer('xpbd.d_bladeContacts'))==6*capacity*CONTACT.itemsize
    count,=struct.unpack_from('<i',archive.pointer('xpbd.d_bladeManifoldCount'))
    buffer=archive.pointer('xpbd.d_bladeManifold')
    assert len(buffer)%CONTACT.itemsize==0
    contacts=np.frombuffer(buffer,dtype=CONTACT)
    ids=np.frombuffer(archive.pointer('xpbd.d_bladeManifoldIds'),dtype='<i4',count=count)
    assert count>0 and np.all((ids>=0)&(ids<len(contacts)))
    active=contacts[ids]
    assert np.all((active['count']>0)&(active['count']<=24)&(active['kind']>=0)&(active['kind']<6))
    assert all(np.all((r['ids'][:r['count']]>=0)&(r['ids'][:r['count']]<n)) for r in active)
    x=active['offset'].copy()
    for k in range(24):
        valid=active['count']>k
        x[valid]+=positions[active['ids'][valid,k]]*active['weights'][valid,k,None]
    shift=(observed.scissors.hinge-active['tool_origin']+
        active['gap_gradient']*(observed.scissors.gap-active['tool_gap'])[:,None])
    gaps=np.einsum('ij,ij->i',x-active['target']-shift,active['normal'])
    return active,x,gaps,dict(native_position_layout_exact=True,contact_stride_bytes=CONTACT.itemsize,
        active_contacts=count,body_contacts=int(np.count_nonzero((active['kind']==3)|(active['kind']==4))))


def deepest_body_sample(surface,geometry,scissors):
    v=surface.vertices;t=surface.triangles;tri=v[t]
    samples=np.concatenate([v,tri.mean(axis=1),(tri[:,0]+tri[:,1])/2,
                            (tri[:,1]+tri[:,2])/2,(tri[:,2]+tri[:,0])/2])
    axis,normal=scissors.axis,scissors.normal;width=np.cross(axis,normal)
    sn=np.clip(scissors.gap/(2*geometry.jaws.length),0,.95);cs=np.sqrt(1-sn*sn)
    best=None
    for side,box in enumerate(geometry.backs):
        sign=1 if side==0 else -1
        frame=np.stack([axis*cs+sign*normal*sn,normal*cs-sign*axis*sn,width])
        orientation=np.stack([box[6:9],box[9:12],np.cross(box[6:9],box[9:12])])
        clearance=box[3:6]-np.abs(((samples-scissors.hinge)@frame.T-box[:3])@orientation.T)
        depths=clearance.min(axis=1);i=int(np.argmax(depths));depth=float(depths[i])
        if best is None or depth>best['depth_cm']:
            best=dict(depth_cm=depth,kind=3+side,sample_index=i,point_cm=samples[i].tolist(),
                sample_source='vertex' if i<len(v) else ['centroid','edge01','edge12','edge20'][(i-len(v))//len(t)],
                source_index=i if i<len(v) else (i-len(v))%len(t))
    return best
