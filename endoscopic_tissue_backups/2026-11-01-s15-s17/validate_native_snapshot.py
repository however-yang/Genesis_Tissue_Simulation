"""Native round-trip probe with exact array and one-step continuation checks."""
import argparse
import ctypes as ct
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from endoscopic_tissue.native_backend import NativeTissue
from endoscopic_tissue.scenario import SCISSORS_SCENES,scissors_settings,stage_start,interpolate,drive_command
from endoscopic_tissue.training.data import atomic_json
from endoscopic_tissue.training.provenance import sha256


class NativeSnapshotAPI:
    def __init__(self,native):
        self.native=native
        self.functions={}
        for name,args,result in [('LCS_StateAbiVersion',[],ct.c_int),
                ('LCS_StateSize',[ct.POINTER(ct.c_size_t)],ct.c_int),
                ('LCS_SaveState',[ct.c_void_p,ct.c_size_t],ct.c_int),
                ('LCS_LoadState',[ct.c_void_p,ct.c_size_t],ct.c_int),
                ('LCS_StateLastError',[],ct.c_char_p),
                ('LCS_StateInventory',[ct.c_void_p,ct.c_size_t],ct.c_int)]:
            fn=native.dll[name];fn.argtypes=args;fn.restype=result;self.functions[name]=fn
        if self.functions['LCS_StateAbiVersion']()!=1:raise ValueError('Snapshot ABI mismatch')
    def check(self,name,*args):
        code=self.functions[name](*args)
        if code:raise RuntimeError(name+': '+self.functions['LCS_StateLastError']().decode('utf8'))
    def save(self):
        n=ct.c_size_t();self.check('LCS_StateSize',ct.byref(n))
        buffer=ct.create_string_buffer(n.value)
        self.check('LCS_SaveState',buffer,n.value)
        return buffer.raw
    def load(self,data):
        self.check('LCS_LoadState',ct.c_char_p(data),len(data))
    def inventory(self):
        buffer=ct.create_string_buffer(131072)
        self.check('LCS_StateInventory',buffer,len(buffer))
        return json.loads(buffer.value)


def arrays_equal(a,b):
    if isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:arrays_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):arrays_equal(x,y)
    else:
        assert a==b,(a,b)


@threadpool_limits.wrap(limits=1,user_api='blas')
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True)
    p.add_argument('--stage',default='settle')
    p.add_argument('--scene',default='wall-traction')
    p.add_argument('--cycles',type=int,default=0)
    args=p.parse_args()
    out=ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/snapshot'/args.name
    out.mkdir(parents=True,exist_ok=False)
    dll=ROOT/'endoscopic_tissue/native/bin-s15-training-snapshot1/LiverCudaSim.dll'
    scene=SCISSORS_SCENES[args.scene]
    settings=replace(scissors_settings(),young_pa=scene.young_pa,contact_model=scene.contact_model)
    grid=scene.build_grid(settings.density_kg_m3)
    record={'dll_sha256':sha256(dll),'stage':args.stage,'scene':args.scene,'passed':False}
    try:
        with NativeTissue(grid,settings,dll,scene.stages[0].command) as native:
            api=NativeSnapshotAPI(native)
            requested=scene.stages[0].command
            observed=native.observe()
            for stage in scene.stages:
                start=stage_start(stage.name,requested,observed)
                for i in range(stage.steps):
                    requested=interpolate(start,stage.command,(i+1)/(stage.steps*stage.ramp))
                    command=drive_command(requested,observed,settings)
                    observed=native.step(command)
                print('prepared',stage.name,native.tick,flush=True)
                if stage.name==args.stage:break
            record['inventory']=api.inventory()
            tick=native.tick
            started=time.perf_counter();saved=api.save();record['save_ms']=(time.perf_counter()-started)*1000
            record['bytes']=len(saved);record['state_sha256']=hashlib.sha256(saved).hexdigest()
            command=drive_command(requested,observed,settings)
            # Original continuation uses its already captured execution graphs.
            source=native.step(command)
            started=time.perf_counter();api.load(saved);record['load_ms']=(time.perf_counter()-started)*1000
            native.tick=tick
            saved2=api.save()
            record['T1_saved_bytes_equal']=saved==saved2
            if saved!=saved2:
                (out/'state-before.bin').write_bytes(saved)
                (out/'state-after.bin').write_bytes(saved2)
            assert saved==saved2,'T1 native bytes changed after round-trip'
            native.topology_revision=None
            arrays_equal(asdict(observed),asdict(native.observe()))
            record['T2_observation_exact']=True
            restored=native.step(command)
            record['T3_max_node_difference_mm']=float(np.linalg.norm(source.nodes-restored.nodes,axis=1).max()*10)
            assert record['T3_max_node_difference_mm']<.01,'T3 one-step continuation exceeded 0.01 mm'
            for offset in ((.03,0,0),(0,.03,0),(0,0,.03)):
                api.load(saved);native.tick=tick;native.topology_revision=None
                native.step(replace(command,hinge=tuple(np.asarray(command.hinge)+offset)))
            api.load(saved);native.tick=tick;native.topology_revision=None
            assert api.save()==saved,'T5 branch contaminated reusable snapshot'
            record['T5_three_branches_exact']=True
            # Invalid bytes must be rejected without corrupting the current state.
            bad=bytearray(saved);bad[-1]^=1
            try:api.load(bytes(bad))
            except RuntimeError:pass
            else:raise AssertionError('Corrupt state was accepted')
            assert api.save()==saved
            record['corruption_rejected_without_mutation']=True
            if args.cycles:
                samples=[];rss=[];process=psutil.Process()
                for i in range(args.cycles):
                    started=time.perf_counter();current=api.save();api.load(current)
                    samples.append((time.perf_counter()-started)*1000)
                    if i%100==0 or i==args.cycles-1:
                        rss.append({'cycle':i+1,'rss':process.memory_info().rss,'allocations':api.inventory()})
                record['cycles']={'count':args.cycles,'save_load_ms_mean':float(np.mean(samples)),
                                  'save_load_ms_p95':float(np.percentile(samples,95)),'memory':rss}
            record['passed']=True
    except Exception as e:
        record['error']=repr(e)
        raise
    finally:
        atomic_json(out/'result.json',record)
        print(json.dumps({k:v for k,v in record.items() if k!='inventory'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
