"""Compare actual frozen-input GPU queries; no live solver state is changed."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
GPU='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'
os.environ['CUDA_VISIBLE_DEVICES']=GPU
sys.path.insert(0,str(ROOT.parent/'20261103-s18-s25/profiling/native-core-25hz-20261104/python-gpu-check'))
from endoscopic_tissue.training.acceptance_queue import GPUReservation

def read(path):
    with path.open('rb') as f:
        magic,count,width,n,triangles=np.fromfile(f,np.int32,5)
        assert magic==0x46525031 and width==20
        ms=float(np.fromfile(f,np.float32,1)[0])
        rows=np.fromfile(f,np.float32,count*width).reshape(count,width)
        delta=np.fromfile(f,np.float32,n*3).reshape(n,3)
        degree=np.fromfile(f,np.int32,n)
        assert not f.read(1)
    return rows[np.argsort(rows[:,0],kind='stable')],delta,degree,ms,int(triangles)

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--case',required=True);p.add_argument('--name',required=True)
    p.add_argument('--angle-budget',type=float,default=.02);p.add_argument('--integral-variant',action='store_true');args=p.parse_args()
    if not 0<args.angle_budget<=1:raise ValueError('Invalid angle budget')
    for value in [args.reference,args.candidate,args.case,args.name]:
        if not value.replace('-','').replace('_','').isalnum():raise ValueError('Invalid label')
    output=ROOT/'outputs'/args.name;output.mkdir(exist_ok=False)
    control=ROOT/'controls'/args.name;control.mkdir(exist_ok=False)
    guard=GPUReservation(GPU,control)
    result=dict(passed=False,reference=args.reference,candidate=args.candidate,case=args.case,cases=[],angle_budget=args.angle_budget,integral_variant=args.integral_variant)
    try:
        libraries=[]
        for label in [args.reference,args.candidate]:
            path=ROOT/'binaries'/(label+'.so')
            manifest=json.loads((ROOT/'results'/('build-'+label+'.json')).read_text())
            assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256']
            dll=ctypes.CDLL(str(path));fn=dll.LCS_ProbeFragmentFile
            fn.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int];fn.restype=ctypes.c_int
            libraries.append((dll,fn))
        for stage in ['drop_settle','regrasp_close']:
            case=ROOT/'outputs'/args.case/(stage+'.fragment')
            # The legacy query-only probe uses its particle query mode. Clear
            # the private copy's grip flags for the held case so it evaluates
            # the same volume geometry; the live regrasp test covers mode 2.
            for release in ([0,1] if stage=='drop_settle' else [1]):
                data=[]
                for index,(dll,fn) in enumerate(libraries):
                    budget=dll.LCS_SetFragmentAngleBudget;budget.argtypes=[ctypes.c_float];budget.restype=ctypes.c_int
                    assert budget(.02 if index==0 else args.angle_budget)==0
                    target=output/f'{stage}-{release}-{index}.probe'
                    assert fn(os.fsencode(case),os.fsencode(target),release)==0
                    data.append(read(target))
                a,b=data;identical=np.array_equal(a[0].view(np.uint32),b[0].view(np.uint32))
                columns=[i for i in range(20) if i not in (2,16)]
                decisions=np.array_equal(a[0][:,columns].view(np.uint32),b[0][:,columns].view(np.uint32))
                delta_error=float(np.max(np.abs(a[1]-b[1])));degrees=np.array_equal(a[2],b[2])
                result['cases'].append(dict(stage=stage,release=release,queries=len(a[0]),triangles=a[4],
                    query_records_bitwise_equal=identical,degrees_equal=degrees,
                    decision_columns_bitwise_equal=decisions,
                    reference_mean_triangles=float(a[0][:,2].mean()),candidate_mean_triangles=float(b[0][:,2].mean()),
                    atomic_delta_max_native_error=delta_error,
                    reference_kernel_ms=a[3],candidate_kernel_ms=b[3]))
                assert decisions and degrees, 'A geometric decision changed'
                if args.angle_budget==.02 and not args.integral_variant:assert identical,'Unchanged angle algorithm must also preserve work and angle records'
                # Atomics can reorder positional sums. Record them separately;
                # this is not a bitwise whole-solver equivalence claim.
                np.testing.assert_allclose(a[1],b[1],rtol=2e-6,atol=1e-7)
        result['passed']=True
    finally:
        guard.close();(ROOT/'results'/('query-'+args.name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
