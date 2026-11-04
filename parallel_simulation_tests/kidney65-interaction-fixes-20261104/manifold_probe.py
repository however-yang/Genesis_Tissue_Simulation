"""GPU production-manifold regression: area subdivision and finite footprints."""
import argparse
import ctypes
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

def main():
    parser=argparse.ArgumentParser();parser.add_argument('label');args=parser.parse_args()
    if not args.label.replace('-','').isalnum():raise ValueError('Invalid label')
    control=ROOT/'controls'/('manifold-'+args.label);control.mkdir(exist_ok=False)
    guard=GPUReservation(GPU,control)
    result=dict(passed=False,cases=['wrist_one_triangle','wrist_four_triangles','box_outside','box_inside'],
                columns=['witness_count','area_weight','force_native','loaded_witnesses'])
    try:
        dll=ctypes.CDLL(str(ROOT/'binaries'/(args.label+'.so')))
        fn=dll.LCS_KidneyManifoldProbe;fn.argtypes=[ctypes.c_void_p,ctypes.c_int];fn.restype=ctypes.c_int
        rows=np.zeros((4,4),np.float32);assert fn(rows.ctypes.data,rows.size)==0
        result['values']=rows.tolist()
        assert rows[0,0]==1 and rows[1,0]==4 and min(rows[0,2],rows[1,2])>0
        np.testing.assert_allclose(rows[0,1:3],rows[1,1:3],rtol=2e-6,atol=1e-7)
        assert rows[2,0]==0 and rows[2,2]==0, 'Out-of-footprint box contact survived refresh'
        assert rows[3,0]>0 and rows[3,2]>0, 'In-footprint contact lost'
        result['passed']=True
    finally:
        guard.close();(ROOT/'results'/('manifold-'+args.label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
