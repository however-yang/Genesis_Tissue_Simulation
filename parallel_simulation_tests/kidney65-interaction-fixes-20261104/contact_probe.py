"""GPU contact fixture, using a diagnostic candidate library on A100 GPU1 only."""
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
sys.path.insert(0,str(ROOT.parents[1]/'parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/python-gpu-check'))
from endoscopic_tissue.training.acceptance_queue import GPUReservation


def main():
    parser=argparse.ArgumentParser();parser.add_argument('label');args=parser.parse_args()
    if not args.label.replace('-','').isalnum():raise ValueError('Invalid build label')
    control=ROOT/'controls'/('probe-'+args.label);control.mkdir(exist_ok=False)
    guard=GPUReservation(GPU,control)
    try:
        lib=ctypes.CDLL(str(ROOT/'binaries'/(args.label+'.so')))
        fn=lib.LCS_KidneyContactProbe;fn.argtypes=[ctypes.c_void_p,ctypes.c_int];fn.restype=ctypes.c_int
        rows=np.empty((6,4),np.float32);assert fn(rows.ctypes.data,rows.size)==0
        result=dict(passed=False,columns=['gap_cm','force_native','normal_y','shape_distance_cm'],
                    cases=['wrist_exit','wrist_contact','wrist_deep_crossing','back_exit','back_contact','back_deep_crossing'],values=rows.tolist())
        for i in (0,3):assert rows[i,0]>=0 and rows[i,1]==0, str(rows)
        for i in (1,2,4,5):assert rows[i,0]<0 and rows[i,1]>0 and rows[i,2]>.5, str(rows)
        result['passed']=True
        (ROOT/'results'/('probe-'+args.label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
    finally:guard.close()


if __name__=='__main__':main()
