"""Compare the actual R8 CUDA algebra with independent SciPy QP solutions."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import minimize


SOURCE = r'''
#include <cuda_runtime.h>
#include <math.h>
#define REPAIR_CONE_ALGEBRA_ONLY
#include "repair_cone.cuh"
__global__ void probe(const double* inputs,double* outputs,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const double* p=inputs+42*i;repair_cone::Matrix h={p[0],p[1],p[2],p[3],p[4],p[5]};
 repair_cone::Vector g={p[6],p[7],p[8]},x;int count=int(p[9]);repair_cone::Plane planes[8];
 for(int j=0;j<count;j++)planes[j]={{p[10+4*j],p[11+4*j],p[12+4*j]},p[13+4*j]};
 bool ok=repair_cone::solve(h,g,planes,count,x);double* q=outputs+4*i;
 q[0]=ok;q[1]=x.x;q[2]=x.y;q[3]=x.z;
}
extern "C" int run_cone(int n,const double* inputs,double* outputs){
 double *a=nullptr,*b=nullptr;cudaError_t e=cudaMalloc(&a,n*42*sizeof(double));
 if(e==cudaSuccess)e=cudaMalloc(&b,n*4*sizeof(double));
 if(e==cudaSuccess)e=cudaMemcpy(a,inputs,n*42*sizeof(double),cudaMemcpyHostToDevice);
 if(e==cudaSuccess){probe<<<(n+63)/64,64>>>(a,b,n);e=cudaDeviceSynchronize();}
 if(e==cudaSuccess)e=cudaMemcpy(outputs,b,n*4*sizeof(double),cudaMemcpyDeviceToHost);
 cudaFree(a);cudaFree(b);return int(e);
}
'''


def main():
    root = Path('/srv/disk2/hryang/genesis-exec')
    here = Path(__file__).resolve().parent
    assert here == root/'runs/a100-cone-repair-20261102-r1'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'
    out = here/'algebra-check'
    out.mkdir(exist_ok=False)
    source = out/'probe.cu'
    source.write_text(SOURCE)
    header = here/'source/vendor/cuda/solver/material/repair_cone.cuh'
    prefix = root/'envs/genesis-py311-cu125-v1'
    library = out/'probe.so'
    command = [str(prefix/'bin/nvcc'), '-std=c++17', '-arch=sm_80', '-shared',
               '-Xcompiler=-fPIC', '-ccbin=/usr/bin/g++-12', '-I'+str(header.parent),
               str(source), '-o', str(library)]
    with (out/'compile.log').open('xb') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    rng = np.random.default_rng(20261102)
    rows = []
    cases = []
    for count in range(9):
        for repeat in range(8):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            h = q@np.diag(10**rng.uniform(0, 2, 3))@q.T
            g = rng.normal(size=3)*10
            a = rng.normal(size=(count, 3))
            if count:
                a /= np.linalg.norm(a, axis=1)[:, None]
            lower = -rng.uniform(0, .2, count)
            if repeat == 0:
                lower[:] = 0
            if count >= 2 and repeat == 1:
                a[1] = a[0]  # Duplicate constraints must not cause a singular solve.
                lower[1] = lower[0]
            if count >= 2 and repeat == 2:
                a[1] = -a[0]
                lower[:2] = 0  # Motion restricted to a plane.
            row = np.zeros(42)
            row[:10] = [h[0,0], h[1,1], h[2,2], h[0,1], h[0,2], h[1,2], *g, count]
            if count:
                row[10:10+4*count] = np.column_stack([a, lower]).ravel()
            rows.append(row)
            cases.append((h, g, a, lower, count, repeat))
    inputs = np.ascontiguousarray(rows, dtype=np.float64)
    outputs = np.zeros((len(rows), 4), dtype=np.float64)
    dll = ctypes.CDLL(str(library))
    ptr = ctypes.POINTER(ctypes.c_double)
    dll.run_cone.argtypes = [ctypes.c_int, ptr, ptr]
    dll.run_cone.restype = ctypes.c_int
    error = dll.run_cone(len(rows), inputs.ctypes.data_as(ptr), outputs.ctypes.data_as(ptr))
    assert error == 0, error
    records = []
    for row, (h, g, a, lower, count, repeat) in zip(outputs, cases):
        def fun(x): return .5*x@h@x+g@x
        constraints = [] if count == 0 else [dict(type='ineq', fun=lambda x: a@x-lower, jac=lambda x: a)]
        reference = minimize(fun, np.zeros(3), jac=lambda x: h@x+g, method='SLSQP',
                             constraints=constraints, options=dict(ftol=1e-11, maxiter=500))
        x = row[1:]
        residual = float((a@x-lower).min(initial=0))
        energy_gap = float(abs(fun(x)-reference.fun))
        passed = bool(row[0] == 1 and reference.success and residual >= -1e-8
                      and energy_gap <= 1e-7*max(1., abs(reference.fun)))
        records.append(dict(constraints=count, repeat=repeat, passed=passed,
            reference_success=bool(reference.success), reference_message=str(reference.message),
            maximum_position_error=float(abs(x-reference.x).max()),
            absolute_energy_error=energy_gap, minimum_constraint_residual=residual))
    record = dict(passed=all(x['passed'] for x in records), cases=len(records),
        header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], tests=records,
        scope='GPU algebra only; not dynamic interaction or performance acceptance')
    (out/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k != 'tests'}), flush=True)
    return 0 if record['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
