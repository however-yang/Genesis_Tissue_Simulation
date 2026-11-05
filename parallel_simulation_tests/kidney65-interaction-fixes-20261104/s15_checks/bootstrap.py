"""Pin the isolated overlay and frozen Python reference before any imports."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
GPU='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'
SOURCE=ROOT.parents[1]/'parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/python-gpu-check'
os.environ.update(CUDA_VISIBLE_DEVICES=GPU,PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0,str(SOURCE))
from endoscopic_tissue import training
training.__path__.insert(0,str(ROOT/'python/endoscopic_tissue/training'))
