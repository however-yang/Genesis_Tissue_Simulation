from pathlib import Path
import runpy
import sys
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from endoscopic_tissue import rendering

with threadpool_limits(limits=1,user_api="blas"):
    runpy.run_path(str(Path(__file__).with_name("render_cases.py")),run_name="__main__")
