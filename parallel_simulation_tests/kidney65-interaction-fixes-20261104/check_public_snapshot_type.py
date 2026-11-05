"""CPU-only regression for the public snapshot loader's object type.

A public snapshot must reach the initialized-session guard, rather than be
rejected solely because an opt-in codec defined its own equivalent class.
This does not load native bytes or certify a GPU restoration.
"""
import argparse
import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--name',required=True);a=parser.parse_args()
    assert a.name.replace('-','').isalnum()
    root=Path(__file__).resolve().parent
    sys.path.insert(0,str(root.parents[1]/'parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/python-gpu-check'))
    from endoscopic_tissue import training
    training.__path__.insert(0,str(root/'python/endoscopic_tissue/training'))
    from endoscopic_tissue.training.snapshot import EnvSnapshot
    from endoscopic_tissue.training.kidney_snapshot import restore_snapshot
    import zstandard as zstd
    compressed=zstd.ZstdCompressor().compress(b'x')
    metadata=dict(schema='environment_snapshot_v1',state_abi=1,native_bytes=1,python_bytes=1,
        native_compressed_sha256=hashlib.sha256(compressed).hexdigest(),python_compressed_sha256=hashlib.sha256(compressed).hexdigest())
    public_snapshot=EnvSnapshot(compressed,compressed,json.dumps(metadata).encode())
    result=dict(passed=False,scope='Public class dispatch only; no native/GPU state created',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        restore_snapshot(SimpleNamespace(native=None,context=None),public_snapshot)
    except RuntimeError as error:
        result['error']=str(error)
        result['passed']='initialized session' in str(error)
    except Exception as error:
        result['error']=repr(error)
    with (root/'results'/(a.name+'.json')).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result));return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
