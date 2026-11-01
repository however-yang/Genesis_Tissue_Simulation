"""Serialize native validation against a frozen copy of the implemented adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

P=Path(__file__).resolve().parent
ROOT=P.parents[1]
OUT=ROOT/'endoscopic_tissue/outputs/training-completion-20261101'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('--batch',action='store_true')
    args=parser.parse_args()
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.tag):
        raise ValueError('Unsafe validation tag')
    evidence=P/args.tag
    evidence.mkdir(exist_ok=False)
    snapshot=evidence/'adapter-source'
    snapshot.mkdir()
    hashes={}
    for source in sorted((ROOT/'endoscopic_tissue/training').glob('*.py')):
        hashes[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source,snapshot/source.name)
    (evidence/'source-hashes.json').write_text(json.dumps(hashes,indent=2))
    specs=['hundred-screen'] if args.batch else ['policy-pilot','frozen-evaluation','friction-pilot','vision-pilot','sensitivity-screen']
    jobs=[]
    for name in specs:
        for file,digest in hashes.items():
            if hashlib.sha256((ROOT/'endoscopic_tissue/training'/file).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Adapter changed during validation; preserve earlier evidence and start a new tag')
        command=[sys.executable,'-X','utf8','-m','endoscopic_tissue.training.experiment',
                 '--spec',str(P/(name+'.json')),'--output',str(OUT/args.tag/name)]
        print('START',args.tag,name,flush=True)
        start=time.perf_counter()
        with (evidence/(name+'.log')).open('w',encoding='utf8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=7200,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        jobs.append(dict(name=name,command=command,exit_code=result.returncode,wall_s=time.perf_counter()-start))
        (evidence/'jobs.json').write_text(json.dumps(jobs,indent=2))
        print('END',args.tag,name,result.returncode,flush=True)
        if result.returncode:raise RuntimeError('Validation process failed')


if __name__=='__main__':main()
