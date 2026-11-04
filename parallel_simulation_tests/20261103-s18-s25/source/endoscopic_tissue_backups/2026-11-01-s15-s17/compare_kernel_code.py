"""Stream disassembly into per-kernel digests; never retain huge SASS dumps."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TOOL=Path(r'D:\develop\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\cuobjdump.exe')
OUT=ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/kernel-code.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kernels(path):
    result={}
    p=subprocess.Popen([str(TOOL),'--dump-sass',str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                       text=True,encoding='utf8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    name=None
    hasher=None
    for line in p.stdout:
        match=re.search(r'Function\s*:\s*(\S+)',line)
        if match:
            if name is not None:result[name]=hasher.hexdigest()
            name=match[1];hasher=hashlib.sha256()
        elif name is not None and '/*' in line:
            hasher.update(line.strip().encode())
    if name is not None:result[name]=hasher.hexdigest()
    error=p.stderr.read()
    if p.wait():raise RuntimeError(error)
    if not result:raise RuntimeError('No kernels found')
    return {'dll_sha256':digest(path),'kernels':result}


def main():
    data={}
    for name in ('bin-contact-groups31','bin-s15-baseline','bin-s15-training-state1','bin-s15-training-snapshot1'):
        print('Inspecting',name,flush=True)
        data[name]=kernels(ROOT/'endoscopic_tissue/native'/name/'LiverCudaSim.dll')
    original=data['bin-contact-groups31']['kernels']
    comparisons={}
    for name,row in data.items():
        current=row['kernels']
        comparisons[name]={'identical_common_kernels':sum(current[k]==v for k,v in original.items() if k in current),
            'changed_common_kernels':[k for k,v in original.items() if k in current and current[k]!=v],
            'missing_kernels':sorted(set(original)-set(current)),'added_kernels':sorted(set(current)-set(original))}
    result={'schema':'streamed_sass_comparison_v1','cuobjdump_sha256':digest(TOOL),
        'note':'Exact instruction listing equality is evidence only for those kernels; host scheduling, physical interaction and timing still require full runtime/video validation.',
        'artifacts':data,'comparisons':comparisons}
    OUT.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
