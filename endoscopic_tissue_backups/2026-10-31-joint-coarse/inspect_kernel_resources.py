"""Read compiled kernel resources; this does not measure occupancy or runtime."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
TOOL = Path('D:/develop/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin/cuobjdump.exe')
DLL = ROOT / 'endoscopic_tissue/native/bin-joint-coarse31-gated/LiverCudaSim.dll'
text = subprocess.check_output([str(TOOL), '--dump-resource-usage', str(DLL)], text=True)
(WORK / 'resources-gated.txt').write_text(text, encoding='utf8')
records = []
arch = None
name = ''
for line in text.splitlines():
    if 'arch =' in line:
        arch = line.strip()
    if line.strip().startswith('Function '):
        name = line.strip()
    elif 'REG:' in line and any(k in name for k in ('k_JointAccumulate', 'k_JointSolve', 'k_JointLine', 'k_JointInteractions')):
        records.append(dict(arch=arch, name=name, resources=line.strip()))
assert len(records) == 4, records
(WORK / 'kernel-resources.json').write_text(json.dumps(dict(
    dll=str(DLL), note='Static compiled resources; do not establish actual occupancy or the cause of the measured slowdown.',
    kernels=records), indent=2), encoding='utf8')
print(json.dumps(records, indent=2))
