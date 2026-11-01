"""Freeze protected inputs and create a read-only profiling driver from the tested harness."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import psutil

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
OLD = ROOT/'endoscopic_tissue_backups/2026-10-31-interaction-fixes'


def main():
    original = (OLD/'time_full_interaction.py').read_text()
    old = "    args=parser.parse_args()"
    new = "    parser.add_argument('--profile-steps',type=int,nargs='*',default=[])\n"+old
    assert original.count(old) == 1
    text = original.replace(old, new)
    old = "profile=args.diagnostic_spikes and first_spike is not None and len(rows)==first_spike"
    assert text.count(old) == 1
    text = text.replace(old, "profile=(len(rows)+1 in args.profile_steps) or (args.diagnostic_spikes and first_spike is not None and len(rows)==first_spike)")
    (P/'profile_release.py').write_text(text, encoding='utf8', newline='\n')
    protection = json.loads((ROOT/'endoscopic_tissue_backups/2026-11-01-training-completion/machine.json').read_text())['protected']
    for name in ('endoscopic_tissue/native/bin-scissors/LiverCudaSim.dll', 'endoscopic_tissue/native/bin/LiverCudaSim.dll'):
        protection[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest for name,digest in protection.items())
    processes=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        if (proc.info['name'] or '').lower() in ('python.exe','pythonw.exe','nvcc.exe','ffmpeg.exe'):
            processes.append(proc.info)
    result={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'protected':protection,'processes':processes,'disk_free_gib':shutil.disk_usage(ROOT).free/2**30,
            'profile_driver':{'original_sha256':hashlib.sha256(original.encode()).hexdigest(),
                              'candidate_sha256':hashlib.sha256((P/'profile_release.py').read_bytes()).hexdigest(),
                              'change':'explicit selected profiling frames only; no physics/settings changes'}}
    target=P/'machine.json'
    if target.exists():
        raise RuntimeError('Do not overwrite initial protection evidence')
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
