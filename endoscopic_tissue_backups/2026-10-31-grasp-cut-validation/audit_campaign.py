"""Run CPU-only independent audits of an existing interaction campaign."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'endoscopic_tissue/outputs/grasp-cut-validation-20261031')
    a=p.parse_args()
    results=[]
    for report in sorted(a.root.glob('*/report.json')):
        r=json.loads(report.read_text(encoding='utf8'))
        m=r.get('metadata')
        if not m or m.get('fixture_valid') is False:continue
        scripts=[]
        if m['kind']=='cut':scripts.append('inspect_cuts.py')
        if m['kind']=='cut' and m['scene'] in ('wall','wall-traction'):scripts.append('inspect_wall_bridges.py')
        if (report.parent/'failure-state.npz').exists():scripts.append('audit_failures.py')
        for script in scripts:
            result=subprocess.run([sys.executable,'-X','utf8',str(HERE/script),str(report.parent)],capture_output=True,text=True,encoding='utf8')
            results.append(dict(run=report.parent.name,audit=script,exit_code=result.returncode))
            if result.returncode:raise RuntimeError(result.stderr or result.stdout)
            print(report.parent.name,script,'OK',flush=True)
    (a.root/'audit-verification.json').write_text(json.dumps(results,indent=2),encoding='utf8')
if __name__=='__main__':main()
