import argparse,json,hashlib,difflib,subprocess
from pathlib import Path
p=Path(__file__).resolve().parents[1];a=argparse.ArgumentParser();a.add_argument('--version',required=True);v=a.parse_args().version
baseline=p/'baseline-source';candidate=p/('candidate-'+v+'-source')
files=set(x.relative_to(baseline) for x in (baseline/'endoscopic_tissue').rglob('*.py'))|set(x.relative_to(candidate) for x in (candidate/'endoscopic_tissue').rglob('*.py'))
patch=[];changes=[]
for path in sorted(files):
 b=baseline/path;c=candidate/path;before=b.read_bytes() if b.exists() else b'';after=c.read_bytes() if c.exists() else b''
 if before==after:continue
 patch.extend(difflib.unified_diff(before.decode().splitlines(keepends=True),after.decode().splitlines(keepends=True),fromfile='a/'+str(path) if b.exists() else '/dev/null',tofile='b/'+str(path) if c.exists() else '/dev/null'))
 changes.append(dict(path=str(path),before_sha256=hashlib.sha256(before).hexdigest() if b.exists() else None,after_sha256=hashlib.sha256(after).hexdigest() if c.exists() else None))
patch_file=p/('candidate-'+v+'.patch');patch_file.write_text(''.join(patch))
check=subprocess.run(['git','apply','--check',str(patch_file)],cwd=baseline,text=True,capture_output=True)
manifest=dict(candidate=str(candidate),baseline=str(baseline),patch=str(patch_file),patch_sha256=hashlib.sha256(patch_file.read_bytes()).hexdigest(),patch_check_returncode=check.returncode,patch_check_output=check.stdout+check.stderr,changes=changes,production_replaced=False,physics_model_changed=False,acceptance_thresholds_changed=False)
(p/('changes-'+v+'.json')).write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if check.returncode:raise SystemExit(check.returncode)
