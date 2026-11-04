"""Create reviewable source patches; never applies or publishes them."""
import json,hashlib,difflib,subprocess,os
from pathlib import Path
p=Path(__file__).resolve().parents[1];os.sched_setaffinity(0,[30,31]);m=json.loads((p/'manifest.json').read_text());choice=json.loads((p/'results/recommended-candidate.json').read_text());base=p.parent/'aggressive-20261104/native-gpu-diagnostics-v2/source';after=Path(choice['native_source']);sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();native_hashes=lambda root:{str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*')) if f.is_file()}
hashes=native_hashes(after);patch=[];changed=[]
for rel in hashes:
 f=after/rel;old=base/rel
 if old.exists() and sha(old)==hashes[rel]:continue
 patch.extend(difflib.unified_diff(old.read_bytes().decode().splitlines(True) if old.exists() else [],f.read_bytes().decode().splitlines(True),fromfile='a/'+rel if old.exists() else '/dev/null',tofile='b/'+rel));changed.append(rel)
file=p/'delivery/native-from-previous.patch';file.write_bytes(''.join(patch).encode());check=subprocess.run(['git','apply','--check',str(file)],cwd=base,capture_output=True,text=True);assert check.returncode==0,(check.stdout,check.stderr)
def variant(d):
 d=dict(d);d['native_source_sha256']=native_hashes(Path(d['native_source']));d['dll_sha256']=sha(Path(d['dll']));return d
recommended=variant(choice);conservative=json.loads((p/'results/conservative-candidate.json').read_text());conservative['harness']='profile_fallback.py';conservative['execution']['bound_cache']=False;conservative=variant(conservative);original=dict(recommended);original['execution']=dict(recommended['execution'],query_mode=0,bound_cache=False)
archives={str(f.relative_to(base)):sha(f)==sha(after/f.relative_to(base)) for f in base.rglob('*') if f.is_file() and any(k in f.name for k in ['state_archive','xpbd_state'])};assert all(archives.values())
manifest=dict(variants=dict(recommended=recommended,conservative=conservative,original=original),python_source=m['baseline_source'],python_source_sha256=m['source_sha256'],patch=str(file),patch_base=str(base),patch_check_returncode=check.returncode,changed_native_files=changed,physical_archive_files_unchanged=archives,physics_parameters_changed=False,quality_thresholds_changed=False,production_replaced=False,formal_physics_domain_approved=False,training_domain_approved=False,harness_sha256={str(f.relative_to(p)):sha(f) for f in sorted((p/'tools').glob('*.py'))})
(p/'delivery/manifest.json').write_text(json.dumps(manifest,indent=2));print('Packaged isolated candidate',choice['dll_sha256'],len(changed),'changed native files; patch check passed')
