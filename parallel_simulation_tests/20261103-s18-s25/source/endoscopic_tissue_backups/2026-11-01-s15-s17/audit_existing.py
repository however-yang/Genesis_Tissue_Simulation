from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue.training.audit_physics import audit_tree
from endoscopic_tissue.training.data import atomic_json
roots = {'wam-precheck': Path('E:/datasets/可微世界模型_试采/precheck-01'),
         'local-records': ROOT/'endoscopic_tissue/outputs'}
out = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/audits'
out.mkdir(parents=True, exist_ok=True)
for name, source in roots.items():
    result = audit_tree(source) if source.exists() else {'source_missing': str(source)}
    atomic_json(out/(name+'.json'), result)
    print(name, len(result.get('records', [])), result.get('flag_mismatches'), len(result.get('errors', [])))
