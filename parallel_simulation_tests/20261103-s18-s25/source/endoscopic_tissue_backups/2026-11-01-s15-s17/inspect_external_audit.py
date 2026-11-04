"""Read NPZ string metadata only; never load pickle/object payloads."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from endoscopic_tissue.training.audit_physics import audit_context
from endoscopic_tissue.training.data import atomic_json
from endoscopic_tissue.training.provenance import sha256
source = Path('E:/datasets/可微世界模型_试采/precheck-01')
results = []
for file in sorted(source.glob('*.npz')):
    row = {'file': str(file), 'sha256': sha256(file), 'metadata_keys': {}, 'audit': None}
    with np.load(file, allow_pickle=False) as archive:
        row['keys'] = list(archive.files)
        for key in archive.files:
            if not any(s in key.lower() for s in ('context', 'meta', 'config')):
                continue
            try:
                array = archive[key]
                row['metadata_keys'][key] = {'shape': array.shape, 'dtype': str(array.dtype)}
                if array.size == 1 and array.dtype.kind in 'US':
                    value = json.loads(str(array.item()))
                    if 'settings' in value and 'scene_config' in value:
                        row['audit'] = audit_context(value)
                    else:
                        row['metadata_keys'][key]['value'] = value
            except (ValueError, TypeError) as error:
                row['metadata_keys'].setdefault(key, {})['error'] = repr(error)
    results.append(row)
atomic_json(ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/audits/wam-npz.json', results)
summary = json.loads((source/'summary.json').read_text(encoding='utf-8-sig'))
audit = {'input': str(source/'summary.json'), 'input_sha256': sha256(source/'summary.json'),
         'complete_context_available': False, 'historical_flag_verifiable': False,
         'inputs_modified': False, 'episodes': [
             {'name': r['name'], 'recorded_start_force_n': r['context_start_force_n'],
              'canonical_start_force_n': .2, 'start_force_changed': r['context_start_force_n'] != .2,
              'physical_defaults_changed_should_be_true': r['context_start_force_n'] != .2}
             for r in summary['episodes']]}
atomic_json(ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/audits/wam-summary.json', audit)
print(json.dumps(results, ensure_ascii=False, indent=2))
