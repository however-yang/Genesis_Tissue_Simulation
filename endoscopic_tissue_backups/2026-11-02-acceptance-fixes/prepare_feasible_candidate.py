"""Reuse the main solver's feasible coarse basis in surface-contact repair only."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]/'endoscopic_tissue_backups/2026-11-01-s15-s17/artifacts/training-snapshot1/source'
PREVIOUS = HERE/'overrides/boundary-snapshot2'
OUT = HERE/'overrides/repair-feasible-snapshot3'
manifest = json.loads((PREVIOUS/'manifest.json').read_text(encoding='utf8'))
for relative, digest in manifest['candidate_sha256'].items():
    if hashlib.sha256((PREVIOUS/relative).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Previous candidate changed: '+relative)
if OUT.exists():
    # Resume only an untouched copy left by an aborted preparation.
    for relative, digest in manifest['candidate_sha256'].items():
        if hashlib.sha256((OUT/relative).read_bytes()).hexdigest() != digest:
            raise FileExistsError('Existing modified candidate is preserved: '+str(OUT))
else:
    shutil.copytree(PREVIOUS, OUT)
relative = 'vendor/cuda/xpbd/xpbd_solver.cu'
path = OUT/relative
text = path.read_text(encoding='utf8')
start = text.index('extern "C" int xpbd_relax_reconstructed_contact(){')
end = text.index('static int unifiedSubstep(', start)
section = text[start:end]
safe = '\n'.join('                '+line for line in (
    '// Keep the existing descent budget and objective. Near-floor tets',
    '// share a coarse weight, as in the main solve, so one compressed',
    '// cell does not clip the entire reconstructed contact correction.',
    'k_InitSafeCoarse<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified);',
    'k_JoinSafeCoarse<<<Gr(g_T),64,0,g_captureStream>>>(d_unified);',
    'k_FlattenSafeCoarse<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified);',
    'k_AccumulateSafeCoarse<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified);',
    'k_FinishSafeCoarse<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified);'))+'\n'
for anchor in (
    '                k_CoarseBegin<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);',
    '                k_CoarseBegin<<<1,1,0,g_captureStream>>>(d_unified,d_coarseBlock);'):
    if section.count(anchor) != 1:
        raise RuntimeError('Non-unique repair insertion point')
    section = section.replace(anchor, safe+anchor)
path.write_text(text[:start]+section+text[end:], encoding='utf8')
manifest['candidate_sha256'] = {relative: hashlib.sha256((OUT/relative).read_bytes()).hexdigest()
                                for relative in manifest['candidate_sha256']}
manifest.update(name='repair-feasible-snapshot3', parent_candidate='boundary-snapshot2',
    change='Material-boundary candidate plus the existing feasible-basis preprocessing in contact repair only',
    isolated_change='Existing volume-feasible coarse-weight preprocessing before the two repair directions',
    new_state_fields=[], material_parameters_changed=False, iterations_changed=False,
    default_replaced=False, accepted=False)
(OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
print(json.dumps(manifest, indent=2))
