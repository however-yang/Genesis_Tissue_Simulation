"""Verify experiment provenance and evidence, not surgical acceptance."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
P = Path(__file__).resolve().parent
PACKAGE = P/'package'
OUT = ROOT/'endoscopic_tissue/outputs/release-next-20261101'


def main():
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    assert branch == 'dev', branch
    machine = json.loads((P/'machine.json').read_text())
    protected = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
                 for name, digest in machine['protected'].items()}
    assert all(protected.values()), protected
    scripts = list(P.glob('*.py'))+[PACKAGE/'build_experiment.py']
    for path in scripts:
        ast.parse(path.read_text(encoding='utf8'), filename=str(path))
    variants = ['release-r21-fragment-submission', 'release-r21-topology-probe',
                'release-r22-repair-empty-manifold', 'release-r23-repair-prelude-proof',
                'release-r24-cut-witness-probe', 'release-r24-material-boundary',
                'release-r25-boundary-contact-identity']
    sources = {}
    for variant in variants:
        manifest = json.loads((PACKAGE/(variant+'.json')).read_text())
        for item in manifest['files']:
            source = PACKAGE/'overrides'/item.get('source_variant', variant)/item['path']
            built_source = PACKAGE/'artifacts'/variant/'source'/item['path']
            assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
            assert hashlib.sha256(built_source.read_bytes()).hexdigest() == item['sha256']
        build = json.loads((PACKAGE/'artifacts'/variant/'build-result.json').read_text())
        assert hashlib.sha256(Path(build['dll']).read_bytes()).hexdigest() == build['sha256']
        assert build['experimental'] and not build['default_replaced']
        sources[variant] = dict(override_count=len(manifest['files']), dll_sha256=build['sha256'])
    evidence = {}
    for revision in ['r21', 'r22', 'r23', 'r24', 'r25']:
        code = json.loads((P/(revision+'-gpu-code.json')).read_text())
        assert code['candidate_functions'] == 301
        if revision in ('r21', 'r22', 'r23'):
            assert code['all_gpu_functions_identical']
        else:
            expected = ['k_BuildCutTriangles', 'k_LookupConnectivity']
            if revision == 'r25':
                expected.append('k_BindSurface')
            assert not code['changed'] and len(code['added']) == len(code['removed']) == len(expected)
            assert all(any(name in key for name in expected) for key in code['added']+code['removed'])
        result = json.loads((P/(revision+'-result.json')).read_text())
        assert not result['accepted'] and not result['default_replaced']
        report = json.loads((OUT/(revision+'-kidney')/'report.json').read_text())
        video = json.loads((OUT/(revision+'-kidney')/'review/video-verification.json').read_text())
        assert report['completed'] and report['steps'] == 1320
        assert video['frames'] == video['decoded'] == report['steps']+1
        overlap = json.loads((OUT/(revision+'-kidney')/'release-overlap-audit.json').read_text())
        assert [row['step'] for row in overlap] == list(range(1244, 1321))
        evidence[revision] = dict(simulation_steps=report['steps'], verified_video_frames=video['decoded'],
            free_release_frames=len(overlap), accepted=False)
    audit = json.loads((OUT/'r23-kidney/repair-dispatch-audit.json').read_text())
    assert audit['audited_calls'] == audit['proven_empty_calls']+audit['possible_contact_calls'] > 0
    probe = json.loads((OUT/'r23-contact-probe-confirm/report.json').read_text())
    assert probe['steps'] == 360 and probe['max_invalid_tets'] == 0
    assert max(probe['retained_constraint_samples']) > 0
    frozen = OUT/'cut-witness-trace-v2'
    for step in (600, 840):
        row = json.loads((frozen/f'offline-topology-{step:04d}.json').read_text())
        assert row['control']['exact_native_feature_incidence']
        assert row['hypothesis']['boundary_edges'] == row['hypothesis']['nonmanifold_edges'] == 0
        assert row['control']['skin_and_cut_witness_uses'] == row['hypothesis']['skin_and_cut_witness_uses']
    identity = json.loads((frozen/'boundary-identity-audit.json').read_text())
    assert identity['changed_cuts_hidden_by_surface_only_key'] > 0
    links = json.loads((OUT/'r25-kidney/vertex-link-audit.json').read_text())['summary']
    assert links['states'] == 95 and links['all_closed'] and links['all_winding_consistent']
    assert links['invalid_links_total'] == 0
    result = dict(branch=branch, protected_files=protected, parsed_python_scripts=len(scripts),
        sources=sources, evidence=evidence, r23_proof_audits=audit['audited_calls'],
        r23_nonempty_constraint_samples=probe['retained_constraint_samples'],
        actual_cut_identity_aliases=identity['changed_cuts_hidden_by_surface_only_key'],
        r25_position_welded_vertex_links=links,
        verification_scope='Evidence completeness and protected defaults only; experimental candidates remain unaccepted.')
    (P/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
