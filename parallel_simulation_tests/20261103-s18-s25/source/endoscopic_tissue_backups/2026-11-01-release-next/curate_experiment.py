"""Create an explicit local Git selection, excluding binaries and raw captures."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
P = Path(__file__).resolve().parent
PACKAGE = P/'package'
OUT = ROOT/'endoscopic_tissue/outputs/release-next-20261101'
EVIDENCE = PACKAGE/'evidence'


def main():
    EVIDENCE.mkdir(exist_ok=True)
    selected = list(P.glob('*.py'))
    selected += [P/n for n in ['plan.md', 'findings.md', 'machine.json', 'controls-jobs.json', 'abba-jobs.json', 'verification.json', 'visual-review.json', 'finish_boundary_reviews.cmd']]
    selected += [ROOT/'endoscopic_tissue/docs'/n for n in ['interaction-release-followup-2026-11-01.md', 'training-completion-results-2026-11-01.md', 'interaction-fixes-2026-10-31.md']]
    selected += [PACKAGE/n for n in ['.gitattributes', 'CMakeLists.txt', 'build_experiment.py', 'README.md']]
    variants = ['release-r21-fragment-submission', 'release-r21-topology-probe',
                'release-r22-repair-empty-manifold', 'release-r23-repair-prelude-proof',
                'release-r24-cut-witness-probe', 'release-r24-material-boundary',
                'release-r25-boundary-contact-identity']
    for variant in variants:
        manifest_path = PACKAGE/(variant+'.json')
        manifest = json.loads(manifest_path.read_text())
        selected.append(manifest_path)
        for item in manifest['files']:
            source = PACKAGE/'overrides'/item.get('source_variant', variant)/item['path']
            assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
            selected.append(source)
        for name in ['build-inputs.json', 'build-result.json']:
            source = PACKAGE/'artifacts'/variant/name
            destination = EVIDENCE/(variant+'-'+name)
            shutil.copyfile(source, destination)
            selected.append(destination)
    for revision in ['r21', 'r22', 'r23', 'r24', 'r25']:
        selected += [P/(revision+s) for s in ['-gpu-code.json', '-result.json']]
        if revision != 'r21':
            selected.append(P/(revision+'-timing-jobs.json'))
        for name in ['report.json', 'release-overlap-audit.json', 'review/video-verification.json']:
            source = OUT/(revision+'-kidney')/name
            destination = EVIDENCE/(revision+'-'+Path(name).name)
            shutil.copyfile(source, destination)
            selected.append(destination)
        if revision in ('r22', 'r23'):
            source = OUT/(revision+'-kidney')/'repair-dispatch-audit.json'
            destination = EVIDENCE/(revision+'-repair-dispatch-audit.json')
            shutil.copyfile(source, destination)
            selected.append(destination)
        if revision == 'r25':
            source = OUT/'r25-kidney/vertex-link-audit.json'
            destination = EVIDENCE/'r25-vertex-link-audit.json'
            shutil.copyfile(source, destination)
            selected.append(destination)
    selected.append(P/'r24-witness-gpu-code.json')
    for name in ['offline-topology-0600.json', 'offline-topology-0840.json',
                 'frozen-boundary-audit.json', 'boundary-identity-audit.json']:
        source = OUT/'cut-witness-trace-v2'/name
        destination = EVIDENCE/name
        shutil.copyfile(source, destination)
        selected.append(destination)
    for source, name in [(OUT/'topology-trace/native-topology-audit.json', 'native-topology-audit.json'),
                         (OUT/'r23-contact-probe-confirm/report.json', 'r23-nonempty-contact-probe.json')]:
        destination = EVIDENCE/name
        shutil.copyfile(source, destination)
        selected.append(destination)
    selected = sorted(set(path.resolve() for path in selected))
    rows = []
    for path in selected:
        assert path.is_relative_to(ROOT)
        assert path.suffix.lower() not in ('.dll', '.lib', '.npz', '.bin', '.rdx', '.mp4', '.pdf')
        content = path.read_bytes()
        rows.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=len(content),
                         sha256=hashlib.sha256(content).hexdigest()))
    result = dict(files=rows, count=len(rows), bytes=sum(r['bytes'] for r in rows),
                  note='Explicit experimental source and compact evidence only; no default promotion or remote push.')
    (P/'selected-files.json').write_text(json.dumps(result, indent=2)+'\n')
    (P/'selected-paths.nul').write_bytes(b''.join((r['path']+'\0').encode('utf8') for r in rows))
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))


if __name__ == '__main__':
    main()
