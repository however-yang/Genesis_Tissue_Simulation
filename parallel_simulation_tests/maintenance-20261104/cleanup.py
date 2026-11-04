"""Project-specific, reversible cleanup. It never changes simulation source."""
import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

ROOT = Path('/srv/disk2/hryang/genesis-exec')
BASE = ROOT / 'parallel_simulation_tests/20261103-s18-s25/profiling'
LATEST = BASE / 'native-core-25hz-20261104'
WORK = ROOT / 'parallel_simulation_tests/maintenance-20261104'
ARCHIVE = ROOT / 'archive'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, data):
    (WORK / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def fingerprint():
    files = {}
    for base in (LATEST / 'python-gpu-check', LATEST / 'native-readonly-v8/source'):
        for path in sorted(base.rglob('*')):
            if path.is_file() and not path.is_symlink():
                files[str(path.relative_to(ROOT))] = sha(path)
    files[str((LATEST / 'native-readonly-v8/build/lib/libLiverCudaSim.so').relative_to(ROOT))] = sha(LATEST / 'native-readonly-v8/build/lib/libLiverCudaSim.so')
    files['builds/0643e10991cf-contact-groups31-sm80/lib/libLiverCudaSim.so'] = sha(ROOT / 'builds/0643e10991cf-contact-groups31-sm80/lib/libLiverCudaSim.so')
    files[str((LATEST / 'delivery/manifest.json').relative_to(ROOT))] = sha(LATEST / 'delivery/manifest.json')
    return files


def prepare():
    assert not WORK.exists(), 'Do not overwrite a previous cleanup run'
    WORK.mkdir()
    ARCHIVE.mkdir(exist_ok=True)
    (ARCHIVE / 'publications').mkdir(exist_ok=True)
    shutil.copy2(__file__, WORK / 'cleanup.py')
    targets = []
    def add(path, reason):
        assert path.exists() and not path.is_symlink(), path
        targets.append({'path': str(path.relative_to(ROOT)), 'reason': reason})
    for name in ['candidate-source', 'candidate-v1-source', 'candidate-v2-source', 'candidate-v3-source', 'candidate-v5-source']:
        add(BASE / name, 'Intermediate or rejected profiling snapshot; retained V4 and baseline remain')
    for name in ['candidate-source'] + [f'candidate-r{i}-source' for i in range(1, 9)]:
        add(BASE / 'aggressive-20261104' / name, 'Historical candidate; independent retained final-source remains')
    for name in ['native-lean-v1', 'native-tool-v2', 'native-query-v3', 'native-query-v4', 'native-coarse-v5', 'native-material-v6', 'native-final-v7', 'python-zero-check']:
        add(LATEST / name, 'Not retained: no performance gain or failed complete interaction validation')
    tools = LATEST / 'tools'
    obsolete_tools = [p for p in tools.iterdir() if p.name.startswith('prepare_')]
    obsolete_tools += [tools / ('build_' + name + '.py') for name in ['coarse_v5','final_v7','lean_v1','material_v6','query_v3','query_v4','tool_v2']]
    obsolete_tools += [tools / name for name in ['add_gpu_body_check.py', 'test_zero_body_check.py', 'run_matrix.py', 'run_comparison.py', 'run_post_checks.py', 'inspect_sass_diffs.py', 'write_report.py', 'freeze_delivery.py']]
    for path in sorted(set(obsolete_tools)):
        add(path, 'One-off generator or orchestration for frozen historical candidates; not a retained runtime entry')
    for name in ['finish_deployment.py','install_environment.py','resolve_dependencies_r1.py','verify_deployment.py','verify_deployment_r1.py']:
        add(ROOT / name, 'Completed one-off environment deployment script; retained for recovery only')
    names = [x['path'] for x in targets]
    assert len(names) == len(set(names))
    for a in names:
        assert not any(a != b and a.startswith(b + '/') for b in names)
    save('plan.json', {'roots': targets, 'never_touch': ['envs', 'releases', 'builds', 'runs', 'all experiment results/logs', 'retained V8 source/library', 'frozen reference sources'], 'github_snapshot': 'd201cfd5543c6caa27556aae6dbfe5fbfb191101'})
    save('runtime-before.json', fingerprint())
    shutil.copy2(ROOT / 'README.md', ARCHIVE / 'deployment-20261102-README.md')
    print(json.dumps({'roots': len(targets), 'runtime_files': len(fingerprint()), 'work': str(WORK)}), flush=True)


def bundle():
    export = ROOT / 'parallel_simulation_tests/github-export-20261104'
    parent_export = ROOT / 'parallel_simulation_tests/github-export-20261103'
    for path in [parent_export, export]:
        for args in [['status','--porcelain','--untracked-files=no'], ['ls-files','--others','--exclude-standard'], ['ls-files','--others','--ignored','--exclude-standard']]:
            result = subprocess.check_output(['git', *args], cwd=path)
            assert not result, ('Unexpected changes; keep export', path, args)
    path = ARCHIVE / 'publications/genesis-history-through-d201cfd.bundle'
    assert not path.exists()
    subprocess.run(['git','bundle','create',str(path),'--all'],cwd=export,check=True)
    verify = WORK / 'git-bundle-verify.git'
    subprocess.run(['git','init','--bare','--quiet',str(verify)],check=True)
    subprocess.run(['git','--git-dir',str(verify),'fetch','--quiet',str(path),'HEAD:refs/heads/verified-snapshot'],check=True)
    fsck = subprocess.run(['git','--git-dir',str(verify),'fsck','--full'],text=True,capture_output=True)
    (WORK / 'bundle-fsck.log').write_text(fsck.stdout + fsck.stderr)
    assert fsck.returncode == 0, fsck.stderr
    commit = subprocess.check_output(['git','--git-dir',str(verify),'rev-parse','refs/heads/verified-snapshot'],text=True).strip()
    assert commit == 'd201cfd5543c6caa27556aae6dbfe5fbfb191101'
    data = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'bytes': path.stat().st_size, 'self_contained': True, 'commit': commit, 'verified_in_empty_repository': True, 'fsck_passed': True}
    save('git-bundle.json', data)
    shutil.copy2(ROOT / 'parallel_simulation_tests/github-publish-receipt-20261104.json', ARCHIVE / 'publications/github-publish-receipt-20261104.json')
    print(json.dumps(data), flush=True)


def pack():
    plan = json.loads((WORK / 'plan.json').read_text())
    path = ARCHIVE / 'retired-code-20261104.tar.gz'
    assert not path.exists()
    entries = {}
    def record(info):
        source = ROOT / info.name
        entry = {'mode': info.mode, 'size': info.size}
        if info.isfile(): entry.update(type='file', sha256=sha(source))
        elif info.isdir(): entry.update(type='directory')
        elif info.issym(): entry.update(type='symlink', target=info.linkname)
        elif info.islnk(): entry.update(type='hardlink', target=info.linkname)
        else: raise RuntimeError('Unsupported archive member ' + info.name)
        assert info.name not in entries, info.name
        entries[info.name] = entry
        return info
    with tarfile.open(path,'w:gz',compresslevel=1,dereference=False) as tar:
        for item in plan['roots']:
            print('Packing ' + item['path'],flush=True)
            tar.add(ROOT / item['path'],arcname=item['path'],filter=record)
    checked = set()
    with tarfile.open(path,'r:gz') as tar:
        for info in tar:
            expected = entries[info.name]
            if info.isfile():
                h = hashlib.sha256()
                with tar.extractfile(info) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):h.update(block)
                assert h.hexdigest() == expected['sha256'], info.name
            elif info.issym() or info.islnk():assert info.linkname == expected['target']
            checked.add(info.name)
    assert checked == set(entries)
    manifest = {'archive': str(path.relative_to(ROOT)), 'archive_sha256': sha(path), 'archive_bytes': path.stat().st_size, 'original_regular_bytes': sum(x['size'] for x in entries.values() if x['type'] == 'file'), 'roots': plan['roots'], 'entries': entries, 'verified': True}
    (ARCHIVE / 'retired-code-20261104-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    save('archive-summary.json', {k:v for k,v in manifest.items() if k!='entries'})
    print(json.dumps({'archive': str(path), 'entries': len(entries), 'compressed_bytes': path.stat().st_size, 'verified': True}),flush=True)


def prune():
    manifest = json.loads((ARCHIVE / 'retired-code-20261104-manifest.json').read_text())
    assert sha(ROOT / manifest['archive']) == manifest['archive_sha256']
    assert fingerprint() == json.loads((WORK / 'runtime-before.json').read_text()), 'Runtime changed; stop cleanup'
    for name, info in manifest['entries'].items():
        path = ROOT / name
        if info['type'] == 'file':assert path.is_file() and sha(path) == info['sha256'], name
        elif info['type'] == 'symlink':assert path.is_symlink() and os.readlink(path) == info['target'], name
    current_names = set()
    for item in manifest['roots']:
        path = ROOT / item['path']
        current_names.add(item['path'])
        if path.is_dir():
            for directory, dirs, files in os.walk(path, followlinks=False):
                for name in dirs + files:
                    current_names.add(str((Path(directory) / name).relative_to(ROOT)))
    assert current_names == set(manifest['entries']), 'Files added or removed after backup; stop cleanup'
    bundle_info = json.loads((WORK / 'git-bundle.json').read_text())
    assert bundle_info['self_contained'] and sha(ROOT / bundle_info['path']) == bundle_info['sha256']
    disk_before = shutil.disk_usage(ROOT).free
    removed=[]
    for item in manifest['roots']:
        path = ROOT / item['path']
        assert path.is_relative_to(ROOT) and not path.is_symlink()
        if path.is_dir():shutil.rmtree(path)
        else:path.unlink()
        removed.append(item['path'])
    for name in ['github-export-20261104','github-export-20261103']:
        path = ROOT / 'parallel_simulation_tests' / name
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=path)
        assert not subprocess.check_output(['git','ls-files','--others','--ignored','--exclude-standard'],cwd=path)
        shutil.rmtree(path)
        removed.append(str(path.relative_to(ROOT)))
    old_bundle = ROOT / 'parallel_simulation_tests/github-publish-20261104.bundle'
    assert hashlib.sha256(old_bundle.read_bytes()).hexdigest() == '78c9ec6712c45093a4c4cc9f535f2f69faca1d871074c86c7ffa0c048dbe29c6'
    old_bundle.unlink()
    removed.append(str(old_bundle.relative_to(ROOT)))
    shutil.rmtree(WORK / 'git-bundle-verify.git')
    after = fingerprint()
    assert after == json.loads((WORK / 'runtime-before.json').read_text())
    save('runtime-after.json', after)
    data={'removed_roots':removed,'runtime_files_identical':len(after),'regular_files_archived':sum(v['type']=='file' for v in manifest['entries'].values()),'disk_free_increase_bytes_since_prune':shutil.disk_usage(ROOT).free-disk_before,'git_history_backed_up':True,'archive_verified':True,'physics_modified':False,'rendering_modified':False,'gpu_tests_run':False}
    save('cleanup-result.json',data)
    print(json.dumps(data,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','bundle','pack','prune']);args=parser.parse_args()
    globals()[args.stage]()
