"""Commit only reviewed implementation/evidence; no push or default adoption."""
import hashlib
import json
from pathlib import Path
import py_compile
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


assert git('branch', '--show-current') == 'dev'
assert not git('diff', '--cached', '--name-only'), 'Do not combine someone else\'s index'
baseline = json.loads((HERE/'baseline.json').read_text(encoding='utf8'))
protected = {n: {'expected': want, 'actual': digest(ROOT/n)} for n, want in baseline['protected'].items()}
assert all(x['expected'] == x['actual'] for x in protected.values())
(OUT/'protected-current.json').write_text(json.dumps({'all_unchanged': True, 'files': protected}, indent=2)+'\n', encoding='utf8')
cpu = json.loads((OUT/'cpu-tests.json').read_text(encoding='utf8'))
assert cpu['passed'] and cpu['tests'] == 79
fault = json.loads((OUT/'host-state-tests/failure-cleanup-final/result.json').read_text(encoding='utf8'))
assert fault['passed'] and fault['archive_source_sha256'] == digest(HERE/'snapshot-native/state_archive.cpp')
sass = json.loads((OUT/'kernel-code.json').read_text(encoding='utf8'))
for variant in ('baseline', 'training-state1', 'training-snapshot1'):
    build = json.loads((HERE/'artifacts'/variant/'build-result.json').read_text(encoding='utf8'))
    assert build['sha256'] == digest(Path(build['dll'])) == sass['artifacts']['bin-s15-'+variant]['dll_sha256']
for file in [*(ROOT/'endoscopic_tissue/training').glob('*.py'), *HERE.glob('*.py')]:
    py_compile.compile(str(file), doraise=True)

allowed = ('endoscopic_tissue/training/', 'endoscopic_tissue/docs/training-s15-s17-',
           'endoscopic_tissue/outputs/s15-s17-20261101/', 'endoscopic_tissue_backups/2026-11-01-s15-s17/')
tracked = git('diff', '--name-only').splitlines()
assert all(n.startswith(allowed) for n in tracked), 'Review unrelated changes separately'
files = {ROOT/n for n in tracked}
files.update(HERE/n for n in ('validation_support.py', 'validation_capture.py', 'acceptance_job.py',
    'validate_environment_snapshot.py', 'validate_snapshot_distribution.py', 'validate_failure_dose.py',
    'prepare_snapshot_jobs.py', 'snapshot-validation-plan.json', 'summarize_snapshot.py', 'summarize_failure.py',
    'prepare_geometry_jobs.py', 'summarize_geometry.py', 'audit_state_layout.py', 'state-layout-audit.json',
    'run_host_state_tests.py', 'host-state-tests/cuda_runtime.h', 'host-state-tests/archive_test.cpp',
    'checkpoint_validation.py'))
files.add(ROOT/'endoscopic_tissue/docs/training-s15-s17-usage-2026-11-01.md')
files.add(OUT/'protected-current.json')
for name in ('failure-cleanup-before-r2', 'failure-cleanup-after', 'failure-cleanup-final'):
    files.update(OUT/'host-state-tests'/name/n for n in ('result.json', 'run.log'))
names = sorted(f.relative_to(ROOT).as_posix() for f in files)
assert all((ROOT/n).is_file() and (ROOT/n).stat().st_size < 2*1024**2 for n in names)
selection = {'files': names, 'bytes': sum((ROOT/n).stat().st_size for n in names),
             'gpu_acceptance_completed': False, 'push_requested_now': False}
(HERE/'validation-selected.json').write_text(json.dumps(selection, indent=2)+'\n', encoding='utf8')
pathspec = HERE/'validation-selected.nul'
pathspec.write_bytes(b'\0'.join(n.encode() for n in names)+b'\0')
subprocess.run(['git', '-c', 'core.safecrlf=false', 'add', '-f', '--pathspec-from-file='+str(pathspec),
                '--pathspec-file-nul'], cwd=ROOT, check=True)
subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
message = HERE/'commit-validation.txt'
message.write_text('training: harden snapshot cleanup and prepare full acceptance\n\n'
    'Retain const-pointee device bindings, quarantine partial restores, prevent stale-pointer double frees, '
    'and reclaim owned buffers after failed restoration. Bind branch controls/vision and preserve preparation telemetry. '
    'Add gated native T1-T8, full distribution, dose and geometry evidence workers.\n\n'
    'Validation: 79 Python CPU checks pass; actual archive CPU fault injection goes from one invalid free and '
    'one leaked allocation to zero, including failure-state export rejection. Isolated CUDA builds compile; '
    '261 common physics kernel listings match the original, and nine default hashes are unchanged. '
    'GPU physics, performance and full video acceptance remain pending. Local experimental rollback point only.\n', encoding='utf8')
subprocess.run(['git', 'commit', '-F', str(message)], cwd=ROOT, check=True)
print(json.dumps({'commit': git('rev-parse', 'HEAD'), **selection}, indent=2))
