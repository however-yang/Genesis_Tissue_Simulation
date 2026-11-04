"""Verify local trial evidence; a completed run is NOT a quality acceptance."""
from pathlib import Path
import ast
import hashlib
import json
import math
import re
import statistics
import subprocess

import cv2

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = ROOT / 'endoscopic_tissue/outputs'
BASE = '0403d319097ca77099edc5c33d51b2d22d4312a0'
FIRST = OUT / 'joint-coarse-20261031'
SECOND = OUT / 'joint-gated-20261031'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.relative_to(ROOT)).replace('\\', '/')


def summary(values):
    values = sorted(values)
    assert values and all(math.isfinite(x) and x >= 0 for x in values)
    p = (len(values) - 1) * .95
    i = int(p)
    return dict(mean=statistics.mean(values),
                p95=values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (p - i),
                maximum=max(values))


def video(path, width):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    assert frames == 1111 and abs(fps - 30) < 1e-8, path
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == width, path
    for n in (0, 210, 480, 750, 1110):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = cap.read()
        assert ok and frame is not None, (path, n)
    cap.release()
    return dict(path=relative(path), frames=frames, fps=fps, bytes=path.stat().st_size,
                note='Playback at simulation time; not measured runtime FPS.')


def kernel(source, name):
    start = source.index('__global__ void ' + name + '(')
    body = source.index('{', start)
    # Mask comments/strings without changing offsets; braces in them are not syntax.
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                    lambda match: ' ' * len(match.group()), source)
    depth = 0
    for i in range(body, len(source)):
        depth += (masked[i] == '{') - (masked[i] == '}')
        if depth == 0:
            return source[start:i + 1]
    raise AssertionError(name)


build = read(WORK / 'build-manifest.json')
assert build['base_commit'] == BASE
for folder, digest in build['dll'].items():
    assert sha(ROOT / 'endoscopic_tissue/native' / folder / 'LiverCudaSim.dll') == digest, folder
for filename, digest in build['source'].items():
    assert sha(WORK / filename) == digest, filename
for path in WORK.glob('*.py'):
    ast.parse(path.read_text(encoding='utf8'), filename=str(path))
assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'dev'
subprocess.run(['git', 'diff', '--exit-code', BASE, '--', 'endoscopic_tissue/native'],
               cwd=ROOT, check=True, capture_output=True)
patches = ['initial-joint.patch', 'initial-joint-probe.patch', 'joint-coarse.patch',
           'joint-probe.patch', 'joint-gated.patch', 'joint-gated-probe.patch']
for filename in patches:
    subprocess.run(['git', 'apply', '--check', str(WORK / filename)],
                   cwd=ROOT, check=True, capture_output=True)

headers = ROOT / 'endoscopic_tissue/native/vendor/cuda'
sources = '\n'.join(p.read_text(encoding='utf8') for p in headers.rglob('*.cuh'))
sources += '\n' + (WORK / 'joint_coarse.cuh').read_text(encoding='utf8')
gated = (WORK / 'joint_fallback_gate.cuh').read_text(encoding='utf8')
gates = read(WORK / 'gate-kernel-manifest.json')
assert len(gates) == 16
for item in gates:
    name = item['name']
    wrapped = kernel(gated, name + 'Gated').replace(name + 'Gated(', name + '(', 1)
    wrapped = wrapped.replace('const int* gate,', '', 1).replace('\n    if(gate && !*gate)return;', '', 1)
    assert wrapped == kernel(sources, name), name

numeric = {}
for filename in ['algebra-results.json', 'guarded-algebra-results.json', 'gated-algebra-results.json']:
    data = read(WORK / filename)
    assert data['tet_cases'] == 180 and data['cubic_cases'] == 2004 and data['cubic_disagreements'] == 0
    assert data['endpoint_only_counterexample_rejected']
    limits = dict(gradient=2e-4, curvature=1e-4, baseline_diagonal=1e-4,
                  baseline_gradient=1e-4, negative_eigenvalue=1e-6)
    assert all(data['normalized_max_errors'][k] < limit for k, limit in limits.items())
    numeric[filename] = data
for filename in ['rounding-results.json', 'gated-rounding-results.json']:
    data = read(WORK / filename)
    assert data['cases'] == 12000 and data['old_path_accepts_below_stored_floor'] > 0
    assert data['stored_guard_accepts_below_floor'] == 0
    numeric[filename] = data

diagnostics = {}
for name, directory in [('initial', FIRST / 'diagnostic-a1'), ('strict_red', FIRST / 'strict-red'),
                        ('guarded_green', FIRST / 'strict-green'), ('gated_green', SECOND / 'strict-green')]:
    data = read(directory / 'joint-diagnostics-summary.json')
    rows = read(directory / 'joint-diagnostics.json')
    assert len(rows) == data['steps']
    assert [r['step'] for r in rows] == list(range(1, len(rows) + 1))
    assert all(r['roots'] == r['accepted'] + r['zero'] + r['fallback'] for r in rows)
    if name == 'strict_red':
        assert data['steps'] < 1110 and data['negative_margin'] > 0
    else:
        assert data['steps'] == 1110 and data['accepted'] > 0
        if name.endswith('green'):
            assert all(data[k] == 0 for k in ('negative_margin', 'volume_violation', 'closing_violation', 'energy_violation'))
    diagnostics[name] = dict(summary=data, path=relative(directory), rows_sha256=sha(directory / 'joint-diagnostics.json'))

settings = read(FIRST / 'baseline-a1/report.json')['settings']
assert (settings['dt'], settings['substeps'], settings['young_pa'], settings['poisson'],
        settings['strain_damping_s']) == (1 / 30, 3, 15000, .25, .04)
result = dict(base_commit=BASE, production_adopted=False, settings=settings, numeric=numeric,
              diagnostics=diagnostics, gate_bodies_verified=len(gates),
              timing_scope='Normal CUDA Graphs, no live display. Pipeline includes control/readback/recorder submission and possible backpressure; excludes stage boundaries and final draining. Capture/render runs are separate.',
              series=[], captures=[])
for label, directory, names in [('joint', FIRST, ['baseline-a1', 'joint-b1', 'joint-b2', 'baseline-a2']),
                                 ('gated', SECOND, ['baseline-a1', 'gated-b1', 'gated-b2', 'baseline-a2'])]:
    runs = []
    for name in names:
        path = directory / name
        rows, report = read(path / 'timing-steps.json'), read(path / 'report.json')
        meta = read(path / 'timing-summary.json')['metadata']
        assert len(rows) == report['steps'] == 1110 and report['metrics']['completed'], path
        assert [r['step'] for r in rows] == list(range(1, 1111))
        assert report['settings'] == settings and report['grid_spacing_cm'] == .1
        assert (meta['total_grid_nodes'], meta['active_nodes'], meta['material_nodes'], meta['movable_active_nodes']) == (354200, 121260, 102710, 118660)
        assert sha(Path(meta['dll'])) == meta['dll_sha256'] == report['dll_sha256']
        stages = {s: {k: summary([r['calls_ms']['LCS_Step'] if k == 'core_ms' else r[k]
                                 for r in rows if r['stage'] == s]) for k in ('core_ms', 'pipeline_ms')}
                  for s in dict.fromkeys(r['stage'] for r in rows)}
        runs.append(dict(name=name, path=relative(path), dll_sha256=meta['dll_sha256'],
                         rows_sha256=sha(path / 'timing-steps.json'), report_sha256=sha(path / 'report.json'),
                         core_ms=summary([r['calls_ms']['LCS_Step'] for r in rows]),
                         pipeline_ms=summary([r['pipeline_ms'] for r in rows]), stages=stages,
                         quality_passed=report['metrics']['acceptance_passed'], metrics=report['metrics']))
    means = {k: dict(baseline=(runs[0][k]['mean'] + runs[3][k]['mean']) / 2,
                     candidate=(runs[1][k]['mean'] + runs[2][k]['mean']) / 2) for k in ('core_ms', 'pipeline_ms')}
    for item in means.values():
        item['change_percent'] = 100 * (item['candidate'] / item['baseline'] - 1)
    result['series'].append(dict(name=label, runs=runs, means=means))

for label, directory, prefix in [('baseline', FIRST, 'baseline'), ('joint', FIRST, 'joint'), ('gated', SECOND, 'gated')]:
    path = directory / (prefix + '-capture')
    report = read(path / 'report.json')
    rows = [json.loads(line) for line in (path / 'motion.jsonl').read_text(encoding='utf8').splitlines()]
    assert report['steps'] == 1110 and report['metrics']['completed'] and report['settings'] == settings
    assert [r['step'] for r in rows] == list(range(1111))
    assert sha(Path(report['dll'])) == report['dll_sha256']
    loads = {}
    for stage in dict.fromkeys(r['stage'] for r in rows):
        group = [r for r in rows if r['stage'] == stage]
        loads[stage] = {k: [max(r[k][side] for r in group) for side in (0, 1)]
                        for k in ('jaw_load_n', 'scissors_load_n')}
    item = dict(name=label, path=relative(path), dll_sha256=report['dll_sha256'],
                quality_passed=report['metrics']['acceptance_passed'], metrics=report['metrics'],
                motion_sha256=sha(path / 'motion.jsonl'), report_sha256=sha(path / 'report.json'),
                peak_loads_by_stage=loads, video=video(directory / (prefix + '-video/interactions.mp4'), 1280))
    audits = read(path / 'surface-audit.json')
    item['topology_audit'] = {key: max(a[key] for a in audits) for key in
                              ('recorded_nonmanifold_edges', 'positional_nonmanifold_edges', 'degenerate_triangles', 'duplicate_triangles')}
    if prefix != 'baseline':
        review = directory / (prefix + '-review')
        manifest, motion = read(review / 'manifest.json'), read(review / 'motion-comparison.json')
        assert manifest['frames'] == motion['frames'] == 1111 and manifest['fps'] == 30
        item['comparison'] = video(review / 'comparison.mp4', 2560)
        item['motion_comparison'] = dict(path=relative(review / 'motion-comparison.json'),
                                         sha256=sha(review / 'motion-comparison.json'))
    result['captures'].append(item)

profiles = read(WORK / 'sections-summary.json')
for name, item in profiles['runs'].items():
    rows = read(WORK / ('sections-' + name + '.json'))
    report = read(SECOND / ('profile-' + name) / 'report.json')
    assert sum(r['steps'] for r in rows) == report['steps'] == item['steps'] == 1110
    assert report['metrics']['completed'] and report['settings'] == settings
    assert sha(Path(report['dll'])) == report['dll_sha256'] == item['dll_sha256']
    for key, value in item['sections_ms_per_step'].items():
        computed = sum(r['steps'] * r['sections_ms_per_step'].get(key, 0) for r in rows) / 1110
        assert math.isfinite(value) and value >= 0 and abs(computed - value) < 1e-10
    item['rows_sha256'] = sha(WORK / ('sections-' + name + '.json'))
result['section_diagnostics'] = profiles
result['build_manifest_sha256'] = sha(WORK / 'build-manifest.json')
result['patches'] = {filename: sha(WORK / filename) for filename in patches}
result['protected_dll_sha256'] = {name: build['dll'][name] for name in ('bin-scissors', 'bin')}
(WORK / 'evidence-summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(dict(timing_runs=8, captures=3, comparisons=2, strict_green_runs=2, section_runs=2,
                      patches_checked=len(patches), gated_original_bodies_verified=len(gates),
                      baseline_preserved=True, production_adopted=False,
                      timings=[dict(name=s['name'], means=s['means']) for s in result['series']]), indent=2))
