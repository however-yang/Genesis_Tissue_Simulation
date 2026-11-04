#!/usr/bin/env python3
"""One entry to the retained simulator; imports no GPU library on status/verify."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent


def configuration():
    return json.loads((PROJECT / 'current/VERSION.json').read_text())


def verify(config):
    experiment = PROJECT / config['experiment']
    manifest = json.loads((experiment / 'delivery/manifest.json').read_text())
    pairs = [
        (PROJECT / 'current/native', experiment / manifest['native_source'], manifest['native_files']),
        (PROJECT / 'current/python', experiment / manifest['python_source'], manifest['python_files']),
    ]
    count = 0
    for alias, original, files in pairs:
        if alias.resolve(strict=True) != original.resolve(strict=True):
            raise ValueError('Current source link differs from audited source: ' + str(alias))
        for name, expected in files.items():
            path = alias / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Source differs from audited version: ' + str(path))
            count += 1
    library = PROJECT / 'current/lib/libLiverCudaSim.so'
    original_library = experiment / manifest['native_dll']
    if library.resolve(strict=True) != original_library.resolve(strict=True):
        raise ValueError('Current library link differs from audited library')
    if hashlib.sha256(library.read_bytes()).hexdigest() != manifest['native_dll_sha256']:
        raise ValueError('Native library differs from audited version')
    return {'verified_source_files': count, 'native_library_sha256': manifest['native_dll_sha256'],
            'gpu_initialized': False, 'physics_acceptance_inferred': False}


def main():
    parser = argparse.ArgumentParser(description='当前保留 V8 的统一入口；不修改物理、渲染或验收配置。')
    commands = parser.add_subparsers(dest='command')
    commands.add_parser('status', help='显示唯一当前版本与实际源码位置，不使用 GPU')
    commands.add_parser('verify', help='核验原交付清单中的 386 个源码文件及共享库，不使用 GPU')
    for kind, description in [('benchmark', '完整 RGB-D 测速'), ('validate', '完整物理/诊断对照'),
                              ('rgbd', '当前帧 RGB-D 回归'), ('snapshot', '快照回归'),
                              ('profile', 'Nsight 热点采集')]:
        item = commands.add_parser(kind, help=description + '；使用原资源守卫')
        item.add_argument('--variant', choices=['readonly', 'reference'], default='readonly')
        item.add_argument('--output', type=Path, help='原实验 results 下尚不存在的新目录')
        item.add_argument('--dry-run', action='store_true', help='只核验并打印原命令，不启动仿真')
    topology = commands.add_parser('topology', help='CPU 复现已知非流形错误；成功表示错误被复现')
    topology.add_argument('--case', type=int, default=0)
    topology.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = configuration()
    command = args.command or 'status'
    if command == 'status':
        print(json.dumps({
            'version': config['version'], 'status': config['status'],
            'python_source': str(PROJECT / 'current/python'),
            'native_source': str(PROJECT / 'current/native'),
            'reference_release': str(PROJECT / config['reference_release']),
            'report': str(PROJECT / config['experiment'] / 'REPORT.md'),
            'measured_full_rgbd_hz': config['measured_full_rgbd_hz'],
            'stable_25hz_approved': False, 'formal_physics_domain_approved': False,
            'training_domain_approved': False, 'gpu_initialized': False,
        }, ensure_ascii=False, indent=2))
        return 0
    checked = verify(config)
    if command == 'verify':
        print(json.dumps(checked, ensure_ascii=False, indent=2))
        return 0
    experiment = PROJECT / config['experiment']
    python = PROJECT / config['python_environment'] / 'bin/python'
    if command == 'topology':
        invocation = [str(python), str(experiment / 'tools/reproduce_nonmanifold.py'), '--case', str(args.case)]
        if args.dry_run:
            print(json.dumps({'command': invocation, 'gpu_initialized': False}, ensure_ascii=False, indent=2))
            return 0
    else:
        invocation = [str(python), str(experiment / 'tools/reproduce.py'), '--kind', command, '--variant', args.variant]
        if args.output is not None:
            invocation += ['--output', str(args.output.absolute())]
        if args.dry_run:
            invocation.append('--dry-run')
    environment = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run(invocation, cwd=experiment, env=environment).returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print('入口检查失败：' + str(error), file=sys.stderr)
        raise SystemExit(2)
