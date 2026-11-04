#!/usr/bin/env python3
"""Restore selected, verified historical code without overwriting current files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
INDEX = PROJECT / 'archive/retired-code-20261104-manifest.json'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='按原路径恢复归档的历史代码；不覆盖现有文件，不启动仿真。')
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--list', action='store_true')
    choice.add_argument('--path', help='清单中某一条完整的项目相对路径')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(INDEX.read_text())
    if args.list:
        print(json.dumps(manifest['roots'], ensure_ascii=False, indent=2))
        return
    allowed = {item['path'] for item in manifest['roots']}
    if args.path not in allowed:
        raise ValueError('请选择 --list 中的一条完整路径')
    archive = PROJECT / manifest['archive']
    if digest(archive) != manifest['archive_sha256']:
        raise ValueError('归档 SHA256 不符')
    target = PROJECT / args.path
    if target.exists() or target.is_symlink():
        raise FileExistsError('目标已存在，拒绝覆盖：' + str(target))
    if not target.parent.resolve().is_relative_to(PROJECT):
        raise ValueError('恢复路径不能离开项目')
    if args.dry_run:
        print(json.dumps({'restore': args.path, 'archive_verified': True, 'writes': False}, ensure_ascii=False))
        return
    selected = {name: info for name, info in manifest['entries'].items()
                if name == args.path or name.startswith(args.path + '/')}
    with tempfile.TemporaryDirectory(prefix='.restore-', dir=PROJECT / 'archive') as folder:
        staging = Path(folder)
        restored = set()
        with tarfile.open(archive, 'r:gz') as stream:
            for member in stream:
                if member.name not in selected:
                    continue
                info = selected[member.name]
                path = staging / member.name
                if not path.resolve().is_relative_to(staging):
                    raise ValueError('Unsafe archive path')
                path.parent.mkdir(parents=True, exist_ok=True)
                if member.isdir():
                    path.mkdir(exist_ok=True)
                elif member.issym():
                    link = Path(os.path.normpath(str((PROJECT / member.name).parent / member.linkname)))
                    if not link.is_relative_to(PROJECT):
                        raise ValueError('Symlink leaves project: ' + member.name)
                    path.symlink_to(member.linkname)
                elif member.isfile() or member.islnk():
                    with stream.extractfile(member) as source, path.open('wb') as destination:
                        shutil.copyfileobj(source, destination)
                    expected = info.get('sha256')
                    if expected and digest(path) != expected:
                        raise ValueError('Restored content differs: ' + member.name)
                else:
                    raise ValueError('Unsupported member: ' + member.name)
                if not path.is_symlink():
                    path.chmod(member.mode & 0o777)
                    os.utime(path, (member.mtime, member.mtime))
                restored.add(member.name)
        if restored != set(selected):
            raise ValueError('Archive members incomplete')
        target.parent.mkdir(parents=True, exist_ok=True)
        (staging / args.path).rename(target)
    print(json.dumps({'restored': args.path, 'verified_entries': len(restored), 'gpu_initialized': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
