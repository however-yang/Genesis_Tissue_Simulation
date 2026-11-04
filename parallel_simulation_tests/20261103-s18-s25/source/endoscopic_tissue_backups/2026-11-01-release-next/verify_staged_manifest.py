"""Check the explicit Git selection and exact staged native-source bytes."""
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    package = directory/'package'
    count = 0
    for path in sorted(package.glob('release-*.json')):
        manifest = json.loads(path.read_text())
        for item in manifest['files']:
            source = package/'overrides'/item.get('source_variant', manifest['variant'])/item['path']
            name = source.relative_to(root).as_posix()
            content = subprocess.check_output(['git', 'show', ':'+name], cwd=root)
            assert hashlib.sha256(content).hexdigest() == item['sha256'], name
            count += 1
    wanted = {row['path'] for row in json.loads((directory/'selected-files.json').read_text())['files']}
    actual = set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root, text=True).splitlines())
    assert actual == wanted, (actual-wanted, wanted-actual)
    print(f'PASS: staged selection {len(actual)} files; pinned native override entries {count}')


if __name__ == '__main__':
    main()
