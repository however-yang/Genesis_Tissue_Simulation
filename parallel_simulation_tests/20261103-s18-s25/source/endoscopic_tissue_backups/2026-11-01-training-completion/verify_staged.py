"""Verify the explicit Git index against reviewed working bytes and protection hashes."""
import hashlib
import json
from pathlib import Path
import subprocess

P = Path(__file__).resolve().parent
ROOT = P.parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    selected = json.loads((P/'selected-files.json').read_text())
    names = [row['path'] for row in selected]
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=ROOT)
    actual_names = {s.decode('utf8') for s in staged.split(b'\0') if s}
    if actual_names != set(names):
        raise RuntimeError({'missing': sorted(set(names)-actual_names),
                            'unexpected': sorted(actual_names-set(names))})
    request = ''.join(':'+name+'\n' for name in names).encode('utf8')
    output = subprocess.run(['git', 'cat-file', '--batch'], input=request,
                            stdout=subprocess.PIPE, check=True, cwd=ROOT).stdout
    cursor = 0
    for row in selected:
        end = output.index(b'\n', cursor)
        header = output[cursor:end].split()
        if len(header) != 3 or header[1] != b'blob':
            raise RuntimeError((row['path'], header))
        size = int(header[2])
        blob = output[end+1:end+1+size]
        cursor = end+2+size
        working = (ROOT/row['path']).read_bytes()
        if len(blob) != row['bytes'] or digest(blob) != row['sha256'] or blob != working:
            raise RuntimeError('Index/source bytes differ: '+row['path'])
    if cursor != len(output):
        raise RuntimeError('Unexpected trailing Git batch output')
    protected = json.loads((P/'machine.json').read_text())['protected']
    for name, expected in protected.items():
        if name in actual_names or digest((ROOT/name).read_bytes()) != expected:
            raise RuntimeError('Default protection changed: '+name)
    result = {'staged_files': len(names), 'staged_bytes': sum(r['bytes'] for r in selected),
              'all_selected_index_working_hashes_equal': True,
              'default_protected_files_unchanged': len(protected)}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
