"""Repeatable source inventory for explicitly reviewed native POD layouts.

This emits evidence, not a C++ parser proof. Native compilation and T1-T8 are
still required. Only the three declared pointer-bearing records are accepted.
"""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'artifacts/training-state1/source/vendor'
SCALARS = {'float', 'float3', 'float4', 'int', 'int3', 'int4', 'unsigned', 'unsigned int',
           'unsigned short', 'unsigned long long', '__half', 'bool', 'double'}
POINTER_TYPES = {'UnifiedState', 'CoupledTool', 'MultigridLevel'}
HOST_TYPES = {'ReconInitDesc', 'PhysicsInitDesc', 'CutInitDesc', 'CutToolDesc', 'HapticParams',
              'XpbdParams', 'XpbdGraspParams', 'XpbdToolState', 'XpbdBladeState',
              'XpbdScissorsState', 'XpbdScissorsBackBox', 'JawMeshBounds', 'MultigridLevel'}
NESTED = {'SurfaceContact': ['ContactRecord', 'ContactShape'],
          'GripSolveStep': ['XpbdToolState'], 'CoupledTool': ['XpbdToolState', 'XpbdScissorsBackBox'],
          'XpbdScissorsState': ['XpbdScissorsBackBox'],
          'ToolFeedbackPacket': ['XpbdToolState', 'XpbdToolFeedback']}


def clean(s):
    return re.sub(r'//[^\n]*|/\*[\s\S]*?\*/', '', s)


def definitions():
    found = {}
    for path in sorted(SOURCE.rglob('*')):
        if path.suffix not in ('.cpp', '.cu', '.h', '.cuh'):
            continue
        text = clean(path.read_text(encoding='utf-8-sig'))
        for match in re.finditer(r'\bstruct\s+(\w+)\s*\{', text):
            i, depth = match.end(), 1
            while depth and i < len(text):
                depth += (text[i] == '{')-(text[i] == '}')
                i += 1
            body = text[match.end():i-1]
            found.setdefault(match[1], []).append((path.relative_to(SOURCE).as_posix(), body))
    return found


def member_statements(body):
    # All state PODs use ordinary data members. Symmetric3 also has methods;
    # skip their complete brace bodies rather than treating multiplication as
    # a pointer declarator. No inherited/virtual/dynamic C++ objects are allowed.
    result, start, depth = [], 0, 0
    for i, ch in enumerate(body):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                start = i+1
        elif ch == ';' and depth == 0:
            statement = body[start:i].strip()
            if statement:
                if '(' in statement:
                    raise ValueError('Unreviewed POD member declaration: '+statement)
                result.append(statement)
            start = i+1
    return result


def main():
    inventory = json.loads((HERE/'state-inventory.json').read_text(encoding='utf8'))
    calls = [a for file in inventory['files'] for a in file['allocations']]
    allocated = {m[1] for a in calls for m in re.finditer(r'sizeof\(([^)]+)\)', a['call'])}
    wanted = allocated | HOST_TYPES | {n for names in NESTED.values() for n in names}
    declarations = definitions()
    records = []
    for name in sorted(wanted):
        if name in SCALARS:
            records.append({'type': name, 'pointer_layout': 'none', 'category': 'CUDA numeric scalar/vector'})
            continue
        if name not in declarations:
            raise ValueError('Missing type definition for review: '+name)
        for file, body in declarations[name]:
            members = member_statements(body)
            pointers = [m for m in members if '*' in m]
            if bool(pointers) != (name in POINTER_TYPES):
                raise ValueError('Pointer-bearing type inventory changed: '+name)
            records.append({'type': name, 'source': file, 'members': members,
                'nested_value_types': NESTED.get(name, []),
                'pointer_layout': pointers or 'none',
                'layout_handling': 'explicit offsetof field layout' if pointers else 'POD bytes; no pointer-valued members',
                'source_sha256': hashlib.sha256((SOURCE/file).read_bytes()).hexdigest()})
    result = {'schema': 'native_pod_layout_review_v1', 'runtime_accepted': False,
        'allocation_sites': len(calls), 'allocated_sizeof_types': sorted(allocated), 'records': records,
        'explicit_pointer_records': sorted(POINTER_TYPES),
        'conservative_scope': 'All live allocations saved. No scratch buffer excluded to meet the size target.',
        'additional_serialization_state': [
            {'state': 'allocation IDs, site hashes, sizes, normalized layouts, nextId', 'category': 'save in archive tables/header'},
            {'state': 'allocation addresses', 'category': 'recompute/rebind to current allocations; T1/new-session tests required'},
            {'state': 'known-layout whitelist', 'category': 'recompute from same DLL declarations during collect'},
            {'state': 'atBoundary', 'category': 'restore to completed-boundary true only after successful load'},
            {'state': 'errorText', 'category': 'diagnostic only; never consumed by physics'}]}
    (HERE/'state-layout-audit.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({'reviewed_types': len(records), 'allocation_sites': len(calls), 'runtime_accepted': False}))


if __name__ == '__main__':
    main()
