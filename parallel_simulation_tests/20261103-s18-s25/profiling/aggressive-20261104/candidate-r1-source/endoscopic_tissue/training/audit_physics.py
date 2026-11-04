"""Read-only audit of historical context.json physics provenance.

python -m endoscopic_tissue.training.audit_physics ROOT --output NEW_REPORT.json
Input records are never edited, even when their historical flag was incorrect.
"""
import argparse
import json
from pathlib import Path

from ..native_backend import Settings
from .physics import PhysicsOverrides, physical_differences, resolve_physics
from .provenance import sha256


def audit_context(context):
    config = context['scene_config']
    settings = Settings(**context['settings'])
    native = context.get('native_physics_parameters')
    native_known = native is not None
    if native is None:
        override = config.get('physics')
        _, native = resolve_physics(settings, PhysicsOverrides(**override) if override else None)
    differences = physical_differences(settings, config, native)
    changed = bool(differences)
    return {'episode_id': context.get('episode_id'),
            'recorded_physical_defaults_changed': context.get('physical_defaults_changed'),
            'audited_physical_defaults_changed': changed,
            'flag_mismatch': context.get('physical_defaults_changed') != changed,
            'overridden_fields': differences,
            'native_parameter_evidence': 'recorded' if native_known else
                'not recorded; historical adapter defaults assumed, native execution not verified',
            'input_modified': False}


def audit_tree(root):
    rows, errors = [], []
    paths = [root] if root.is_file() else sorted(root.rglob('context.json'))
    for path in paths:
        try:
            row = audit_context(json.loads(path.read_text(encoding='utf-8-sig')))
            rows.append({'path': str(path.resolve()), 'input_sha256': sha256(path), **row})
        except (KeyError, ValueError, TypeError, OSError) as error:
            errors.append({'path': str(path), 'error': repr(error)})
    return {'schema': 'physics_provenance_audit_v1', 'records': rows, 'errors': errors,
            'flag_mismatches': sum(row['flag_mismatch'] for row in rows), 'inputs_modified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.root.exists():
        parser.error('Input root does not exist')
    result = audit_tree(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents accidentally overwriting an input or prior audit.
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'records': len(result['records']), 'mismatches': result['flag_mismatches'],
                      'errors': len(result['errors'])}))


if __name__ == '__main__':
    main()
