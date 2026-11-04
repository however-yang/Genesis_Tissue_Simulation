"""Freeze three geometry variants before viewing outcomes; queue 5 full runs each."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue.training.data import atomic_json
from endoscopic_tissue.training.provenance import json_hash, sha256
from endoscopic_tissue.training.geometry import WallGeometry


def main():
    out = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/geometry-batches'
    out.mkdir(parents=True, exist_ok=True)
    queue = json.loads((HERE/'gpu-queue.json').read_text(encoding='utf8'))
    prepared = []
    for purpose, variants in [('collection', ('wall-24mm', 'wall-25mm')),
                              ('evaluation', ('wall-26mm-heldout',))]:
        batch = out/purpose
        batch.mkdir(exist_ok=True)
        cases = []
        for variant in variants:
            geometry_file = HERE/'geometries'/(variant+'.json')
            geometry = WallGeometry(**json.loads(geometry_file.read_text(encoding='utf8')))
            for repeat in range(5):
                name = f'geometry-{variant}-r{repeat}'
                cases.append({'id': name, 'seed': repeat, 'geometry': json.loads(geometry_file.read_text(encoding='utf8')),
                              'geometry_recipe_sha256': json_hash(geometry.recipe()),
                              'geometry_file_sha256': sha256(geometry_file)})
                args = [str(HERE/'validate_damage.py'), '--name', name, '--seed', str(repeat),
                        '--dll', 'bin-contact-groups31', '--geometry', str(geometry_file),
                        '--probe-dir', str(batch/name/'episode')]
                if repeat == 0:
                    args += ['--capture']
                if not any(j['name'] == name for j in queue['jobs']):
                    queue['jobs'].append({'name': name, 'args': args})
                    prepared.append(name)
        spec = {'schema': 'geometry_acceptance_probe_v1', 'purpose': purpose, 'episodes': cases,
                'data_scope': 'one adjacent transition for split integration; complete original three-snip script is checked separately with every quality check enabled',
                'full_script_results': 'endoscopic_tissue/outputs/s15-s17-20261101/damage',
                'evaluation_frozen_before_results': True}
        path = batch/'spec.json'
        if path.exists() and json.loads(path.read_text(encoding='utf8')) != spec:
            raise ValueError('Existing frozen geometry batch differs; do not silently overwrite')
        atomic_json(path, spec)
    atomic_json(HERE/'gpu-queue.json', queue)
    print(json.dumps({'queued': prepared, 'total_jobs': len(queue['jobs'])}, indent=2))


if __name__ == '__main__':
    main()
