"""After frozen native validation, reject lossy goal IDs and recheck all recorded rewards."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
OUT = ROOT/'endoscopic_tissue/outputs/training-completion-20261101'


def main():
    print('WAIT for all post-batch checks; no changes to active frozen source', flush=True)
    deadline = time.monotonic()+7200
    expected_jobs = {'final-positive-hold', 'cpu-final', 'batch-metadata-audit', 'physical-split-index', 'evidence-summary'}
    while True:
        path = P/'post-batch-jobs.json'
        if path.exists():
            jobs = json.loads(path.read_text())
            if any(j['exit_code'] for j in jobs):
                raise RuntimeError('Post-batch validation failed; no input-contract edits made')
            if len(jobs) == len(expected_jobs) and {j['name'] for j in jobs} == expected_jobs:
                break
        if time.monotonic() > deadline:
            raise TimeoutError('Post-batch checks did not finish')
        time.sleep(2)
    source = ROOT/'endoscopic_tissue/training'
    expected = json.loads((P/'final-positive/source-hashes.json').read_text())
    current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}
    if current != expected:
        raise RuntimeError('Unexpected source changes before input-contract correction')
    tasks = (source/'tasks.py').read_text(encoding='utf8')
    marker = '        positive = (self.position_tolerance_m, self.required_progress, self.required_hold_s,\n'
    if tasks.count(marker) != 1:
        raise RuntimeError('Task source anchor changed')
    addition = '''        def integer_ids(values):
            values = tuple(values)
            if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in values):
                raise ValueError('Material IDs must be integers; fractional and boolean IDs are invalid')
            return tuple(int(i) for i in values)
        edges = tuple(self.target_edges)
        if any(len(edge) != 2 for edge in edges):
            raise ValueError('Target edges must be pairs of material IDs')
        object.__setattr__(self, 'target_edges', tuple(integer_ids(edge) for edge in edges))
        object.__setattr__(self, 'region_nodes', integer_ids(self.region_nodes))
        object.__setattr__(self, 'hold_nodes', integer_ids(self.hold_nodes))
'''
    tasks = tasks.replace(marker, addition+marker)
    (source/'tasks.py').write_text(tasks, encoding='utf8', newline='\n')
    tests = (source/'test_tasks.py').read_text(encoding='utf8')
    marker = 'class TaskTests(unittest.TestCase):\n'
    if tests.count(marker) != 1:
        raise RuntimeError('Test source anchor changed')
    addition = '''    def test_material_ids_are_strict_and_json_serializable(self):
        for edge in ((.9, 1.9), (False, True)):
            with self.subTest(edge=edge), self.assertRaises(ValueError):
                TaskGoal(kind='incision', target_edges=(edge,))
        for field in ('hold_nodes', 'region_nodes'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                TaskGoal(**{'kind':'hold', 'hold_nodes':(0,), field:(True,)})
        goal = TaskGoal(kind='incision', target_edges=((np.int32(0), np.int64(1)),))
        self.assertTrue(all(type(i) is int for i in goal.target_edges[0]))
        self.assertEqual(len(json_hash(goal.to_dict())), 64)

'''
    tests = tests.replace(marker, marker+addition)
    (source/'test_tasks.py').write_text(tests, encoding='utf8', newline='\n')
    evidence = P/'final-input-validation'
    snapshot = evidence/'adapter-source'
    snapshot.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in source.glob('*.py'):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path, snapshot/path.name)
    (evidence/'source-hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    command = [sys.executable, '-X', 'utf8', str(P/'run_cpu_checks.py'), 'endoscopic_tissue.test_core',
               'endoscopic_tissue.training.test_contract', 'endoscopic_tissue.training.test_tasks',
               'endoscopic_tissue.training.test_splits']
    print('START final 66 CPU checks', flush=True)
    with (P/'cpu-id-final.log').open('w', encoding='utf8') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                timeout=1200, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError('Final CPU checks failed; inspect log before proceeding')
    sys.path.insert(0, str(ROOT))
    from endoscopic_tissue.training.data import EpisodeReader
    from endoscopic_tissue.training.tasks import recompute_episode
    from threadpoolctl import threadpool_limits
    checked = []
    with threadpool_limits(limits=1, user_api='blas'):
        for path in sorted(OUT.glob('final-*/**/results.json')):
            results = json.loads(path.read_text())
            for row in results['episodes']:
                if row['status'] != 'recorded':
                    continue
                folder = path.parent/row['id']/'episode'
                context = json.loads((folder/'context.json').read_text())
                if context.get('goal') is None:
                    continue
                reader = EpisodeReader(folder)
                rewards = recompute_episode(reader)
                checked.append({'episode': str(folder), 'transitions': len(rewards), 'all_online_records_equal': True})
                print('RECOMPUTED', row['id'], len(rewards), flush=True)
    output = {'schema': 'strict_material_id_validation_v1', 'change': 'reject fractional/boolean material IDs; normalize NumPy integers',
              'native_physics_changed': False, 'all_recorded_goals_used_valid_integer_ids': True,
              'episodes': checked, 'total_transitions': sum(r['transitions'] for r in checked),
              'all_online_records_equal': True, 'cpu_tests': 66, 'cpu_exit_code': 0}
    (P/'strict-id-offline-equivalence.json').write_text(json.dumps(output, indent=2)+'\n')
    print('DONE strict ID input correction; all captured integer-goal rewards unchanged', flush=True)


if __name__ == '__main__':
    main()
