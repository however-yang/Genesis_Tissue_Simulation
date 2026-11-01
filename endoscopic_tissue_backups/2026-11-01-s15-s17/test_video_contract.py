"""Catch preparation omissions and mislabeled full videos without a GPU."""
import json
import hashlib
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

from render_validation import ROOT, preflight
from replay_archive import ReplayArchiveWriter
from endoscopic_tissue.native_backend import Settings


class VideoContract(unittest.TestCase):
    def setUp(self):
        root = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/cpu-temp'
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.folder = Path(self.temp.name)
        frames = [(0, 'initial'), (1, 'settle'), (2, 'incise_1')]
        stream = ReplayArchiveWriter(self.folder/'replay.rdx')
        pickle.dump((Settings(), 'wall-traction', 'CPU evidence fixture'), stream, protocol=5)
        self.rows = []
        for step, stage in frames:
            obs = SimpleNamespace(grip_points=step, jaw_load_n=(step*.2, step*.1))
            stream.write_frame(pickle.dumps((step, stage, None, obs), protocol=5))
            if step:
                self.rows.append({'step': step, 'stage': stage, 'grip_points': obs.grip_points,
                                  'jaw_load_n': list(obs.jaw_load_n)})
        stream.close()
        self.write_rows(self.rows)
        (self.folder/'report.json').write_text(json.dumps({'steps': 2, 'completed': False,
            'scope': 'CPU alignment fixture', 'expected_steps': 2, 'completed_requested_continuation': True}), encoding='utf8')

    def write_rows(self, rows):
        (self.folder/'trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf8')

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_recorded_continuation_is_not_claimed_as_full_original_scenario(self):
        result = preflight(self.folder)
        self.assertTrue(result['passed'])
        self.assertEqual(result['frames'], 3)
        self.assertFalse(result['full_original_scenario'])
        self.assertTrue(result['completed_requested_continuation'])

    def test_omitted_preparation_annotation_is_rejected(self):
        self.write_rows(self.rows[1:])
        with self.assertRaisesRegex(ValueError, 'preparation'):
            preflight(self.folder)

    def test_correct_ticks_with_wrong_frame_loads_are_rejected(self):
        self.rows[0]['jaw_load_n'] = [9., 9.]
        self.write_rows(self.rows)
        with self.assertRaisesRegex(ValueError, 'different states'):
            preflight(self.folder)


if __name__ == '__main__':
    out = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101'
    with (out/'video-contract-cpu.txt').open('w', encoding='utf8') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(VideoContract))
    summary = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'passed': result.wasSuccessful(), 'gpu_used': False,
        'scope': 'CPU replay/annotation contract only; no real interaction or renderer acceptance',
        'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ('test_video_contract.py', 'render_validation.py')}}
    (out/'video-contract-cpu.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf8')
    print(json.dumps(summary))
    raise SystemExit(0 if result.wasSuccessful() else 1)
