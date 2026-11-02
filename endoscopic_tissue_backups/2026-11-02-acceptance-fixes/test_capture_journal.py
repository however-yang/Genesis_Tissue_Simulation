"""Real process-exit checks for optional capture; no native DLL/GPU is loaded."""
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'endoscopic_tissue_backups/2026-11-01-s15-s17'), str(ROOT)]
import validation_capture as capture_module
from endoscopic_tissue.native_backend import Settings
from replay_archive import open_replay
import numpy as np


def child(folder, complete):
    class FakeNative:
        def __init__(self):
            self.settings = Settings()
            self.tick = 0
            self.grid = SimpleNamespace(inside=np.array([1]), neighbors=np.full((1, 6), -1))

        def surface(self):
            return None

        def observe(self):
            return SimpleNamespace(grip_points=1, jaw_load_n=(.2, .3), neighbors=self.grid.neighbors)

        def step(self, command):
            self.tick += 1
            return self.observe()

    class FakeMonitor:
        def __init__(self, *args):
            pass

        def assess_native(self, observed):
            return {'witness_tangential_slip_m': 0., 'witness_ids': [0],
                    'witness_retained_fraction': 1.}

    capture_module.entry.NativeTissue = FakeNative
    capture_module.GraspMonitor = FakeMonitor
    capture_module.pack_frame = lambda stream, step, stage, surface, observed: stream.write_frame(
        pickle.dumps((step, stage, surface, observed), protocol=5))
    capture = capture_module.PairedCapture(folder, 'wall-traction', 'CPU interruption fixture')
    capture.expected_end_step = 2
    native = capture_module.entry.NativeTissue()
    native.step(None)
    if not complete:
        # Bypass Python finalizers, exactly as a supervised native stall does.
        os._exit(23)
    native.step(None)
    capture.close()


class CaptureJournal(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_child(self, complete):
        result = subprocess.run([sys.executable, '-X', 'utf8', __file__, '--child',
                                 str(self.folder), 'complete' if complete else 'interrupt'],
                                cwd=ROOT, capture_output=True, text=True, timeout=30,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.assertEqual(result.returncode, 0 if complete else 23, result.stderr)

    def frames(self, arm):
        frames = []
        with open_replay(arm/'replay.bin') as stream:
            pickle.load(stream)
            while header := stream.read(4):
                size, = struct.unpack('<I', header)
                frames.append(pickle.loads(zlib.decompress(stream.read(size))))
        return frames

    def test_abrupt_exit_preserves_last_annotated_frame_and_never_claims_completion(self):
        self.run_child(False)
        for name in ('original', 'restored'):
            arm = self.folder/name
            rows = [json.loads(s) for s in (arm/'trajectory.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual([r['step'] for r in rows], [1])
            self.assertEqual([f[0] for f in self.frames(arm)], [0, 1])
            progress = json.loads((arm/'capture-progress.json').read_text(encoding='utf8'))
            self.assertFalse(progress['closed'])
            self.assertFalse(progress['completed_requested_continuation'])
            self.assertEqual(progress['last_recorded_step'], 1)
            self.assertFalse((arm/'report.json').exists())

    def test_normal_close_preserves_exact_archive_and_existing_completion_semantics(self):
        self.run_child(True)
        for name in ('original', 'restored'):
            arm = self.folder/name
            report = json.loads((arm/'report.json').read_text(encoding='utf8'))
            self.assertFalse(report['completed'])
            self.assertTrue(report['completed_requested_continuation'])
            self.assertEqual([f[0] for f in self.frames(arm)], [0, 1, 2])
            archive = json.loads((arm/'replay-archive.json').read_text(encoding='utf8'))
            self.assertTrue(archive['exact_encoded_byte_roundtrip'])


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        child(Path(sys.argv[2]), sys.argv[3] == 'complete')
    else:
        unittest.main()
