"""Complete native-frame video inputs for paired snapshot trajectories."""
from dataclasses import asdict
from pathlib import Path
import json
import pickle
import sys

from validation_support import ROOT
sys.path.insert(0, str(ROOT/'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation'))
sys.path.insert(0, str(ROOT/'endoscopic_tissue_backups/2026-10-31-interaction-fixes'))
import numpy as np
from replay_archive import ReplayArchiveWriter
from validate_strict import pack_frame
from endoscopic_tissue.training import env as entry
from endoscopic_tissue.training.grasp import GraspMonitor
from endoscopic_tissue.training.data import atomic_json


class PairedCapture:
    def __init__(self, root, scene, name):
        self.scene, self.name = scene, name
        self.label, self.arm = 'preparation', None
        self.expected_end_step = None
        self.streams, self.rows, self.stages, self.journals = {}, {}, {}, {}
        self.root = Path(root)
        for arm in ('original', 'restored'):
            folder = self.root/arm
            folder.mkdir()
            self.streams[arm] = ReplayArchiveWriter(folder/'replay.rdx')
            self.rows[arm], self.stages[arm] = [], []
            self.journals[arm] = (folder/'trajectory.jsonl').open('x', encoding='utf8', buffering=1)
            self.progress(arm)
        original = entry.NativeTissue
        capture = self

        class Recorder(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                capture.native = self
                capture.monitor = GraspMonitor(self.settings, self.grid.inside)
                for stream in capture.streams.values():
                    pickle.dump((self.settings, capture.scene, capture.name), stream, protocol=5)
                    pack_frame(stream, 0, 'initial', self.surface(), self.observe())

            def step(self, command):
                observed = super().step(command)
                capture.frame(self, observed)
                return observed

        self.original_class = original
        entry.NativeTissue = Recorder

    def progress(self, arm, *, closed=False):
        last = self.rows[arm][-1]['step'] if self.rows[arm] else 0
        atomic_json(self.root/arm/'capture-progress.json', {
            'closed': closed, 'last_recorded_step': last,
            'expected_steps': self.expected_end_step,
            'completed_requested_continuation': bool(closed and self.expected_end_step is not None
                                                      and last == self.expected_end_step),
            'physical_or_performance_approved': False})

    def frame(self, native, observed):
        grasp = self.monitor.assess_native(observed)
        surface = native.surface()
        row = {'step': native.tick, 'stage': self.label,
            'pad_slip_mm': grasp['witness_tangential_slip_m']*1000,
            'grip_points': observed.grip_points, 'bite_nodes': len(grasp['witness_ids']),
            'held_bite_nodes': round(len(grasp['witness_ids'])*grasp['witness_retained_fraction']),
            'cut_edges': int(np.count_nonzero((native.grid.neighbors[:, 1::2] >= 0)&(observed.neighbors[:, 1::2] < 0))),
            'jaw_load_n': list(observed.jaw_load_n), 'grasp': grasp}
        arms = self.streams.keys() if self.arm is None else (self.arm,)
        for arm in arms:
            pack_frame(self.streams[arm], native.tick, self.label, surface, observed)
            # Diagnostic capture only. Flush the corresponding frame before
            # committing its annotation; a native stall can bypass close().
            self.streams[arm].flush()
            self.journals[arm].write(json.dumps(row)+'\n')
            self.rows[arm].append(row)
            if not self.stages[arm] or self.stages[arm][-1]['name'] != self.label:
                self.stages[arm].append({'name': self.label, 'end': {'step': native.tick}})
            self.stages[arm][-1]['end']['step'] = native.tick
            self.progress(arm)

    def close(self):
        entry.NativeTissue = self.original_class
        for arm, stream in self.streams.items():
            self.journals[arm].close()
            stream.close()
            folder = self.root/arm
            last = self.rows[arm][-1]['step'] if self.rows[arm] else 0
            assert [r['step'] for r in self.rows[arm]] == list(range(1, last+1)), 'Capture lost or repeated native frames'
            atomic_json(folder/'report.json', {'metadata': {'case': self.name+'/'+arm}, 'stages': self.stages[arm],
                'steps': last, 'completed': False, 'expected_steps': self.expected_end_step,
                'completed_requested_continuation': self.expected_end_step is not None and last == self.expected_end_step,
                'scope': 'complete preparation/fork prefix and tested continuation; not the full three-snip scenario'})
            self.progress(arm, closed=True)
