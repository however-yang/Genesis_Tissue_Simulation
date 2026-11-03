"""Single-trajectory counterpart of the existing paired full-frame capture."""
import json
import pickle

import numpy as np

from validation_capture import ReplayArchiveWriter, pack_frame, GraspMonitor, atomic_json, entry


class NativeCapture:
    def __init__(self, folder, scene, name):
        self.folder, self.scene, self.name = folder, scene, name
        self.label, self.expected = 'preparation', None
        self.rows, self.stages = [], []
        self.stream = ReplayArchiveWriter(folder/'replay.rdx')
        self.journal = (folder/'trajectory.jsonl').open('x', buffering=1, encoding='utf8')
        self.original = entry.NativeTissue
        capture = self

        class Recorder(self.original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                capture.monitor = GraspMonitor(self.settings, self.grid.inside)
                pickle.dump((self.settings, scene, name), capture.stream, protocol=5)
                pack_frame(capture.stream, 0, 'initial', self.surface(), self.observe())

            def step(self, command):
                observed = super().step(command)
                grasp = capture.monitor.assess_native(observed)
                row = dict(step=self.tick, stage=capture.label,
                    pad_slip_mm=grasp['witness_tangential_slip_m']*1000,
                    grip_points=observed.grip_points, bite_nodes=len(grasp['witness_ids']),
                    held_bite_nodes=round(len(grasp['witness_ids'])*grasp['witness_retained_fraction']),
                    cut_edges=int(np.count_nonzero((self.grid.neighbors[:,1::2]>=0)&(observed.neighbors[:,1::2]<0))),
                    jaw_load_n=list(observed.jaw_load_n), grasp=grasp)
                pack_frame(capture.stream, self.tick, capture.label, self.surface(), observed)
                capture.stream.flush()
                capture.journal.write(json.dumps(row)+'\n')
                capture.rows.append(row)
                if not capture.stages or capture.stages[-1]['name'] != capture.label:
                    capture.stages.append(dict(name=capture.label,end=dict(step=self.tick)))
                capture.stages[-1]['end']['step'] = self.tick
                capture.progress(False)
                return observed

        entry.NativeTissue = Recorder
        self.progress(False)

    def progress(self, closed):
        last = self.rows[-1]['step'] if self.rows else 0
        atomic_json(self.folder/'capture-progress.json',dict(closed=closed,last_recorded_step=last,
            expected_steps=self.expected,completed_requested_continuation=bool(closed and last==self.expected),
            physical_or_performance_approved=False))

    def close(self):
        entry.NativeTissue = self.original
        self.journal.close()
        self.stream.close()
        last = self.rows[-1]['step'] if self.rows else 0
        assert [r['step'] for r in self.rows] == list(range(1,last+1))
        atomic_json(self.folder/'report.json',dict(metadata=dict(case=self.name),stages=self.stages,
            steps=last,completed=False,expected_steps=self.expected,
            completed_requested_continuation=last==self.expected,
            scope='Full preparation plus requested dose, or all states up to a rejected preparation/quality limit'))
        self.progress(True)
