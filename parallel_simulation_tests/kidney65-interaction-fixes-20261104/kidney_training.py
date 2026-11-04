"""Opt-in kidney snapshot adapter; frozen training files are not overwritten."""
from pathlib import Path
from endoscopic_tissue import training

ROOT=Path(__file__).resolve().parent
training.__path__.insert(0,str(ROOT/'python/endoscopic_tissue/training'))
from endoscopic_tissue.training.env import NativeEnvironment
from endoscopic_tissue.training.kidney_snapshot import EnvSnapshot,take_snapshot,restore_snapshot


class KidneyEnvironment(NativeEnvironment):
    def snapshot(self):
        with self._state_lock:
            return take_snapshot(self)

    def restore(self,snapshot,*,branch_label=None,**kwargs):
        with self._state_lock:
            return restore_snapshot(self,snapshot,branch_label=branch_label,**kwargs)
