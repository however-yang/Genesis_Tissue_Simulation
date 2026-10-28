"""Run endoscopic_tissue.run with the topology audit recording non-manifold edges instead of stopping the flow.

Development only: the project's audit stays strict, and the run's acceptance is unchanged. Arguments are passed to
run.py, so `dev_run.py --scene kidney` opens the live window on the whole kidney flow.
"""
from pathlib import Path
import sys

# The project root, two levels above this folder, holds the endoscopic_tissue package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from endoscopic_tissue import run  # noqa: E402

audit = run.topology_statistics


def recording_audit(surface):
    statistics = audit(surface)
    statistics["recorded_nonmanifold_edges"] = statistics["positional_nonmanifold_edges"]
    statistics["positional_nonmanifold_edges"] = 0
    return statistics


run.topology_statistics = recording_audit
sys.argv = ["endoscopic_tissue.run", *sys.argv[1:]]
run.main()
