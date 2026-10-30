"""Run a scissors flow with a different number of solver substeps per frame: SUBSTEPS (default 2) replaces the 3 of
scenario.scissors_settings. With --probe, the run goes through realism_probe.py (its first argument is the output jsonl);
without, like dev_run.py, with the topology audit recording non-manifold edges instead of stopping the flow.

Usage: python substeps.py [--probe <output jsonl>] [run.py arguments...]
"""
import os
import runpy
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from endoscopic_tissue import run, scenario  # noqa: E402

SUBSTEPS = int(os.environ.get("SUBSTEPS", "2"))
scissors_settings = scenario.scissors_settings


def settings_with_substeps():
    return replace(scissors_settings(), substeps=SUBSTEPS)


scenario.scissors_settings = settings_with_substeps
run.scissors_settings = settings_with_substeps
if sys.argv[1:2] == ["--probe"]:
    sys.argv = [str(Path(__file__).resolve().parents[1] / "2026-10-29-stage3" / "realism_probe.py"), *sys.argv[2:]]
    runpy.run_path(sys.argv[0], run_name="__main__")
else:
    audit = run.topology_statistics

    def recording_audit(surface):
        statistics = audit(surface)
        statistics["recorded_nonmanifold_edges"] = statistics["positional_nonmanifold_edges"]
        statistics["positional_nonmanifold_edges"] = 0
        return statistics

    run.topology_statistics = recording_audit
    sys.argv = ["endoscopic_tissue.run", *sys.argv[1:]]
    run.main()
