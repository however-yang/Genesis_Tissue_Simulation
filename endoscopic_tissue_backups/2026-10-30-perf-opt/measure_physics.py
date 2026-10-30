"""Observe the unchanged default flow with CUDA graphs enabled.

Usage: python measure_baseline.py <new output directory> [run.py options]
Uses the existing kidney dev entrypoint; its recorded topology failures are
retained. No solver settings, validation thresholds, or DLLs are changed.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from endoscopic_tissue import native_backend, run

output = Path(sys.argv[1]).resolve()
run_args = sys.argv[2:]
rows, stages = [], []
state = {"stage": None, "row": None, "last_submit": None}
metadata, record_ms = {}, {}
boundary_costs = defaultdict(Counter)
check = native_backend.NativeTissue.check
step = native_backend.NativeTissue.step
surface = native_backend.NativeTissue.surface
stage_start = run.stage_start
flow_open, flow_run = run.Flow.open, run.Flow.run
submit, record, drain = run.StepRecorder.submit, run.StepRecorder.record, run.StepRecorder.drain
savez, finish_stage, audit = np.savez_compressed, run.Sequence.finish_stage, run.topology_statistics


def close_stage():
    if state["stage"] is not None:
        stages.append({"stage": state["stage"],
                       "wall_ms": 1000 * (time.perf_counter() - state["stage_started"]),
                       "boundary_calls_ms": dict(boundary_costs[state["stage"]])})
        state["stage"] = None


def timed_stage(name, *args):
    close_stage()
    result = stage_start(name, *args)
    state.update(stage=name, stage_started=time.perf_counter(), last_submit=time.perf_counter())
    return result


def opened(self, tissue):
    flow_open(self, tissue)
    g = self.grid
    metadata.update(total_grid_nodes=len(g.positions), active_nodes=int(np.count_nonzero(g.active)),
                    material_nodes=int(np.count_nonzero(g.inside)),
                    movable_active_nodes=int(np.count_nonzero(g.active & ~g.pinned.astype(bool))),
                    cells=len(g.corners), cell_size_cm=g.cell_size.tolist(),
                    dll=str(tissue.dll_path.resolve()),
                    dll_sha256=hashlib.sha256(tissue.dll_path.read_bytes()).hexdigest())
    print('[PERF] ' + json.dumps(metadata), flush=True)


def checked(self, name, *args):
    start = time.perf_counter()
    try:
        return check(self, name, *args)
    finally:
        if state["in_step"]:
            state["row"]["calls_ms"][name] += 1000 * (time.perf_counter() - start)


def stepped(self, command):
    started = time.perf_counter()
    row = {"step": self.tick + 1, "stage": state["stage"], "calls_ms": Counter(),
           "control_ms": 1000 * (started - state["last_submit"]), "surface_ms": 0.0}
    state.update(row=row, in_step=True, iteration_start=state["last_submit"])
    try:
        obs = step(self, command)
    finally:
        row["native_step_ms"] = 1000 * (time.perf_counter() - started)
        state["in_step"] = False
    row.update(jaw_gap_mm=obs.jaw_gap * 10, jaw_load_n=list(obs.jaw_load_n), grip_points=obs.grip_points)
    rows.append(row)
    return obs


def surfaced(self):
    start = time.perf_counter()
    result = surface(self)
    if state["row"] is not None:
        state["row"]["surface_ms"] += 1000 * (time.perf_counter() - start)
    return result


def submitted(self, *args):
    start = time.perf_counter()
    result = submit(self, *args)
    end = time.perf_counter()
    state["row"].update(submit_ms=1000 * (end - start),
                        pipeline_ms=1000 * (end - state["iteration_start"]),
                        recorder_queue_size=self.steps.qsize())
    state.update(last_submit=end, row=None)
    return result


def recorded(self, stage, number, *args):
    start = time.perf_counter()
    try:
        return record(self, stage, number, *args)
    finally:
        record_ms[number] = 1000 * (time.perf_counter() - start)


def flowed(self, *args):
    try:
        return flow_run(self, *args)
    finally:
        close_stage()


def boundary_timed(name, function):
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            boundary_costs[state["stage"]][name] += 1000 * (time.perf_counter() - start)
    return wrapped


state["in_step"] = False
native_backend.NativeTissue.check = checked
native_backend.NativeTissue.step = stepped
native_backend.NativeTissue.surface = surfaced
run.stage_start = timed_stage
run.Flow.open, run.Flow.run = opened, flowed
run.StepRecorder.submit, run.StepRecorder.record = submitted, recorded
run.StepRecorder.drain = boundary_timed("recorder_drain", drain)
np.savez_compressed = boundary_timed("snapshot_compression", savez)
run.Sequence.finish_stage = boundary_timed("stage_acceptance", finish_stage)
run.topology_statistics = boundary_timed("topology_audit", audit)
sys.argv = ["dev_run.py", "--output", str(output), *run_args]
try:
    runpy.run_path(str(ROOT / "endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py"), run_name="__main__")
finally:
    if output.is_dir():
        for row in rows:
            row["async_record_ms"] = record_ms.get(row["step"])
        (output / "timing-steps.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["stage"]].append(row)
        summary = []
        for stage in stages:
            values = grouped[stage["stage"]]
            if not values:
                continue
            n = len(values)
            result = dict(stage, steps=n, wall_ms_per_step=stage["wall_ms"] / n)
            for name in ("pipeline_ms", "native_step_ms", "surface_ms", "control_ms", "async_record_ms"):
                samples = np.array([r[name] for r in values if r.get(name) is not None])
                if len(samples):
                    result[name] = dict(mean=float(samples.mean()), median=float(np.median(samples)),
                                        p95=float(np.percentile(samples, 95)), maximum=float(samples.max()))
            calls = Counter()
            for row in values:
                calls.update(row["calls_ms"])
            result["calls_ms_per_step"] = {k: v / n for k, v in calls.items()}
            result["stage_boundary_ms"] = stage["wall_ms"] - sum(r.get("pipeline_ms", 0) for r in values)
            summary.append(result)
            print('[PERF] ' + stage["stage"] + ' ' + json.dumps(result), flush=True)
        (output / "timing-summary.json").write_text(json.dumps({"metadata": metadata, "stages": summary}, indent=2), encoding="utf8")
