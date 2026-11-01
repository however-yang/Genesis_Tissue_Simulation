"""Bounded lossless recording and checked continuous-window reading."""
from io import BytesIO
import json
import os
from pathlib import Path
from queue import Full, Queue
import threading
import time

import numpy as np
import zstandard

from .provenance import json_hash, sha256


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_arrays(path, arrays):
    raw = BytesIO()
    np.savez(raw, **arrays)
    encoded = zstandard.ZstdCompressor(level=1).compress(raw.getbuffer())
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return sha256(path)


def load_arrays(path):
    raw = zstandard.ZstdDecompressor().decompress(path.read_bytes())
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def owned_state(state, include_topology=True):
    return {"metadata": json.loads(json.dumps(state.metadata, allow_nan=False)),
            "arrays": {k: v.copy() for k, v in state.arrays.items()},
            "topology": {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state.topology.items()
                         if include_topology or k == "revision"}}


class EpisodeWriter:
    """An episode directory is never reused. Backpressure is measured, never dropped."""
    def __init__(self, directory, context, initial_state, static_arrays, preparation=(), queue_size=4):
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        if initial_state.metadata["step_id"] != 0 or initial_state.metadata["episode_id"] != context["episode_id"]:
            raise ValueError("Writer needs this episode's actual s0")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.episode_id = context["episode_id"]
        self.context_hash = context["context_sha256"]
        self.queue = Queue(maxsize=queue_size)
        self.error = None
        self.closed = False
        self.next_step = 0
        self.states = 0
        self.transitions = 0
        self.failed_attempts = 0
        self.terminal_reason = None
        self.enqueue_ms = 0.
        self.write_ms = 0.
        self.topologies = {}
        self.queued_topologies = {initial_state.topology["revision"]}
        self.array_files = {}
        atomic_json(self.directory / "context.json", context)
        atomic_json(self.directory / "preparation.json", list(preparation))
        self.array_files["static.npz.zst"] = save_arrays(self.directory / "static.npz.zst", static_arrays)
        atomic_json(self.directory / "manifest.json", {"status": "recording", "episode_id": self.episode_id})
        self.worker = threading.Thread(target=self._work, name="tissue-episode-writer", daemon=True)
        self.worker.start()
        self._put(("initial", owned_state(initial_state)))

    def _put(self, value):
        if self.closed:
            raise RuntimeError("Recorder is closed")
        started = time.perf_counter()
        while True:
            if self.error:
                raise RuntimeError("Recorder failed: " + self.error)
            try:
                self.queue.put(value, timeout=.1)
                break
            except Full:
                if not self.worker.is_alive():
                    raise RuntimeError("Recorder worker stopped")
                if time.perf_counter()-started > 30:
                    raise TimeoutError("Recorder backpressure exceeded 30s; no sample was dropped")
        self.enqueue_ms += (time.perf_counter()-started)*1000

    def append(self, result):
        if (result.before.metadata["episode_id"] != self.episode_id
                or result.before.metadata["step_id"] != self.next_step):
            raise ValueError("Transition crosses an episode or is out of order")
        if self.terminal_reason is not None:
            raise ValueError("Cannot append after terminal transition")
        after = result.after
        if after is not None and (after.metadata["step_id"] != self.next_step+1
                                  or after.metadata["episode_id"] != self.episode_id):
            raise ValueError("Transition lacks its adjacent next state")
        record = {"episode_id": self.episode_id, "step_id": self.next_step,
                  "context_sha256": self.context_hash, "before": self.next_step,
                  "after": self.next_step+1 if after is not None else None,
                  "t_sim_s": result.before.metadata["t_sim_s"], "dt_s": result.before.metadata["dt_s"],
                  "action": result.action, "terminated": result.terminated, "truncated": result.truncated,
                  "reason": result.reason, "timings_ms": result.timings_ms,
                  "reward": result.task['reward'] if result.task is not None else None,
                  "task_success": result.task['success'] if result.task is not None else None,
                  "task_failure": result.task['failure'] if result.task is not None else None,
                  "task": result.task, "bootstrap_allowed": result.bootstrap_allowed}
        # Native failures preserve an attempted action, not a fabricated next state.
        new_topology = after is not None and after.topology["revision"] not in self.queued_topologies
        value = (json.loads(json.dumps(record, allow_nan=False)), owned_state(after, new_topology) if after is not None else None,
                 None if result.failure_nodes_native_cm is None else result.failure_nodes_native_cm.copy())
        self._put(("transition", value))
        if after is not None:
            self.queued_topologies.add(after.topology["revision"])
            self.next_step += 1
        if result.terminated or result.truncated:
            self.terminal_reason = result.reason or "terminal"

    def _write_state(self, state):
        revision = state["topology"]["revision"]
        if revision not in self.topologies:
            name = f"topology-{revision:08}.npz.zst"
            digest = save_arrays(self.directory / name, {k: v for k, v in state["topology"].items() if k != "revision"})
            self.topologies[revision] = name
            self.array_files[name] = digest
        index = state["metadata"]["step_id"]
        name = f"state-{index:08}.npz.zst"
        self.array_files[name] = save_arrays(self.directory / name, state["arrays"])
        metadata = {**state["metadata"], "arrays_file": name, "topology_file": self.topologies[revision]}
        atomic_json(self.directory / f"state-{index:08}.json", metadata)
        self.states += 1

    def _work(self):
        try:
            self._write_loop()
        except Exception as error:
            self.error = repr(error)
            # The producer polls this error; close also handles an already-dead worker.

    def _write_loop(self):
        with (self.directory / "transitions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            while True:
                item = self.queue.get()
                started = time.perf_counter()
                try:
                    if item is None:
                        stream.flush()
                        os.fsync(stream.fileno())
                        return
                    if self.error is not None:
                        continue
                    kind, value = item
                    if kind == "initial":
                        self._write_state(value)
                    else:
                        record, state, raw_failure = value
                        if state is not None:
                            self._write_state(state)
                            self.transitions += 1
                        else:
                            self.failed_attempts += 1
                        if raw_failure is not None:
                            name = f"failure-{record['step_id']:08}.npz.zst"
                            self.array_files[name] = save_arrays(self.directory / name, {"nodes_native_cm": raw_failure})
                            record["failure_arrays_file"] = name
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                except Exception as error:
                    self.error = repr(error)
                    if item is None:
                        return
                finally:
                    self.write_ms += (time.perf_counter()-started)*1000
                    self.queue.task_done()

    def close(self, reason=None):
        if self.closed:
            return
        # Worker drains the bounded queue even after a write error.
        if self.worker.is_alive():
            self.queue.put(None, timeout=30)
        self.worker.join(timeout=60)
        self.closed = True
        if self.worker.is_alive():
            raise TimeoutError("Recorder did not confirm its final writes; manifest stays incomplete")
        result = {"status": "write_failure" if self.error else "closed", "episode_id": self.episode_id,
                  "normal_transitions": self.transitions, "failed_attempts": self.failed_attempts,
                  "states": self.states, "reason": self.terminal_reason or reason or "collector_stop",
                  "context_sha256": self.context_hash, "arrays_sha256": self.array_files,
                  "record_enqueue_ms": self.enqueue_ms, "record_worker_ms": self.write_ms, "error": self.error}
        if not self.error and self.states != self.transitions+1:
            result["status"] = "integrity_failure"
        result["metadata_sha256"] = {p.name: sha256(p) for p in self.directory.iterdir()
                                     if p.suffix in (".json", ".jsonl") and p.name != "manifest.json"}
        atomic_json(self.directory / "manifest.json", result)
        if result["status"] != "closed":
            raise RuntimeError("Episode recording did not complete: " + str(result))

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        self.close("collector_exception" if error is not None else None)


class EpisodeReader:
    def __init__(self, directory, verify_hashes=True):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.context = json.loads((self.directory / "context.json").read_text(encoding="utf-8"))
        if self.manifest["status"] != "closed":
            raise ValueError("Incomplete or failed recording cannot supply training windows")
        expected = self.context["context_sha256"]
        if json_hash({k: v for k, v in self.context.items() if k != "context_sha256"}) != expected:
            raise ValueError("Context content/hash mismatch")
        if self.manifest["context_sha256"] != expected:
            raise ValueError("Episode version mismatch")
        self.transitions = [json.loads(line) for line in (self.directory / "transitions.jsonl").read_text(encoding="utf-8").splitlines()]
        expected_state = 0
        ended = False
        normal = 0
        state_meta = [json.loads((self.directory/f"state-{i:08}.json").read_text(encoding="utf-8"))
                      for i in range(self.manifest["states"])]
        if not state_meta or state_meta[0]["step_id"] != 0:
            raise ValueError("Missing actual s0")
        dt = state_meta[0]["dt_s"]
        for i, state in enumerate(state_meta):
            if (state["step_id"] != i or state["episode_id"] != self.context["episode_id"]
                    or not np.isfinite(dt) or dt <= 0 or state["dt_s"] != dt
                    or not np.isclose(state["t_sim_s"], i*dt, rtol=0, atol=1e-12)
                    or state["native_tick"] != state_meta[0]["native_tick"]+i):
                raise ValueError("State identity or simulation time alignment failed")
        for record in self.transitions:
            if (ended or record["episode_id"] != self.context["episode_id"] or record["context_sha256"] != expected
                    or record["before"] != expected_state or record["step_id"] != expected_state):
                raise ValueError("Episode mismatch, gap, reordering or transition after termination")
            if record["dt_s"] != dt or not np.isclose(record["t_sim_s"], expected_state*dt, rtol=0, atol=1e-12):
                raise ValueError("Action/state timing mismatch")
            if record["after"] is not None:
                if record["after"] != expected_state+1:
                    raise ValueError("Nonadjacent transition")
                expected_state += 1
                normal += 1
            elif not record["terminated"]:
                raise ValueError("Missing next state without explicit terminal failure")
            ended = record["terminated"] or record["truncated"]
        if (normal != self.manifest["normal_transitions"] or self.manifest["states"] != normal+1
                or len(self.transitions)-normal != self.manifest["failed_attempts"]):
            raise ValueError("N transitions / N+1 states contract failed")
        if verify_hashes:
            for name, digest in {**self.manifest["arrays_sha256"], **self.manifest["metadata_sha256"]}.items():
                if Path(name).name != name or sha256(self.directory/name) != digest:
                    raise ValueError("Array path or hash mismatch: "+name)

    def state(self, index):
        return self._state(index, {})

    def _state(self, index, topology_cache):
        metadata = json.loads((self.directory/f"state-{index:08}.json").read_text(encoding="utf-8"))
        if metadata["step_id"] != index or metadata["episode_id"] != self.context["episode_id"]:
            raise ValueError("State identity mismatch")
        for field in ("arrays_file", "topology_file"):
            if Path(metadata[field]).name != metadata[field] or metadata[field] not in self.manifest["arrays_sha256"]:
                raise ValueError("State references an unknown array file")
        arrays = load_arrays(self.directory/metadata["arrays_file"])
        name = metadata["topology_file"]
        if name not in topology_cache:
            topology_cache[name] = load_arrays(self.directory/name)
        topology = topology_cache[name]
        for value in (*arrays.values(), *topology.values()):
            value.flags.writeable = False
        return metadata, arrays, topology

    def windows(self, length, *, allow_unvalidated_scope=False, allow_diagnostic=False):
        if length < 1:
            raise ValueError("Window length must be positive")
        if not allow_unvalidated_scope and self.context["domain_validation"] != "approved":
            raise ValueError("Action domain is not approved; explicitly opt into exploratory data")
        if self.context.get("diagnostic_continue") and not allow_diagnostic:
            raise ValueError("Diagnostic replay requires an explicit reader opt-in")
        state_cache, topology_cache = {}, {}
        for start in range(len(self.transitions)-length+1):
            records = self.transitions[start:start+length]
            if any(r["after"] is None for r in records):
                continue
            if any(r["terminated"] or r["truncated"] for r in records[:-1]):
                continue
            indices = [records[0]["before"]] + [r["after"] for r in records]
            state_cache = {i: s for i, s in state_cache.items() if i in indices}
            for index in indices:
                if index not in state_cache:
                    state_cache[index] = self._state(index, topology_cache)
            states = [state_cache[i] for i in indices]
            needed = {s[0]["topology_file"] for s in states}
            topology_cache = {k: v for k, v in topology_cache.items() if k in needed}
            if any(not s[0]["quality"]["checked_metrics_passed"] or not s[0]["quality"]["within_declared_local_scope"] for s in states):
                continue
            # Only a sliding window stays cached. Consumers receive private metadata
            # and read-only arrays so overlapping windows cannot change one another.
            safe_states = [(json.loads(json.dumps(s[0])), s[1].copy(), s[2].copy()) for s in states]
            yield {"context": self.context, "transitions": records, "states": safe_states}
