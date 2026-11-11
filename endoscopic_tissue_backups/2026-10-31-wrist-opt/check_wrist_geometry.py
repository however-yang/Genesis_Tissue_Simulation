"""Check the GPU wrist helper against an independent float64 constrained LS oracle."""
from dataclasses import asdict, dataclass
import ctypes as ct
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits


@dataclass
class Case:
    name: str
    values: list
    fallback: bool = False


@dataclass
class Result:
    cases: int
    changed_cases: int
    baseline_identical_cases: int
    required_fallback_cases: int
    max_analytic_oracle_error_cm: float
    max_distance_worse_than_baseline_cm: float
    max_barycentric_sum_error: float


def cases():
    result = []
    triangle = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    known = (
        ("unique", triangle, [.2, .2, .2], [.3, .3, .8], False),
        ("axis_crossing", triangle, [.2, .2, -1], [.2, .2, 1], True),
        ("zero_axis", triangle, [.2, .2, .1], [.2, .2, .1], True),
        ("parallel_plane", triangle, [.2, .2, .1], [.8, .1, .1], True),
        ("near_parallel", triangle, [.2, .2, .1], [.8, .1, .100001], True),
        ("point_triangle", [[0, 0, 0]] * 3, [.2, .2, .2], [.3, .3, .8], True),
        ("collinear", [[0, 0, 0], [.5, 0, 0], [1, 0, 0]], [.2, .2, .2], [.3, .3, .8], True),
        ("sliver", [[0, 0, 0], [1, 0, 0], [.5, .000001, 0]], [.2, .2, .2], [.3, .3, .8], True),
    )
    for name, vertices, a, b, fallback in known:
        for order in itertools.permutations(range(3)):
            result.append(Case(name, [value for i in order for value in vertices[i]] + a + b + [.1], fallback))
    rng = np.random.default_rng(20261031)
    for i in range(1200):
        center = rng.uniform(-3, 3, 3)
        vertices = center + rng.normal(size=(3, 3)) * 10 ** rng.uniform(-2, .3)
        a = center + rng.normal(size=3) * .5
        b = a + rng.normal(size=3) * 10 ** rng.uniform(-2, .3)
        result.append(Case("random", [value for point in vertices.tolist() for value in point] + a.tolist() + b.tolist() + [.1]))
    for i in range(150):
        epsilon = 10 ** rng.uniform(-10, -1)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [.5, epsilon, 0]])
        a, b = [.2, .2, .2], [.3, .3, .8]
        result.append(Case("aspect_sweep", [value for point in vertices.tolist() for value in point] + a + b + [.1]))
        result.append(Case("parallel_sweep", [value for p in triangle for value in p] +
                           [.2, .2, .1, .8, .1, .1 + epsilon, .1]))
        result.append(Case("short_axis_sweep", [value for p in triangle for value in p] +
                           [.2, .2, .1, .2, .2, .1 + epsilon, .1]))
    return result


def oracle_faces():
    # x=(segment t, triangle u, triangle v). All faces of its convex feasible polytope.
    constraints = np.array([[-1., 0, 0], [1., 0, 0], [0, -1., 0], [0, 0, -1.], [0, 1., 1.]])
    bounds = np.array([0., 1., 0., 0., 1.])
    faces = [(np.zeros(3), np.eye(3))]
    for size in range(1, 4):
        for subset in itertools.combinations(range(5), size):
            matrix = constraints[list(subset)]
            target = bounds[list(subset)]
            particular = np.linalg.lstsq(matrix, target, rcond=None)[0]
            if np.max(np.abs(matrix @ particular - target)) > 1e-12:
                continue
            _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
            rank = np.count_nonzero(singular > 1e-12)
            faces.append((particular, vh[rank:].T))
    return constraints, bounds, faces


def reference_distance(case, constraints, bounds, faces):
    points = case[:9].reshape(3, 3)
    a, b = case[9:12], case[12:15]
    matrix = np.column_stack((b - a, points[0] - points[1], points[0] - points[2]))
    residual = a - points[0]
    best = np.inf
    for particular, nullspace in faces:
        solution = particular
        if nullspace.shape[1]:
            solution = particular + nullspace @ np.linalg.lstsq(matrix @ nullspace, -residual - matrix @ particular, rcond=None)[0]
        if np.max(constraints @ solution - bounds) <= 1e-8:
            best = min(best, np.linalg.norm(residual + matrix @ solution))
    assert np.isfinite(best)
    return best - case[15]


def main():
    sample = cases()
    values = np.array([case.values for case in sample], dtype=np.float32)
    output = np.empty((len(sample), 12), dtype=np.float32)
    library = ct.CDLL(str(Path(sys.argv[1]).resolve()))
    library.LCS_ProbeWristGeometry.argtypes = (ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_float))
    library.LCS_ProbeWristGeometry.restype = ct.c_int
    status = library.LCS_ProbeWristGeometry(len(sample), values.ctypes.data_as(ct.POINTER(ct.c_float)),
                                           output.ctypes.data_as(ct.POINTER(ct.c_float)))
    assert status == 0, status
    assert np.isfinite(output).all()
    old, new, exact = output[:, :3], output[:, 3:6], output[:, 6:9]
    assert np.min(new) >= -1e-6 and np.max(np.abs(new.sum(axis=1) - 1)) <= 1e-5
    fallback = np.all(new == old, axis=1)
    required = np.array([case.fallback for case in sample])
    assert np.all(fallback[required]), [sample[i].name for i in np.flatnonzero(required & ~fallback)]
    assert np.count_nonzero(~fallback) >= 500, "Analytic branch is not exercised"
    np.testing.assert_array_equal(new[~fallback], exact[~fallback])
    constraints, bounds, faces = oracle_faces()
    reference = np.array([reference_distance(case, constraints, bounds, faces) for case in values.astype(np.float64)])
    tolerance = 1e-5 * np.maximum(1, np.max(np.abs(values[:, :15]), axis=1))
    errors = np.abs(output[:, 10] - reference)
    assert np.all(errors[~fallback] <= tolerance[~fallback]), np.max(errors[~fallback] / tolerance[~fallback])
    worse = output[:, 10] - output[:, 9]
    assert np.all(worse <= tolerance), np.max(worse / tolerance)
    result = Result(len(sample), int(np.count_nonzero(~fallback)), int(np.count_nonzero(fallback)),
                    int(np.count_nonzero(required)), float(np.max(errors[~fallback])),
                    float(np.max(worse)), float(np.max(np.abs(new.sum(axis=1) - 1))))
    Path(sys.argv[2]).write_text(json.dumps(asdict(result), indent=2), encoding="utf8")
    print(json.dumps(asdict(result)), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1, user_api="blas"):
        main()
