"""Small constraint-generation QP reference; diagnostic, not a simulator."""
from itertools import combinations

import numpy as np


def solve_constraints(H, gradient, unconstrained, normals, lower):
    inverse_normals = np.linalg.solve(H, normals.T).T
    best = np.zeros(3)
    best_energy = 0.
    for count in range(min(3, len(normals))+1):
        for subset in combinations(range(len(normals)), count):
            if count:
                ids = np.array(subset)
                C = normals[ids]@inverse_normals[ids].T
                if np.linalg.det(C) <= 1e-12*np.diag(C).prod():
                    continue
                multipliers = np.linalg.solve(C, lower[ids]-normals[ids]@unconstrained)
                candidate = unconstrained+inverse_normals[ids].T@multipliers
            else:
                candidate = unconstrained
            if (normals@candidate < lower-1e-9).any():
                continue
            energy = .5*candidate@H@candidate+gradient@candidate
            if np.isfinite(energy) and energy < best_energy:
                best, best_energy = candidate.copy(), float(energy)
    return best


def small_cone(H, gradient, A, margin, *, passes=8):
    unconstrained = -np.linalg.solve(H, gradient)
    candidate = unconstrained.copy()
    chosen = []
    norms = np.linalg.norm(A, axis=1)
    assert (norms > 0).all() and (margin >= 0).all()
    def bound(direction):
        derivative = A@direction
        limits = np.ones(len(A))
        np.divide(margin, -derivative, out=limits, where=derivative < 0)
        return np.clip(limits, 0, 1)
    original = unconstrained*bound(unconstrained).min(initial=1)
    for _ in range(passes):
        limits = bound(candidate)
        worst = int(limits.argmin())
        if limits[worst] >= 1 or worst in chosen:
            break
        chosen.append(worst)
        ids = np.asarray(chosen)
        candidate = solve_constraints(H, gradient, unconstrained, A[ids]/norms[ids, None],
                                      -margin[ids]/norms[ids])
    final_scale = bound(candidate).min(initial=1)
    candidate *= final_scale
    def energy(direction): return float(.5*direction@H@direction+gradient@direction)
    fallback = energy(candidate) > energy(original)
    if fallback:
        candidate = original
    return dict(direction_cm=candidate.tolist(), selected_constraints=chosen,
                final_scale=float(final_scale), quadratic_change=energy(candidate),
                original_quadratic_change=energy(original), fallback=fallback,
                worst_residual=float((A@candidate+margin).min()))
