"""Compare tet energy differences E(p+a*v)-E(p): the solver's current form (float positions, double energy), the
stable single-precision expansion planned for the device, and an all-double reference."""
import numpy as np

rng = np.random.default_rng(1)
f32, f64 = np.float32, np.float64


def energy(p, b, r, mu, lam, damping_c, hardening, minimum):
    """Tet energy of unifiedTetEnergy evaluated in the dtype of p (barrier left out: J stays far above it)."""
    e = p[1:] - p[0]
    det = np.dot(b[0], np.cross(b[1], b[2]))
    F = (e.T @ b)                                  # columns f[c] = sum_k e_k b_k[c]
    J = np.dot(e[0], np.cross(e[1], e[2])) * det
    E = .5 * mu * (np.sum(F * F) - 3) - mu * (J - 1) + .5 * lam * (J - 1) ** 2
    G = F @ r                                      # g[c] = sum_a f[a] r_a[c]
    S = G.T @ G - np.eye(3)
    E += damping_c * np.sum(S * S)
    rest = np.array([np.zeros(3), np.cross(b[1], b[2]) / det, np.cross(b[2], b[0]) / det, np.cross(b[0], b[1]) / det])
    for j in range(4):
        for k in range(j + 1, 4):
            l2, r2 = np.sum((p[j] - p[k]) ** 2), np.sum((rest[j] - rest[k]) ** 2)
            if l2 > r2:
                s = np.sqrt(l2 / r2) - 1
                E += .5 * mu * hardening * s ** 6
    return E


def delta_stable(p, v, a, b, r, mu, lam, damping_c, hardening):
    """Single-precision expansion of E(p+a*v)-E(p) in powers of a, term by term."""
    p, v, b, r = p.astype(f32), v.astype(f32), b.astype(f32), r.astype(f32)
    mu, lam, damping_c, hardening, a = f32(mu), f32(lam), f32(damping_c), f32(hardening), f32(a)
    e, d = p[1:] - p[0], v[1:] - v[0]
    det = np.dot(b[0], np.cross(b[1], b[2]))
    F, dF = e.T @ b, d.T @ b
    FdF, dF2 = np.sum(F * dF), np.sum(dF * dF)
    j0 = np.dot(e[0], np.cross(e[1], e[2])) * det
    j1 = (np.dot(d[0], np.cross(e[1], e[2])) + np.dot(e[0], np.cross(d[1], e[2])) + np.dot(e[0], np.cross(e[1], d[2]))) * det
    j2 = (np.dot(d[0], np.cross(d[1], e[2])) + np.dot(d[0], np.cross(e[1], d[2])) + np.dot(e[0], np.cross(d[1], d[2]))) * det
    j3 = np.dot(d[0], np.cross(d[1], d[2])) * det
    dJ = a * (j1 + a * (j2 + a * j3))
    out = mu * (a * FdF + f32(.5) * a * a * dF2) - mu * dJ + f32(.5) * lam * dJ * (f32(2) * (j0 - f32(1)) + dJ)
    G, dG = F @ r, dF @ r
    S0 = G.T @ G - np.eye(3, dtype=f32)
    S1 = G.T @ dG + dG.T @ G
    S2 = dG.T @ dG
    out += damping_c * (f32(2) * a * np.sum(S0 * S1) + a * a * (np.sum(S1 * S1) + f32(2) * np.sum(S0 * S2))
                        + f32(2) * a ** 3 * np.sum(S1 * S2) + a ** 4 * np.sum(S2 * S2))
    rest = np.array([np.zeros(3), np.cross(b[1], b[2]) / det, np.cross(b[2], b[0]) / det, np.cross(b[0], b[1]) / det], dtype=f32)
    for j in range(4):
        for k in range(j + 1, 4):
            x, y = p[j] - p[k], v[j] - v[k]
            r2 = np.sum((rest[j] - rest[k]) ** 2)
            d0 = np.sum(x * x) - r2
            da = d0 + a * (f32(2) * np.dot(x, y) + a * np.dot(y, y))
            strain = lambda dl: dl / (r2 * (np.sqrt(f32(1) + dl / r2) + f32(1))) if dl > 0 else f32(0)
            out += f32(.5) * mu * hardening * (strain(da) ** 6 - strain(d0) ** 6)
    return out


cells = .1
rest_tet = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], float) * cells
B = np.linalg.inv((rest_tet[1:] - rest_tet[0]).T).T      # rows b_k with F = E @ B
errors = {"current": [], "stable": []}
for trial in range(2000):
    # A deformed tet far from the origin (world positions up to 5 cm), with 10% strain and a rotation.
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    strain = np.eye(3) + .1 * rng.normal(size=(3, 3))
    p = (rest_tet @ (q @ strain).T) + rng.uniform(-5, 5, size=3)
    p = p.astype(f32).astype(f64)                         # the solver stores float positions
    v = (rng.normal(size=(4, 3)) * 10 ** rng.uniform(-5, -2)).astype(f32).astype(f64)
    a = [1, .5, .25, .125][trial % 4]
    prev_strain = np.eye(3) + .05 * rng.normal(size=(3, 3))
    r = np.linalg.inv(q @ prev_strain).astype(f32).astype(f64)   # previous inverse, F_prev^-1
    mu, lam, damping_c, hardening = 1.0, 4.0, .3, 20.0
    reference = energy(p + a * v, B, r, mu, lam, damping_c, hardening, 0) - energy(p, B, r, mu, lam, damping_c, hardening, 0)
    # Current device form: trial positions rounded to float, energy in double.
    q32 = (p.astype(f32) + v.astype(f32) * f32(a)).astype(f64)
    current = energy(q32, B.astype(f32).astype(f64), r, mu, lam, damping_c, hardening, 0) - energy(p, B.astype(f32).astype(f64), r, mu, lam, damping_c, hardening, 0)
    stable = delta_stable(p, v, a, B, r, mu, lam, damping_c, hardening)
    scale = abs(reference) + 1e-30
    errors["current"].append((current - reference) / scale)
    errors["stable"].append((float(stable) - reference) / scale)
for name, values in errors.items():
    values = np.array(values)
    print(f"{name:8s} signed relative error: mean {values.mean():+.2e}  median {np.median(values):+.2e}  share positive {np.mean(values > 0):.2f}")
