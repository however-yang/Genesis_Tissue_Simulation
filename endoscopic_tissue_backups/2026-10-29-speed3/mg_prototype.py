"""Offline prototype: does an aggregation multigrid preconditioner reach the PCG quality of the solver in fewer iterations?

Usage: python mg_prototype.py <dump.bin> [more dumps...]

Reads a material PCG system dumped by a debug_material_dump.py build, rebuilds its matrix on the largest component,
checks the rebuild against the solver's own block-Jacobi PCG result, and reports the quadratic-model decrease share
(the fraction of the converged decrease of 0.5 x'Ax - b'x) by iteration for block-Jacobi PCG and for multigrid PCG.
"""
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla

LOADED, FREE = 32, 80


def load(path):
    raw = open(path, "rb").read()
    n, nx, ny, nz = np.frombuffer(raw, np.int32, 4, 0)
    offset = 16
    def take(dtype, count):
        nonlocal offset
        array = np.frombuffer(raw, dtype, count, offset)
        offset += array.nbytes
        return array
    matrix = take(np.float16, 72 * n).reshape(72, n).astype(np.float64)
    scale = take(np.float32, 8 * n).reshape(8, n).astype(np.float64)
    mask = take(np.uint16, n)
    rhs = take(np.float32, 3 * n).reshape(n, 3).astype(np.float64)
    step = take(np.float32, 3 * n).reshape(n, 3).astype(np.float64)
    diagonal = take(np.float32, 6 * n).reshape(n, 6).astype(np.float64)
    component = take(np.int32, n)
    grasp = take(np.int32, n)
    return dict(n=n, dims=(nx, ny, nz), matrix=matrix, scale=scale, mask=mask, rhs=rhs, step=step, diagonal=diagonal,
                component=component, grasp=grasp)


def offset_of(bits, nx, ny):
    return (bits & 1) + nx * (((bits >> 1) & 1) + ny * ((bits >> 2) & 1))


def build_matrix(d, nodes):
    """The block matrix of the PCG on `nodes` (global ids), from the stored half-precision stencil."""
    n, (nx, ny, nz) = d["n"], d["dims"]
    local = -np.ones(n, np.int64)
    local[nodes] = np.arange(len(nodes))
    rows, cols, vals = [], [], []
    mask = d["mask"][nodes].astype(np.int64)
    for slot in range(15):
        has = (mask >> slot) & 1 == 1
        i = nodes[has]
        bits = abs(slot - 7)
        if slot == 7:
            j, owner, block = i, i, 0
        elif slot > 7:
            j = i + offset_of(bits, nx, ny); owner, block = i, bits
        else:
            j = i - offset_of(bits, nx, ny); owner, block = j, bits
        a = d["matrix"][9 * block:9 * block + 9, owner] * d["scale"][block, owner]  # (9, m), a[3u+v]
        li, lj = local[i], local[j]
        assert (lj >= 0).all()
        for u in range(3):
            for v in range(3):
                k = 3 * u + v if slot >= 7 else 3 * v + u
                rows.append(3 * li + u); cols.append(3 * lj + v); vals.append(a[k])
    m = 3 * len(nodes)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m, m))
    return A


def block_jacobi(d, nodes):
    D = d["diagonal"][nodes]  # xx,yy,zz,xy,xz,yz
    blocks = np.zeros((len(nodes), 3, 3))
    blocks[:, 0, 0], blocks[:, 1, 1], blocks[:, 2, 2] = D[:, 0], D[:, 1], D[:, 2]
    blocks[:, 0, 1] = blocks[:, 1, 0] = D[:, 3]
    blocks[:, 0, 2] = blocks[:, 2, 0] = D[:, 4]
    blocks[:, 1, 2] = blocks[:, 2, 1] = D[:, 5]
    return np.linalg.inv(blocks)


def apply_blocks(inverse, r):
    return np.einsum("nij,nj->ni", inverse, r.reshape(-1, 3)).reshape(-1)


def pcg(A, b, precondition, iterations, x_star=None):
    """PCG from x=0; returns the iterate after each iteration."""
    x = np.zeros_like(b); r = b.copy(); z = precondition(r); p = z.copy(); rz = r @ z
    history = []
    for _ in range(iterations):
        Ap = A @ p; alpha = rz / (p @ Ap)
        x = x + alpha * p; r = r - alpha * Ap
        z = precondition(r); rz_next = r @ z; p = z + (rz_next / rz) * p; rz = rz_next
        history.append(x.copy())
    return history


def share(A, b, x, x_star):
    q = lambda v: 0.5 * v @ (A @ v) - b @ v
    return q(x) / q(x_star)  # q(0) = 0


class Multigrid:
    """Aggregation multigrid on a lattice: aggregates are the connected nodes of each 2x2x2 block of sites, Galerkin
    coarse operators, damped block-Jacobi smoothing, a direct solve at the coarsest level."""

    def __init__(self, A, sites, omega=0.6, smoothed=False, coarsest=3000, pre=1, post=1, components=None, coarse_sweeps=0):
        self.levels, self.omega, self.pre, self.post, self.coarse_sweeps = [], omega, pre, post, coarse_sweeps
        while A.shape[0] > coarsest:
            m = A.shape[0] // 3
            diagonal = np.array([A[3 * k:3 * k + 3, 3 * k:3 * k + 3].toarray() for k in range(m)]) if m < 0 else None
            inverse = self._diagonal_inverse(A)
            block = sites // 2
            key = block[:, 0] + (1 << 20) * block[:, 1] + (1 << 40) * block[:, 2]
            # Node graph: nonzero coupling between nodes of the same 2x2x2 block.
            S = sp.csr_matrix((np.abs(A.data), A.indices, A.indptr), shape=A.shape)
            G = (sp.kron(sp.eye(m), np.ones((1, 3))) @ S @ sp.kron(sp.eye(m), np.ones((3, 1)))).tocoo()
            same = key[G.row] == key[G.col]
            G = sp.csr_matrix((np.ones(same.sum()), (G.row[same], G.col[same])), shape=(m, m))
            if components is None:
                count, label = csgraph.connected_components(G, directed=False)
            else:
                _, label = np.unique(np.stack([key, components]), axis=1, return_inverse=True)
                label = label.reshape(-1); count = label.max() + 1
            T = sp.csr_matrix((np.ones(m), (np.arange(m), label)), shape=(m, count))
            P = sp.kron(T, sp.eye(3)).tocsr()
            if smoothed:
                Dinv = sp.block_diag([inverse[k] for k in range(m)]).tocsr()
                P = (P - (4.0 / 3.0) / self._spectral_radius(Dinv @ A) * (Dinv @ (A @ P))).tocsr()
            Ac = (P.T @ A @ P).tocsr()
            self.levels.append((A, inverse, P))
            # Coarse sites: the block of each aggregate.
            first = np.zeros(count, np.int64); first[label] = np.arange(m)
            sites = block[first]
            if components is not None: components = components[first]
            A = Ac
        self.coarse = spla.factorized(A.tocsc()) if not coarse_sweeps else None
        self.sizes = [L[0].shape[0] // 3 for L in self.levels] + [A.shape[0] // 3]
        self.coarse_A, self.coarse_inverse = A, self._diagonal_inverse(A)

    @staticmethod
    def _diagonal_inverse(A):
        m = A.shape[0] // 3
        blocks = np.zeros((m, 3, 3))
        Acoo = A.tocoo()
        same = Acoo.row // 3 == Acoo.col // 3
        np.add.at(blocks, (Acoo.row[same] // 3, Acoo.row[same] % 3, Acoo.col[same] % 3), Acoo.data[same])
        return np.linalg.inv(blocks)

    @staticmethod
    def _spectral_radius(M):
        return abs(spla.eigs(M, k=1, which="LM", return_eigenvectors=False, maxiter=200, tol=1e-3)[0])

    def apply(self, r, level=0):
        if level == len(self.levels):
            if self.coarse is not None: return self.coarse(r)
            x = np.zeros_like(r)
            for _ in range(self.coarse_sweeps): x = x + self.omega * apply_blocks(self.coarse_inverse, r - self.coarse_A @ x)
            return x
        A, inverse, P = self.levels[level]
        x = np.zeros_like(r)
        for _ in range(self.pre):
            x = x + self.omega * apply_blocks(inverse, r - A @ x)
        x = x + P @ self.apply(P.T @ (r - A @ x), level + 1)
        for _ in range(self.post):
            x = x + self.omega * apply_blocks(inverse, r - A @ x)
        return x


def analyze(path):
    d = load(path)
    nx, ny, nz = d["dims"]
    cg_nodes = np.flatnonzero((d["mask"] >> 7) & 1)
    roots, counts = np.unique(d["component"][cg_nodes], return_counts=True)
    root = roots[np.argmax(counts)]
    nodes = cg_nodes[d["component"][cg_nodes] == root]
    loaded = bool(d["grasp"][root] & 7)
    budget = LOADED if loaded else FREE
    A = build_matrix(d, nodes)
    b = d["rhs"][nodes].reshape(-1)
    inverse = block_jacobi(d, nodes)
    jacobi = lambda r: apply_blocks(inverse, r)
    print(f"== {path}: {len(nodes)} CG nodes in the largest of {len(roots)} components, "
          f"{'loaded' if loaded else 'free'} ({budget} iterations), symmetric error {abs(A - A.T).max():.2e}")
    history = pcg(A, b, jacobi, budget)
    gpu = d["step"][nodes].reshape(-1)
    print(f"   rebuild check: |offline - solver| / |solver| after {budget} iterations = "
          f"{np.linalg.norm(history[-1] - gpu) / np.linalg.norm(gpu):.2e}")
    sites = np.stack([nodes % nx, (nodes // nx) % ny, nodes // (nx * ny)], axis=1)
    mg = Multigrid(A, sites)
    reference = pcg(A, b, mg.apply, 60)[-1]
    target = share(A, b, history[-1], reference)
    print(f"   block-Jacobi PCG decrease share: " + ", ".join(
        f"{k}:{share(A, b, history[k - 1], reference):.4f}" for k in (4, 8, 16, 32, 48, 64, 80) if k <= budget))
    comp = d["component"][nodes]
    variants = (("block+component, coarsest<=60 direct, omega 0.7", Multigrid(A, sites, components=comp, coarsest=180, omega=0.7).apply),)
    for label, preconditioner in variants:
        mg_history = pcg(A, b, preconditioner, 20)
        shares = [share(A, b, x, reference) for x in mg_history]
        reach = next((k + 1 for k, s in enumerate(shares) if s >= target), None)
        print(f"   {label} levels {preconditioner.__self__.sizes}")
        print(f"   {label} MG-PCG: " + ", ".join(f"{k + 1}:{s:.4f}" for k, s in enumerate(shares[:12])) +
              f"  -> reaches the solver's {target:.4f} at iteration {reach}")


for path in sys.argv[1:]:
    analyze(path)
