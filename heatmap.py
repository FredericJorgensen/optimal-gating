"""Four-start suffix heatmap: 800 iterations, with fine-grid checkpoint selection."""
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import ndtr, softmax
from core import K, DT, SIGMA, C, gamma_suffix, adjoint, grid, suffix_starts

SCALE = (C/2)**2
CHECKPOINTS = (30, 60, 100, 200, 400, 800)


def model(ranges, B):
    """Use a normalized pulse and per-bin background coordinate.

    The 1e-12 Fisher ridge below is defined in these nuisance coordinates.
    """
    ps, js = [], []
    left = np.arange(K, dtype=float)*DT
    shifts = np.arange(-2, 3, dtype=float)[:, None]*(K*DT)
    for R in ranges:
        zr = (left[None, :]+DT-2*R/C-shifts)/SIGMA
        zl = (left[None, :]-2*R/C-shifts)/SIGMA
        f0 = (ndtr(zr)-ndtr(zl)).sum(axis=0)
        df0 = ((np.exp(-.5*zl**2)-np.exp(-.5*zr**2))/(SIGMA*np.sqrt(2*np.pi))).sum(axis=0)
        Z, dZ = float(f0.sum()), float(df0.sum())
        f, df = f0/Z, (df0*Z-f0*dZ)/Z**2
        a = .5*(7.5/R)**2
        lam = a*f+B/K
        grad = np.column_stack((f, a*df, np.ones(K)))
        weight = 1/np.maximum(np.expm1(lam), 1e-14)
        ps.append(-np.expm1(-lam))
        js.append(weight[:, None, None]*grad[:, :, None]*grad[:, None, :])
    return np.asarray(ps), np.asarray(js)


@njit(cache=True)
def evaluate(q, ps, js, D, want_grad=True):
    beta = np.cumsum(q/q.sum())
    phis = np.empty(len(ps)); grads = np.zeros((len(ps), K))
    for i in range(len(ps)):
        p, J = ps[i], js[i]
        gamma = np.clip(gamma_suffix(p, beta, D), 0., 1.)
        F = np.zeros((3, 3))
        for r in range(K):
            for a in range(3):
                for b in range(3):
                    F[a, b] += gamma[r]*J[r, a, b]
        F = .5*(F+F.T)+1e-12*np.eye(3)
        h = np.linalg.solve(F, np.array([0., 1., 0.]))
        phis[i] = h[1]
        if want_grad:
            v = np.empty(K)
            for r in range(K):
                v[r] = -(h[0]*h[0]*J[r,0,0]+h[1]*h[1]*J[r,1,1]+h[2]*h[2]*J[r,2,2]
                         +2*h[0]*h[1]*J[r,0,1]+2*h[0]*h[2]*J[r,0,2]+2*h[1]*h[2]*J[r,1,2])
            eta = adjoint(p, beta, D, v)
            prev = np.sum((p*gamma)[K-D:]) if D else 0.
            gb = np.empty(K)
            for r in range(K):
                gb[r] = eta[r]*(1-prev)
                if r < D:
                    prev -= p[K-D+r]*gamma[K-D+r]
            gq = np.cumsum(gb[::-1])[::-1].copy()
            grads[i] = q*(gq-np.dot(q, gq))
            grads[i] -= grads[i].mean()
    return phis, grads


def solve(B, D):
    """Return best q and [free-running, optimized] range CRBs (m²).

    Optimize on 36 Gauss–Legendre nodes; rank every checkpoint on 120 nodes.
    Exact free-running is a fallback, and is optimal analytically when D=0.
    """
    pe, Je = model(grid(120), B)
    qfree = np.zeros(K); qfree[0] = 1.
    free = float(evaluate(qfree, pe, Je, D, False)[0].max()*SCALE)
    best_q, worst = qfree, free
    if D == 0:
        return best_q, np.array([free, worst])
    p, J = model(grid(36), B)
    norm = float(evaluate(qfree, p, J, D, False)[0].max())

    def objective(z):
        phis, grads = evaluate(softmax(z-z.mean()), p, J, D)
        center = phis.max()
        raw = np.exp(np.clip(35*(phis-center), -745., 50.))
        grad = np.dot(raw/raw.sum(), grads)
        grad -= grad.mean()
        return float((center+np.log(raw.sum())/35)/norm), grad/norm

    for z0 in suffix_starts(early_bias=True):
        snapshots = {}; nit = 0

        def checkpoint(z):
            nonlocal nit
            nit += 1
            if nit in CHECKPOINTS:
                snapshots[nit] = z.copy()

        result = minimize(objective, z0-z0.mean(), jac=True, method='L-BFGS-B',
                          callback=checkpoint, options=dict(maxiter=800, maxfun=16000,
                          ftol=1e-9, gtol=1e-7, maxls=50))
        snapshots[int(result.nit)] = result.x
        for _, z in sorted(snapshots.items()):
            q = softmax(z-z.mean())
            loss = float(evaluate(q, pe, Je, D, False)[0].max()*SCALE)
            if loss < worst:
                best_q, worst = q, loss
    return best_q, np.array([free, worst])
