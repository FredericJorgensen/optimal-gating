"""Convex pointwise range oracles with a dual-gap stopping rule."""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize, minimize_scalar
from core import C, base_model, gamma_suffix


def constraints(p, dead):
    """Sparse recurrence for gamma + cyclic_sum(p*gamma, dead) <= 1."""
    k = len(p)
    if not 0 <= dead < k:
        raise ValueError('Expected 0 <= dead < number of bins')
    r = np.arange(k-1)
    old = (r-dead) % k
    eq = sparse.coo_matrix((np.r_[-p[r], p[old], -np.ones(k-1), np.ones(k-1)],
        (np.tile(r, 4), np.r_[r, old, k+r, k+r+1])), shape=(k, 2*k))
    anchor = np.r_[1., -p[k-dead:]]
    eq += sparse.coo_matrix((anchor,
        (np.full(dead+1, k-1), np.r_[k, np.arange(k-dead, k)])), shape=(k, 2*k))
    ub = sparse.hstack([sparse.eye(k), sparse.eye(k)], format='csr')
    return eq.tocsr(), ub


def activity_load(gamma, p, dead):
    """Evaluate the original cyclic inequalities without auxiliary variables."""
    y = p*gamma
    cumulative = np.r_[0., np.cumsum(np.r_[y, y])]
    phase = len(p)+np.arange(len(p))
    return gamma + cumulative[phase]-cumulative[phase-dead]


def feasible(gamma, p, dead):
    """Remove numerical LP infeasibility using a feasible radial contraction."""
    gamma = np.maximum(gamma, 0.)
    # A has nonnegative entries, so nonnegative gamma / max(A gamma) is feasible.
    # A small inward margin covers the roundoff of the sums and convex mixtures.
    margin = 128*np.finfo(float).eps*len(p)
    return gamma*(1-margin)/max(1., float(activity_load(gamma, p, dead).max()))


def solve(distance, background, dead, tolerance=1.e-6, maxiter=300):
    lam, jac = base_model([distance], background)
    lam, jac = lam[0], jac[0]
    p = -np.expm1(-lam)
    k = len(p)
    J = jac[:, :, None]*jac[:, None, :]/np.expm1(lam)[:, None, None]
    eq, ub = constraints(p, dead)
    gamma = feasible(gamma_suffix(p, np.ones(k), dead), p, dead)
    F = np.einsum('r,rij->ij', gamma, J)
    # Whiten by the free-running Fisher matrix and normalize its loss to one.
    inv_chol = np.linalg.solve(np.linalg.cholesky(F), np.eye(3))
    target = inv_chol @ np.array([0., C/2, 0.])
    scale = float(target @ target)
    target /= np.sqrt(scale)
    S = np.einsum('ai,rij,bj->rab', inv_chol, J, inv_chol)
    vertices = [gamma]
    matrices = [np.einsum('r,rij->ij', gamma, S)]
    weights = np.ones(1)

    for _ in range(maxiter):
        H = np.einsum('r,rij->ij', gamma, S)
        h = np.linalg.solve(H, target)
        value = float(target @ h)
        gradient = -np.einsum('i,rij,j->r', h, S, h)
        gscale = float(np.max(np.abs(gradient)))
        cost = np.r_[gradient/gscale, np.zeros(k)]
        lp = linprog(cost, A_ub=ub, b_ub=np.ones(k), A_eq=eq,
            b_eq=np.zeros(k), bounds=(0., 1.), method='highs',
            options={'dual_feasibility_tolerance': 1.e-9,
                     'primal_feasibility_tolerance': 1.e-9})
        if not lp.success:
            raise RuntimeError(lp.message)
        # For any y <= 0, z free, and 0 <= x <= 1 with Ux <= 1, Ex = 0:
        # c.x >= sum(y) + sum(min(c-U.T@y-E.T@z, 0)).
        y = np.minimum(lp.ineqlin.marginals, 0.)
        residual = cost-ub.T@y-eq.T@lp.eqlin.marginals
        linear_lower = gscale*(y.sum()+np.minimum(residual, 0.).sum())
        # Subtract a roundoff allowance for the dual dot products/sums.
        allowance = 128*np.finfo(float).eps*k*gscale*(
            1.+np.abs(y).sum()+np.abs(lp.eqlin.marginals).sum())
        lower = value + linear_lower - float(np.sum(gradient*gamma)) - allowance
        gap = max(0., (value-lower)/value)
        violation = max(0., float(activity_load(gamma, p, dead).max()-1.),
                        -float(gamma.min()))
        if gap <= tolerance:
            if violation > 1.e-11:
                raise RuntimeError(f'Oracle primal infeasibility {violation:g}')
            return value*scale

        vertex = feasible(lp.x[:k], p, dead)
        matrix = np.einsum('r,rij->ij', vertex, S)
        vertices.append(vertex)
        matrices.append(matrix)
        M = np.array(matrices)

        def objective(w):
            H = np.einsum('v,vij->ij', w, M)
            try:
                np.linalg.cholesky(H)
                h = np.linalg.solve(H, target)
            except np.linalg.LinAlgError:
                return np.inf, np.zeros(len(w))
            return float(target @ h), -np.einsum('i,vij,j->v', h, M, h)

        line = minimize_scalar(lambda t: objective(np.r_[weights*(1-t), t])[0],
            bounds=(0., 1.), method='bounded', options={'xatol':1.e-12})
        initial = np.r_[weights*(1-line.x), line.x]
        fit = minimize(objective, initial, jac=True, method='SLSQP',
            bounds=[(0., 1.)]*len(M), constraints={'type':'eq',
            'fun':lambda w: w.sum()-1., 'jac':lambda w: np.ones_like(w)},
            options={'ftol':1.e-13, 'maxiter':300})
        proposed = np.maximum(fit.x, 0.)
        proposed /= proposed.sum()
        weights = proposed if objective(proposed)[0] <= objective(initial)[0] else initial
        keep = weights > 1.e-12
        weights = weights[keep]/weights[keep].sum()
        vertices = list(np.array(vertices)[keep])
        matrices = list(M[keep])
        gamma = np.einsum('v,vr->r', weights, np.array(vertices))
    raise RuntimeError(f'Oracle did not reach {tolerance:g}: gap={gap:g}, '
                       f'R={distance:g}, B={background:g}, D={dead}')
