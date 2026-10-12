"""Independent-bin opening policies (Section 5)."""
import nlopt
import numpy as np
from numba import njit
from core import K, base_model, gamma_suffix, adjoint


def _data(ranges, B):
    lam, derivatives = base_model(np.atleast_1d(ranges), B)
    derivatives[:, :, 2] *= K  # nuisance parameter: background/bin
    J = (derivatives[:, :, :, None] * derivatives[:, :, None, :]
         / np.maximum(np.expm1(lam), 1e-14)[:, :, None, None])
    return -np.expm1(-lam), J


@njit(cache=True)
def _evaluate(alpha, ps, js, D, want_grad):
    values = np.empty(len(ps)); gradients = np.zeros((len(ps), K))
    for i in range(len(ps)):
        # Independent thinning gives live=(I+T_D diag(p*alpha))^{-1}1.
        p, J = ps[i], js[i]
        live = gamma_suffix(p * alpha, np.ones(K), D)
        gamma = alpha * live
        F = np.zeros((3, 3))
        for r in range(K):
            F += gamma[r] * J[r]
        h = np.linalg.solve((F + F.T) / 2 + 1e-12*np.eye(3),
                            np.array([0., 1., 0.]))
        values[i] = h[1]
        if want_grad:
            v = np.empty(K)
            for r in range(K):
                v[r] = -(h @ J[r] @ h)
            eta = adjoint(p * alpha, np.ones(K), D, alpha * v)
            gradients[i] = eta * live / alpha
    return values, gradients


def evaluate(alpha, ranges, B, D, want_grad=False):
    """Return full-nuisance delay variances in ns² and derivatives wrt alpha."""
    return _evaluate(np.clip(alpha, 1e-4, 1.), *_data(ranges, B), D, want_grad)


def optimize(ranges, B, D, starts, maxeval, eval_ranges=None):
    ps, js = _data(ranges, B)
    pe, je = _data(ranges if eval_ranges is None else eval_ranges, B)
    winner = np.ones(K)
    validation_best = _evaluate(winner, pe, je, D, False)[0].max()
    for start in starts:
        alpha0 = np.clip(np.full(K, start) if np.ndim(start) == 0 else
                         np.asarray(start, dtype=float), 1e-4, 1.)
        best = alpha0.copy()
        best_loss = _evaluate(best, ps, js, D, False)[0].max()
        scale = max(best_loss, 1e-30)

        def objective(alpha, grad):
            nonlocal best, best_loss
            alpha = np.clip(alpha, 1e-4, 1.)
            values = _evaluate(alpha, ps, js, D, False)[0]
            active = int(values.argmax()); value = values[active]
            if grad.size:
                grad[:] = _evaluate(alpha, ps[active:active+1],
                                    js[active:active+1], D, True)[1][0] / scale
            if np.isfinite(value) and value < best_loss:
                best, best_loss = alpha.copy(), value
            return float(value / scale)

        opt = nlopt.opt(nlopt.LD_LBFGS, K)
        opt.set_lower_bounds(1e-4); opt.set_upper_bounds(1.)
        opt.set_min_objective(objective); opt.set_maxeval(maxeval)
        opt.set_ftol_rel(1e-8); opt.set_xtol_rel(1e-6)
        opt.set_initial_step(.05)
        try:
            result = opt.optimize(alpha0)
            objective(result, np.empty(0))
        except (RuntimeError, nlopt.RoundoffLimited, nlopt.ForcedStop):
            pass  # NLopt can stop at a nonsmooth maximum; retain best iterate.
        validation = _evaluate(best, pe, je, D, False)[0].max()
        if validation < validation_best:
            winner, validation_best = best.copy(), validation
    return winner
