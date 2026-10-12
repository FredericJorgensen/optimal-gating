"""Section 5 relative-error remark and Appendix C optical attenuation."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import K, DT, base_model, eval_grid


def evaluate(r, B, D, q, u=1., relative=False):
    lam, grad = base_model(r, B)
    f = eval_grid(lam, grad, u, np.cumsum(q), D)[0]
    return f*np.asarray(r)**-2 if relative else f


def extrema(B, D, q, u=1., relative=False):
    r = np.linspace(2., 5., 3001)
    f = evaluate(r, B, D, q, u, relative)
    peaks = np.flatnonzero((f[1:-1] > f[:-2]) & (f[1:-1] >= f[2:]))+1
    points = [(r[0], f[0]), (r[-1], f[-1])]
    for i in peaks:
        sol = minimize_scalar(lambda x: -evaluate([x], B, D, q, u, relative)[0],
                              bounds=(r[i-1], r[i+1]), method='bounded',
                              options={'xatol': 1.e-10})
        points.append((sol.x, -sol.fun))
    return r, f, np.array(points)


def optimize(B, D, relative=False, joint=False, seed=0):
    r = np.linspace(2., 5., 151)
    support = np.arange(0, K, 10)
    q = np.zeros(K)
    q[support] = 1. if seed == 0 else np.exp(-.5*((support*DT-18.)/5.5)**2)
    q /= q.sum()
    u = 1. if seed == 0 else .3
    if not joint:
        u = 1.
    norm = evaluate(r, B, D, np.eye(1, K)[0], relative=relative).max()
    for exchange in range(6):
        lam, grad = base_model(r, B)
        w = r**-2 if relative else np.ones(len(r))
        def full(q, u):
            f, gb, gu = eval_grid(lam, grad, u, np.cumsum(q), D, True)
            gq = np.cumsum(gb[:, ::-1], axis=1)[:, ::-1]
            return f*w/norm, gq*w[:, None]/norm, gu*w/norm
        for column in range(30):
            n = len(support)
            memo = [None, None]
            def ev(x):
                if memo[0] is None or not np.array_equal(x[:-1], memo[0]):
                    qt = np.zeros(K); qt[support] = x[:n]
                    f, g, h = full(qt, x[n] if joint else u)
                    g = np.column_stack([g[:, support], h]) if joint else g[:, support]
                    memo[:] = [x[:-1].copy(), (f, g)]
                return memo[1]
            x0 = np.r_[q[support]/q[support].sum(), [u] if joint else [], 1.]
            x0[-1] = 1.001*ev(x0)[0].max()
            sol = minimize(lambda x: x[-1], x0, method='SLSQP',
                jac=lambda x: np.r_[np.zeros(len(x)-1), 1.],
                bounds=[(0., 1.)]*n+([(1.e-4, 1.)] if joint else [])+[(1.e-10, None)],
                constraints=[
                    {'type': 'eq', 'fun': lambda x: x[:n].sum()-1.,
                     'jac': lambda x: np.r_[np.ones(n), np.zeros(int(joint)+1)]},
                    {'type': 'ineq', 'fun': lambda x: x[-1]-ev(x)[0],
                     'jac': lambda x: np.column_stack([-ev(x)[1], np.ones(len(r))])}],
                options={'maxiter': 500, 'ftol': 2.e-11})
            q = np.zeros(K); q[support] = np.clip(sol.x[:n], 0., 1.); q /= q.sum()
            u = float(sol.x[n]) if joint else u
            f, g, h = full(q, u)
            jac = np.column_stack([g, h]) if joint else g
            at = np.r_[q, u] if joint else q
            lp = linprog(np.r_[np.zeros(len(at)), 1.],
                A_ub=np.column_stack([jac, -np.ones(len(r))]), b_ub=np.dot(jac,at)-f,
                A_eq=[np.r_[np.ones(K), np.zeros(int(joint)+1)]], b_eq=[1.],
                bounds=[(0., 1.)]*K+([(1.e-4, 1.)] if joint else [])+[(None, None)],
                method='highs')
            if not lp.success:
                raise RuntimeError(lp.message)
            gap = max(0., f.max()-lp.fun)
            if gap < 1.e-6:
                break
            support = np.unique(np.r_[np.flatnonzero(q > 1.e-10),
                                      np.flatnonzero(lp.x[:K] > 1.e-9)])
        _, _, peaks = extrema(B, D, q, u, relative)
        violation = peaks[:, 1].max()/norm-f.max()
        if gap < 1.e-6 and violation < 2.e-7:
            return q, u, {'tangent_gap': float(gap), 'range_gap': float(violation)}
        r = np.unique(np.r_[r, peaks[:, 0]])
    raise RuntimeError(f'Minimax refinement did not converge: {gap=}, {violation=}')


def main():
    out = Path(__file__).with_name('output'); out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    report = {}
    qfree = np.eye(1, K)[0]
    for ax, B, D in zip(axes, [1., 10.], [100, 400]):
        candidates = [optimize(B, D, relative=True, seed=s) for s in range(2)]
        q, _, checks = min(candidates, key=lambda z: extrema(B, D, z[0], relative=True)[2][:, 1].max())
        r, f, peaks = extrema(B, D, q, relative=True)
        _, free, free_peaks = extrema(B, D, qfree, relative=True)
        relgain = 100*(1-peaks[:, 1].max()/free_peaks[:, 1].max())
        joint = [optimize(B, D, joint=True, seed=s) for s in range(2)]
        qj, uj, joint_checks = min(joint, key=lambda z: extrema(B, D, z[0], z[1])[2][:, 1].max())
        rr = np.linspace(2., 5., 301)
        objective = lambda u: evaluate(rr, B, D, qfree, u).max()
        opt = minimize_scalar(objective, bounds=(.005, 1.), method='bounded', options={'xatol': 1.e-9})
        uf = min([1., opt.x], key=objective)
        free_gain = 100*(1-extrema(B, D, qfree, uf)[2][:, 1].max()/extrema(B, D, qfree)[2][:, 1].max())
        M = float(base_model(r, B)[0].max())
        report[str(B)] = {'relative_reduction_percent': relgain,
            'relative_worst': float(peaks[:, 1].max()), 'relative_checks': checks,
            'joint_transmissions': [z[1] for z in joint], 'joint_checks': joint_checks,
            'free_transmission': float(uf), 'free_attenuation_reduction_percent': free_gain,
            'max_bin_mean': M, 'attenuation_bound_percent': 100*(1-(M/(2*np.sinh(M/2)))**2)}
        np.savez_compressed(out/f'remarks_B{B:g}.npz', ranges=r, relative=f,
                            free_relative=free, relative_q=q, joint_q=qj, joint_u=uj)
        ax.plot(r, free, '--', label='Free running')
        ax.plot(r, f, label='Relative minimax suffix')
        ax.set(xlabel='Range (m)', ylabel=r'$L_R^{\mathrm{period}}/R^2$',
               title=rf'$B={B:g},\ t_d={D*DT:g}$ ns; reduction {relgain:.1f}%')
        ax.legend(frameon=False); ax.grid(alpha=.2)
    fig.savefig(out/'relative_error.png', dpi=180)
    plt.close(fig)
    (out/'remarks.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
