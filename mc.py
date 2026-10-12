"""Exact suffix trajectories and robust matched pilot + one Fisher-scoring step."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
from pathlib import Path
import numpy as np
from numba import njit, prange, set_num_threads
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import K, DT, SIGMA, C, base_model, eval_grid

REGIMES = [('lpu', 'Low pile-up', 1, 100), ('hpu', 'High pile-up', 10, 400)]
RANGES = np.array([2., 3.5, 5.])
OUTPUT = Path(__file__).with_name('output')


@njit(parallel=True, cache=True)
def trajectories(lam, q, dead, periods, seeds):
    """Exponential integrated-hazard jumps; interval differences count live bins."""
    k = len(lam)
    hazard = np.concatenate((np.zeros(1), np.cumsum(lam)))
    cdf = np.cumsum(q); cdf[-1] = 1.
    active = np.zeros((len(seeds), k), np.int64)
    counts = np.zeros_like(active)
    for m in prange(len(seeds)):
        np.random.seed(seeds[m])
        diff = np.zeros(k + 1, np.int64)
        ready = 0
        for cycle in range(periods):
            start = np.searchsorted(cdf, np.random.random(), side='right')
            r = max(start, ready - cycle*k)
            while r < k:
                h = hazard[r] + np.random.exponential()
                hit = np.searchsorted(hazard, h, side='right') - 1
                stop = min(hit + 1, k)
                diff[r] += 1; diff[stop] -= 1
                if hit >= k:
                    break
                counts[m, hit] += 1
                ready = cycle*k + hit + dead + 1
                r = hit + dead + 1
        active[m] = np.cumsum(diff[:-1])
    return active, counts


def pulse(tau):
    lo = (np.arange(K)*DT - tau - np.array([-K*DT, 0., K*DT])[:, None])/SIGMA
    hi = lo + DT/SIGMA
    f = (ndtr(hi) - ndtr(lo)).sum(axis=0)
    df = ((np.exp(-lo*lo/2) - np.exp(-hi*hi/2)) / (SIGMA*np.sqrt(2*np.pi))).sum(axis=0)
    return f, df


def estimate(active, counts):
    """Exposure-weighted robust pilot; all three parameters are inferred."""
    rate = -np.log1p(-(counts + .5)/(active + 1))
    weight = active / active.max()
    well_exposed = weight > .1
    background = np.median(rate[well_exposed])
    centered = rate - background
    f0, _ = pulse(0.)
    correlate = lambda x, f: np.fft.irfft(np.fft.rfft(x)*np.fft.rfft(f).conj(), n=K)
    numerator = correlate(weight*centered, f0)
    denominator = np.maximum(correlate(weight, f0*f0), 1e-30)
    score = numerator / np.sqrt(denominator)
    score[denominator < .01*denominator.max()] = -np.inf
    tau = DT*np.argmax(score)

    def match(t):
        f, _ = pulse(t)
        return -np.dot(weight*centered, f)/np.sqrt(np.dot(weight, f*f))

    tau = minimize_scalar(match, bounds=(tau-DT, tau+DT), method='bounded',
                          options={'xatol': 1e-8}).x % (K*DT)
    f, df = pulse(tau)
    amplitude = max(np.dot(weight*centered, f)/np.dot(weight, f*f), 1e-12)
    background = max(np.median((rate-amplitude*f)[well_exposed]), 1e-12)
    pilot = np.array([amplitude, tau, background])
    lam = amplitude*f + background
    jac = np.column_stack((f, amplitude*df, np.ones(K)))
    fisher = np.einsum('ri,r,rj->ij', jac, active/np.expm1(lam), jac)
    score = jac.T @ (counts/(-np.expm1(-lam)) - active)
    return pilot, pilot + np.linalg.solve(fisher, score)


def run(args):
    set_num_threads(args.workers)
    rng = np.random.default_rng(args.seed)
    raw = dict(periods=args.periods, seed=args.seed)
    with np.load(OUTPUT/'suffix.npz') as gates:
        for key, _, background, dead in REGIMES:
            q = raw[key+'_q'] = gates[key+'_q']
            for distance in RANGES:
                lam, _ = base_model([distance], background)
                seeds = rng.integers(0, 2**32, args.runs, dtype=np.uint32)
                active, counts = trajectories(lam[0], q, dead, args.periods, seeds)
                raw[f'{key}_{distance:g}'] = np.array([estimate(a, c) for a, c in zip(active, counts)])
    np.savez_compressed(OUTPUT/'mc.npz', **raw)


def plot():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex='col')
    grid = np.linspace(2, 5, 200)
    with np.load(OUTPUT/'mc.npz') as data:
        for col, (key, title, background, dead) in enumerate(REGIMES):
            squared = np.array([(data[f'{key}_{r:g}'][:, :, 1]*(C/2)-r)**2 for r in RANGES])
            runs = squared.shape[1]
            mse, se = squared.mean(axis=1), squared.std(axis=1, ddof=1)/np.sqrt(runs)
            def bound(ranges):
                lam, jac = base_model(ranges, background)
                return eval_grid(lam, jac, 1., np.cumsum(data[key+'_q']), dead)[0]/data['periods']
            theory = bound(RANGES)
            top, bottom = axes[:, col]
            top.plot(grid, 1e6*bound(grid), 'k--', label='Fisher prediction')
            for i, label, marker in [(0, 'Robust pilot', '^'), (1, 'Robust + OSE', 'o')]:
                top.errorbar(RANGES, 1e6*mse[:, i], yerr=2e6*se[:, i], fmt=marker,
                             capsize=4, label=label)
            top.set(title=title, ylabel=r'Range MSE (mm$^2$)', ylim=(0, None))
            bottom.axhline(1, color='k', ls='--')
            bottom.errorbar(RANGES, mse[:, 1]/theory, yerr=2*se[:, 1]/theory, fmt='C1o', capsize=4)
            bottom.set(xlabel='True range (m)', ylabel='OSE MSE / Fisher prediction', xticks=RANGES)
        fig.suptitle(f"{runs:,} trajectories of {int(data['periods']):,} periods per case; error bars: ±2 SE")
    axes[0, 0].legend(loc='lower center')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT/f'mc_validation.{ext}', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true', help='Plot existing output/mc.npz')
    parser.add_argument('--periods', type=int, default=100_000)
    parser.add_argument('--runs', type=int, default=5_000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261009)
    args = parser.parse_args()
    if not args.plot:
        run(args)
    plot()
