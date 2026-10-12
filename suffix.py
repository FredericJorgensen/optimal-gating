import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve, splu
from scipy.special import ndtr, softmax
from core import K, DT, SIGMA, C, base_model, eval_grid, gamma_suffix, grid
from alpha import optimize as optimize_alpha, evaluate as evaluate_alpha
from oracle import solve as solve_oracle
from plot_suffix import plot
from heatmap import solve as solve_heatmap

SCALE = (C/2)**2


def fisher_data(ranges, B, D):
    cases = []
    left = np.arange(K, dtype=float)*DT
    shifts = np.arange(-2,3, dtype=float)[:,None]*(K*DT)
    lags = np.arange(1,D+1)
    for R in ranges:
        lo = (left[None,:]-2*float(R)/C-shifts)/SIGMA
        hi = ((left+DT)[None,:]-2*float(R)/C-shifts)/SIGMA
        f0 = (ndtr(hi)-ndtr(lo)).sum(axis=0)
        pl, pr = np.exp(-.5*lo**2)/np.sqrt(2*np.pi), np.exp(-.5*hi**2)/np.sqrt(2*np.pi)
        df0 = ((-pr+pl)/SIGMA).sum(axis=0)
        Z, dZ = float(f0.sum()), float(df0.sum())
        f, df = f0/Z, (df0*Z-f0*dZ)/Z**2
        a = .5*(7.5/float(R))**2
        lam = a*f+B/K
        p, w = -np.expm1(-lam), 1/np.maximum(np.expm1(lam),1e-14)
        jac = np.column_stack((f,a*df,np.ones(K)))
        J = w[:,None,None]*jac[:,:,None]*jac[:,None,:]
        lower = sparse.eye(K,format='csr')+sparse.diags([p[:K-d] for d in lags],-lags,shape=(K,K),format='csr')
        wrap = sparse.diags([p[K-d:] for d in lags],K-lags,shape=(K,K),format='csr')
        cases.append((J,lower,wrap))
    return cases


def evaluate(q, cases, gradient=False, pool=None):
    if gradient: q = q/q.sum()
    beta = np.ones(K) if q is None else np.cumsum(q/q.sum())
    def one(case):
        J, lower, wrap = case
        mat = (lower+wrap.multiply(beta[:,None])).tocsc()
        gamma = splu(mat).solve(beta) if q is None else spsolve(mat,beta)
        gamma = np.clip(gamma,0,1)
        F = np.einsum('r,rij->ij',gamma,J,optimize=True)
        try:
            h = np.linalg.solve(.5*(F+F.T)+1e-12*np.eye(3),[0.,1.,0.])
        except np.linalg.LinAlgError:
            return 1e30,np.zeros(K) if gradient else np.empty(0)
        if not np.isfinite(h[1]) or not 0 < h[1] < 1e29:
            return 1e30,np.zeros(K) if gradient else np.empty(0)
        g = np.empty(0)
        if gradient:
            v = -np.einsum('i,rij,j->r',h,J,h,optimize=True)
            gb = spsolve(mat.T,v)*(1-wrap@gamma)
            gq = np.cumsum(gb[::-1])[::-1]
            g = q*(gq-np.dot(q,gq)); g -= g.mean()
        return h[1],g
    losses, gradients = zip(*(map(one,cases) if pool is None else pool.map(one,cases)))
    return np.array(losses), np.array(gradients)


def optimize(B, D, workers=4):
    with ThreadPoolExecutor(workers) as pool:
        train = fisher_data(grid(72),B,D)
        norm = evaluate(None,train,pool=pool)[0].max()

        def objective(z):
            q = softmax(z-z.mean())
            f, gradients = evaluate(q,train,True,pool)
            center = f.max()
            raw = np.exp(np.clip(35*(f-center),-745.,50.))
            g = np.dot(raw/raw.sum(),gradients); g -= g.mean()
            return (center+np.log(raw.sum())/35)/norm, g/norm

        near = np.full(K,.05/(K-1)); near[0] = .95
        qs = [near]+[softmax(-.5*((np.arange(K)/(K-1)-c)/w)**2)
                     for c,w in ((.18,.055),(.28,.07))]
        starts = [np.zeros(K)]+[np.log(np.maximum(q/q.sum(),1e-14)) for q in qs]
        candidates = []
        for z in starts:
            result = minimize(objective, z-z.mean(), jac=True, method='L-BFGS-B',
                options={'maxiter':80, 'maxfun':1600, 'ftol':1e-9,
                         'gtol':1e-7, 'maxls':50})
            q = softmax(result.x-result.x.mean())
            worst = evaluate(q,train,True,pool)[0].max()
            candidates.append((worst,q))
        q = min(candidates, key=lambda x:x[0])[1]
        cases = fisher_data(grid(160),B,D)
        free = evaluate(None,cases,pool=pool)[0].max()*SCALE
        worst = evaluate(q,cases,pool=pool)[0].max()*SCALE
        if worst > free:
            q = np.zeros(K); q[0]=1.; worst=free
        return q, np.array([free,worst])


def heatmap_cell(task):
    B, dead = task
    _, (free, worst) = solve_heatmap(B, int(round(dead/DT)))
    return worst, 100*(1-worst/free)


def compute(workers, size):
    data = dict(R_curve=np.linspace(2.,5.,200), R_marks=np.array([2.,3.5,5.]),
                t=(np.arange(K)+.5)*DT)
    for key,B,D in [('lpu',1,100), ('hpu',10,400)]:
        q, worst = optimize(B,D,workers)
        alpha = optimize_alpha(grid(72),B,D,
                               [np.full(K,x) for x in [.999,.8,.5,.3]],180,grid(160))
        data[key+'_q'],data[key+'_alpha'] = q,alpha
        alpha_worst = evaluate_alpha(alpha,grid(160),B,D)[0].max()*SCALE
        data[key+'_worst'] = np.r_[worst,alpha_worst]
        for scheme,beta in [('free',np.ones(K)), ('opt',np.cumsum(q))]:
            lam,grad = base_model(data['R_curve'],B)
            data[key+'_'+scheme] = eval_grid(lam,grad,1.,beta,D)[0]
            lm,_ = base_model(data['R_marks'],B)
            data[key+'_'+scheme+'_gamma'] = np.array([
                gamma_suffix(-np.expm1(-l),beta,D) for l in lm])
        ranges = np.linspace(2.,5.,80)
        with ProcessPoolExecutor(workers) as pool:
            values = list(pool.map(solve_oracle, ranges, [B]*len(ranges), [D]*len(ranges)))
        data[key+'_oracle_R'], data[key+'_oracle'] = ranges, np.array(values)
    bs=np.geomspace(.3,20,size); ds=np.linspace(0,60,size)
    with ProcessPoolExecutor(workers) as pool:
        values=list(pool.map(heatmap_cell,[(b,d) for b in bs for d in ds]))
    v=np.array(values).reshape(size,size,2)
    data.update(B_grid=bs,dead_grid=ds,objective=v[:,:,0],reduction=v[:,:,1])
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plot',action='store_true',help='plot existing output without recomputing')
    parser.add_argument('--heatmap',type=int,default=64,metavar='SIZE')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    out=Path(__file__).with_name('output'); out.mkdir(exist_ok=True)
    if args.plot:
        with np.load(out/'suffix.npz') as z: data=dict(z)
    else:
        data=compute(args.workers,args.heatmap)
        np.savez_compressed(out/'suffix.npz',**data)
    plot(data,out)
