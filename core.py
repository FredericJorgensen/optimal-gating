import numpy as np
from scipy.special import ndtr, softmax
from numba import njit

K=1000; DT=.1; SIGMA=2.; C=.299792458

def grid(n):
    return 3.5 + 1.5*np.polynomial.legendre.leggauss(n)[0]

def suffix_starts(early_bias=False):
    near = np.full(K, .05/(K-1)); near[0] = .95
    qs = [np.ones(K)/K, near]
    phase = np.arange(K)/(K-1)
    for center, width in ((.18, .055), (.28, .07)):
        qs.append(softmax(-.5*((phase-center)/width)**2))
    if early_bias:
        qs = [qs[0], *qs[2:], softmax(2-1.2*np.arange(K)/K)]
    return [np.log(np.maximum(q, 1e-14)) for q in qs]

def base_model(ranges,B=10.):
    ranges=np.asarray(ranges)
    tau=2*ranges/C; a=.5*(7.5/ranges)**2
    left=np.arange(K)*DT
    f=np.zeros((len(ranges),K)); df=np.zeros_like(f)
    for shift in [-100.,0.,100.]:
        lo=(left[None,:]-tau[:,None]-shift)/SIGMA
        hi=lo+DT/SIGMA
        f+=ndtr(hi)-ndtr(lo)
        df+=(np.exp(-lo**2/2)-np.exp(-hi**2/2))/(SIGMA*np.sqrt(2*np.pi))
    lam=a[:,None]*f+B/K
    grad=np.stack([f,a[:,None]*df,np.ones_like(f)/K],axis=-1)
    return lam,grad

@njit(cache=True)
def gamma_suffix(p,beta,D):
    if D == 0: return beta.copy()
    K=len(p); y=np.zeros(K); g=np.zeros(K)
    for iteration in range(1000):
        prev=np.sum(y[K-D:]) if D else 0.
        curr=0.; change=0.
        for r in range(K):
            old=y[r]
            g[r]=beta[r]*(1-prev)-curr
            value=p[r]*g[r]
            if r<D:
                prev-=y[K-D+r]
            else:
                curr-=y[r-D]
            curr+=value
            y[r]=value
            change=max(change,abs(old-value))
        if change<1.e-14:
            return g
    raise RuntimeError('gamma iteration failed')

@njit(cache=True)
def adjoint(p,beta,D,v):
    if D == 0: return v.copy()
    K=len(p); z=np.zeros(K)
    for it in range(1000):
        prev=np.sum(beta[:D]*z[:D]) if D else 0.
        curr=0.; change=0.
        for j in range(K-1,-1,-1):
            old=z[j]; value=v[j]-p[j]*(prev+curr)
            if j+D>=K:
                prev-=beta[j+D-K]*z[j+D-K]
            else:
                curr-=z[j+D]
            curr+=value; z[j]=value
            change=max(change,abs(value-old))
        if change<1.e-13*max(1.,np.max(np.abs(z))):
            return z
    raise RuntimeError('adjoint iteration failed')

@njit(cache=True)
def eval_grid(lam,grad,u,beta,D,want_grad=False):
    nr=len(lam); losses=np.zeros(nr); grad_beta=np.zeros((nr,K)); grad_u=np.zeros(nr)
    for i in range(nr):
        p=-np.expm1(-u*lam[i]); dp=lam[i]*np.exp(-u*lam[i])
        g=gamma_suffix(p,beta,D)
        J=np.zeros((K,3,3)); F=np.zeros((3,3))
        for r in range(K):
            J[r]=u*u*np.outer(grad[i,r],grad[i,r])/np.expm1(u*lam[i,r])
            F+=g[r]*J[r]
        e=np.array([0.,1.,0.]); h=np.linalg.solve(F,e)
        losses[i]=(C/2)**2*h[1]
        if want_grad:
            v=np.zeros(K)
            for r in range(K):
                v[r]=-(C/2)**2*h@J[r]@h
            z=adjoint(p,beta,D,v)
            # M_u gamma = within-period sum dp*g + beta*wrap sum dp*g.
            yg=dp*g; pu=np.sum(yg[K-D:]) if D else 0.; cu=0.
            oldpg=p*g; prev=np.sum(oldpg[K-D:]) if D else 0.
            for r in range(K):
                grad_beta[i,r]=z[r]*(1-prev)
                grad_u[i]-=z[r]*(cu+beta[r]*pu)
                grad_u[i]+=v[r]*g[r]*(2/u-lam[i,r]/p[r])
                if r<D:
                    pu-=yg[K-D+r]; prev-=oldpg[K-D+r]
                else:
                    cu-=yg[r-D]
                cu+=yg[r]
    return losses,grad_beta,grad_u
