# -*- coding: utf-8 -*-
"""CUDA-enabled LegendreDecomposition calculations"""
from types import ModuleType
from typing import Dict, Tuple, List
import itertools
import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp as scipy_logsumexp
import os
try:
    if os.getenv("LEGENDRE_DECOMP_DISABLE_CUPY") == "1":
        raise ImportError()
    import cupy as cp
    from cupyx.scipy.special import logsumexp as cupy_logsumexp
    def xp_get(val):
        return val.get()
except ImportError:
    import numpy as cp
    from scipy.special import logsumexp as cupy_logsumexp
    def get_array_module(X):
        return np
    cp.get_array_module = get_array_module
    def xp_get(val):
        return val



def kl(P: NDArray[np.float64], Q: NDArray[np.float64], xp: ModuleType = cp) -> np.float64:
    """Kullback-Leibler divergence.

    Args:
        P: P tensor
        Q: Q tensor
        xp (ModuleType): Array module, either numpy (CPU) or cupy

    Returns:
        KL divergence.
    """
    return xp.sum(P * xp.log(P / Q)) - xp.sum(P) + xp.sum(Q)



def get_eta(Q: NDArray[np.float64], D: int, xp: ModuleType = cp) -> NDArray[np.float64]:
    """Eta tensor.

    Args:
        Q: Q tensor
        D: Dimensionality
        xp (ModuleType): Array module, either numpy (CPU) or cupy

    Returns:
        Eta tensor.
    """
    for i in range(D):
        Q = xp.flip(xp.cumsum(xp.flip(Q, axis=i), axis=i), axis=i)
    return Q


def get_h(theta: NDArray[np.float64], D: int, xp: ModuleType = cp) -> NDArray[np.float64]:
    """H tensor.

    Args:
        theta: Theta tensor
        D: Dimensionality
        xp (ModuleType): Array module, either numpy (CPU) or cupy

    Returns:
        Updated theta.
    """
    for i in range(D):
        theta = xp.cumsum(theta, axis=i)
    return theta
  

def get_q(h: NDArray[np.float64], gpu=True, xp: ModuleType = cp) -> NDArray[np.float64]:
    """Q tensor.

    Args:
        H: H tensor

    Returns:
        Updated Q.
    """
    if gpu:
        logsumexp = cupy_logsumexp
    else:
        logsumexp = scipy_logsumexp
    return xp.exp(h - logsumexp(h))


def get_slice(key, D):
    indices = [0 for _ in range(D)]
    for k in key:
        indices[k]=slice(None)
    return tuple(indices)

def init_theta(keys,shape,theta0_flag=False, xp: ModuleType = cp):
    theta=xp.random.normal(0,0.1,shape)
    mask=xp.zeros(shape)
    for key in keys:
        s=get_slice(key,len(shape))
        mask[s]=1
    if theta0_flag:
        i=tuple([0]*len(shape))
        mask[i]=0
    return theta*mask, mask

def compute_G(eta, mask, xp: ModuleType = cp):
    I=xp.meshgrid(*[xp.arange(n) for n in eta.shape], indexing='ij')
    I_=xp.array(I).transpose(tuple(list(range(1,len(eta.shape)+1))+[0]))

    I_masked = I_[mask==1,:]
    eta_b = eta[mask==1]

    I_i = I_masked[:, xp.newaxis, :]  # (n,1,d)
    I_j = I_masked[xp.newaxis, :, :]  # (1,n,d)

    # maximum index: (n,n,d)
    k_ = xp.maximum(I_i, I_j)

    eta_k = eta[tuple(k_.transpose(2,0,1))]  # shape (n,n)

    # outer product using broadcasting
    eta_prod = eta_b[:, xp.newaxis] * eta_b[xp.newaxis, :]

    return eta_k-eta_prod,eta_b,I_masked

def LD_MBA(
    X: NDArray[np.float64],
    I: List[Tuple[int, ...]]|None = None,
    order: int = 2,
    n_iter: int = 100,
    lr: float = 1.0,
    eps: float = 1.0e-5,
    error_tol: float = 1.0e-5,
    ngd: bool = True,
    ngd_lstsq =True,
    verbose: bool = True,
    gpu: bool=True, # Use GPU (CUDA or ROCm depending on the installed CuPy version).
    dtype: np.dtype | None = None #By default, the data-type is inferred from the input data.

) -> tuple[list[list[float]], np.float64, NDArray[np.float64], NDArray[np.float64]]:
    """Compute many-body tensor approximation.

    Args:
        X: Input tensor.
        I: A list of pairs of indices that represent slices with nonzero elements in the parameter tensor.
           e.g. [(0,1),(2,),(1,3)]
        n_iter: Maximum number of iteration.
        lr: Learning rate.
        eps: (see paper).
        error_tol: KL divergence tolerance for the iteration.
        ngd: Use natural gradient.
        verbose: Print debug messages.

    Returns:
        all_history_kl: KL divergence history.
        scaleX: Scaled X tensor.
        Q: Q tensor.
        theta: Theta.
    """
    if gpu:
        X = cp.asarray(X, dtype=dtype)
        eps = cp.asarray(eps, dtype=dtype)
        lr = cp.asarray(lr, dtype=dtype)
        logsumexp = cupy_logsumexp
        xp=cp
        if cp==np and verbose:
            print("GPU mode is disabled because cupy module does not installed")
    else:
        logsumexp = scipy_logsumexp
        xp=np
    all_history_kl = []
    D = len(X.shape)
    S = X.shape
    if I is None:
        I = [e for e in itertools.combinations(list(range(D)), order)]
    scaleX = xp.sum(X + eps)
    P = (X + eps) / scaleX
    # update: theta => H => Q
    theta, theta_mask=init_theta(I, S, xp=xp)
    h=get_h(theta,D, xp=xp)
    Q=get_q(h,gpu, xp=xp)
    # evaluation
    history_kl = []
    prev_kld = None
    kld = kl(P, Q, xp=xp)
    history_kl.append(kld)
    if verbose:
        mse=xp.mean((P - Q) ** 2)
        print("iter=", 0, "kl=", kld, "mse=", mse)
    ### eta_hat
    eta_hat = get_eta(P, D, xp=xp)
    eta_hat_b=eta_hat[theta_mask==1]
    tol_cnt=0
    for i in range(n_iter):
        eta = get_eta(Q, D, xp=xp)
        # update theta using eta
        if ngd:
            theta_b=theta[theta_mask==1]
            G,eta_b,I_masked=compute_G(eta,theta_mask, xp=xp)
            # theta_b[1:] -= lr*cp.linalg.pinv(G[1:,1:])@(eta_b[1:]-eta_hat_b[1:])
            if ngd_lstsq:
                v = xp.linalg.lstsq(G[1:, 1:], lr * (eta_b[1:] - eta_hat_b[1:]), rcond=None)[0]
                theta_b[1:] -= v
            else:
                v = xp.linalg.solve(G[1:, 1:], lr * (eta_b[1:] - eta_hat_b[1:]))
                theta_b[1:] -= v
            theta[tuple(I_masked.T)] = theta_b
        else:
            theta-=lr*(eta-eta_hat)*theta_mask
        # update: theta => H => Q
        h=get_h(theta,D, xp=xp)
        Q=get_q(h,gpu, xp=xp)
        # evaluation
        kld = kl(P, Q, xp=xp)
        history_kl.append(kld)
        if verbose:
            print("iter=", i + 1, "kl=", kld, "mse=", xp.mean((P - Q) ** 2))
        if prev_kld is not None and prev_kld - kld < error_tol:
            break
        if xp.isnan(kld) or xp.isinf(kld) or xp.isneginf(kld):
            break
        prev_kld = kld
    all_history_kl.append([float(x) for x in history_kl])
    if gpu:
        scaleX = xp_get(scaleX)  # type: ignore
        P = xp_get(P)  # type: ignore
        Q = xp_get(Q)  # type: ignore
        theta = xp_get(theta)  # type: ignore
    return all_history_kl, scaleX, P, Q, theta  # type: ignore


def get_weight(shape, I_x=None, order=2, xp: ModuleType = cp):
    W=xp.zeros(shape)
    D=len(shape)
    if I_x is None:
        I_x=[e for e in itertools.combinations(list(range(D)), order)]
    for e in I_x:
        s=[0]*D
        for i in e:
            s[i]=slice(None)
        W[tuple(s)]+=1
    return W,I_x

def compute_nbody(theta, shape, I_x=None, order=2, dtype=None, gpu=True, verbose=False):
    if gpu:
        theta = cp.asarray(theta, dtype=dtype)
        if cp==np and verbose:
            print("GPU mode is disabled because cupy module does not installed")
        xp = cp
    else:
        xp = np
    W,I_x=get_weight(shape, I_x, order,xp=xp)
    D=len(shape)
    h=get_h(theta, D, xp=xp)
    logz=-theta[tuple([0]*D)]
    #logz=logsumexp(h)

    X_out=[]
    for e in I_x:
        s=[0]*D
        for i in e:
            s[i]=slice(None)
        h_=h[tuple(s)]/W[tuple(s)]-logz/len(I_x)
        x_=xp.exp(h_)
        if gpu:
          x_ = xp_get(x_)  # type: ignore
        X_out.append((e,x_))
    ## transepose
    new_X_out=[]
    for k,x_ in X_out:
        temp=[(org_order,new_i) for new_i,(_,org_order) in enumerate(sorted((v,org_order)for org_order,v in enumerate(k)))]
        tpl=tuple([new_i for _,new_i in sorted(temp)])
        y_=xp.transpose(x_,tpl)
        new_X_out.append((k,y_))
    return new_X_out

def recons_nbody(X_out, D, rescale=True, dtype=None, gpu=True):
    if gpu:
        xp = cp
    else:
        xp = np
    l=[]
    for e,x_ in X_out:
        if gpu:
          x_ = cp.asarray(x_, dtype=dtype)
        l.append(x_)
        l.append(e)
    l.append(tuple(list(range(D))))
    Q2=xp.einsum(*l)
    if rescale:
        Q2=Q2/xp.sum(Q2)
    if gpu:
        Q2 = xp_get(Q2)  # type: ignore
    return Q2

