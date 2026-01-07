import dataclasses
from legendre_decomp import LD_MBA
from types import ModuleType
from typing import Dict, Tuple, List
from numpy.typing import NDArray
from scipy.special import logsumexp as scipy_logsumexp
from legendre_decomp.module_mba import initialize_theta, get_h, get_q, kl
import numpy as np
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


@dataclasses.dataclass
class LDComponent:
    I: List[Tuple[int, ...]]
    theta:  NDArray[np.float64]|None=None
    theta_mask:  NDArray[np.float64]|None=None
    Q:  NDArray[np.float64]|None=None
    gamma: NDArray[np.float64]|None=None
    pi: float=1.0

def mixQ(components, xp: ModuleType = cp):
  Q=xp.zeros_like(components[0].Q)
  for comp in components:
    Q+=comp.Q*comp.pi
  return Q

def MixLD_MBA(
    X: NDArray[np.float64],
    components: LDComponent,
    n_round: int=300,
    n_iter: int = 100,
    lr: float = 1.0,
    eps: float = 1.0e-5,
    error_tol: float = 1.0e-5,
    em_tol: float = 1.0e-5,
    ngd: bool = True,
    ngd_lstsq =True,
    verbose: bool = True,
    verbose_ld: bool = True,
    gpu: bool=True, # Use GPU (CUDA or ROCm depending on the installed CuPy version).
    dtype: np.dtype | None = None #By default, the data-type is inferred from the input data.

) -> tuple[list[list[float]], np.float64, NDArray[np.float64], NDArray[np.float64]]:
    """Compute many-body tensor approximation.
    Args:
        X: Input tensor.
        I: A list of pairs of indices that represent slices with nonzero elements in the parameter tensor.
           e.g. [(0,1),(2,),(1,3)]
        n_round: Maximum number of EM rounds.
        n_iter: Maximum number of iteration.
        lr: Learning rate.
        eps: (see paper).
        error_tol: KL divergence tolerance for the iteration.
        em_tol: KL divergence tolerance for the EM round.
        ngd: Use natural gradient.
        verbose: Print debug messages.
        verbose_ld: Print debug messages.

    Returns:
        all_history_kl: KL divergence history.
        scaleX: Scaled X tensor.
        Q: Q tensor.
        theta: Theta.
    """
    if gpu:
        X = cp.asarray(X, dtype=dtype)
        #eps = cp.asarray(eps, dtype=dtype)
        #lr = cp.asarray(lr, dtype=dtype)
        xp: ModuleType = cp
    else:
        xp: ModuleType = np


    mix_history_kl = []
    D = len(X.shape)
    S = X.shape
    scaleX = xp.sum(X + eps)
    P = (X + eps) / scaleX
    K=len(components)

    for comp in components:
      comp.theta, comp.theta_mask=initialize_theta(comp.I, S,  xp = xp)
      comp.gamma=xp.ones(S)
      h=get_h(comp.theta, D, xp)
      comp.Q=get_q(h,gpu, xp)
      comp.pi=1/K
    prev_kld=None
    for i in range(n_round):
      # Expectation
      ps=mixQ(components, xp)
      for comp in components:
        comp.gamma=comp.Q*comp.pi/ps

      # Maximization
      s_pi=0
      for comp in components:
        X_=X*comp.gamma
        all_history_kl, scaleX, _, Q,theta = LD_MBA(X_,comp.I,init_theta=comp.theta,init_theta_mask=comp.theta_mask,n_iter=n_iter,lr=lr, error_tol=error_tol,gpu=gpu, verbose=verbose_ld)

        if gpu:
          Q = xp.asarray(Q, dtype=dtype)
          theta = xp.asarray(theta, dtype=dtype)
        comp.Q=Q
        comp.theta=theta
        comp.pi=xp.sum(X_)
        s_pi+=comp.pi
      for comp in components:
        comp.pi/=s_pi
      # evaluation
      Q=mixQ(components, xp)
      kld = kl(P, Q, xp)
      mix_history_kl.append(float(kld))
      if verbose:
          print("round=", i + 1, "kl=", kld, "mse=", xp.mean((P - Q) ** 2))
      if prev_kld is None:
        prev_kld=kld
      elif em_tol>0:
        if prev_kld-kld<em_tol and i>1:
          break
      prev_kld=kld
    return mix_history_kl,  scaleX, P, Q, components

