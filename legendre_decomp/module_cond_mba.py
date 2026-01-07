from legendre_decomp import LD_MBA
from legendre_decomp.module_mba import compute_nbody
from legendre_decomp.module_mba import recons_nbody
from legendre_decomp.module_mba import get_slice
from legendre_decomp.module_mba import kl
import itertools
from typing import Dict, Tuple, List, Any
from numpy.typing import NDArray
import numpy as np

def decomp_given_tensor(
    X_: NDArray[np.float64],
    I: List[Tuple[int, ...]],
    n_iter: int = 100,
    lr: float = 1.0,
    eps: float = 1.0e-5,
    error_tol: float = 1.0e-5,
    ngd: bool = True,
    ngd_lstsq: bool =True,
    gpu: bool=True,
    verbose: bool=True,
    ):
  all_history_kl, scaleX, P, Q,theta = LD_MBA(
      X_, I,
      n_iter=n_iter,
      lr = lr,
      eps = eps,
      error_tol= error_tol,
      ngd = ngd,
      ngd_lstsq =ngd_lstsq,
      gpu=gpu,
      verbose=verbose)
  X_out=compute_nbody(theta,X_.shape,I_x=I,gpu=False)
  Q2=recons_nbody(X_out, len(X_.shape),gpu=False)
  #MAE between X and Q2
  metric_X_Q2=np.mean(np.abs(scaleX*Q2-X_))
  metric_P_Q=np.mean(np.abs(Q-P))
  metric_kl_P_Q2=kl(P,Q2,xp=np)
  result={"given_tensor_MAE_X_Q2":metric_X_Q2,
          "given_tensor_MAE_P_Q":metric_P_Q,
          "given_tensor_kl_P_Q2":metric_kl_P_Q2,
          }
  return X_out, theta, result


def build_condition(
    theta_dict: Dict[Tuple[int, ...], NDArray[np.float64]],
    I: List[Tuple[int, ...]],
    shape: Tuple[int, ...]):
  # theta_dict: {(1,2)=>theta1, (2,3)=>theta2}
  # I: I for MBA: e.g. [(0,1),(3,1)]
  theta=np.random.normal(0,0.1,shape)
  mask=np.zeros(shape)
  for key in I:
      s=get_slice(key,len(shape))
      mask[s]=1
  theta=theta*mask
  for prior_index, theta_prior in theta_dict.items():
      prior_slice=get_slice(prior_index,len(shape))
      theta[prior_slice]+=theta_prior
  return theta, mask


def conditional_decomp_tensor(
    X_: NDArray[np.float64],
    I: List[Tuple[int, ...]],
    I_c: Tuple[int, ...],
    theta: NDArray[np.float64],
    mask: NDArray[np.float64],
    X_out_given_: Dict[Any,Any],
    theta_given: NDArray[np.float64],
    n_iter: int = 100,
    lr: float = 1.0,
    eps: float = 1.0e-5,
    error_tol: float = 1.0e-5,
    ngd: bool = True,
    ngd_lstsq: bool =True,
    verbose: bool = True,
    verbose_ld: bool = True,
    gpu: bool=True,
    ):
  all_history_kl, scaleX, P, Q,theta = LD_MBA(
      X_, I,init_theta=theta,init_theta_mask=mask,
      lr = lr,
      eps = eps,
      error_tol= error_tol,
      ngd = ngd,
      ngd_lstsq =ngd_lstsq,
      gpu=gpu,
      verbose=verbose)
  theta_new=theta.copy()
  I_sc=get_slice(I_c,len(theta.shape))
  theta_new[I_sc]-=theta_given
  X_out=compute_nbody(theta_new,X_.shape,I_x=I,gpu=False)
  if X_out_given_ is not None:
    X_out=X_out+X_out_given_
  if len(X_out)==1:
    Q2=X_out[0][1]
  else:
    #for s,x in X_out:
    #  print(s, x.shape)
    Q2=recons_nbody(X_out, len(X_.shape),gpu=False)
  metric_X_Q2=np.mean(np.abs(scaleX*Q2-X_))
  metric_P_Q=np.mean(np.abs(Q-P))
  metric_kl_P_Q2=kl(P,Q2,xp=np)
  metric_kl_P_Q=kl(P,Q,xp=np)
  result={"MAE_X_Q2":metric_X_Q2,
          "MAE_P_Q":metric_P_Q,
          "kl_P_Q2":metric_kl_P_Q2,
          "kl_P_Q":metric_kl_P_Q,
          }

  return all_history_kl, scaleX, P, Q,theta, X_out, Q2, result
  

# thetaを条件として使用するために変換する
# I=[(0,1),(1,2)]
# I_c=(0,2)
def CLD_MBA(
    X: NDArray[np.float64],
    I: List[Tuple[int, ...]],
    Y: NDArray[np.float64],
    I_c: Tuple[int, ...],
    n_iter: int = 100,
    lr: float = 1.0,
    eps: float = 1.0e-5,
    error_tol: float = 1.0e-5,
    ngd: bool = True,
    ngd_lstsq:bool =True,
    verbose: bool = False,
    gpu: bool=True):
  order_y=2
  D=len(Y.shape)
  I_Y = [e for e in itertools.combinations(list(range(D)), order_y)]
  # 与えられたテンソルのthetaを計算するために分解する
  if verbose:
      print("=== condition MBA ===")
  X_out_given, theta_given, result1 = decomp_given_tensor(
      Y, I_Y,
      lr = lr,
      eps = eps,
      error_tol= error_tol,
      ngd = ngd,
      ngd_lstsq =ngd_lstsq,
      gpu=gpu,
      verbose=verbose)
  # 結果保存
  theta_dict={}
  theta_dict[I_c]=theta_given
  # 条件付き分解
  if verbose:
      print("=== target MBA ===")
  theta, mask = build_condition(theta_dict=theta_dict, I=I, shape=X.shape)
  all_history_kl, scaleX, P, Q,theta, X_out, Q2, result = conditional_decomp_tensor(
      X, I, I_c, theta, mask, X_out_given, theta_given,
      lr = lr,
      eps = eps,
      error_tol= error_tol,
      ngd = ngd,
      ngd_lstsq =ngd_lstsq,
      gpu=gpu,
      verbose=verbose)

  return all_history_kl, scaleX, P, Q,theta, X_out, Q2, result, result1


