# -*- coding: utf-8 -*-
"""Implementations found in the original Jupyter Notebook"""
import itertools

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.special import logsumexp


def kl(P: NDArray[np.float_], Q: NDArray[np.float_]) -> np.float_:
    """Kullback-Leibler divergence.

    Args:
        P: P tensor
        Q: Q tensor

    Returns:
        KL divergence.
    """
    return np.sum(P * np.log(P / Q)) - np.sum(P) + np.sum(Q)


def get_eta(Q: NDArray[np.float_], D: int) -> NDArray[np.float_]:
    """Eta tensor.

    Args:
        Q: Q tensor
        D: Dimensionality

    Returns:
        Eta tensor.
    """
    for i in range(D):
        Q = np.flip(np.cumsum(np.flip(Q, axis=i), axis=i), axis=i)
    return Q


def get_h(theta: NDArray[np.float_], D: int) -> NDArray[np.float_]:
    """H tensor.

    Args:
        theta: Theta tensor
        D: Dimensionality

    Returns:
        Updated theta.
    """
    for i in range(D):
        theta = np.cumsum(theta, axis=i)
    return theta


def LD(
    X: NDArray[np.float_],
    B: NDArray[np.intp] | list[tuple[int, ...]] | None = None,
    order: int = 2,
    n_iter: int = 10,
    lr: float = 1.0,
    eps: float = 1.0e-5,
    error_tol: float = 1.0e-5,
    ngd: bool = True,
    verbose: bool = True,
) -> tuple[list[list[float]], np.float_, NDArray[np.float_], NDArray[np.float_]]:
    """Compute many-body tensor approximation.

    Args:
        X: Input tensor.
        B: B tensor.
        order: Order of default tensor B, if not provided.
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
    all_history_kl = []
    D = len(X.shape)
    S = X.shape

    if B is None:

        def constraint(I):
            I = np.array(I)
            if np.sum(I != 0) <= order:
                return True
            else:
                return False

        B = [I for I in itertools.product(*[list(range(S[d])) for d in range(D)]) if constraint(I)]

    scaleX = np.sum(X + eps)
    P = (X + eps) / scaleX

    Q = np.ones(P.shape)
    Q = Q / np.sum(Q)
    ### eta
    eta_hat = get_eta(P, D)
    eta_hat_b = np.empty((len(B),))
    for u, I in enumerate(B):
        eta_hat_b[u] = eta_hat[I]
    ###
    eta_b = np.empty((len(B),))
    theta_b = np.zeros((len(B),))
    G = np.zeros((len(B), len(B)))
    history_kl = []
    prev_kld = None
    # evaluation
    kld = kl(P, Q)
    history_kl.append(kld)

    if verbose:
        print("iter=", 0, "kl=", kld, "mse=", np.mean((P - Q) ** 2))
    for i in range(n_iter):
        # compute eta
        eta = get_eta(Q, D)
        for u, I in enumerate(B):
            eta_b[u] = eta[tuple(I)]

        # compute G
        for u, I in enumerate(B):
            for v in range(u + 1):
                J = B[v]
                I_ = np.array(I)
                J_ = np.array(J)
                K_ = np.maximum(I_, J_)
                G[u, v] = eta[tuple(K_)] - eta[I] * eta[J]

        # update theta_b
        if ngd:
            # theta_b[1:] -= lr*np.linalg.pinv(G[1:,1:])@(eta_b[1:]-eta_hat_b[1:])
            v = scipy.linalg.solve(
                G[1:, 1:], lr * (eta_b[1:] - eta_hat_b[1:]), lower=True, assume_a="sym"
            )
            theta_b[1:] -= v
        else:
            theta_b -= lr * (eta_b - eta_hat_b)
        # theta_b=>theta
        theta = np.zeros(S)
        for u, I in enumerate(B):
            theta[I] = theta_b[u]
        # theta => H => Q
        Hq = get_h(theta, D)

        # without logsum exp
        # Q_=np.exp(Hq)
        # Q=(Q_+eps)/np.sum(Q_+eps)

        # with logsumexp
        logQ_ = Hq
        logQ = logQ_ - logsumexp(logQ_)
        Q = np.exp(logQ) + eps

        # evaluation
        kld = kl(P, Q)
        history_kl.append(kld)
        if verbose:
            print("iter=", i + 1, "kl=", kld, "mse=", np.mean((P - Q) ** 2))
        if prev_kld is not None and prev_kld - kld < error_tol:
            break
        prev_kld = kld

    all_history_kl.append(history_kl)
    return all_history_kl, scaleX, Q, theta  # type: ignore
