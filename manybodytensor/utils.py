# -*- coding: utf-8 -*-
"""Utility functions"""
import itertools
from types import ModuleType
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def default_B(shape: Sequence[int], order: int, xp: ModuleType = np) -> NDArray[np.intp]:
    """Vectorized implementation of the default B tensor.

    Args:
        shape: Shape of the corresponding X tensor.
        order: Order of the B tensor.
        xp (ModuleType): Array module, either numpy (CPU) or cupy (CUDA/GPU)

    Returns:
        array-like: Default B tensor of specified order.
    """
    B = xp.indices(shape).reshape(len(shape), -1).T
    mask = (B != 0).sum(axis=1) <= order
    return B[mask]


def ring_decomposition_B(shape: Sequence[int]) -> list[tuple[int, ...]]:
    """Construct B tensor for ring decomposition.

    Args:
        shape: Shape of the corresponding X tensor.

    Returns:
        B tensor for ring decomposition.
    """

    def constraint(I):
        I = np.array(I)
        idx = np.where(I)[0]
        if len(idx) == 2:
            if idx[1] - idx[0] == 1 or (idx[1] == 3 and idx[0] == 0):
                return True
            else:
                return False
        else:
            return False

    return [
        I
        for I in itertools.product(*[list(range(shape[d])) for d in range(len(shape))])
        if constraint(I)
    ]


def index_channel_B(shape: Sequence[int]) -> list[tuple[int, ...]]:
    """２次の近似から、index - channel 結合を切断する.

    Args:
        shape: Shape of the corresponding X tensor.

    Returns:
        B tensor with broken index-channel bond for larger than 2D.
    """

    def constraint(I):
        I = np.array(I)
        if np.sum(I != 0) <= 2:
            if I[0] != 0 and I[3] != 0:
                return False
            else:
                return True
        else:
            return False

    return [
        I
        for I in itertools.product(*[list(range(shape[d])) for d in range(len(shape))])
        if constraint(I)
    ]
