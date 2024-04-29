# -*- coding: utf-8 -*-
"""Test consistency between original and current utility functions."""
import itertools

import numpy as np
import pytest

from manybodytensor.utils import default_B, index_channel_B, ring_decomposition_B


@pytest.mark.parametrize("order", [1, 2, 3])
def test_default(order, random_X_array):
    def constraint(I):
        I = np.array(I)
        if np.sum(I != 0) <= order:
            return True
        else:
            return False

    shape = random_X_array.shape
    B_a = [
        I
        for I in itertools.product(*[list(range(shape[d])) for d in range(len(shape))])
        if constraint(I)
    ]
    B_b = default_B(shape, order)

    for b_a, b_b in zip(B_a, B_b):
        assert (b_a == b_b).all()


def test_ring_decomposition(coil_100_array):
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

    shape = coil_100_array.shape
    B_a = [
        I
        for I in itertools.product(*[list(range(shape[d])) for d in range(len(shape))])
        if constraint(I)
    ]
    B_b = ring_decomposition_B(shape)
    for b_a, b_b in zip(B_a, B_b):
        assert b_a == b_b


def test_index_channel(coil_100_array):
    def constraint(I):
        I = np.array(I)
        if np.sum(I != 0) <= 2:
            if I[0] != 0 and I[3] != 0:
                return False
            else:
                return True
        else:
            return False

    shape = coil_100_array.shape
    B_a = [
        I
        for I in itertools.product(*[list(range(shape[d])) for d in range(len(shape))])
        if constraint(I)
    ]
    B_b = index_channel_B(shape)
    for b_a, b_b in zip(B_a, B_b):
        assert b_a == b_b
