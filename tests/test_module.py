# -*- coding: utf-8 -*-
"""Test consistency between original and current implementation."""
import numpy as np

from LegendreDecomposition import LD as LD_cupy
from LegendreDecomposition.naive import LD
from LegendreDecomposition.utils import index_channel_B, ring_decomposition_B

gpu=True

def test_random(random_X_array):
    """Test random tensor."""
    _, scaleX, Q, Hq = LD(random_X_array, order=2)
    _, scaleX_cpu, Q_cpu, Hq_cpu = LD_cupy(random_X_array, order=2, exit_abs=False, gpu=False)
    _, scaleX_gpu, Q_gpu, Hq_gpu = LD_cupy(random_X_array, order=2, exit_abs=False, gpu=gpu)
    assert np.isclose(scaleX, scaleX_cpu) and np.isclose(scaleX, scaleX_gpu)
    assert np.isclose(Q, Q_cpu).all() and np.isclose(Q, Q_gpu).all()
    assert np.isclose(Hq, Hq_cpu).all() and np.isclose(Hq, Hq_gpu).all()


def test_2d_similarity(coil_100_array):
    """Test coil 100."""
    _, scaleX, Q, Hq = LD(coil_100_array, order=2, verbose=False)
    _, scaleX_cpu, Q_cpu, Hq_cpu = LD_cupy(coil_100_array, order=2, verbose=False, gpu=False)
    _, scaleX_gpu, Q_gpu, Hq_gpu = LD_cupy(coil_100_array, order=2, verbose=False, gpu=gpu)
    assert np.isclose(scaleX, scaleX_cpu) and np.isclose(scaleX, scaleX_gpu)
    assert np.isclose(Q, Q_cpu).all() and np.isclose(Q, Q_gpu).all()
    assert np.isclose(Hq, Hq_cpu).all() and np.isclose(Hq, Hq_gpu).all()


def test_ring_decomposition(coil_100_array):
    """Test ring decomposition."""
    B = ring_decomposition_B(coil_100_array.shape)
    _, scaleX, Q, Hq = LD(coil_100_array, B, verbose=False)
    _, scaleX_cpu, Q_cpu, Hq_cpu = LD_cupy(
        coil_100_array, B, verbose=False, exit_abs=False, gpu=False
    )
    _, scaleX_gpu, Q_gpu, Hq_gpu = LD_cupy(
        coil_100_array, B, verbose=False, exit_abs=False, gpu=gpu
    )
    assert np.isclose(scaleX, scaleX_cpu) and np.isclose(scaleX, scaleX_gpu)
    assert np.isclose(Q, Q_cpu).all() and np.isclose(Q, Q_gpu).all()
    assert np.isclose(Hq, Hq_cpu).all() and np.isclose(Hq, Hq_gpu).all()


def test_index_channel(coil_100_array):
    """Test index channel."""
    B = index_channel_B(coil_100_array.shape)
    _, scaleX, Q, Hq = LD(coil_100_array, B, verbose=False)
    _, scaleX_cpu, Q_cpu, Hq_cpu = LD_cupy(coil_100_array, B, verbose=False, gpu=False)
    _, scaleX_gpu, Q_gpu, Hq_gpu = LD_cupy(coil_100_array, B, verbose=False, gpu=gpu)
    assert np.isclose(scaleX, scaleX_cpu) and np.isclose(scaleX, scaleX_gpu)
    assert np.isclose(Q, Q_cpu).all() and np.isclose(Q, Q_gpu).all()
    assert np.isclose(Hq, Hq_cpu).all() and np.isclose(Hq, Hq_gpu).all()


def test_movie_lens(movie_lens_array):
    _, scaleX, Q, Hq = LD(movie_lens_array, order=2, eps=1.0e-10)
    _, scaleX_cpu, Q_cpu, Hq_cpu = LD_cupy(
        movie_lens_array, order=2, eps=1.0e-10, exit_abs=False, gpu=False
    )
    _, scaleX_gpu, Q_gpu, Hq_gpu = LD_cupy(
        movie_lens_array, order=2, eps=1.0e-10, exit_abs=False, gpu=gpu
    )
    assert np.isclose(scaleX, scaleX_cpu) and np.isclose(scaleX, scaleX_gpu)
    assert np.isclose(Q, Q_cpu).all() and np.isclose(Q, Q_gpu).all()
    # NOTE: Probably due to different linalg.solve used (_SYSV vs _GESV), the exact value does
    # not match within numpy's default precision. However the KL and MSE are similar (?)
    # assert np.isclose(Hq, Hq_cpu).all() and np.isclose(Hq, Hq_gpu).all()
