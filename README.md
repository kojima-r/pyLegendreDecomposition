# Legendre decomposition implementation

[![GitHub Pages](https://github.com/kojima-r/pyLegendreDecomposition/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/kojima-r/pyLegendreDecomposition/actions/workflows/gh-pages.yml)

This project contains the following three python implementation of Legendre decomposition and Many-body Approximation for Non-negative Tensors.

- Naïve implementation:
  - straightforward implementation by using SciPy/NumPy

- Faster implementation:
  - Vectorization using equivalent NumPy code

- GPU implementation:
  - Using CuPy to further change NumPy code to GPU-executable code
  - The same method is used for the Faster implementation and GPU, and switching is done by enabling or disabling the GPU.
## Installation

Installation of pyLegendreDecomposition
```
pip install git+https://github.com/kojima-r/pyLegendreDecomposition.git
```

Installation of Cupy
```
pip install  cupy-cuda12x
```
Please change the `cuda12x` as appropriate.
See the cupy manual for details.

https://docs.cupy.dev/en/stable/install.html

## Simple Usage
```python
from legendre_decomp import LD
all_history_kl,scaleX,Q,Hq = LD(X,order=2)
```

- X: a original input tensor
- order: an order parameter of many-body approximation
- all_history_kl: KL divergence history
- scaleX: Scaled X tensor
- Q: Q tensor
- theta: Theta

See [API reference of LD](https://kojima-r.github.io/pyLegendreDecomposition/autoapi/legendre_decomp/module/index.html#legendre_decomp.module.LD) for details.
## Documents / notebook

[API reference](https://kojima-r.github.io/pyLegendreDecomposition/autoapi/index.html)

[Benchmark results](https://kojima-r.github.io/pyLegendreDecomposition/benchmarks.html)

[Colaboratory notebook](https://colab.research.google.com/drive/10P1hVJZvEv51GdnGNjMKk2qTjIZ7XwpF?usp=sharing)

## Project structure
- `legendre_decomp`: Main module
- `docs`: Documentation
- `tests`: Functional tests
- `scripts`: Benchmark scripts
- `notebooks`: Benchmark Jupyter Notebooks

## Reference
Many-body Approximation for Non-negative Tensors:
https://arxiv.org/abs/2209.15338
```
@article{ghalamkari2024many,
  title={Many-body Approximation for Non-negative Tensors},
  author={Ghalamkari, Kazu and Sugiyama, Mahito and Kawahara, Yoshinobu},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
Legendre Decomposition for Tensors:
https://arxiv.org/abs/1802.04502
```
@article{sugiyama2018legendre,
  title={Legendre decomposition for tensors},
  author={Sugiyama, Mahito and Nakahara, Hiroyuki and Tsuda, Koji},
  journal={Advances in Neural Information Processing Systems},
  volume={31},
  year={2018}
}
```

## How to build documentation (need `sphinx`)
Installation of sphinx modules
```shell
pip install sphinx-autoapi
pip install pydata-sphinx-theme
```

Building documents
```shell
sphinx-build -M html docs/source docs/build
```

or

```shell
cd docs
make html
```

The documentation including installation manual can then be accessed from `docs/build/html/index.html`

