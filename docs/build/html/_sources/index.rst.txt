User guide
==========

.. toctree::
   :maxdepth: 2
   :caption: Contents

   benchmarks

Installation
------------

Installing CuPy
^^^^^^^^^^^^^^^

(Recommended) install one of the binary wheels
depending on the system's installed CUDA Toolkit version.

.. code-block:: console

   $ pip install cupy-cuda11x  # CUDA 11.x
   $ pip install cupy-cuda12x  # CUDA 12.x
   $ pip install cupy-rocm-4-3  # AMD ROCm 4.3
   $ pip install cupy-rocm-5-0  # AMD ROCm 5.0

Other options for installing CuPy can be found
`here <https://docs.cupy.dev/en/stable/install.html>`_

Installing manybodytensor
^^^^^^^^^^^^^^^^^^^^^^^^^

Install using either ``poetry`` or ``pip``.

Poetry
""""""

.. code-block:: console

   $ poetry install

Pip
"""

.. code-block:: console

   $ pip install [-e] .

Project structure
-----------------

- ``manybodytensor/``: Main module
- ``docs/``: Documentation
- ``tests/``: Functional tests
- ``scripts/``: Benchmark scripts
- ``notebooks/``: Benchmark Jupyter Notebooks 

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
