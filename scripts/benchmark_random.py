#!/usr/bin/env python
"""Benchmark with random tensor"""
import argparse
import sys

import numpy as np

from legendre_decomp import LD
from legendre_decomp.naive import LD as LD_naive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shape", nargs="+", type=int, help="Shape of random input matrix")
    parser.add_argument("-o", "--order", type=int, default=2, help="Order of B matrix")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--naive", action="store_true", help="Use naive implementation")
    args = parser.parse_args()

    if args.gpu and args.old:
        sys.exit("Old implementation does not support GPU. Exiting.")

    X = np.random.uniform(0, 0.1, size=args.shape)
    if args.naive:
        LD_naive(X, order=args.order, verbose=False)
    else:
        LD(X, order=args.order, gpu=args.gpu, verbose=False)


if __name__ == "__main__":
    main()
