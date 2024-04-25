#!/usr/bin/env python
"""Benchmark with random tensor"""
import argparse
import sys

import numpy as np

from manybodytensor import MBTA
from manybodytensor.old import MBTA as MBTA_old


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shape", nargs="+", type=int, help="Shape of random input matrix")
    parser.add_argument("-o", "--order", type=int, default=2, help="Order of B matrix")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--old", action="store_true", help="Use old implementation")
    args = parser.parse_args()

    if args.gpu and args.old:
        sys.exit("Old implementation does not support GPU. Exiting.")

    X = np.random.uniform(0, 0.1, size=args.shape)
    if args.old:
        MBTA_old(X, order=args.order, verbose=False)
    else:
        MBTA(X, order=args.order, gpu=args.gpu, verbose=False)


if __name__ == "__main__":
    main()
