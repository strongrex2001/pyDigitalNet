# -*- coding: utf-8 -*-
"""
Integrator
----------
| Created on Sun Mar 29 15:52:12 2020
@author: eliphat

| This module provides main APIs for numerical integration.
| NOTICE: This package is in preview. APIs are subject to change.

References
----------
| [1]
  Tuffin, B. (2004).
  Randomization of Quasi-Monte Carlo Methods for Error Estimation:
  Survey and Normal Approximation, Monte Carlo Methods and Applications,
  10(3-4), 617-628. doi: https://doi.org/10.1515/mcma.2004.10.3-4.617

License
-------
   Copyright 2020 eliphat@sjtu.edu.cn

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import numpy

from .digital_sequences import digital_sequence
from .sequences.faure import faure_generating_matrices, naive_next_prime


def get_n(epsilon, base):
    return int(-numpy.log(epsilon) / numpy.log(base)) * 2


def welford_update(stats, current):
    """
    Welford's Algorithm

    References
    ----------
    | [1]
      B. P. Welford (1962),  Note on a Method for Calculating
      Corrected Sums of Squares and Products, Technometrics,
      4:3, 419-420, DOI: 10.1080/00401706.1962.10490022
    """
    (n, mean, M2) = stats
    n += 1
    d1 = current - mean
    mean += d1 / n
    d2 = current - mean
    M2 += d1 * d2

    return (n, mean, M2)


def integrate_basic(func, ranges, epsilon=1e-8, robust_coeff=3):
    """
    Integrates numerically a n-dim function.

    Parameters
    ----------
    func: callable [x1, x2, ..., xS] -> scalar
      Base, or radix of generated digital sequence.
    ranges: array-like of shape (S, 2)
      Ranges of each variable.
    epsilon: float
      Required precision. (may be estimated in some situations)
    limit: int
      If variance < `epsilon`
      holds for continuously `limit` times of evaluation,
      the result is returned.

    Returns
    -------
    v: float
      Result of integration.
    """
    ranges = numpy.array(ranges, copy=False)
    s = ranges.shape[0]
    b = naive_next_prime(s)
    N = get_n(epsilon, b)
    Cs = faure_generating_matrices(s, N, b)
    dx = numpy.prod(ranges[:, 1] - ranges[:, 0])
    n = 0
    su = numpy.zeros([robust_coeff])
    shifts = numpy.random.uniform(size=[robust_coeff, s])
    tar_std = epsilon / dx
    for x in digital_sequence(b, Cs):
        for i, shift in enumerate(shifts):
            x_s = (x + shift) % 1.0
            x_d = x_s * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
            su[i] += func(x_d)

        n += 1
        if n > 2 and numpy.std(su) < tar_std * (n - 1):
            return numpy.mean(su) / n * dx
    raise ValueError("Integration precision can't be met.")
