# -*- coding: utf-8 -*-
"""
Integrator
----------
| Created on Sun Mar 29 15:52:12 2020
@author: eliphat

| This module provides main APIs for numerical integration.
| NOTICE: This package is in preview. APIs are subject to change.

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
    return naive_next_prime(int(-numpy.log(epsilon) / numpy.log(base)))


def integrate_basic(func, ranges, epsilon=1e-8, limit=10):
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
    n = 0
    su = 0.0
    oldavg = numpy.nan
    b = naive_next_prime(s)
    Cs = faure_generating_matrices(s, get_n(epsilon, b), b)
    digs = digital_sequence(b, Cs)
    dx = numpy.prod(ranges[:, 1] - ranges[:, 0])
    nvar = 0
    for i, x in enumerate(digs):
        x_d = x * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
        result = func(x_d)

        su += result
        n += 1
        if abs(su / n - oldavg) * dx < epsilon:
            nvar += 1
            if nvar >= limit:
                return su / n * dx
        else:
            nvar = 0
        oldavg = su / n
    raise ValueError("Integration precision can't be met.")
