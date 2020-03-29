# -*- coding: utf-8 -*-
"""
Faure Sequences Generator
-------------------------
| Created on Sun Mar 29 10:24:45 2020
@author: eliphat

| "A Faure sequence is a digital (0, s)-sequence",
  according to Dick, J., Kuo, F. and Sloan, I. (2013).
| This module provides computation of generator matrices for
  the Faure sequence.

References
----------
| [1]
  Henri Faure.
  Discrépance de suites associées à un système de numération (en dimension s).
  Acta Arithmetica, 41:337–351, 1982.
| [2]
  Dick, J., Kuo, F. and Sloan, I. (2013).
  High-dimensional integration: The quasi-Monte Carlo way.
  In Acta Numerica, Vol. 22, Cambridge University Press, pp. 133–288.

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


def pascal_matrix(b, N, dtype=numpy.int32):
    """
    Returns a N-by-N Pascal Matrix under modular b.

    Parameters
    ----------
    b: int
      Modular used. Should be at least 2.
    N: int
      Number of rows and columns of the output.
    dtype: data-type, optional
      The desired data-type for the array.
      Recommended to be a int type. Default is numpy.int32.

    Returns
    -------
    P: numpy.ndarray of shape (N, N)
      An array containing the requested Pascal Matrix under modular.

    Remarks
    -------
    | A Pascal Matrix should be prepared to calculate the Faure sequences.
      It is by definition a lower-triangular matrix.
    """
    P = numpy.zeros((N, N), dtype=dtype)
    for n in range(N):
        # C(n, m) = C(n - 1, m) + C(n - 1, m - 1)
        P[n, 0] = 1
        P[n, 1: n + 1] = (P[n - 1, 1: n + 1] + P[n - 1, 0: n]) % b
    return P


def naive_next_prime(n):
    """
    Returns the smallest prime that is greater than n.

    Parameters
    ----------
    n: int
      The number starting from which to find the next prime.

    Returns
    -------
    p: int
       The smallest prime that is greater than n.

    Remarks
    -------
    | It is a naive algorithm that iterates a naive prime check,
      which is slow when n is very large (>> 2e9).
    """
    if n < 2:
        return 2
    p = n
    while True:
        p += 1
        for i in range(2, int(p ** 0.5) + 1):
            if p % i == 0:
                break
        else:
            return p


def faure_generating_matrices(s, N=32, b=None, dtype=numpy.int32):
    """
    Returns the generating matrices for a Faure sequence.

    Parameters
    ----------
    s: int
      The s value of the target sequence.
      In numerical integration, it is the same as
      the number of integrated variables.
    N: int
      Dimension of generating matrices.
      It is related with accuracy of this step with roughly
      error = s ** (-N). Default to 32.
    b: int, optional
      Modular (or base, radix) of generating matrices.
      If b is `None`, the next prime from s is used.
    dtype: data-type, optional
      The desired data-type for the array.
      Recommended to be a int type. Default is numpy.int32.

    Returns
    -------
    P: numpy.ndarray of shape (s, N, N), dtype numpy.int32
      An array containing the requested s generating matrices.

    Remarks
    -------
    | For more details about Faure sequences and paper references,
      see the documentation of this module.
    """
    b = b or naive_next_prime(s)
    P = numpy.transpose(pascal_matrix(b, N, dtype=dtype))
    Cs = numpy.zeros((s, N, N), dtype=dtype)
    A = numpy.eye(N, dtype=dtype)  # P ** 0
    for i in range(s):
        Cs[i, :, :] = A
        A = numpy.matmul(A, P) % b
    return Cs
