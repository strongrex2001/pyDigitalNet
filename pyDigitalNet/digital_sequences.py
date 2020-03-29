# -*- coding: utf-8 -*-
"""
Digital Sequences
-----------------
| Created on Sun Mar 29 13:56:22 2020
@author: eliphat

| This module provides generation of digital sequences from defition.

References
----------
| [1]
  Goda, Takashi & Suzuki, Kosuke. (2019).
  Recent advances in higher order quasi-Monte Carlo methods.

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


def digital_sequence(b, Cs):
    """
    Yields a digital sequence generated from matrices Cs.

    Parameters
    ----------
    b: int
      Base, or radix of generated digital sequence.
    Cs: numpy.ndarray of shape (s, N, N)
      Generating Matrices for the sequence.

    Returns
    -------
    S: iter(numpy.ndarray of shape (s))
      An iterator through the digital sequence.

    Remarks
    -------
    | For more details about Digital Sequences and paper references,
      see the documentation of this module.
    """
    N = Cs.shape[1]
    Cs = numpy.transpose(Cs, (0, 2, 1))
    weights = numpy.zeros((N,))
    for i in range(N):
        weights[i] = b ** (-i - 1)
    for h in range(1, b ** N):
        # B-adic Representation
        # TODO: Use Matmul and Modular Inversion for better performance
        eta = numpy.zeros((N,), dtype=Cs.dtype)
        for i in range(N):
            eta[i] = h % b
            h //= b
        # Matrix Transform
        xi = numpy.matmul(eta, Cs) % b
        # Invert Representation
        yield numpy.dot(xi, weights)
