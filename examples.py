# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:50:36 2020

@author: eliphat
"""
import pyDigitalNet
import numpy
import scipy.integrate as iint

pi = numpy.pi


def tri(x):
    return x[0] ** 3


def sin(x):
    return numpy.sin(numpy.prod(x))


def qsin(*a):
    return numpy.sin(numpy.prod(a))


def qua(x):
    return x ** 4


# print(pyDigitalNet.integrate_basic(tri, [[0, 2]]))
print(pyDigitalNet.integrate_basic(sin, [[0, 2]] * 4, 1e-4))
print(iint.nquad(qsin, [(0, 2)] * 4,
                 opts={'epsabs': 1e-4, 'epsrel': 1e-4}))
