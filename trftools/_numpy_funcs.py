"""Add NDVar support to common numpy functions"""
import math
from numbers import Number
from typing import Union, Sequence

from eelbrain import NDVar
import numpy


MUV = Union[NDVar, numpy.array, Sequence, Number]


def element_wise(element_func, numpy_func):
    def func(x: MUV, name: str = None, info: dict = None):
        if isinstance(x, NDVar):
            return NDVar(numpy_func(x.x), x.dims, name, info)
        elif isinstance(x, numpy.ndarray):
            return numpy_func(x)
        elif isinstance(x, Sequence):
            return [element_func(xi) for xi in x]
        else:
            return element_func(x)
    return func


# Numpy functions
# https://numpy.org/doc/stable/reference/routines.math.html
# Trigonometric functions
sin = element_wise(math.sin, numpy.sin)
cos = element_wise(math.cos, numpy.cos)
tan = element_wise(math.tan, numpy.tan)
arcsin = element_wise(math.asin, numpy.arcsin)
arccos = element_wise(math.acos, numpy.arccos)
arctan = element_wise(math.atan, numpy.arctan)
# Hyperbolic functions
sinh = element_wise(math.sinh, numpy.sinh)
cosh = element_wise(math.cosh, numpy.cosh)
tanh = element_wise(math.tanh, numpy.tanh)
arcsinh = element_wise(math.asinh, numpy.arcsinh)
arccosh = element_wise(math.acosh, numpy.arccosh)
arctanh = element_wise(math.atanh, numpy.arctanh)
