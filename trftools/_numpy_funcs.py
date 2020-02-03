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


arctanh = element_wise(math.atanh, numpy.arctanh)
