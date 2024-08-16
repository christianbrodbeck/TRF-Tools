# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from functools import cached_property, reduce
import operator
from typing import Tuple, Union

import numpy
import sklearn

from eelbrain import NDVar, Categorial
from eelbrain._data_obj import Dimension


DimArg = Union[str, tuple]


class NDVarInterface:

    def __init__(self, x: NDVar, samples: DimArg):
        if isinstance(samples, str):
            samples = (samples,)
        elif not isinstance(samples, tuple):
            raise TypeError(f"samples={samples!r}")
        self.x = x
        self.samples = samples
        dimnames = x.get_dimnames(first=samples)
        self.dims = x.get_dims(dimnames[len(samples):])

    @cached_property
    def flat_shape(self):
        return -1, reduce(operator.mul, self.shape)

    @cached_property
    def dimnames(self):
        return [dim.name for dim in self.dims]

    @cached_property
    def shape(self):
        return [len(dim) for dim in self.dims]

    @staticmethod
    def component_dim(n, name='component'):
        return Categorial(name, map(str, range(n)))

    def flatten(self, x: NDVar) -> (numpy.ndarray, Tuple[Dimension, ...]):
        "Flatten the input data"
        dimnames = x.get_dimnames(last=self.dimnames)
        sample_dims = x.get_dims(dimnames[:-len(self.dimnames)])
        # in case x dimensions are superset of PCA
        index = {}
        for dim in self.dims:
            x_dim = x.get_dim(dim.name)
            if x_dim != dim:
                index[dim.name] = dim
        if index:
            x = x.sub(**index)
        return x.get_data(dimnames).reshape(self.flat_shape), sample_dims

    @staticmethod
    def flatten_sources(x: NDVar) -> (numpy.ndarray, Tuple[Dimension, ...]):
        "Flatten an NDVar representing sources"
        dimnames = x.get_dimnames(last='component')
        sample_dims = x.get_dims(dimnames[:-1])
        data = x.get_data(dimnames)
        return data.reshape((-1, data.shape[-1])), sample_dims

    def consume(self):
        out = self.flatten(self.x)
        self.x = None
        return out

    @staticmethod
    def package_labels(x: numpy.ndarray, sample_dims: Tuple[Dimension, ...]):
        shape = [len(dim) for dim in sample_dims]
        return NDVar(x.reshape(shape), sample_dims, 'labels')

    def package_components(self, x: numpy.ndarray, name='component') -> NDVar:  # (n_components, n_features)
        x = x.reshape((x.shape[0], *self.shape))
        component = self.component_dim(x.shape[0], name)
        return NDVar(x, (component, *self.dims))

    def package_sources(self, x: numpy.ndarray, sample_dims: Tuple[Dimension, ...]):  # (n_samples, n_components)
        component = self.component_dim(x.shape[-1])
        dims = (*sample_dims, component)
        shape = [len(dim) for dim in dims]
        return NDVar(x.reshape(shape), dims)

    def package_original(self, x: numpy.ndarray, sample_dims: Tuple[Dimension, ...]):  # (n_samples, n_features)
        shape = [len(dim) for dim in sample_dims]
        return NDVar(x.reshape((*shape, *self.shape)), (*sample_dims, *self.dims))

    def __getstate__(self):
        return {'samples': self.samples, 'dims': self.dims}

    def __setstate__(self, state):
        self.x = None
        self.samples = state['samples']
        self.dims = state['dims']


class Wrapper:
    skl_version = sklearn.__version__
    _ndvar = None

    def __getstate__(self):
        return {
            'skl_version': self.skl_version,
            'skl_instance': self.skl_instance,
            'ndvar': self._ndvar,
        }

    def __setstate__(self, state):
        self.skl_version = state['skl_version']
        self.skl_instance = state['skl_instance']
        self._ndvar = state['ndvar']

    def __repr__(self):
        if self._ndvar is None:
            return f'<{self.__class__.__name__}>'
        dims = ', '.join([f'{len(dim)} {dim.name}' for dim in self._ndvar.dims])
        return f'<{self.__class__.__name__}: {dims}>'
