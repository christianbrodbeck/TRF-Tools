# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from functools import cached_property, wraps

import numpy
import sklearn.decomposition

from eelbrain import NDVar

from ._interface import DimArg, NDVarInterface, Wrapper


# FIXME: adds docs to base class??
def wrap_init(original_class):
    "Decorator for inheriting documentation"
    # print(f"{original_class} wraps {original_class._skl_class}")
    original_class.__init__ = wraps(original_class._skl_class.__init__)(original_class.__init__)
    return original_class


class PCAWrapper(Wrapper):
    """Attributes
    ----------
    components : NDVar, (component, *dims)
        PCA components.
    """
    _skl_class = None

    def __init__(self, *args, **kwargs):
        self.skl_instance = self._skl_class(*args, **kwargs)

    def fit(self, x: NDVar, samples: DimArg):
        self._ndvar = NDVarInterface(x, samples)
        data, _ = self._ndvar.consume()
        self.skl_instance.fit(data)

    def fit_transform(self, x: NDVar, samples: DimArg):
        self._ndvar = NDVarInterface(x, samples)
        data, sample_dims = self._ndvar.consume()
        xt = self.skl_instance.fit_transform(data)
        return self._ndvar.package_sources(xt, sample_dims)

    def transform(self, x: NDVar):
        data, sample_dims = self._ndvar.flatten(x)
        xt = self.skl_instance.transform(data)
        return self._ndvar.package_sources(xt, sample_dims)

    def inverse_transform(self, x: NDVar, add_mean: bool = False):
        data, sample_dims = self._ndvar.flatten_sources(x)
        # pad missing components
        if data.shape[1] < self.skl_instance.n_components_:
            data_ = numpy.zeros((data.shape[0], self.skl_instance.n_components_))
            data_[:, :data.shape[1]] = data
            data = data_
        elif data.shape[1] > self.skl_instance.n_components_:
            raise RuntimeError(f"Data with {data.shape[1]} components for PCA with {self.skl_instance.n_components_} components")
        # inverse transform
        if add_mean:
            data_orig = self.skl_instance.inverse_transform(data)
        elif self.skl_instance.whiten:
            raise NotImplementedError
        else:
            data_orig = numpy.dot(data, self.skl_instance.components_)
        # package output
        return self._ndvar.package_original(data_orig, sample_dims)

    @cached_property
    def components(self):
        return self._ndvar.package_components(self.skl_instance.components_)


@wrap_init
class FastICA(PCAWrapper):
    _skl_class = sklearn.decomposition.FastICA


@wrap_init
class PCA(PCAWrapper):
    _skl_class = sklearn.decomposition.PCA


@wrap_init
class SparsePCA(PCAWrapper):
    _skl_class = sklearn.decomposition.SparsePCA
