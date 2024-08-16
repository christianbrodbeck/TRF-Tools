# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from functools import cached_property, wraps

import sklearn.cluster

from eelbrain import NDVar

from ._interface import DimArg, NDVarInterface, Wrapper


class KMeans(Wrapper):
    """Attributes
    ----------
    cluster_centers : NDVar, (cluster, *dims)
        Cluster centers.
    """
    _skl_class = sklearn.cluster.KMeans

    @wraps(_skl_class.__init__)
    def __init__(self, *args, **kwargs):
        self.skl_instance = self._skl_class(*args, **kwargs)

    def fit(self, x: NDVar, samples: DimArg):
        self._ndvar = NDVarInterface(x, samples)
        data, _ = self._ndvar.consume()
        self.skl_instance.fit(data)

    def fit_predict(self, x: NDVar, samples: DimArg):
        "Predict the closest cluster each sample in x belongs to"
        self._ndvar = NDVarInterface(x, samples)
        data, sample_dims = self._ndvar.consume()
        labels = self.skl_instance.fit_predict(data)
        return self._ndvar.package_labels(labels, sample_dims)

    def predict(self, x: NDVar):
        "Predict the closest cluster each sample in x belongs to"
        data, sample_dims = self._ndvar.flatten(x)
        labels = self.skl_instance.predict(data)
        return self._ndvar.package_labels(labels, sample_dims)

    @cached_property
    def cluster_centers(self):
        # cluster_centers_ndarray of shape (n_clusters, n_features)
        return self._ndvar.package_components(self.skl_instance.cluster_centers_, 'cluster')
