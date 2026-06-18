from __future__ import annotations

from .estimator import Estimator
from eelbrain._experiment.mne_experiment import TestDims


class NCRFEstimator(Estimator):
    """
    NCRF estimator configuration.

    Parameters
    ----------
    mu
        Regularization strength, or ``'auto'`` to determine it during fitting.
    n_iter
        Number of outer iterations.
    n_iterf
        Number of forward iterations.
    n_iterc
        Number of Champ-Lasso iterations.
    normalize
        Normalize predictors before fitting.
    in_place
        Allow in-place operations during fitting.
    """
    def __init__(self, *, mu: float | str = 'auto', n_iter: int = None, n_iterf: int = None, n_iterc: int = None, normalize: bool = True, in_place: bool = True):
        self.mu = mu
        self.n_iter = n_iter
        self.n_iterf = n_iterf
        self.n_iterc = n_iterc
        self.normalize = normalize
        self.in_place = in_place

    def parameters_for_partial(self) -> dict[str, object]:
        params = {
            "mu": self.mu,
            "n_iter": self.n_iter,
            "n_iterf": self.n_iterf,
            "n_iterc": self.n_iterc,
            "normalize": self.normalize,
            "in_place": self.in_place,
        }
        return {key: value for key, value in params.items() if value is not None}

    def normalize_trf_args(self, experiment, data, mask, state):
        state = dict(state)
        # NCRF always operates on sensor data and should not inherit
        # source-space configuration from the public TRF API.
        data = TestDims('sensor')
        mask = None
        return data, mask, state
