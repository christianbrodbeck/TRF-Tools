from __future__ import annotations

"""
Boosting estimator for TRF pipeline.

Use in experiment definition:

    estimators = {
        'boosting': BoostingEstimator(basis=0.050),
        'boosting-l2': BoostingEstimator(error='l2', partitions=5),
        'boosting-backward': BoostingEstimator(backward=True),
    }
    # Then: e.load_trf('acoustic_envelop', 0, 0.5, estimator='boosting')

tstart & tstop are parameters of load_trf/load_trfs, not of the estimator.
"""

from .estimator import Estimator


class BoostingEstimator(Estimator):
    """
    Boosting estimator configuration.

    Parameters
    ----------
    delta
        Boosting step size.
    mindelta
        Minimum boosting step size.
    error
        Error function passed to boosting.
    basis
        Basis window width in seconds.
    partitions
        Number of partitions used for fitting.
    test
        Whether to use cross-validation during fitting.
    selective_stopping
        Stop boosting each predictor separately.
    partition_results
        Keep partition-level fit results.
    backward
        Fit a backward model.
    """
    def __init__(
        self,
        *,
        delta: float = 0.005,
        mindelta: float | None = None,
        error: str = "l1",
        basis: float = 0.050,
        partitions: int | None = None,
        test: bool = True,
        selective_stopping: int = 0,
        partition_results: bool = False,
        backward: bool = False,
    ):
        self.delta = delta
        self.mindelta = mindelta
        self.error = error
        self.basis = basis
        self.partitions = partitions
        self.test = test
        self.selective_stopping = selective_stopping
        self.partition_results = partition_results
        self.backward = backward

    def parameters_for_partial(self) -> dict[str, object]:
        return {
            "delta": self.delta,
            "mindelta": self.mindelta,
            "error": self.error,
            "basis": self.basis,
            "partitions": self.partitions,
            "test": self.test,
            "selective_stopping": self.selective_stopping,
            "partition_results": self.partition_results,
            "backward": self.backward,
        }
