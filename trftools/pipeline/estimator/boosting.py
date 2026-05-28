from __future__ import annotations

"""
Boosting estimator for TRF pipeline.

Use in experiment definition:

    estimators = {
        'boosting': BoostingEstimator(basis=0.050),
        'boosting-l2': BoostingEstimator(error='l2', partitions=5),
    }
    # Then: e.load_trf('acoustic_envelop', 0, 0.5, estimator='boosting')

tstart & tstop are parameters of load_trf/load_trfs, not of the estimator.
"""

from .estimator import Estimator


class BoostingEstimator(Estimator):
    """
    Boosting estimator configuration.

    Boosting builds the TRF incrementally, unlike NCRF which solves a
    regularized inverse problem.
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
    ):
        self.delta = delta
        self.mindelta = mindelta
        self.error = error
        self.basis = basis
        self.partitions = partitions
        self.test = test
        self.selective_stopping = selective_stopping
        self.partition_results = partition_results

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
        }
