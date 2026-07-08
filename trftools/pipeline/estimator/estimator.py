from typing import Any


class Estimator:
    """Base for TRF estimator configs."""

    def parameters_for_partial(self) -> dict[str, Any]:
        """Return kwargs for partial(fitter, ..., **kwargs)."""
        raise NotImplementedError

    def normalize_trf_args(self, experiment, data, mask, state):
        """Normalize estimator-specific TRF args while preserving public API compatibility."""
        return data, mask, dict(state)
