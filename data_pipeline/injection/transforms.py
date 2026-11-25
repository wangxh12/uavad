"""Elementary transformations for synthetic time-series anomalies."""

import numpy as np


def inject_bias(
    series: np.ndarray,
    start: int,
    end: int,
    magnitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Add a constant bias over [start, end) and return point-wise labels."""
    injected = series.copy()
    injected[start:end] += magnitude

    labels = np.zeros(len(series), dtype=np.int64)
    labels[start:end] = 1
    return injected, labels
