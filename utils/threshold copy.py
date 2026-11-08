import numpy as np


_SOURCE_SPLITS = {
    "train": ("train",),
    "val": ("val",),
    "train_val": ("train", "val"),
    "train_test": ("train", "test"),  # Original TSLib threshold protocol.
}


def _as_scores(values, split):
    """Validate a one-dimensional, nonempty, finite score array."""
    if values is None:
        raise ValueError(f"{split}_scores is required by the selected source")
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError(f"{split}_scores must be a nonempty 1D array, got {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError(f"{split}_scores contains NaN or Inf")
    return scores


def calculate_threshold(
    train_scores=None,
    val_scores=None,
    test_scores=None,
    *,
    method="percentile",
    source="val",
    anomaly_ratio=1.0,
    sigma=3.0,
    mad_k=3.0,
    iqr_k=1.5,
):
    """Compute an upper-tail anomaly threshold from the selected score split(s).

    Parameters
    ----------
    method : {'percentile', '3sigma', 'mad', 'iqr'}
        Threshold estimator. Higher scores are assumed more anomalous.
    source : {'train', 'val', 'train_val', 'train_test'}
        Which score distribution(s) determine the threshold.
        'train_test' plus 'percentile' reproduces the TSLib protocol.
    anomaly_ratio : float
        Percentage, not a fraction: 1.0 means the highest 1% of scores.
    sigma, mad_k, iqr_k : float
        Positive multipliers for the corresponding estimators.

    Returns
    -------
    float
        Threshold; use `test_scores > threshold` for binary predictions.
    """
    if source not in _SOURCE_SPLITS:
        raise ValueError(f"Unknown source {source!r}; choose from {tuple(_SOURCE_SPLITS)}")

    arrays = {"train": train_scores, "val": val_scores, "test": test_scores}
    reference = np.concatenate(
        [_as_scores(arrays[name], name) for name in _SOURCE_SPLITS[source]]
    )

    if method == "percentile":
        if not np.isfinite(anomaly_ratio) or not 0 < anomaly_ratio < 100:
            raise ValueError("anomaly_ratio must be a percentage strictly between 0 and 100")
        threshold = np.percentile(reference, 100.0 - anomaly_ratio)
    elif method == "3sigma":
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive and finite")
        threshold = reference.mean() + sigma * reference.std(ddof=0)
    elif method == "mad":
        if not np.isfinite(mad_k) or mad_k <= 0:
            raise ValueError("mad_k must be positive and finite")
        median = np.median(reference)
        mad = np.median(np.abs(reference - median))
        threshold = median + mad_k * 1.4826 * mad
    elif method == "iqr":
        if not np.isfinite(iqr_k) or iqr_k <= 0:
            raise ValueError("iqr_k must be positive and finite")
        q1, q3 = np.percentile(reference, [25, 75])
        threshold = q3 + iqr_k * (q3 - q1)
    else:
        raise ValueError("Unknown method {!r}; choose from ('percentile', '3sigma', 'mad', 'iqr')".format(method))

    return float(threshold)
