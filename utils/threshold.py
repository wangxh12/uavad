import numpy as np


def calculate_threshold(
    val_scores,
    method='percentile',
    anomaly_ratio=1.0,
    sigma=3.0,
    mad_k=3.0,
    iqr_k=1.5,
):
    """Calculate anomaly threshold using validation scores only."""
    scores = np.asarray(val_scores, dtype=np.float64)

    if method == 'percentile':
        if not 0 < anomaly_ratio < 100:
            raise ValueError("anomaly_ratio must be in (0, 100)")
        threshold = np.percentile(scores, 100 - anomaly_ratio)

    elif method == '3sigma':
        threshold = scores.mean() + sigma * scores.std()

    elif method == 'mad':
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        threshold = median + mad_k * 1.4826 * mad

    elif method == 'iqr':
        q1, q3 = np.percentile(scores, [25, 75])
        threshold = q3 + iqr_k * (q3 - q1)

    else:
        raise ValueError(f"Unsupported threshold method: {method}")

    return float(threshold)