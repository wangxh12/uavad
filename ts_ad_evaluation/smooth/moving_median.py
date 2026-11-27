import pandas as pd


def moving_median(residuals, args):
    """Apply a centered moving median with args as the window length.

    Boundary windows use the available samples to preserve residual alignment.
    """
    window = pd.Series(residuals).rolling(int(args), center=True, min_periods=1)
    return window.median().to_numpy()
