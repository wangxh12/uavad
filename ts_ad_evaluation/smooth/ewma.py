import pandas as pd


def ewma(residuals, args):
    """Apply causal EWMA with args as the current residual weight alpha.

    Initialize with the first residual, then use
        s[t] = alpha * r[t] + (1 - alpha) * s[t - 1].
    
    Smaller alpha values give stronger smoothing.
    """
    return pd.Series(residuals).ewm(alpha=args, adjust=False).mean().to_numpy()
