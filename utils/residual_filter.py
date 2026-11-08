import numpy as np


def bi_aewma(residuals, rho):
    r = np.asarray(residuals, dtype=np.float64)
    n = r.size
    if n == 0:
        return r.copy()

    forward = np.empty(n, dtype=np.float64)
    backward = np.empty(n, dtype=np.float64)

    # Eq. (28): F_1 = R_1; F_t = [rho * F_(t-1) + (1-rho) * R_t] / (1-rho**t)
    forward[0] = r[0]
    for i in range(1, n):
        forward[i] = (rho * forward[i - 1] + (1.0 - rho) * r[i]) / (1.0 - rho ** (i + 1))

    # Eq. (29): S_L = R_L; S_t = [rho * S_(t+1) + (1-rho) * R_t] / (1-rho**(L-t+1))
    backward[-1] = r[-1]
    for i in range(n - 2, -1, -1):
        backward[i] = (rho * backward[i + 1] + (1.0 - rho) * r[i]) / (1.0 - rho ** (n - i))

    # Eq. (31): R_smooth,t = min(F_t, S_t)
    return np.minimum(forward, backward)
