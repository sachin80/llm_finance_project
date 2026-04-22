import numpy as np
import pandas as pd

from .config import EWMA_HALFLIFE

# Use Exponential Weighted Moving Average (EWMA) estimator with a forgetting
# factor between (0, 1). Assume that the daily mean return is negligible and 
# so just use the second moment of the return. 
def ewma_covariance(returns, halflife=EWMA_HALFLIFE):
    """Estimate a covariance matrix using exponentially weighted returns 
    assumming on a daily basis the expected returns are small relative to
    volatility so we can just use the second moment with a forgetting factor.
    """
    # Convert half-life, H, into a forgetting forgetting_factor, where 
    # forgetting_factor^H = 1/2. 
    forgetting_factor = np.exp(np.log(0.5) / halflife)

    # Returns matrix where rows are time and columns assets
    x = returns.fillna(0.0).to_numpy()
    n_obs, _ = x.shape

    # Exponentially decaying weights
    weights = np.array([forgetting_factor ** (n_obs - 1 - i) for i in range(n_obs)])
    weights = weights / weights.sum()

    # Sigma_hat = Σ_t w_t * r_t r_t^T
    cov = (x * weights[:, None]).T @ x
    cov = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    return cov
