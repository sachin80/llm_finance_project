import cvxpy as cp
import numpy as np

from .config import RISK_AVERSION, TURNOVER_PENALTY, MAX_WEIGHT

# Solve the convex portfolio problem used for both signal methods.
def optimize_portfolio(alpha, cov_matrix, previous_weights=None):
    """
    Inputs
    ------  
    alpha:  Cross-sectional alpha signal for the current rebalance date. 
            Larger alpha then the optimizer allocates more weight to that asset.
    
    cov_matrix: Estimate covariance matrix of assert returns. This is 
                the risk model. 

    previous_weights:   Portfolio weights held at the previous rebalance date.
                        Used to penalize large changes in holdings. 

    Returns
    -------
    Optimal portfolio weights. 

    
    This solves a long-only, fully invested Markowitz-like problem:
        
        maximize    alpha^T w - lambda * w^T Sigma w - gamma * ||w - w_prev||_1

        subject to  sum(w) = 1
                    w_i >= 0
                    w_i <= max_weight

    where 
    - alpha^T w rewards exposure to alpha
    - lambda * w^T Sigma w penalizes risk
    - ||w - w_prev||_1 penalizes turnover
    - w >= 0 long-only portfolio
    - w <= MAX_WEIGHT prevents concentration in one name
    """
    # Convert alpha to a 1D numpy array
    alpha = np.asarray(alpha, dtype=float).reshape(-1)

    # Convert covariance matrix to numpy array
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    # Number of assets in the universe for this rebalance step
    n = len(alpha)

    # If no previous portfolio, then start from equal weights. 
    if previous_weights is None:
        previous_weights = np.repeat(1.0 / n, n)

    # Make covariance matrix symmetric to reduce numerical issues.
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)

    # Optimization variale is portfolio weights, w_i. 
    w = cp.Variable(n)

    # Reward alpha exposure.
    expected_return = alpha @ w
    # Quadratic risk penalty depends on portfolio variance. 
    risk_penalty = cp.quad_form(w, cov_matrix)

    # Discourage large reallocations at rebalance. Regularizes solution. 
    turnover_penalty = cp.norm1(w - previous_weights)

    # Objective: Maximize signal exposure minus risk minus turnover penalty. 
    objective = cp.Maximize(
        expected_return
        - RISK_AVERSION * risk_penalty
        - TURNOVER_PENALTY * turnover_penalty
    )

    # Constraints: 
    # 1. Fully-invested
    # 2. Long-only
    # 3. Max position cap. 
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= MAX_WEIGHT,
    ]

    # Build the convex optimization problem.
    problem = cp.Problem(objective, constraints)

    # Solve it. 
    problem.solve(solver="CLARABEL")

    # Raise error if solver fails. 
    if w.value is None:
        raise RuntimeError(f"Optimization failed with status: {problem.status}")
    
    # Return weights as a flattened numpy array. 
    return np.asarray(w.value).ravel()
