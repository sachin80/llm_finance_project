import numpy as np

# Compute a few standard metrics. 
def compute_perf_stats(returns):
    # If no returns, then return empty hashmap. 
    if returns.empty:
        return {}
    
    # Compute the annualized compounded return. 
    # Where (1 + returns).prod() is compounded return over sample
    # And (252 / len(returns)) annualizes. 
    annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1

    # Compute the annualized volatility using population std. 
    annual_vol = returns.std(ddof=0) * np.sqrt(252)

    # Compute a simplified Sharpe ratio:
    if annual_vol > 0:
        sharpe = annual_return / annual_vol
    else:
        sharpe = np.nan

    # Cumulative wealth curve is if you started with $1 and then 
    # the cumulative(t) is portfolio value after compounding all returns
    # up to time, t. 
    cumulative = (1 + returns).cumprod()

    # Measure how far below the running peak the current portfolio is. 
    drawdown = cumulative / cumulative.cummax() - 1
    
    # Worst drawdown. 
    max_drawdown = drawdown.min()

    return {
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }
