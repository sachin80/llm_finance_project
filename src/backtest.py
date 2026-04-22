import backtrader as bt
import numpy as np
import pandas as pd

from .config import LOOKBACK_DAYS, INITIAL_CASH, COMMISSION
from .data_ingest import compute_returns
from .optimizer import optimize_portfolio
from .risk_model import ewma_covariance

# An equal-weight portfolio is included as a benchmark in 
# order to provide a simple signal-free baseline. 
def equal_weight_weights(tickers):
    """Simple equal-weight benchmark."""
    n = len(tickers)

    # Every ticker gets a w_i = 1/n. 
    return pd.Series(np.repeat(1.0 / n, n), index=tickers)


def build_weight_schedule(prices, alpha):
    """Build month-end optimized target weights for one alpha signal."""
    # Compute returns. 
    returns = compute_returns(prices)


    # Take tickers that are in the alpha and returns indexes. 
    tickers = [ticker for ticker in alpha.index if ticker in returns.columns]

    prices = prices[tickers].copy()
    returns = returns[tickers].dropna(how="all")
    alpha = alpha.loc[tickers]

    # Compute the month end dates then resample to get month end labels. 
    month_ends = prices.resample("ME").last().index

    # Rebalance monthly on only trading days. 
    rebalance_dates = prices.index.intersection(month_ends)

    # Initial portfolio used. 
    previous_weights = np.repeat(1.0 / len(tickers), len(tickers))

    rows = []

    for rebalance_date in rebalance_dates:

        # Use history up to the current rebalance date then keep
        # the keep the recent LOOBACK_DAYS observations
        history = returns.loc[:rebalance_date].tail(LOOKBACK_DAYS)

        # Skip due to covariance not fully updated. 
        if len(history) < 60:
            continue

        # Estimate covariance matrux using EWMA
        cov_matrix = ewma_covariance(history)

        # Solve the portfolio optimization problem. 
        weights = optimize_portfolio(
            alpha.values,
            cov_matrix.values,
            previous_weights=previous_weights,
        )

        # Store optimized weights as one row with rebalance_date label.
        rows.append(pd.Series(weights, index=tickers, name=rebalance_date))

        # These are used for the next rebalance. 
        previous_weights = weights

    return pd.DataFrame(rows)


def build_all_weight_schedules(prices, alpha_tfidf, alpha_embed):
    """Build target-weight schedules for the three strategies being compared."""

    # Use common tickers that appear in: TF-IDF alpha, embedding alpha, and price data.
    tickers = sorted(set(alpha_tfidf.index).intersection(alpha_embed.index).intersection(prices.columns))

    prices = prices[tickers].copy()
    alpha_tfidf = alpha_tfidf.loc[tickers]
    alpha_embed = alpha_embed.loc[tickers]

    # Build optimized schedule for each alpha method. 
    optimized_tfidf = build_weight_schedule(prices, alpha_tfidf)
    optimized_embed = build_weight_schedule(prices, alpha_embed)

    # Choose the rebalance calendar for the equal-weight benchmark. If the
    # embedding schedule exists prefer using thos rebalance dates. 
    if len(optimized_embed.index) > 0:
        rebalance_dates = optimized_embed.index
    else:
        rebalance_dates = optimized_tfidf.index

    eq_rows = []
    for rebalance_date in rebalance_dates:
        eq_rows.append(pd.Series(equal_weight_weights(tickers).values, index=tickers, name=rebalance_date))

    equal_weight = pd.DataFrame(eq_rows)

    # Return all strategies as a dict. 
    schedules = {
        "equal_weight": equal_weight,
        "optimized_tfidf": optimized_tfidf,
        "optimized_embedding": optimized_embed,
    }

    return schedules

# Use a simplified vectorized backtest instead of Backtrader. 
def run_simple_backtest(prices, target_weights, initial_cash=INITIAL_CASH):
    """Forward-fill monthly weights to daily frequency,
    then compute daily portfolio return as weighted sum of asset returns."""
    returns = compute_returns(prices)

    # Forward-fill monthly rebalance weights to every trading day
    daily_weights = (
        target_weights
        .reindex(returns.index, method="ffill")
        .dropna(how="all")
    )

    # Align dates
    returns, daily_weights = returns.align(daily_weights, join="inner", axis=0)

    # Daily portfolio return = sum of weight_i * return_i
    portfolio_returns = (returns * daily_weights).sum(axis=1)

    # Portfolio value over time
    portfolio_values = initial_cash * (1 + portfolio_returns).cumprod()

    return portfolio_returns, portfolio_values


def run_all_backtests(prices, schedules):
    all_returns = {}
    all_values = {}

    for name, schedule in schedules.items():
        if schedule.empty:
            continue
        returns, values = run_simple_backtest(prices[schedule.columns], schedule)
        all_returns[name] = returns
        all_values[name] = values

    returns_df = pd.DataFrame(all_returns).sort_index()
    values_df = pd.DataFrame(all_values).sort_index()

    return returns_df, values_df