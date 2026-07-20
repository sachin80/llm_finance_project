"""
Custom signal diagnostics for the LLM finance project.

This module is the baseline signal-diagnostics layer.

It does not replace the existing project modules.

Existing project modules keep their jobs:

    data_ingest.py
        loads the universe
        downloads Yahoo prices
        computes ordinary daily returns

    alpha_model.py
        winsorizes raw thematic scores
        z-scores thematic scores
        builds normalized alpha

    risk_model.py
        estimates covariance matrices

    optimizer.py
        turns alpha and risk into portfolio weights

    backtest.py
        evaluates realized portfolio returns

This module adds the missing signal-evaluation layer:

    forward returns
    monthly IC / rank IC
    quantile returns
    top-minus-bottom spread
    signal turnover
    sector exposure
    IC summary statistics

The comments are intentionally very detailed because this code is for a
research-paper project and should be easy to audit.
"""

# Import numpy for numerical operations such as NaN handling and square roots.
import numpy as np

# Import pandas for tabular data, dates, alignment, correlations, and grouping.
import pandas as pd


# -----------------------------------------------------------------------------
# Project integration helpers
# -----------------------------------------------------------------------------


def normalize_ticker_index(index):
    """Normalize ticker labels to uppercase strings."""

    # Convert the incoming labels into a pandas Index so we get vectorized string methods.
    tickers = pd.Index(index)

    # Convert labels to strings, strip extra spaces, and uppercase all tickers.
    tickers = tickers.astype(str).str.strip().str.upper()

    # Return the cleaned ticker index.
    return tickers


def build_static_alpha_from_scores(
    df,
    score_col,
    ticker_col="Ticker",
    alpha_builder=None,
):
    """Build a static alpha Series from one raw score column.

    This helper connects the diagnostics layer to alpha_model.py.

    The existing project already has alpha_model.build_alpha(...), so this
    function uses that existing alpha builder instead of reimplementing
    winsorization and z-scoring here.
    """

    # Check that the ticker column exists before trying to use it.
    if ticker_col not in df.columns:
        # Raise a clear error if the expected ticker column is missing.
        raise ValueError(f"Missing ticker column: {ticker_col}")

    # Check that the raw score column exists before trying to use it.
    if score_col not in df.columns:
        # Raise a clear error if the expected score column is missing.
        raise ValueError(f"Missing score column: {score_col}")

    # If no alpha builder is passed, use the existing project function.
    if alpha_builder is None:
        # Import lazily so tests can pass a fake builder without importing the full project.
        from .alpha_model import build_alpha as alpha_builder

    # Keep only the ticker and raw score columns.
    scores = df[[ticker_col, score_col]].copy()

    # Normalize ticker strings before using them as the Series index.
    scores[ticker_col] = scores[ticker_col].astype(str).str.strip().str.upper()

    # Drop duplicate tickers so each ticker has one raw score.
    scores = scores.drop_duplicates(subset=[ticker_col])

    # Convert the raw score column into a numeric Series indexed by ticker.
    score_series = scores.set_index(ticker_col)[score_col]

    # Convert bad score values to NaN.
    score_series = pd.to_numeric(score_series, errors="coerce")

    # Drop tickers that do not have usable scores.
    score_series = score_series.dropna()

    # Use the existing project alpha builder to winsorize and z-score the raw scores.
    alpha = alpha_builder(score_series)

    # Normalize the alpha index to uppercase tickers.
    alpha.index = normalize_ticker_index(alpha.index)

    # Give the Series a stable name for debugging and CSV output.
    alpha.name = "alpha"

    # Return the static alpha Series.
    return alpha


def load_prices_for_alpha(
    alpha,
    start_date=None,
    end_date=None,
    download_func=None,
):
    """Load prices for the tickers in a static alpha Series.

    This helper connects the diagnostics layer to data_ingest.py.

    The existing project already has data_ingest.download_prices(...), so this
    function calls that instead of duplicating yfinance logic here.
    """

    # Normalize alpha tickers and convert them to a list for the downloader.
    tickers = normalize_ticker_index(alpha.index).tolist()

    # If dates are not passed, use the existing project config values.
    if start_date is None or end_date is None:
        # Import lazily so tests can pass explicit dates without importing project config.
        from .config import START_DATE, END_DATE

        # Use project START_DATE if no start date was passed.
        if start_date is None:
            start_date = START_DATE

        # Use project END_DATE if no end date was passed.
        if end_date is None:
            end_date = END_DATE

    # If no downloader is passed, use the existing project Yahoo price downloader.
    if download_func is None:
        # Import lazily so tests can pass a fake downloader.
        from .data_ingest import download_prices as download_func

    # Call the existing project downloader.
    prices = download_func(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    # Clean the downloaded prices into the standard matrix format.
    prices = clean_prices(prices)

    # Return the cleaned adjusted-close price matrix.
    return prices


def align_static_alpha_and_prices(alpha, prices):
    """Keep only tickers that appear in both alpha and prices."""

    # Check that static alpha is a pandas Series.
    if not isinstance(alpha, pd.Series):
        # Raise a clear error if the wrong object type is passed.
        raise TypeError("alpha must be a pandas Series")

    # Clean the prices before aligning tickers.
    prices = clean_prices(prices)

    # Copy alpha so the caller's object is not modified.
    alpha = alpha.copy()

    # Normalize alpha tickers.
    alpha.index = normalize_ticker_index(alpha.index)

    # Convert alpha values to floats.
    alpha = alpha.astype(float)

    # Find tickers that are available in both alpha and price columns.
    common_tickers = sorted(set(alpha.index).intersection(set(prices.columns)))

    # Require at least one overlapping ticker.
    if len(common_tickers) == 0:
        # Raise a clear error if alignment would produce empty data.
        raise ValueError("No common tickers between alpha and prices")

    # Keep only common tickers in alpha.
    alpha = alpha.loc[common_tickers]

    # Keep only common tickers in prices and preserve the same ticker order.
    prices = prices.loc[:, common_tickers]

    # Return aligned alpha and prices.
    return alpha, prices


def prepare_static_signal_inputs(
    df,
    score_col,
    ticker_col="Ticker",
    prices=None,
    start_date=None,
    end_date=None,
    alpha_builder=None,
    download_func=None,
):
    """Prepare alpha and prices for static signal diagnostics.

    This is the convenience bridge from the existing project pipeline to the
    diagnostics functions.

    It builds alpha from a score column, loads or cleans prices, and aligns
    tickers.
    """

    # Build normalized alpha from the requested score column.
    alpha = build_static_alpha_from_scores(
        df=df,
        score_col=score_col,
        ticker_col=ticker_col,
        alpha_builder=alpha_builder,
    )

    # If prices are not already provided, load them using the project downloader.
    if prices is None:
        # Load adjusted-close prices for the alpha tickers.
        prices = load_prices_for_alpha(
            alpha=alpha,
            start_date=start_date,
            end_date=end_date,
            download_func=download_func,
        )

    # If prices are provided, just clean them.
    else:
        # Clean user-provided prices into the standard format.
        prices = clean_prices(prices)

    # Align alpha and prices to the same tickers.
    alpha, prices = align_static_alpha_and_prices(alpha, prices)

    # Return the analysis-ready objects.
    return alpha, prices


def build_sector_map_from_universe(
    universe_df,
    ticker_col="Ticker",
    sector_col=None,
):
    """Build ticker-to-sector map from the S&P 500 universe DataFrame."""

    # Check that the ticker column exists.
    if ticker_col not in universe_df.columns:
        # Raise a clear error if ticker labels are missing.
        raise ValueError(f"Missing ticker column: {ticker_col}")

    # If sector column is not specified, search common sector column names.
    if sector_col is None:
        # List common sector column names from S&P/GICS datasets.
        possible_sector_cols = [
            "Sector",
            "GICS Sector",
            "GICS_Sector",
            "sector",
            "gics_sector",
        ]

        # Try each possible sector column.
        for col in possible_sector_cols:
            # Use the first matching column.
            if col in universe_df.columns:
                # Store the detected sector column.
                sector_col = col

                # Stop searching after finding a match.
                break

    # Check that a usable sector column was found.
    if sector_col is None or sector_col not in universe_df.columns:
        # Raise a clear error if sector information is unavailable.
        raise ValueError("Could not find sector column. Pass sector_col explicitly.")

    # Keep only ticker and sector columns.
    sector_map = universe_df[[ticker_col, sector_col]].copy()

    # Normalize ticker labels.
    sector_map[ticker_col] = sector_map[ticker_col].astype(str).str.strip().str.upper()

    # Clean sector names.
    sector_map[sector_col] = sector_map[sector_col].astype(str).str.strip()

    # Drop duplicate tickers.
    sector_map = sector_map.drop_duplicates(subset=[ticker_col])

    # Convert to a Series indexed by ticker, and select `sector_col` as Series. 
    sector_map = sector_map.set_index(ticker_col)[sector_col]

    # Return ticker-to-sector Series.
    return sector_map


# -----------------------------------------------------------------------------
# General diagnostic helpers
# -----------------------------------------------------------------------------


def clean_prices(prices):
    """Clean a wide adjusted-close price matrix."""

    # Check that prices are stored as a DataFrame.
    if not isinstance(prices, pd.DataFrame):
        # Raise a clear error for incorrect input type.
        raise TypeError("prices must be a pandas DataFrame")

    # Check that the DataFrame has at least some data.
    if prices.empty:
        # Raise a clear error for empty input.
        raise ValueError("prices is empty")

    # Copy prices so the caller's object is not modified.
    prices = prices.copy()

    # Convert the index to pandas datetime values.
    prices.index = pd.to_datetime(prices.index)

    # Sort dates from oldest to newest.
    prices = prices.sort_index()

    # Normalize price column names as tickers.
    prices.columns = normalize_ticker_index(prices.columns)

    # Convert price values to numeric values.
    prices = prices.apply(pd.to_numeric, errors="coerce")

    # Drop tickers whose price column is entirely missing.
    prices = prices.dropna(axis=1, how="all")

    # Check that at least one usable price column remains.
    if prices.empty:
        # Raise a clear error if cleaning removed everything.
        raise ValueError("prices has no usable columns after cleaning")

    # Return the cleaned price matrix.
    return prices


def get_month_end_trading_dates(index):
    """Return the actual last available trading date in each calendar month."""

    # Convert the input index into a sorted DateTimeIndex.
    dates = pd.DatetimeIndex(index).sort_values()

    # If there are no dates, return an empty DateTimeIndex.
    if len(dates) == 0:
        # Return empty output for empty input.
        return pd.DatetimeIndex([])

    # Group actual trading dates by calendar month and keep the last trading date.
    month_end_dates = (
        pd.Series(dates, index=dates)
        .groupby(dates.to_period("M"))
        .max()
    )

    # Convert the result back into a DateTimeIndex.
    month_end_dates = pd.DatetimeIndex(month_end_dates.values)

    # Return actual month-end trading dates.
    return month_end_dates


def get_alpha_for_date(alpha, date):
    """Return alpha values available at one evaluation date."""

    # Handle the static Yahoo embedding case.
    if isinstance(alpha, pd.Series):
        # Copy alpha so the caller's object is not modified.
        alpha_today = alpha.copy()

        # Normalize ticker labels.
        alpha_today.index = normalize_ticker_index(alpha_today.index)

        # Convert values to floats and return.
        return alpha_today.astype(float)

    # Handle the future time-varying SEC filing case.
    if isinstance(alpha, pd.DataFrame):
        # Copy alpha history so the caller's object is not modified.
        alpha = alpha.copy()

        # Convert signal dates to pandas datetime values.
        alpha.index = pd.to_datetime(alpha.index)

        # Sort signal dates from oldest to newest.
        alpha = alpha.sort_index()

        # Normalize ticker columns.
        alpha.columns = normalize_ticker_index(alpha.columns)

        # Keep only signal rows known on or before the evaluation date.
        available = alpha.loc[:date]

        # Return an empty Series if no signal was known yet.
        if available.empty:
            # Return empty float Series for no available signal.
            return pd.Series(dtype=float)

        # Use the latest available signal row.
        alpha_today = available.iloc[-1].copy()

        # Normalize ticker labels on the selected row.
        alpha_today.index = normalize_ticker_index(alpha_today.index)

        # Convert values to floats and return.
        return alpha_today.astype(float)

    # Raise a clear error for unsupported alpha type.
    raise TypeError("alpha must be a pandas Series or DataFrame")


def align_alpha_and_returns(alpha_values, return_values):
    """Align alpha values and forward returns by ticker."""

    # Copy alpha values so the caller's object is not modified.
    alpha_values = alpha_values.copy()

    # Copy return values so the caller's object is not modified.
    return_values = return_values.copy()

    # Normalize alpha ticker labels.
    alpha_values.index = normalize_ticker_index(alpha_values.index)

    # Normalize return ticker labels.
    return_values.index = normalize_ticker_index(return_values.index)

    # Combine alpha and future return into one DataFrame aligned by ticker.
    data = pd.concat(
        [
            alpha_values.rename("alpha"),
            return_values.rename("forward_return"),
        ],
        axis=1,
        join="inner",
    )

    # Drop tickers where either alpha or future return is missing.
    data = data.dropna()

    # Return the aligned cross-section.
    return data


def assign_alpha_quantiles(data, n_quantiles):
    """Assign alpha quantiles to one cross-section."""

    # If there are fewer names than quantiles, quantile assignment is not useful.
    if len(data) < n_quantiles:
        # Return None to signal that assignment failed.
        return None

    # Copy data so the caller's object is not modified.
    data = data.copy()

    # Rank alpha values before qcut so ties do not break quantile assignment.
    alpha_rank = data["alpha"].rank(method="first")

    # Assign integer quantile labels from 1 through n_quantiles.
    data["quantile"] = pd.qcut(
        alpha_rank,
        q=n_quantiles,
        labels=list(range(1, n_quantiles + 1)),
    )

    # Convert categorical quantile labels to integers.
    data["quantile"] = data["quantile"].astype(int)

    # Return data with quantile labels.
    return data


def clean_sector_map(sector_map):
    """Convert a sector map into a clean Series indexed by ticker."""

    # Convert dictionary input into a Series.
    if isinstance(sector_map, dict):
        # Dictionary keys are tickers and values are sectors.
        sector_map = pd.Series(sector_map)

    # Copy Series input directly.
    elif isinstance(sector_map, pd.Series):
        # Copy so the caller's object is not modified.
        sector_map = sector_map.copy()

    # Convert DataFrame input into a Series.
    elif isinstance(sector_map, pd.DataFrame):
        # Require standard column names for DataFrame input.
        if "ticker" not in sector_map.columns or "sector" not in sector_map.columns:
            # Raise a clear error for malformed sector DataFrame.
            raise ValueError("sector_map DataFrame must have columns: ticker, sector")

        # Convert DataFrame to Series indexed by ticker.
        sector_map = sector_map.set_index("ticker")["sector"]

    # Reject unsupported input types.
    else:
        # Raise a clear error for unsupported input type.
        raise TypeError("sector_map must be a dict, Series, or DataFrame")

    # Normalize ticker labels.
    sector_map.index = normalize_ticker_index(sector_map.index)

    # Convert sector labels to strings.
    sector_map = sector_map.astype(str)

    # Return cleaned sector map.
    return sector_map


# -----------------------------------------------------------------------------
# Public diagnostic function 1
# -----------------------------------------------------------------------------


def compute_forward_returns(prices, horizons=(21, 63)):
    """Compute forward returns from adjusted close prices."""

    # Clean the price matrix before computing returns.
    prices = clean_prices(prices)

    # Create an output dictionary keyed by horizon.
    forward_returns = {}

    # Loop over each requested holding horizon.
    for horizon in horizons:
        # Convert horizon to integer trading days.
        horizon = int(horizon)

        # Require positive horizons.
        if horizon <= 0:
            # Raise a clear error for invalid horizon.
            raise ValueError("horizons must be positive integers")

        # Shift future prices backward so row t contains price at t+horizon.
        future_prices = prices.shift(-horizon)

        # Compute forward return from current price to future price.
        returns = future_prices / prices - 1.0

        # Store the forward-return matrix for this horizon.
        forward_returns[horizon] = returns

    # Return all forward-return matrices.
    return forward_returns


# -----------------------------------------------------------------------------
# Public diagnostic function 2
# -----------------------------------------------------------------------------


def compute_monthly_ic(alpha, forward_returns, method="spearman"):
    """Compute monthly cross-sectional information coefficient."""

    # Validate the requested correlation method.
    if method not in ["spearman", "pearson", "kendall"]:
        # Raise a clear error for unsupported method.
        raise ValueError("method must be 'spearman', 'pearson', or 'kendall'")

    # Require forward_returns to be a dictionary keyed by horizon.
    if not isinstance(forward_returns, dict):
        # Raise a clear error for incorrect input type.
        raise TypeError("forward_returns must be a dictionary")

    # Require at least one horizon.
    if len(forward_returns) == 0:
        # Raise a clear error for empty input.
        raise ValueError("forward_returns is empty")

    # Create dictionary to hold one IC Series per horizon.
    ic_by_horizon = {}

    # Loop over each horizon and forward-return matrix.
    for horizon, returns in forward_returns.items():
        # Check that the forward-return object is a DataFrame.
        if not isinstance(returns, pd.DataFrame):
            # Raise a clear error for malformed forward_returns.
            raise TypeError("each forward_returns value must be a DataFrame")

        # Copy returns so the caller's object is not modified.
        returns = returns.copy()

        # Convert dates to pandas datetime values.
        returns.index = pd.to_datetime(returns.index)

        # Sort dates from oldest to newest.
        returns = returns.sort_index()

        # Normalize ticker columns.
        returns.columns = normalize_ticker_index(returns.columns)

        # Evaluate IC on the last trading date of each month.
        evaluation_dates = get_month_end_trading_dates(returns.index)

        # Store IC values for this horizon.
        ic_values = []

        # Loop over each monthly evaluation date.
        for evaluation_date in evaluation_dates:
            # Get forward returns starting at this evaluation date.
            returns_today = returns.loc[evaluation_date]

            # Get alpha values known at this evaluation date.
            alpha_today = get_alpha_for_date(alpha, evaluation_date)

            # Align alpha and forward returns by ticker.
            data = align_alpha_and_returns(alpha_today, returns_today)

            # Require at least three names for a meaningful correlation.
            if len(data) < 3:
                # Use NaN when the cross-section is too small.
                ic = np.nan

            # Otherwise compute the cross-sectional correlation.
            else:
                # Use the requested correlation method.
                ic = data["alpha"].corr(data["forward_return"], method=method)

            # Store this month's IC value.
            ic_values.append(ic)

        # Build the output column name, such as 21D.
        column_name = f"{int(horizon)}D"

        # Store the IC Series for this horizon.
        ic_by_horizon[column_name] = pd.Series(
            ic_values,
            index=evaluation_dates,
            name=column_name,
        )

    # Combine all horizon IC Series into one DataFrame.
    ic = pd.DataFrame(ic_by_horizon)

    # Sort output by date.
    ic = ic.sort_index()

    # Return monthly IC table.
    return ic


# -----------------------------------------------------------------------------
# Public diagnostic function 3
# -----------------------------------------------------------------------------


def compute_quantile_returns(alpha, forward_returns, n_quantiles=5):
    """Compute average forward return by alpha quantile."""

    # Require forward_returns to be a dictionary keyed by horizon.
    if not isinstance(forward_returns, dict):
        # Raise a clear error for incorrect input type.
        raise TypeError("forward_returns must be a dictionary")

    # Require at least two quantiles.
    if n_quantiles < 2:
        # Raise a clear error for invalid quantile count.
        raise ValueError("n_quantiles must be at least 2")

    # Create list to collect tidy output rows.
    rows = []

    # Loop over each horizon and forward-return matrix.
    for horizon, returns in forward_returns.items():
        # Check that forward returns are in DataFrame format.
        if not isinstance(returns, pd.DataFrame):
            # Raise a clear error for malformed input.
            raise TypeError("each forward_returns value must be a DataFrame")

        # Copy returns so the caller's object is not modified.
        returns = returns.copy()

        # Convert dates to pandas datetime values.
        returns.index = pd.to_datetime(returns.index)

        # Sort dates from oldest to newest.
        returns = returns.sort_index()

        # Normalize ticker columns.
        returns.columns = normalize_ticker_index(returns.columns)

        # Use the actual last trading date of each month.
        evaluation_dates = get_month_end_trading_dates(returns.index)

        # Loop over each monthly evaluation date.
        for evaluation_date in evaluation_dates:
            # Get future returns starting from this evaluation date.
            returns_today = returns.loc[evaluation_date]

            # Get alpha values known at this evaluation date.
            alpha_today = get_alpha_for_date(alpha, evaluation_date)

            # Align alpha and future returns by ticker.
            data = align_alpha_and_returns(alpha_today, returns_today)

            # If too few names exist, write NaNs for all quantiles.
            if len(data) < n_quantiles:
                # Loop over all quantile labels.
                for quantile in range(1, n_quantiles + 1):
                    # Add an empty-result row for this quantile.
                    rows.append({
                        "date": evaluation_date,
                        "horizon": int(horizon),
                        "quantile": quantile,
                        "mean_forward_return": np.nan,
                        "n_assets": 0,
                    })

                # Move to the next evaluation date.
                continue

            # Assign each ticker to an alpha quantile.
            data = assign_alpha_quantiles(data, n_quantiles)

            # Loop over each quantile bucket.
            for quantile in range(1, n_quantiles + 1):
                # Select the tickers in this quantile.
                # This gets overwritten per loop. 
                group = data.loc[data["quantile"] == quantile]

                # Add one tidy row for this date, horizon, and quantile.
                rows.append({
                    "date": evaluation_date,
                    "horizon": int(horizon),
                    "quantile": quantile,
                    "mean_forward_return": group["forward_return"].mean(),
                    "n_assets": len(group),
                })

    # Convert collected rows into a DataFrame.
    quantile_returns = pd.DataFrame(rows)

    # Sort non-empty output for readability.
    if not quantile_returns.empty:
        # Sort by date, horizon, and quantile.
        quantile_returns = quantile_returns.sort_values(
            ["date", "horizon", "quantile"]
        ).reset_index(drop=True)

    # Return tidy quantile-return table.
    return quantile_returns


# -----------------------------------------------------------------------------
# Public diagnostic function 4
# -----------------------------------------------------------------------------


def compute_top_bottom_spread(quantile_returns):
    """Compute top-minus-bottom quantile return spread."""

    # Define required input columns.
    required_columns = {
        "date",
        "horizon",
        "quantile",
        "mean_forward_return",
    }

    # Identify missing columns.
    missing = required_columns.difference(quantile_returns.columns)

    # Check that all required columns exist.
    if missing:
        # Raise a clear error for malformed input.
        raise ValueError(f"quantile_returns is missing columns: {missing}")

    # Return empty output for empty input.
    if quantile_returns.empty:
        # Return empty DataFrame.
        return pd.DataFrame()

    # Copy input so the caller's object is not modified.
    qret = quantile_returns.copy()

    # Convert date column to pandas datetime values.
    qret["date"] = pd.to_datetime(qret["date"])

    # Create dictionary to store one spread Series per horizon.
    spread_by_horizon = {}

    # Loop over horizons in sorted order.
    for horizon in sorted(qret["horizon"].dropna().unique()):
        # Keep rows for one horizon.
        one_horizon = qret.loc[qret["horizon"] == horizon]

        # Pivot quantile returns so columns are quantile numbers.
        pivot = one_horizon.pivot_table(
            index="date",
            columns="quantile",
            values="mean_forward_return",
            aggfunc="mean",
        )

        # Skip this horizon if pivot is empty.
        if pivot.empty:
            # Continue to next horizon.
            continue

        # Identify lowest alpha quantile.
        bottom_quantile = pivot.columns.min()

        # Identify highest alpha quantile.
        top_quantile = pivot.columns.max()

        # Compute top-minus-bottom spread.
        spread = pivot[top_quantile] - pivot[bottom_quantile]

        # Name the Series using horizon label.
        spread.name = f"{int(horizon)}D"

        # Store this horizon's spread Series.
        spread_by_horizon[f"{int(horizon)}D"] = spread

    # Combine all spread Series into one DataFrame.
    spread_df = pd.DataFrame(spread_by_horizon)

    # Sort output by date.
    spread_df = spread_df.sort_index()

    # Return spread table.
    return spread_df


# -----------------------------------------------------------------------------
# Public diagnostic function 5
# -----------------------------------------------------------------------------


def compute_signal_turnover(alpha_history):
    """Compute simple rank-based signal turnover through time."""

    # Static alpha has no time-series turnover.
    if isinstance(alpha_history, pd.Series):
        # Return empty Series for static signal.
        return pd.Series(dtype=float, name="signal_turnover")

    # Time-varying alpha must be a DataFrame.
    if not isinstance(alpha_history, pd.DataFrame):
        # Raise a clear error for unsupported input type.
        raise TypeError("alpha_history must be a pandas Series or DataFrame")

    # Empty alpha history has no turnover.
    if alpha_history.empty:
        # Return empty Series.
        return pd.Series(dtype=float, name="signal_turnover")

    # Copy input so the caller's object is not modified.
    alpha_history = alpha_history.copy()

    # Convert signal dates to pandas datetime values.
    alpha_history.index = pd.to_datetime(alpha_history.index)

    # Sort signal dates from oldest to newest.
    alpha_history = alpha_history.sort_index()

    # Normalize ticker columns.
    alpha_history.columns = normalize_ticker_index(alpha_history.columns)

    # Convert alpha values into within-date percentile ranks.
    ranks = alpha_history.rank(axis=1, pct=True)

    # Compute absolute rank changes from one signal date to the next.
    rank_changes = ranks.diff().abs()

    # Average rank changes across tickers for each date.
    turnover = rank_changes.mean(axis=1, skipna=True)

    # Name the output Series.
    turnover.name = "signal_turnover"

    # Drop the first date because it has no previous date.
    turnover = turnover.dropna()

    # Return turnover Series.
    return turnover


# -----------------------------------------------------------------------------
# Public diagnostic function 6
# -----------------------------------------------------------------------------


def compute_sector_exposure(alpha, sector_map):
    """Summarize alpha exposure by sector."""

    # Clean the sector mapping.
    sectors = clean_sector_map(sector_map)

    # Define helper for one cross-section.
    def one_date_sector_exposure(alpha_values, date=None):
        # Copy alpha values so the caller's object is not modified.
        alpha_values = alpha_values.copy()

        # Normalize ticker labels.
        alpha_values.index = normalize_ticker_index(alpha_values.index)

        # Convert alpha values to floats.
        alpha_values = alpha_values.astype(float)

        # Align alpha values and sectors by ticker.
        data = pd.concat(
            [
                alpha_values.rename("alpha"),
                sectors.rename("sector"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        # Return empty output if no tickers have sector data.
        if data.empty:
            # Return empty DataFrame.
            return pd.DataFrame()

        # Compute total absolute alpha for concentration shares.
        total_abs_alpha = data["alpha"].abs().sum()

        # Create list to collect sector rows.
        rows = []

        # Loop over sectors.
        for sector, group in data.groupby("sector"):
            # Avoid division by zero if all alpha values are zero.
            if total_abs_alpha == 0:
                # Use NaN when concentration share is undefined.
                abs_alpha_share = np.nan

            # Otherwise compute the sector share of total absolute alpha.
            else:
                # Sum absolute alpha in this sector and divide by total.
                abs_alpha_share = group["alpha"].abs().sum() / total_abs_alpha

            # Build one sector summary row.
            row = {
                "sector": sector,
                "n_assets": len(group),
                "mean_alpha": group["alpha"].mean(),
                "median_alpha": group["alpha"].median(),
                "std_alpha": group["alpha"].std(ddof=0),
                "min_alpha": group["alpha"].min(),
                "max_alpha": group["alpha"].max(),
                "positive_alpha_share": (group["alpha"] > 0).mean(),
                "abs_alpha_share": abs_alpha_share,
            }

            # Include date if this is a time-varying alpha input.
            if date is not None:
                # Add date to the row.
                row["date"] = date

            # Store the sector row.
            rows.append(row)

        # Convert sector rows into a DataFrame.
        result = pd.DataFrame(rows)

        # Return empty output if no rows were built.
        if result.empty:
            # Return empty DataFrame.
            return result

        # Put columns in a stable order for time-varying output.
        if date is not None:
            # Reorder columns with date first.
            result = result[
                [
                    "date",
                    "sector",
                    "n_assets",
                    "mean_alpha",
                    "median_alpha",
                    "std_alpha",
                    "min_alpha",
                    "max_alpha",
                    "positive_alpha_share",
                    "abs_alpha_share",
                ]
            ]

        # Sort sectors by largest alpha concentration.
        result = result.sort_values("abs_alpha_share", ascending=False)

        # Return sector summary.
        return result

    # Handle static alpha.
    if isinstance(alpha, pd.Series):
        # Compute sector exposure for the static cross-section.
        result = one_date_sector_exposure(alpha)

        # Use sector as the index for static output.
        if not result.empty:
            # Set sector index.
            result = result.set_index("sector")

        # Return static sector exposure table.
        return result

    # Handle time-varying alpha.
    if isinstance(alpha, pd.DataFrame):
        # Copy alpha history so the caller's object is not modified.
        alpha = alpha.copy()

        # Convert signal dates to pandas datetime values.
        alpha.index = pd.to_datetime(alpha.index)

        # Sort signal dates from oldest to newest.
        alpha = alpha.sort_index()

        # Normalize ticker columns.
        alpha.columns = normalize_ticker_index(alpha.columns)

        # Create list to collect date-sector tables.
        all_rows = []

        # Loop over each signal date.
        for date, row in alpha.iterrows():
            # Compute sector exposure for this date.
            exposure = one_date_sector_exposure(row, date=date)

            # Keep non-empty date-sector output.
            if not exposure.empty:
                # Store this date's sector exposure table.
                all_rows.append(exposure)

        # Return empty output if no date-sector tables were built.
        if not all_rows:
            # Return empty DataFrame.
            return pd.DataFrame()

        # Combine all date-sector tables.
        result = pd.concat(all_rows, ignore_index=True)

        # Sort by date and alpha concentration.
        result = result.sort_values(["date", "abs_alpha_share"], ascending=[True, False])

        # Return time-varying sector exposure table.
        return result

    # Raise a clear error for unsupported alpha type.
    raise TypeError("alpha must be a pandas Series or DataFrame")


# -----------------------------------------------------------------------------
# Optional statistical summary
# -----------------------------------------------------------------------------


def summarize_ic(monthly_ic, nw_lags=3):
    """Summarize IC series and optionally use statsmodels HAC t-stats."""

    # Check that monthly IC is a DataFrame.
    if not isinstance(monthly_ic, pd.DataFrame):
        # Raise a clear error for incorrect input type.
        raise TypeError("monthly_ic must be a pandas DataFrame")

    # Create list to collect summary rows.
    rows = []

    # Loop over each horizon column.
    for column in monthly_ic.columns:
        # Drop missing IC values for this horizon.
        series = monthly_ic[column].dropna().astype(float)

        # Handle empty IC series.
        if len(series) == 0:
            # Add empty summary row.
            rows.append({
                "horizon": column,
                "n_months": 0,
                "mean_ic": np.nan,
                "std_ic": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "method": "empty",
            })

            # Move to the next horizon.
            continue

        # Compute mean IC.
        mean_ic = series.mean()

        # Compute sample standard deviation of IC.
        std_ic = series.std(ddof=1)

        # Try statsmodels for Newey-West/HAC inference.
        try:
            # Import statsmodels only when this optional function is called.
            import statsmodels.api as sm

            # Regress IC on a constant to estimate mean IC.
            x = np.ones(len(series))

            # Create ordinary least squares model.
            model = sm.OLS(series.values, x)

            # Fit model with HAC covariance to handle autocorrelation.
            fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

            # Extract t-statistic for the constant mean.
            t_stat = float(fit.tvalues[0])

            # Extract p-value for the constant mean.
            p_value = float(fit.pvalues[0])

            # Record inference method.
            method = "statsmodels_hac"

        # Fall back if statsmodels is not installed or fails.
        except Exception:
            # Check that iid t-stat is defined.
            if len(series) > 1 and std_ic > 0:
                # Compute simple iid t-stat for mean IC.
                t_stat = mean_ic / (std_ic / np.sqrt(len(series)))

            # Otherwise t-stat cannot be computed.
            else:
                # Use NaN for undefined t-stat.
                t_stat = np.nan

            # P-value is not computed in fallback mode.
            p_value = np.nan

            # Record fallback method.
            method = "simple_iid"

        # Add summary row for this horizon.
        rows.append({
            "horizon": column,
            "n_months": len(series),
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "t_stat": t_stat,
            "p_value": p_value,
            "method": method,
        })

    # Convert summary rows into a DataFrame.
    summary = pd.DataFrame(rows)

    # Return IC summary table.
    return summary


# List public functions for explicit imports.
__all__ = [
    "build_static_alpha_from_scores",
    "load_prices_for_alpha",
    "align_static_alpha_and_prices",
    "prepare_static_signal_inputs",
    "build_sector_map_from_universe",
    "compute_forward_returns",
    "compute_monthly_ic",
    "compute_quantile_returns",
    "compute_top_bottom_spread",
    "compute_signal_turnover",
    "compute_sector_exposure",
    "summarize_ic",
]
