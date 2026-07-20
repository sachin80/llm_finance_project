"""
Alphalens-Reloaded signal diagnostics for the LLM finance project.

This module is the Alphalens comparison layer.

The custom `src/signal_diagnostics.py` module should remain the baseline
because it is transparent and easy to explain in a research paper.

This module connects the same project pipeline to Alphalens-Reloaded:

    data_ingest.py
        load_universe(...)
        download_prices(...)

    signal_methods.py
        add_tfidf_similarity(...)
        add_embeddings(...)
        add_embedding_similarity(...)

    alpha_model.py
        build_alpha(...)

    signal_diagnostics_alphalens.py
        build Alphalens factor_data
        run Alphalens-style IC and quantile-return diagnostics
        expose functions with the same public names as the custom baseline

Alphalens expects a factor Series indexed by:

    (date, asset) -> factor value

The current Yahoo embedding and TF-IDF signals are static:

    ticker -> alpha score

So this module repeats the static alpha across monthly evaluation dates.
Later, the SEC 10-K / 10-Q signal can be passed as a time-varying alpha
DataFrame indexed by signal date.

This file is intentionally heavily commented.
"""

# Import numpy for NaN values and simple numerical operations.
import numpy as np

# Import pandas for DataFrame manipulation, date handling, grouping, and indexing.
import pandas as pd


# -----------------------------------------------------------------------------
# Optional Alphalens import
# -----------------------------------------------------------------------------


try:
    # Import Alphalens utility functions such as get_clean_factor_and_forward_returns.
    from alphalens import utils as al_utils

    # Import Alphalens performance functions such as factor_information_coefficient.
    from alphalens import performance as al_perf

    # Record that Alphalens is available in the current environment.
    ALPHALENS_AVAILABLE = True

except Exception:
    # If Alphalens is not installed, keep the module importable.
    al_utils = None

    # Keep performance module placeholder as None.
    al_perf = None

    # Record that Alphalens is not available.
    ALPHALENS_AVAILABLE = False


def require_alphalens():
    """Raise a helpful error if alphalens-reloaded is not installed."""

    # Check whether the optional import succeeded.
    if not ALPHALENS_AVAILABLE:
        # Raise a clear installation error.
        raise ImportError(
            "alphalens-reloaded is not installed. "
            "Install it with: pip install alphalens-reloaded"
        )


# -----------------------------------------------------------------------------
# Project integration helpers
# -----------------------------------------------------------------------------


def normalize_ticker_index(index):
    """Normalize ticker labels to uppercase strings."""

    # Convert incoming labels to a pandas Index.
    tickers = pd.Index(index)

    # Convert labels to strings, strip spaces, and uppercase.
    tickers = tickers.astype(str).str.strip().str.upper()

    # Return cleaned tickers.
    return tickers


def build_static_alpha_from_scores(
    df,
    score_col,
    ticker_col="Ticker",
    alpha_builder=None,
):
    """Build a static alpha Series from a raw score column.

    This helper connects to alpha_model.py.

    It uses alpha_model.build_alpha(...) by default so this module does not
    reimplement winsorization and z-scoring.
    """

    # Check that the ticker column exists.
    if ticker_col not in df.columns:
        # Raise a clear error if the input DataFrame is missing tickers.
        raise ValueError(f"Missing ticker column: {ticker_col}")

    # Check that the score column exists.
    if score_col not in df.columns:
        # Raise a clear error if the input DataFrame is missing scores.
        raise ValueError(f"Missing score column: {score_col}")

    # If no alpha builder was passed, use the existing project function.
    if alpha_builder is None:
        # Import lazily so tests can provide a fake builder without project imports.
        from .alpha_model import build_alpha as alpha_builder

    # Keep the two columns needed to build the Series.
    scores = df[[ticker_col, score_col]].copy()

    # Clean ticker labels.
    scores[ticker_col] = scores[ticker_col].astype(str).str.strip().str.upper()

    # Keep one row per ticker.
    scores = scores.drop_duplicates(subset=[ticker_col])

    # Build a raw score Series indexed by ticker.
    score_series = scores.set_index(ticker_col)[score_col]

    # Convert raw score values to numeric values.
    score_series = pd.to_numeric(score_series, errors="coerce")

    # Drop tickers with missing score values.
    score_series = score_series.dropna()

    # Use the project alpha builder to create normalized alpha.
    alpha = alpha_builder(score_series)

    # Normalize ticker index again for safety.
    alpha.index = normalize_ticker_index(alpha.index)

    # Give the alpha Series a stable name.
    alpha.name = "alpha"

    # Return alpha.
    return alpha


def load_prices_for_alpha(
    alpha,
    start_date=None,
    end_date=None,
    download_func=None,
):
    """Load prices for the tickers in alpha using the existing project downloader."""

    # Convert alpha index into normalized ticker list.
    tickers = normalize_ticker_index(alpha.index).tolist()

    # Use project config dates if dates were not passed.
    if start_date is None or end_date is None:
        # Import lazily so tests do not need project config.
        from .config import START_DATE, END_DATE

        # Use project START_DATE if missing.
        if start_date is None:
            start_date = START_DATE

        # Use project END_DATE if missing.
        if end_date is None:
            end_date = END_DATE

    # Use the existing project Yahoo price downloader if no downloader was passed.
    if download_func is None:
        # Import lazily so tests can inject a fake downloader.
        from .data_ingest import download_prices as download_func

    # Download adjusted-close prices through the existing project function.
    prices = download_func(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    # Clean the prices into the standard matrix format.
    prices = clean_prices(prices)

    # Return cleaned prices.
    return prices


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
    """Build alpha and prices for a static Yahoo-description signal.

    This is a convenience bridge from the existing project pipeline to the
    Alphalens diagnostics layer.
    """

    # Build normalized static alpha from the selected score column.
    alpha = build_static_alpha_from_scores(
        df=df,
        score_col=score_col,
        ticker_col=ticker_col,
        alpha_builder=alpha_builder,
    )

    # If prices are not supplied, load them using the existing project function.
    if prices is None:
        # Load prices for the alpha tickers.
        prices = load_prices_for_alpha(
            alpha=alpha,
            start_date=start_date,
            end_date=end_date,
            download_func=download_func,
        )

    # If prices are supplied, just clean them.
    else:
        # Clean supplied prices.
        prices = clean_prices(prices)

    # Align alpha and prices to common tickers.
    alpha, prices = align_static_alpha_and_prices(alpha, prices)

    # Return analysis-ready objects.
    return alpha, prices


def build_sector_map_from_universe(
    universe_df,
    ticker_col="Ticker",
    sector_col=None,
):
    """Build a ticker-to-sector map from the universe DataFrame."""

    # Check that the ticker column exists.
    if ticker_col not in universe_df.columns:
        # Raise a clear error if tickers are missing.
        raise ValueError(f"Missing ticker column: {ticker_col}")

    # If no sector column was supplied, search common sector names.
    if sector_col is None:
        # Define common sector column names used by S&P/GICS files.
        possible_sector_cols = [
            "Sector",
            "GICS Sector",
            "GICS_Sector",
            "sector",
            "gics_sector",
        ]

        # Loop over possible sector columns.
        for col in possible_sector_cols:
            # Use the first column that exists.
            if col in universe_df.columns:
                # Store detected sector column.
                sector_col = col

                # Stop searching after finding a match.
                break

    # Check that a sector column was found.
    if sector_col is None or sector_col not in universe_df.columns:
        # Raise a clear error if sector data is unavailable.
        raise ValueError("Could not find sector column. Pass sector_col explicitly.")

    # Keep ticker and sector columns.
    sector_map = universe_df[[ticker_col, sector_col]].copy()

    # Normalize tickers.
    sector_map[ticker_col] = sector_map[ticker_col].astype(str).str.strip().str.upper()

    # Clean sector names.
    sector_map[sector_col] = sector_map[sector_col].astype(str).str.strip()

    # Keep one row per ticker.
    sector_map = sector_map.drop_duplicates(subset=[ticker_col])

    # Convert to Series indexed by ticker.
    sector_map = sector_map.set_index(ticker_col)[sector_col]

    # Return ticker-to-sector mapping.
    return sector_map


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def clean_prices(prices):
    """Clean a wide adjusted-close price matrix."""

    # Require prices to be a DataFrame.
    if not isinstance(prices, pd.DataFrame):
        # Raise a clear error for wrong type.
        raise TypeError("prices must be a pandas DataFrame")

    # Require at least one row and one column.
    if prices.empty:
        # Raise a clear error for empty prices.
        raise ValueError("prices is empty")

    # Copy prices to avoid modifying caller input.
    prices = prices.copy()

    # Convert the index to datetime.
    prices.index = pd.to_datetime(prices.index)

    # Sort dates oldest to newest.
    prices = prices.sort_index()

    # Normalize ticker columns.
    prices.columns = normalize_ticker_index(prices.columns)

    # Convert price values to numeric values.
    prices = prices.apply(pd.to_numeric, errors="coerce")

    # Remove tickers with no usable price data.
    prices = prices.dropna(axis=1, how="all")

    # Check that at least one price column remains.
    if prices.empty:
        # Raise a clear error if cleaning removed everything.
        raise ValueError("prices has no usable columns after cleaning")

    # Return cleaned price matrix.
    return prices


def align_static_alpha_and_prices(alpha, prices):
    """Keep only tickers that appear in both alpha and prices."""

    # Require static alpha to be a Series.
    if not isinstance(alpha, pd.Series):
        # Raise a clear error for wrong alpha type.
        raise TypeError("alpha must be a pandas Series")

    # Copy alpha to avoid modifying caller input.
    alpha = alpha.copy()

    # Normalize alpha tickers.
    alpha.index = normalize_ticker_index(alpha.index)

    # Convert alpha values to floats.
    alpha = alpha.astype(float)

    # Clean prices.
    prices = clean_prices(prices)

    # Find common tickers.
    common_tickers = sorted(set(alpha.index).intersection(set(prices.columns)))

    # Require at least one common ticker.
    if len(common_tickers) == 0:
        # Raise a clear error for empty alignment.
        raise ValueError("No common tickers between alpha and prices.")

    # Keep common alpha tickers.
    alpha = alpha.loc[common_tickers]

    # Keep common price columns.
    prices = prices.loc[:, common_tickers]

    # Return aligned objects.
    return alpha, prices


def get_month_end_trading_dates(index):
    """Return the actual last trading date in each calendar month."""

    # Convert input to sorted DateTimeIndex.
    dates = pd.DatetimeIndex(index).sort_values()

    # Return empty DateTimeIndex for empty input.
    if len(dates) == 0:
        # Return empty dates.
        return pd.DatetimeIndex([])

    # Group actual trading dates by calendar month and keep the last one.
    month_end_dates = (
        pd.Series(dates, index=dates)
        .groupby(dates.to_period("M"))
        .max()
    )

    # Convert output to DateTimeIndex.
    month_end_dates = pd.DatetimeIndex(month_end_dates.values)

    # Return actual month-end trading dates.
    return month_end_dates


def trim_evaluation_dates_for_forward_returns(evaluation_dates, prices, periods):
    """Drop dates without enough future prices for the largest horizon."""

    # If there are no evaluation dates, return immediately.
    if len(evaluation_dates) == 0:
        # Return empty dates.
        return evaluation_dates

    # Convert periods to integer trading-day horizons.
    periods = [int(period) for period in periods]

    # Find the largest required horizon.
    max_period = max(periods)

    # Get sorted price dates.
    price_dates = pd.DatetimeIndex(prices.index).sort_values()

    # If there are too few price dates, return empty dates.
    if len(price_dates) <= max_period:
        # Return empty output.
        return pd.DatetimeIndex([])

    # Last date that has max_period future observations available.
    last_usable_date = price_dates[-max_period - 1]

    # Keep only evaluation dates with enough future prices.
    evaluation_dates = evaluation_dates[evaluation_dates <= last_usable_date]

    # Return trimmed evaluation dates.
    return pd.DatetimeIndex(evaluation_dates)


def get_alpha_for_date(alpha, date):
    """Return alpha values known at one evaluation date."""

    # Static alpha case.
    if isinstance(alpha, pd.Series):
        # Copy alpha.
        alpha_today = alpha.copy()

        # Normalize ticker labels.
        alpha_today.index = normalize_ticker_index(alpha_today.index)

        # Return float values.
        return alpha_today.astype(float)

    # Time-varying alpha case.
    if isinstance(alpha, pd.DataFrame):
        # Copy alpha history.
        alpha = alpha.copy()

        # Convert signal dates to datetime.
        alpha.index = pd.to_datetime(alpha.index)

        # Sort signal dates.
        alpha = alpha.sort_index()

        # Normalize ticker columns.
        alpha.columns = normalize_ticker_index(alpha.columns)

        # Keep signals known at or before date.
        available = alpha.loc[:date]

        # Return empty Series if no signal was available.
        if available.empty:
            # Return empty float Series.
            return pd.Series(dtype=float)

        # Use latest available signal row.
        alpha_today = available.iloc[-1].copy()

        # Normalize ticker labels.
        alpha_today.index = normalize_ticker_index(alpha_today.index)

        # Return float values.
        return alpha_today.astype(float)

    # Reject unsupported alpha type.
    raise TypeError("alpha must be a pandas Series or DataFrame")


def clean_sector_map(sector_map):
    """Clean sector mapping into Series indexed by ticker."""

    # Convert dict to Series.
    if isinstance(sector_map, dict):
        # Dictionary keys are tickers and values are sector names.
        sector_map = pd.Series(sector_map)

    # Copy Series input.
    elif isinstance(sector_map, pd.Series):
        # Copy input so caller is not modified.
        sector_map = sector_map.copy()

    # Convert DataFrame to Series.
    elif isinstance(sector_map, pd.DataFrame):
        # Require standard columns.
        if "ticker" not in sector_map.columns or "sector" not in sector_map.columns:
            # Raise clear error for malformed DataFrame.
            raise ValueError("sector_map DataFrame must have columns: ticker, sector")

        # Convert DataFrame to Series indexed by ticker.
        sector_map = sector_map.set_index("ticker")["sector"]

    # Reject unsupported type.
    else:
        # Raise clear error.
        raise TypeError("sector_map must be dict, Series, or DataFrame")

    # Normalize ticker index.
    sector_map.index = normalize_ticker_index(sector_map.index)

    # Convert sector labels to strings.
    sector_map = sector_map.astype(str)

    # Return cleaned map.
    return sector_map


def make_static_factor_for_alphalens(alpha, prices, periods=(21, 63)):
    """Repeat a static alpha Series across monthly Alphalens evaluation dates."""

    # Clean prices.
    prices = clean_prices(prices)

    # Copy alpha.
    alpha = alpha.copy()

    # Normalize alpha tickers.
    alpha.index = normalize_ticker_index(alpha.index)

    # Convert alpha values to floats.
    alpha = alpha.astype(float)

    # Align alpha and prices.
    alpha, prices = align_static_alpha_and_prices(alpha, prices)

    # Get actual month-end trading dates.
    evaluation_dates = get_month_end_trading_dates(prices.index)

    # Remove dates that do not have enough future prices.
    evaluation_dates = trim_evaluation_dates_for_forward_returns(
        evaluation_dates=evaluation_dates,
        prices=prices,
        periods=periods,
    )

    # Create list to hold per-date factor Series.
    rows = []

    # Loop over each evaluation date.
    for date in evaluation_dates:
        # Copy the static alpha for this date.
        one_date = alpha.copy()

        # Convert index to Alphalens MultiIndex: date, asset.
        one_date.index = pd.MultiIndex.from_product(
            [[date], one_date.index],
            names=["date", "asset"],
        )

        # Store this date's factor values.
        rows.append(one_date)

    # If no rows were built, return empty factor Series.
    if not rows:
        # Return empty Series with expected MultiIndex.
        empty_index = pd.MultiIndex.from_arrays(
            [[], []],
            names=["date", "asset"],
        )

        # Return empty float Series.
        return pd.Series(dtype=float, index=empty_index, name="factor")

    # Concatenate all date-level factor Series.
    factor = pd.concat(rows)

    # Name the factor Series.
    factor.name = "factor"

    # Return Alphalens factor Series.
    return factor


def make_time_varying_factor_for_alphalens(alpha_history, prices, periods=(21, 63)):
    """Convert a time-varying alpha DataFrame into Alphalens factor format."""

    # Clean prices.
    prices = clean_prices(prices)

    # Require DataFrame alpha history.
    if not isinstance(alpha_history, pd.DataFrame):
        # Raise clear error.
        raise TypeError("alpha_history must be a pandas DataFrame")

    # Copy alpha history.
    alpha_history = alpha_history.copy()

    # Convert signal dates to datetime.
    alpha_history.index = pd.to_datetime(alpha_history.index)

    # Sort signal dates.
    alpha_history = alpha_history.sort_index()

    # Normalize ticker columns.
    alpha_history.columns = normalize_ticker_index(alpha_history.columns)

    # Keep only dates that are in the price index.
    signal_dates = alpha_history.index.intersection(prices.index)

    # Trim dates to ensure enough future prices.
    signal_dates = trim_evaluation_dates_for_forward_returns(
        evaluation_dates=signal_dates,
        prices=prices,
        periods=periods,
    )

    # Create list to hold factor rows.
    rows = []

    # Loop over signal dates.
    for date in signal_dates:
        # Get signal values for this date.
        one_date = alpha_history.loc[date].dropna().astype(float)

        # Keep only tickers that appear in prices.
        one_date = one_date.loc[one_date.index.intersection(prices.columns)]

        # Skip empty rows.
        if one_date.empty:
            # Continue to next date.
            continue

        # Convert to MultiIndex.
        one_date.index = pd.MultiIndex.from_product(
            [[date], one_date.index],
            names=["date", "asset"],
        )

        # Store this row.
        rows.append(one_date)

    # If no rows were built, return empty factor.
    if not rows:
        # Create empty MultiIndex.
        empty_index = pd.MultiIndex.from_arrays(
            [[], []],
            names=["date", "asset"],
        )

        # Return empty Series.
        return pd.Series(dtype=float, index=empty_index, name="factor")

    # Combine factor rows.
    factor = pd.concat(rows)

    # Name the factor.
    factor.name = "factor"

    # Return Alphalens factor Series.
    return factor


def make_groupby_for_alphalens(sector_map, tickers):
    """Create Alphalens groupby dictionary from ticker-to-sector map."""

    # Return None if no sector map is available.
    if sector_map is None:
        # No groups.
        return None

    # Clean sector map.
    sector_map = clean_sector_map(sector_map)

    # Normalize tickers.
    tickers = normalize_ticker_index(tickers)

    # Keep sectors for available tickers only.
    sector_map = sector_map.reindex(tickers).dropna()

    # Convert to dictionary expected by Alphalens.
    groupby = sector_map.to_dict()

    # Return dictionary.
    return groupby


# -----------------------------------------------------------------------------
# Alphalens factor_data builders
# -----------------------------------------------------------------------------


def build_alphalens_factor_data_from_prices(
    alpha,
    prices,
    periods=(21, 63),
    n_quantiles=5,
    sector_map=None,
    max_loss=0.50,
):
    """Build Alphalens factor_data from alpha and price matrix.

    This is the main Alphalens-Reloaded bridge.

    It uses:

        alphalens.utils.get_clean_factor_and_forward_returns(...)
    """

    # Require Alphalens for this true Alphalens builder.
    require_alphalens()

    # Clean prices.
    prices = clean_prices(prices)

    # Choose factor conversion based on alpha type.
    if isinstance(alpha, pd.Series):
        # Convert static alpha to repeated monthly Alphalens factor.
        factor = make_static_factor_for_alphalens(
            alpha=alpha,
            prices=prices,
            periods=periods,
        )

    # Handle time-varying SEC-style alpha.
    elif isinstance(alpha, pd.DataFrame):
        # Convert time-varying alpha to Alphalens factor.
        factor = make_time_varying_factor_for_alphalens(
            alpha_history=alpha,
            prices=prices,
            periods=periods,
        )

    # Reject unsupported alpha type.
    else:
        # Raise clear error.
        raise TypeError("alpha must be a pandas Series or DataFrame")

    # Raise clear error if factor is empty.
    if factor.empty:
        # No factor data can be built.
        raise ValueError("Alphalens factor Series is empty.")

    # Create optional sector/group mapping.
    groupby = make_groupby_for_alphalens(
        sector_map=sector_map,
        tickers=prices.columns,
    )

    # Call the core Alphalens cleaning function.
    factor_data = al_utils.get_clean_factor_and_forward_returns(
        factor=factor,
        prices=prices,
        groupby=groupby,
        quantiles=n_quantiles,
        periods=periods,
        max_loss=max_loss,
    )

    # Return Alphalens factor_data.
    return factor_data


def build_factor_data_from_forward_returns(
    alpha,
    forward_returns,
    n_quantiles=5,
    sector_map=None,
):
    """Build an Alphalens-style factor_data table from precomputed forward returns.

    This helper supports the public function signatures that mirror the custom
    baseline:

        compute_monthly_ic(alpha, forward_returns)
        compute_quantile_returns(alpha, forward_returns)

    It does not call get_clean_factor_and_forward_returns because forward
    returns have already been computed.
    """

    # Require forward_returns to be a dictionary.
    if not isinstance(forward_returns, dict):
        # Raise clear error.
        raise TypeError("forward_returns must be a dictionary")

    # Require at least one horizon.
    if len(forward_returns) == 0:
        # Raise clear error.
        raise ValueError("forward_returns is empty")

    # Use first horizon to define evaluation dates.
    first_returns = next(iter(forward_returns.values())).copy()

    # Convert index to datetime.
    first_returns.index = pd.to_datetime(first_returns.index)

    # Sort by date.
    first_returns = first_returns.sort_index()

    # Normalize ticker columns.
    first_returns.columns = normalize_ticker_index(first_returns.columns)

    # Use actual last trading date of each month.
    evaluation_dates = get_month_end_trading_dates(first_returns.index)

    # Clean sector map if provided.
    sectors = None if sector_map is None else clean_sector_map(sector_map)

    # Create list for per-date factor_data rows.
    rows = []

    # Loop over evaluation dates.
    for evaluation_date in evaluation_dates:
        # Get alpha values known at this date.
        alpha_today = get_alpha_for_date(alpha, evaluation_date)

        # Skip dates without alpha.
        if alpha_today.empty:
            # Move to next date.
            continue

        # Start a one-date table with alpha values.
        one_date = alpha_today.rename("factor").to_frame()

        # Normalize ticker index.
        one_date.index = normalize_ticker_index(one_date.index)

        # Loop over forward return horizons.
        for horizon, returns in forward_returns.items():
            # Copy returns.
            returns = returns.copy()

            # Convert dates to datetime.
            returns.index = pd.to_datetime(returns.index)

            # Sort by date.
            returns = returns.sort_index()

            # Normalize ticker columns.
            returns.columns = normalize_ticker_index(returns.columns)

            # Build column name like 21D.
            column = f"{int(horizon)}D"

            # If evaluation date is missing, store NaNs.
            if evaluation_date not in returns.index:
                # Assign missing returns.
                one_date[column] = np.nan

            # Otherwise attach returns for this date.
            else:
                # Get returns for this date.
                returns_today = returns.loc[evaluation_date]

                # Normalize return tickers.
                returns_today.index = normalize_ticker_index(returns_today.index)

                # Align return values to factor index.
                one_date[column] = returns_today.reindex(one_date.index)

        # Get return columns.
        return_columns = [col for col in one_date.columns if str(col).endswith("D")]

        # Drop rows missing factor or all forward returns.
        one_date = one_date.dropna(subset=["factor"] + return_columns)

        # Skip empty dates.
        if one_date.empty:
            # Move to next date.
            continue

        # Assign factor quantiles using alpha rankings.
        one_date = assign_factor_quantiles(one_date, n_quantiles=n_quantiles)

        # Skip date if quantiles cannot be assigned.
        if one_date is None:
            # Move to next date.
            continue

        # Attach sector/group labels if available.
        if sectors is not None:
            # Add Alphalens-style group column.
            one_date["group"] = sectors.reindex(one_date.index)

        # Convert index to Alphalens MultiIndex.
        one_date.index = pd.MultiIndex.from_product(
            [[evaluation_date], one_date.index],
            names=["date", "asset"],
        )

        # Store one-date factor_data rows.
        rows.append(one_date)

    # Return empty DataFrame if no rows were built.
    if not rows:
        # Return empty output.
        return pd.DataFrame()

    # Combine all date rows into factor_data.
    factor_data = pd.concat(rows).sort_index()

    # Return factor_data.
    return factor_data


def assign_factor_quantiles(data, n_quantiles=5):
    """Assign factor quantiles to a cross-section."""

    # Return None if too few names exist.
    if len(data) < n_quantiles:
        # Too few assets for requested quantiles.
        return None

    # Copy input.
    data = data.copy()

    # Rank factor values to handle ties.
    ranks = data["factor"].rank(method="first")

    # Assign quantile labels.
    data["factor_quantile"] = pd.qcut(
        ranks,
        q=n_quantiles,
        labels=list(range(1, n_quantiles + 1)),
    )

    # Convert labels to integers.
    data["factor_quantile"] = data["factor_quantile"].astype(int)

    # Return data with quantiles.
    return data


# -----------------------------------------------------------------------------
# Public function 1
# -----------------------------------------------------------------------------


def compute_forward_returns(prices, horizons=(21, 63)):
    """Compute forward returns.

    When Alphalens is installed, this function tries to use
    alphalens.utils.compute_forward_returns(...).

    If that API is unavailable or fails, it falls back to explicit pandas
    arithmetic so the project remains runnable.
    """

    # Clean prices.
    prices = clean_prices(prices)

    # Convert horizons to integers.
    horizons = tuple(int(horizon) for horizon in horizons)

    # Require positive horizons.
    if any(horizon <= 0 for horizon in horizons):
        # Raise clear error.
        raise ValueError("horizons must be positive integers")

    # Try Alphalens implementation if installed and available.
    if ALPHALENS_AVAILABLE and hasattr(al_utils, "compute_forward_returns"):
        # Use try/except for compatibility across Alphalens versions.
        try:
            # Create dummy factor rows for every date and ticker.
            rows = []

            # Loop over dates.
            for date in prices.index:
                # Create one dummy factor value per asset.
                one_date = pd.Series(0.0, index=prices.columns)

                # Convert to MultiIndex.
                one_date.index = pd.MultiIndex.from_product(
                    [[date], one_date.index],
                    names=["date", "asset"],
                )

                # Store dummy factor values.
                rows.append(one_date)

            # Combine dummy factor.
            dummy_factor = pd.concat(rows)

            # Name dummy factor.
            dummy_factor.name = "factor"

            # Ask Alphalens to compute forward returns.
            al_forward = al_utils.compute_forward_returns(
                factor=dummy_factor,
                prices=prices,
                periods=horizons,
            )

            # Create output dictionary.
            output = {}

            # Convert each Alphalens period column to wide DataFrame.
            for horizon in horizons:
                # Build expected column name.
                column = f"{int(horizon)}D"

                # Check that Alphalens returned this horizon.
                if column not in al_forward.columns:
                    # Force fallback if column naming differs.
                    raise KeyError(f"Missing Alphalens column: {column}")

                # Unstack assets into wide matrix.
                output[horizon] = al_forward[column].unstack("asset")

            # Return Alphalens-derived forward returns.
            return output

        # Fall back if Alphalens computation fails.
        except Exception:
            # Continue to explicit fallback.
            pass

    # Explicit fallback computation.
    output = {}

    # Loop over horizons.
    for horizon in horizons:
        # Future price at t+horizon aligned to row t.
        future_prices = prices.shift(-horizon)

        # Compute forward returns.
        returns = future_prices / prices - 1.0

        # Store horizon matrix.
        output[horizon] = returns

    # Return forward returns.
    return output


# -----------------------------------------------------------------------------
# Public function 2
# -----------------------------------------------------------------------------


def compute_monthly_ic(alpha, forward_returns, method="spearman"):
    """Compute monthly IC using Alphalens performance functions where possible."""

    # Validate method.
    if method not in ["spearman", "pearson", "kendall"]:
        # Raise clear error.
        raise ValueError("method must be 'spearman', 'pearson', or 'kendall'")

    # Build Alphalens-style factor_data from precomputed returns.
    factor_data = build_factor_data_from_forward_returns(
        alpha=alpha,
        forward_returns=forward_returns,
        n_quantiles=5,
    )

    # Return empty output if factor_data is empty.
    if factor_data.empty:
        # Return empty DataFrame.
        return pd.DataFrame()

    # Identify forward-return columns.
    return_columns = [col for col in factor_data.columns if str(col).endswith("D")]

    # Use Alphalens IC function for Spearman rank IC.
    if method == "spearman" and ALPHALENS_AVAILABLE and al_perf is not None:
        # Try true Alphalens IC.
        try:
            # Compute Alphalens factor IC.
            ic = al_perf.factor_information_coefficient(
                factor_data,
                group_adjust=False,
                by_group=False,
            )

            # Keep expected return columns.
            ic = ic[return_columns]

            # Sort by date.
            ic = ic.sort_index()

            # Return Alphalens IC.
            return ic

        # Fall back if Alphalens IC fails.
        except Exception:
            # Continue to explicit fallback.
            pass

    # Explicit fallback for Pearson/Kendall or missing Alphalens.
    rows = []

    # Group factor_data by date.
    for date, group in factor_data.groupby(level="date"):
        # Create row dictionary for this date.
        row = {}

        # Loop over forward-return columns.
        for column in return_columns:
            # Keep factor and one return column.
            values = group[["factor", column]].dropna()

            # Require at least three assets.
            if len(values) < 3:
                # Use NaN if cross-section is too small.
                row[column] = np.nan

            # Otherwise compute correlation.
            else:
                # Compute requested correlation.
                row[column] = values["factor"].corr(values[column], method=method)

        # Store this date row.
        rows.append(pd.Series(row, name=date))

    # Combine date rows.
    ic = pd.DataFrame(rows)

    # Sort by date.
    ic = ic.sort_index()

    # Return IC table.
    return ic


# -----------------------------------------------------------------------------
# Public function 3
# -----------------------------------------------------------------------------


def compute_quantile_returns(alpha, forward_returns, n_quantiles=5):
    """Compute mean forward return by factor quantile."""

    # Build Alphalens-style factor_data.
    factor_data = build_factor_data_from_forward_returns(
        alpha=alpha,
        forward_returns=forward_returns,
        n_quantiles=n_quantiles,
    )

    # Return empty tidy table if no factor_data exists.
    if factor_data.empty:
        # Return empty output with stable columns.
        return pd.DataFrame(
            columns=[
                "date",
                "horizon",
                "quantile",
                "mean_forward_return",
                "n_assets",
            ]
        )

    # Identify forward-return columns.
    return_columns = [col for col in factor_data.columns if str(col).endswith("D")]

    # Try Alphalens mean_return_by_quantile when available.
    if ALPHALENS_AVAILABLE and al_perf is not None:
        # Use try/except for API compatibility.
        try:
            # Compute mean return by quantile and date.
            mean_returns, _ = al_perf.mean_return_by_quantile(
                factor_data,
                by_date=True,
                by_group=False,
                demeaned=False,
                group_adjust=False,
            )

            # Create output rows.
            rows = []

            # Loop over Alphalens output rows.
            for index, values in mean_returns.iterrows():
                # Skip unexpected index format.
                if not isinstance(index, tuple):
                    # Continue to next row.
                    continue

                # First level is date.
                date = index[0]

                # Second level is factor quantile.
                quantile = index[1]

                # Count assets in this date/quantile.
                mask = (
                    (factor_data.index.get_level_values("date") == date)
                    & (factor_data["factor_quantile"] == quantile)
                )

                # Count assets.
                n_assets = int(mask.sum())

                # Loop over return columns.
                for column in return_columns:
                    # Parse horizon integer from column label like 21D.
                    horizon = int(str(column).replace("D", ""))

                    # Add tidy output row.
                    rows.append({
                        "date": date,
                        "horizon": horizon,
                        "quantile": int(quantile),
                        "mean_forward_return": values[column],
                        "n_assets": n_assets,
                    })

            # Convert rows to DataFrame.
            result = pd.DataFrame(rows)

            # Sort output if non-empty.
            if not result.empty:
                # Sort by date, horizon, quantile.
                result = result.sort_values(
                    ["date", "horizon", "quantile"]
                ).reset_index(drop=True)

            # Return tidy table.
            return result

        # Fall back if Alphalens function fails.
        except Exception:
            # Continue to explicit fallback.
            pass

    # Explicit fallback.
    rows = []

    # Group by date and factor quantile.
    grouped = factor_data.groupby(
        [
            factor_data.index.get_level_values("date"),
            factor_data["factor_quantile"],
        ]
    )

    # Loop over groups.
    for (date, quantile), group in grouped:
        # Loop over return columns.
        for column in return_columns:
            # Parse horizon from column label.
            horizon = int(str(column).replace("D", ""))

            # Add tidy output row.
            rows.append({
                "date": date,
                "horizon": horizon,
                "quantile": int(quantile),
                "mean_forward_return": group[column].mean(),
                "n_assets": len(group),
            })

    # Convert rows to DataFrame.
    result = pd.DataFrame(rows)

    # Sort if non-empty.
    if not result.empty:
        # Sort stable output.
        result = result.sort_values(
            ["date", "horizon", "quantile"]
        ).reset_index(drop=True)

    # Return output.
    return result


# -----------------------------------------------------------------------------
# Public function 4
# -----------------------------------------------------------------------------


def compute_top_bottom_spread(quantile_returns):
    """Compute top-minus-bottom quantile return spread."""

    # Define required columns.
    required_columns = {
        "date",
        "horizon",
        "quantile",
        "mean_forward_return",
    }

    # Find missing columns.
    missing = required_columns.difference(quantile_returns.columns)

    # Validate input table.
    if missing:
        # Raise clear error.
        raise ValueError(f"quantile_returns is missing columns: {missing}")

    # Handle empty input.
    if quantile_returns.empty:
        # Return empty DataFrame.
        return pd.DataFrame()

    # Copy input.
    data = quantile_returns.copy()

    # Convert dates.
    data["date"] = pd.to_datetime(data["date"])

    # Create output dictionary.
    spreads = {}

    # Loop over horizons.
    for horizon in sorted(data["horizon"].dropna().unique()):
        # Keep one horizon.
        one_horizon = data.loc[data["horizon"] == horizon]

        # Pivot quantiles into columns.
        wide = one_horizon.pivot_table(
            index="date",
            columns="quantile",
            values="mean_forward_return",
            aggfunc="mean",
        )

        # Skip empty pivot.
        if wide.empty:
            # Continue to next horizon.
            continue

        # Identify bottom quantile.
        bottom_quantile = wide.columns.min()

        # Identify top quantile.
        top_quantile = wide.columns.max()

        # Compute top-minus-bottom spread.
        spreads[f"{int(horizon)}D"] = wide[top_quantile] - wide[bottom_quantile]

    # Combine spreads.
    result = pd.DataFrame(spreads)

    # Sort by date.
    result = result.sort_index()

    # Return spread table.
    return result


# -----------------------------------------------------------------------------
# Public function 5
# -----------------------------------------------------------------------------


def compute_signal_turnover(alpha_history):
    """Compute Alphalens-style top-quantile signal turnover."""

    # Static alpha has no turnover history.
    if isinstance(alpha_history, pd.Series):
        # Return empty Series.
        return pd.Series(dtype=float, name="signal_turnover")

    # Require DataFrame for time-varying alpha.
    if not isinstance(alpha_history, pd.DataFrame):
        # Raise clear error.
        raise TypeError("alpha_history must be a pandas Series or DataFrame")

    # Empty input has no turnover.
    if alpha_history.empty:
        # Return empty Series.
        return pd.Series(dtype=float, name="signal_turnover")

    # Copy input.
    alpha_history = alpha_history.copy()

    # Convert dates.
    alpha_history.index = pd.to_datetime(alpha_history.index)

    # Sort dates.
    alpha_history = alpha_history.sort_index()

    # Normalize ticker columns.
    alpha_history.columns = normalize_ticker_index(alpha_history.columns)

    # Create factor_data-like rows.
    rows = []

    # Loop over each signal date.
    for date, alpha_today in alpha_history.iterrows():
        # Create one-date DataFrame.
        one_date = alpha_today.rename("factor").to_frame().dropna()

        # Assign five quantiles.
        one_date = assign_factor_quantiles(one_date, n_quantiles=5)

        # Skip dates with too few names.
        if one_date is None:
            # Continue to next date.
            continue

        # Convert index to Alphalens MultiIndex.
        one_date.index = pd.MultiIndex.from_product(
            [[date], one_date.index],
            names=["date", "asset"],
        )

        # Store row.
        rows.append(one_date)

    # Return empty if no rows were built.
    if not rows:
        # Return empty Series.
        return pd.Series(dtype=float, name="signal_turnover")

    # Combine rows.
    factor_data = pd.concat(rows).sort_index()

    # Extract quantile factor Series.
    quantile_factor = factor_data["factor_quantile"]

    # Identify top quantile.
    top_quantile = int(quantile_factor.max())

    # Try Alphalens quantile turnover if available.
    if ALPHALENS_AVAILABLE and al_perf is not None:
        # Use try/except for compatibility.
        try:
            # Compute Alphalens quantile turnover.
            turnover = al_perf.quantile_turnover(
                quantile_factor,
                quantile=top_quantile,
                period=1,
            )

            # Name the output.
            turnover.name = "signal_turnover"

            # Return non-missing turnover.
            return turnover.dropna()

        # Fall back if Alphalens turnover fails.
        except Exception:
            # Continue to explicit fallback.
            pass

    # Explicit fallback:
    # fraction of today's top-quantile names not in previous top quantile.
    dates = factor_data.index.get_level_values("date").unique().sort_values()

    # Create list to hold turnover values.
    values = []

    # Keep prior top names.
    previous_names = None

    # Loop over dates.
    for date in dates:
        # Get one date of factor_data.
        today = factor_data.xs(date, level="date")

        # Get names in the top quantile today.
        today_names = set(today.index[today["factor_quantile"] == top_quantile])

        # Skip first date because there is no prior date.
        if previous_names is None:
            # Store current top names.
            previous_names = today_names

            # Move to next date.
            continue

        # Handle empty top quantile.
        if len(today_names) == 0:
            # Turnover undefined.
            turnover = np.nan

        # Otherwise compute share of new names.
        else:
            # Names in today's top quantile that were not previously top quantile.
            new_names = today_names.difference(previous_names)

            # Turnover is new names divided by today's top names.
            turnover = len(new_names) / len(today_names)

        # Store turnover value.
        values.append((date, turnover))

        # Update previous names.
        previous_names = today_names

    # Build output Series.
    turnover = pd.Series(
        [value for _, value in values],
        index=[date for date, _ in values],
        name="signal_turnover",
    )

    # Drop missing values.
    turnover = turnover.dropna()

    # Return turnover.
    return turnover


# -----------------------------------------------------------------------------
# Public function 6
# -----------------------------------------------------------------------------


def compute_sector_exposure(alpha, sector_map):
    """Summarize alpha exposure by sector.

    Alphalens uses sector information through the `groupby` argument in
    factor_data construction and tear sheets.

    This standalone summary keeps the same API as the custom baseline so the
    sector exposure output can be compared directly.
    """

    # Clean sector map.
    sectors = clean_sector_map(sector_map)

    # Define helper for one date.
    def one_date_exposure(alpha_values, date=None):
        # Copy alpha values.
        alpha_values = alpha_values.copy()

        # Normalize ticker index.
        alpha_values.index = normalize_ticker_index(alpha_values.index)

        # Convert alpha to floats.
        alpha_values = alpha_values.astype(float)

        # Align alpha with sector labels.
        data = pd.concat(
            [
                alpha_values.rename("alpha"),
                sectors.rename("sector"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        # Return empty table if no sectors match.
        if data.empty:
            # Return empty DataFrame.
            return pd.DataFrame()

        # Compute total absolute alpha.
        total_abs_alpha = data["alpha"].abs().sum()

        # Create output rows.
        rows = []

        # Loop over sectors.
        for sector, group in data.groupby("sector"):
            # Handle zero total alpha.
            if total_abs_alpha == 0:
                # Undefined concentration share.
                abs_alpha_share = np.nan

            # Otherwise compute concentration share.
            else:
                # Sector absolute alpha divided by total absolute alpha.
                abs_alpha_share = group["alpha"].abs().sum() / total_abs_alpha

            # Build one sector row.
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

            # Add date for time-varying alpha.
            if date is not None:
                # Store date.
                row["date"] = date

            # Store row.
            rows.append(row)

        # Convert rows to DataFrame.
        result = pd.DataFrame(rows)

        # Sort by concentration.
        result = result.sort_values("abs_alpha_share", ascending=False)

        # Return sector exposure table.
        return result

    # Static alpha case.
    if isinstance(alpha, pd.Series):
        # Compute static exposure.
        result = one_date_exposure(alpha)

        # Set sector as index for static output.
        if not result.empty:
            # Set index.
            result = result.set_index("sector")

        # Return static exposure.
        return result

    # Time-varying alpha case.
    if isinstance(alpha, pd.DataFrame):
        # Copy input.
        alpha = alpha.copy()

        # Convert dates.
        alpha.index = pd.to_datetime(alpha.index)

        # Sort dates.
        alpha = alpha.sort_index()

        # Normalize columns.
        alpha.columns = normalize_ticker_index(alpha.columns)

        # Store per-date exposures.
        all_rows = []

        # Loop over rows.
        for date, row in alpha.iterrows():
            # Compute one-date sector exposure.
            exposure = one_date_exposure(row, date=date)

            # Store non-empty exposure.
            if not exposure.empty:
                # Append table.
                all_rows.append(exposure)

        # Handle no exposure rows.
        if not all_rows:
            # Return empty DataFrame.
            return pd.DataFrame()

        # Combine per-date exposures.
        result = pd.concat(all_rows, ignore_index=True)

        # Sort by date and concentration.
        result = result.sort_values(["date", "abs_alpha_share"], ascending=[True, False])

        # Return time-varying sector exposure.
        return result

    # Reject unsupported alpha type.
    raise TypeError("alpha must be a pandas Series or DataFrame")


# -----------------------------------------------------------------------------
# Direct Alphalens performance wrappers
# -----------------------------------------------------------------------------


def alphalens_information_coefficient(factor_data):
    """Compute Alphalens information coefficient from factor_data."""

    # Require Alphalens.
    require_alphalens()

    # Compute IC with Alphalens.
    ic = al_perf.factor_information_coefficient(
        factor_data,
        group_adjust=False,
        by_group=False,
    )

    # Return IC.
    return ic


def alphalens_mean_information_coefficient(factor_data, by_time="M"):
    """Compute Alphalens mean IC grouped by time."""

    # Require Alphalens.
    require_alphalens()

    # Compute mean IC.
    mean_ic = al_perf.mean_information_coefficient(
        factor_data,
        group_adjust=False,
        by_group=False,
        by_time=by_time,
    )

    # Return mean IC.
    return mean_ic


def alphalens_mean_return_by_quantile(factor_data, by_date=True):
    """Compute Alphalens mean return by quantile."""

    # Require Alphalens.
    require_alphalens()

    # Compute mean return by quantile.
    mean_returns, std_errors = al_perf.mean_return_by_quantile(
        factor_data,
        by_date=by_date,
        by_group=False,
        demeaned=False,
        group_adjust=False,
    )

    # Return both Alphalens outputs.
    return mean_returns, std_errors


# Define public API.
__all__ = [
    "ALPHALENS_AVAILABLE",
    "require_alphalens",
    "build_static_alpha_from_scores",
    "load_prices_for_alpha",
    "prepare_static_signal_inputs",
    "build_sector_map_from_universe",
    "make_static_factor_for_alphalens",
    "make_time_varying_factor_for_alphalens",
    "make_groupby_for_alphalens",
    "build_alphalens_factor_data_from_prices",
    "build_factor_data_from_forward_returns",
    "compute_forward_returns",
    "compute_monthly_ic",
    "compute_quantile_returns",
    "compute_top_bottom_spread",
    "compute_signal_turnover",
    "compute_sector_exposure",
    "alphalens_information_coefficient",
    "alphalens_mean_information_coefficient",
    "alphalens_mean_return_by_quantile",
]
