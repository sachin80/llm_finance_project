
"""
Point-in-time panel construction for SEC filing signals.

This module begins where `sec_signal.py` ends.

Input:
    A flat filing-level signal table, usually created by
    sec_signal.signal_records_to_dataframe(...)

Output:
    Monthly point-in-time signal panels.

The module has five clear responsibilities:

    1. Clean filing-level signal tables.
    2. Convert filing dates into conservative signal availability dates.
    3. Carry the latest available filing signal forward through time.
    4. Build raw date-by-ticker signal panels.
    5. Build cross-sectionally normalized alpha panels for diagnostics.

The module does not:

    - call EdgarTools,
    - call OpenAI,
    - create embeddings,
    - compute forward returns,
    - run IC or quantile diagnostics,
    - construct portfolios.

Those jobs remain in their own project modules.

Design principles
-----------------
SEC filings arrive at irregular event times. Signal diagnostics and portfolio
backtests usually evaluate signals on a regular calendar, such as month-end
trading dates.

This module performs that translation:

    filing-event signal table
        -> latest known signal by evaluation date
        -> date-by-ticker panel

A ticker receives a new signal when a filing becomes available. That value is
then carried forward until a newer filing becomes available.

The first version uses simple date logic and an optional calendar-day lag.
A later production version can replace this with a full trading-calendar rule.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Default signal columns
# -----------------------------------------------------------------------------


# These columns are created by sec_signal.signal_records_to_dataframe(...).
# Not every experiment has to use every column, but keeping this list explicit
# makes the first point-in-time diagnostic run easier to inspect.
DEFAULT_SIGNAL_COLUMNS = [
    "summary_theme_similarity",
    "exposure_label_score",
    "direction_score",
    "evidence_strength_score",
    "similarity_x_exposure_score",
    "similarity_x_evidence_strength_score",
    "similarity_x_direction_score",
    "direct_or_indirect_similarity",
]


# -----------------------------------------------------------------------------
# Small cleaning helpers
# -----------------------------------------------------------------------------


def normalize_ticker_label(value: Any) -> str:
    """Normalize one ticker label to the project's uppercase convention."""

    return str(value).strip().upper()


def normalize_ticker_index(index: Any) -> pd.Index:
    """Normalize a pandas Index of ticker labels."""

    return pd.Index(index).astype(str).str.strip().str.upper()


def _coerce_datetime_series(values: Any) -> pd.Series:
    """Convert a Series-like object to timezone-naive pandas timestamps.

    This helper accepts ordinary dates, strings, pandas timestamps, and
    timezone-aware timestamps. The output is normalized to midnight dates,
    which is sufficient for the first monthly point-in-time panels.
    """

    series = pd.to_datetime(
        values,
        errors="coerce",
        utc=True,
    )

    # Convert timezone-aware timestamps to timezone-naive UTC timestamps.
    # This avoids mixing timezone-aware and timezone-naive objects in joins.
    if hasattr(series, "dt"):
        series = series.dt.tz_convert(None)

    return series.dt.normalize()


def clean_signal_table(
    signal_table: pd.DataFrame,
    *,
    required_signal_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Clean a flat filing-level SEC signal table.

    Required columns:
        ticker
        filing_date

    Optional columns:
        acceptance_datetime
        accession_number

    The requested signal columns are converted to numeric values. Bad values
    are converted to NaN rather than raising, so downstream diagnostics can
    choose how to handle partial coverage.
    """

    if not isinstance(signal_table, pd.DataFrame):
        raise TypeError("signal_table must be a pandas DataFrame")

    if signal_table.empty:
        raise ValueError("signal_table is empty")

    required_columns = [
        "ticker",
        "filing_date",
    ]

    if required_signal_columns is not None:
        required_columns.extend(list(required_signal_columns))

    missing_columns = [
        column
        for column in required_columns
        if column not in signal_table.columns
    ]

    if missing_columns:
        raise ValueError(
            "signal_table is missing required columns: "
            f"{missing_columns}"
        )

    table = signal_table.copy()

    table["ticker"] = table["ticker"].map(normalize_ticker_label)

    table["filing_date"] = _coerce_datetime_series(
        table["filing_date"]
    )

    if "acceptance_datetime" in table.columns:
        table["acceptance_datetime"] = _coerce_datetime_series(
            table["acceptance_datetime"]
        )

    else:
        table["acceptance_datetime"] = pd.NaT

    if "accession_number" not in table.columns:
        table["accession_number"] = ""

    table["accession_number"] = (
        table["accession_number"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if required_signal_columns is not None:
        for column in required_signal_columns:
            table[column] = pd.to_numeric(
                table[column],
                errors="coerce",
            )

    table = table.dropna(
        subset=[
            "ticker",
            "filing_date",
        ]
    )

    return table.reset_index(drop=True)


def signal_available_date_from_row(
    row: pd.Series,
    *,
    availability_lag_days: int = 1,
) -> pd.Timestamp:
    """Return the date on which a filing signal becomes available.

    The first version uses the SEC acceptance date when it is present. If that
    field is missing, the filing date is used.

    A conservative calendar-day lag is applied by default. For monthly
    diagnostics this usually has little effect, but it avoids treating a filing
    as known at the exact same date it was accepted.

    Later, this can be replaced by a trading-calendar rule:
        accepted before/after market close -> next trading session.
    """

    if int(availability_lag_days) < 0:
        raise ValueError("availability_lag_days must be non-negative")

    acceptance_datetime = row.get("acceptance_datetime", pd.NaT)

    if pd.notna(acceptance_datetime):
        base_date = pd.Timestamp(acceptance_datetime)

    else:
        base_date = pd.Timestamp(row["filing_date"])

    base_date = base_date.normalize()

    return base_date + pd.Timedelta(days=int(availability_lag_days))


def add_signal_available_date(
    signal_table: pd.DataFrame,
    *,
    availability_lag_days: int = 1,
) -> pd.DataFrame:
    """Add a signal_available_date column to a cleaned signal table."""

    table = signal_table.copy()

    table["signal_available_date"] = table.apply(
        lambda row: signal_available_date_from_row(
            row,
            availability_lag_days=availability_lag_days,
        ),
        axis=1,
    )

    return table


# -----------------------------------------------------------------------------
# Evaluation-date helpers
# -----------------------------------------------------------------------------


def get_month_end_trading_dates(index: Any) -> pd.DatetimeIndex:
    """Return the last available date in each calendar month."""

    dates = pd.DatetimeIndex(
        pd.to_datetime(index, errors="coerce")
    )

    dates = dates.dropna().sort_values()

    if len(dates) == 0:
        return pd.DatetimeIndex([])

    month_end_dates = (
        pd.Series(dates, index=dates)
        .groupby(dates.to_period("M"))
        .max()
    )

    return pd.DatetimeIndex(month_end_dates.values)


def build_evaluation_dates(
    *,
    prices: pd.DataFrame | None = None,
    evaluation_dates: Sequence[Any] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    frequency: str = "month_end",
) -> pd.DatetimeIndex:
    """Build evaluation dates for signal panels.

    Preferred input:
        prices
            Use actual month-end trading dates from the price matrix.

    Alternative input:
        evaluation_dates
            Use an explicitly supplied list of dates.

    Optional start_date and end_date filters can be applied to either source.
    """

    if prices is None and evaluation_dates is None:
        raise ValueError(
            "Provide either prices or evaluation_dates"
        )

    if prices is not None and evaluation_dates is not None:
        raise ValueError(
            "Provide either prices or evaluation_dates, not both"
        )

    if prices is not None:
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame")

        if prices.empty:
            raise ValueError("prices is empty")

        if frequency != "month_end":
            raise ValueError(
                "Only frequency='month_end' is supported when prices are used"
            )

        dates = get_month_end_trading_dates(prices.index)

    else:
        dates = pd.DatetimeIndex(
            pd.to_datetime(evaluation_dates, errors="coerce")
        )

        dates = dates.dropna().sort_values().unique()
        dates = pd.DatetimeIndex(dates)

    if start_date is not None:
        start = pd.Timestamp(start_date)
        dates = dates[dates >= start]

    if end_date is not None:
        end = pd.Timestamp(end_date)
        dates = dates[dates <= end]

    return pd.DatetimeIndex(dates).sort_values()


# -----------------------------------------------------------------------------
# Raw point-in-time panel construction
# -----------------------------------------------------------------------------


def build_signal_panel(
    signal_table: pd.DataFrame,
    *,
    signal_col: str,
    prices: pd.DataFrame | None = None,
    evaluation_dates: Sequence[Any] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    availability_lag_days: int = 1,
) -> pd.DataFrame:
    """Build one raw point-in-time signal panel.

    Output:
        index:
            evaluation dates, usually month-end trading dates.

        columns:
            tickers.

        values:
            latest known filing signal for each ticker as of that evaluation
            date.

    The function does not cross-sectionally normalize values. Use
    build_alpha_panel(...) for normalized alpha panels.
    """

    table = clean_signal_table(
        signal_table,
        required_signal_columns=[signal_col],
    )

    table = add_signal_available_date(
        table,
        availability_lag_days=availability_lag_days,
    )

    dates = build_evaluation_dates(
        prices=prices,
        evaluation_dates=evaluation_dates,
        start_date=start_date,
        end_date=end_date,
    )

    if len(dates) == 0:
        return pd.DataFrame()

    # Keep only rows with usable numeric signal values.
    table = table.dropna(
        subset=[
            signal_col,
            "signal_available_date",
        ]
    )

    if table.empty:
        return pd.DataFrame(
            index=dates,
        )

    # Sort so duplicate ticker/date rows keep the latest filing/accession.
    table = table.sort_values(
        [
            "signal_available_date",
            "ticker",
            "filing_date",
            "accession_number",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    )

    # If two filings for the same ticker become available on the same date,
    # keep the later row after sorting. This is rare, but the rule is explicit.
    table = table.drop_duplicates(
        subset=[
            "signal_available_date",
            "ticker",
        ],
        keep="last",
    )

    event_panel = table.pivot(
        index="signal_available_date",
        columns="ticker",
        values=signal_col,
    )

    event_panel = event_panel.sort_index()
    event_panel.columns = normalize_ticker_index(event_panel.columns)

    # Reindex to the union of event dates and evaluation dates so forward-fill
    # can carry filing-event values to month-end evaluation dates.
    combined_index = event_panel.index.union(dates).sort_values()

    panel = (
        event_panel
        .reindex(combined_index)
        .ffill()
        .reindex(dates)
    )

    panel.index.name = "date"
    panel.columns.name = None

    panel = panel.sort_index()
    panel = panel.reindex(
        columns=sorted(panel.columns)
    )

    return panel


def build_signal_panels(
    signal_table: pd.DataFrame,
    *,
    signal_cols: Sequence[str] | None = None,
    prices: pd.DataFrame | None = None,
    evaluation_dates: Sequence[Any] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    availability_lag_days: int = 1,
) -> dict[str, pd.DataFrame]:
    """Build raw point-in-time panels for several signal columns."""

    if signal_cols is None:
        signal_cols = [
            column
            for column in DEFAULT_SIGNAL_COLUMNS
            if column in signal_table.columns
        ]

    if not signal_cols:
        raise ValueError("No signal columns were supplied or found")

    panels: dict[str, pd.DataFrame] = {}

    for signal_col in signal_cols:
        panels[signal_col] = build_signal_panel(
            signal_table,
            signal_col=signal_col,
            prices=prices,
            evaluation_dates=evaluation_dates,
            start_date=start_date,
            end_date=end_date,
            availability_lag_days=availability_lag_days,
        )

    return panels


# -----------------------------------------------------------------------------
# Alpha-panel construction
# -----------------------------------------------------------------------------


def _resolve_alpha_builder(
    alpha_builder: Callable[[pd.Series], pd.Series] | None,
) -> Callable[[pd.Series], pd.Series]:
    """Return an injected alpha builder or the project default."""

    if alpha_builder is not None:
        return alpha_builder

    # Import lazily so tests can inject a simple fake builder.
    from .alpha_model import build_alpha

    return build_alpha


def build_alpha_panel(
    raw_panel: pd.DataFrame,
    *,
    alpha_builder: Callable[[pd.Series], pd.Series] | None = None,
    min_non_null: int = 2,
) -> pd.DataFrame:
    """Cross-sectionally normalize a raw signal panel by date.

    For each evaluation date:
        1. take the available ticker signals,
        2. pass them through alpha_model.build_alpha(...),
        3. put the normalized values back into the same date-by-ticker shape.

    Missing signals remain missing.
    """

    if not isinstance(raw_panel, pd.DataFrame):
        raise TypeError("raw_panel must be a pandas DataFrame")

    if int(min_non_null) < 1:
        raise ValueError("min_non_null must be at least 1")

    alpha_builder = _resolve_alpha_builder(alpha_builder)

    panel = raw_panel.copy()
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    panel.columns = normalize_ticker_index(panel.columns)

    alpha_panel = pd.DataFrame(
        index=panel.index,
        columns=panel.columns,
        dtype=float,
    )

    for date, row in panel.iterrows():
        available = pd.to_numeric(
            row,
            errors="coerce",
        ).dropna()

        if len(available) < int(min_non_null):
            continue

        alpha_values = alpha_builder(available)

        alpha_values = pd.to_numeric(
            alpha_values,
            errors="coerce",
        )

        alpha_panel.loc[date, alpha_values.index] = alpha_values

    alpha_panel.index.name = "date"

    return alpha_panel


def build_alpha_panels(
    raw_panels: dict[str, pd.DataFrame],
    *,
    alpha_builder: Callable[[pd.Series], pd.Series] | None = None,
    min_non_null: int = 2,
    suffix: str = "_alpha",
) -> dict[str, pd.DataFrame]:
    """Build normalized alpha panels from raw signal panels."""

    alpha_panels: dict[str, pd.DataFrame] = {}

    for name, raw_panel in raw_panels.items():
        alpha_panels[f"{name}{suffix}"] = build_alpha_panel(
            raw_panel,
            alpha_builder=alpha_builder,
            min_non_null=min_non_null,
        )

    return alpha_panels


# -----------------------------------------------------------------------------
# Coverage and stale-signal diagnostics
# -----------------------------------------------------------------------------


def compute_panel_coverage(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Compute simple coverage statistics for a signal panel."""

    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")

    if panel.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "n_available",
                "n_total",
                "coverage_rate",
            ]
        )

    n_available = panel.notna().sum(axis=1)
    n_total = pd.Series(
        len(panel.columns),
        index=panel.index,
    )

    coverage = pd.DataFrame(
        {
            "date": panel.index,
            "n_available": n_available.astype(int).values,
            "n_total": n_total.astype(int).values,
            "coverage_rate": (n_available / n_total).values,
        }
    )

    return coverage.reset_index(drop=True)


def build_signal_age_panel(
    signal_table: pd.DataFrame,
    *,
    prices: pd.DataFrame | None = None,
    evaluation_dates: Sequence[Any] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    availability_lag_days: int = 1,
) -> pd.DataFrame:
    """Build a panel of signal age in calendar days.

    The age is:

        evaluation date - latest signal availability date

    This helps distinguish a fresh filing signal from an old carried-forward
    filing signal.
    """

    table = clean_signal_table(signal_table)

    table = add_signal_available_date(
        table,
        availability_lag_days=availability_lag_days,
    )

    dates = build_evaluation_dates(
        prices=prices,
        evaluation_dates=evaluation_dates,
        start_date=start_date,
        end_date=end_date,
    )

    if len(dates) == 0:
        return pd.DataFrame()

    table = table.dropna(
        subset=[
            "signal_available_date",
        ]
    )

    if table.empty:
        return pd.DataFrame(
            index=dates,
        )

    table = table.sort_values(
        [
            "signal_available_date",
            "ticker",
            "filing_date",
            "accession_number",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    )

    table = table.drop_duplicates(
        subset=[
            "signal_available_date",
            "ticker",
        ],
        keep="last",
    )

    table["available_date_value"] = table["signal_available_date"]

    event_panel = table.pivot(
        index="signal_available_date",
        columns="ticker",
        values="available_date_value",
    )

    event_panel = event_panel.sort_index()
    event_panel.columns = normalize_ticker_index(event_panel.columns)

    combined_index = event_panel.index.union(dates).sort_values()

    latest_available_dates = (
        event_panel
        .reindex(combined_index)
        .ffill()
        .reindex(dates)
    )

    age_panel = pd.DataFrame(
        index=latest_available_dates.index,
        columns=latest_available_dates.columns,
        dtype=float,
    )

    for date, row in latest_available_dates.iterrows():
        for ticker, available_date in row.items():
            if pd.isna(available_date):
                age_panel.loc[date, ticker] = np.nan

            else:
                age_panel.loc[date, ticker] = (
                    pd.Timestamp(date) - pd.Timestamp(available_date)
                ).days

    age_panel.index.name = "date"

    return age_panel


def summarize_signal_age(
    age_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize signal age by evaluation date."""

    if not isinstance(age_panel, pd.DataFrame):
        raise TypeError("age_panel must be a pandas DataFrame")

    if age_panel.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "mean_age_days",
                "median_age_days",
                "max_age_days",
            ]
        )

    summary = pd.DataFrame(
        {
            "date": age_panel.index,
            "mean_age_days": age_panel.mean(axis=1, skipna=True).values,
            "median_age_days": age_panel.median(axis=1, skipna=True).values,
            "max_age_days": age_panel.max(axis=1, skipna=True).values,
        }
    )

    return summary


# -----------------------------------------------------------------------------
# Loading and saving panels
# -----------------------------------------------------------------------------


def load_signal_table_pickle(
    path: str | Path,
) -> pd.DataFrame:
    """Load a flat SEC signal table from a pickle file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    table = pd.read_pickle(path)

    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            "signal table pickle must contain a pandas DataFrame"
        )

    return table


def save_panel_pickle(
    panel: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Save one panel DataFrame to a pickle file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    panel.to_pickle(path)

    return path


def save_panel_dict_pickle(
    panels: dict[str, pd.DataFrame],
    path: str | Path,
) -> Path:
    """Save a dictionary of panel DataFrames to one pickle file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pd.to_pickle(panels, path)

    return path


def load_panel_dict_pickle(
    path: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load and validate a dictionary of panel DataFrames."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    panels = pd.read_pickle(path)

    if not isinstance(panels, dict):
        raise TypeError("panel dictionary pickle must contain a dict")

    for name, panel in panels.items():
        if not isinstance(name, str):
            raise TypeError("panel dictionary keys must be strings")

        if not isinstance(panel, pd.DataFrame):
            raise TypeError(
                f"panel dictionary value for {name!r} is not a DataFrame"
            )

    return panels


__all__ = [
    "DEFAULT_SIGNAL_COLUMNS",
    "normalize_ticker_label",
    "normalize_ticker_index",
    "clean_signal_table",
    "signal_available_date_from_row",
    "add_signal_available_date",
    "get_month_end_trading_dates",
    "build_evaluation_dates",
    "build_signal_panel",
    "build_signal_panels",
    "build_alpha_panel",
    "build_alpha_panels",
    "compute_panel_coverage",
    "build_signal_age_panel",
    "summarize_signal_age",
    "load_signal_table_pickle",
    "save_panel_pickle",
    "save_panel_dict_pickle",
    "load_panel_dict_pickle",
]
