"""
Reusable plotting and diagnostic helpers for static Yahoo alpha signals and
point-in-time SEC filing alpha panels.

Run from the project root, for example:

    python src/signal_diagnostic_plots.py \
        --scored-universe outputs/scored_universe.pkl \
        --prices outputs/prices.pkl \
        --sec-alpha-panels outputs/sec_pit_openai/panels/sec_alpha_signal_panels.pkl \
        --universe data/SP500.xlsx \
        --output-dir outputs/signal_diagnostics/static_vs_sec_pit

The script writes:
    output_dir/csv/*.csv
    output_dir/plots/*.png

Design notes:
    - Static signals are passed as pandas Series indexed by ticker.
    - PIT signals are passed as pandas DataFrames indexed by evaluation date and
      with tickers as columns.
    - The same rank-IC, quantile return, top-minus-bottom spread, coverage, and
      sector-concentration plots are generated for both.
    - No SEC filings or OpenAI summary calls are made here.
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SignalLike = Union[pd.Series, pd.DataFrame]


DEFAULT_HORIZONS = (21, 63)
DEFAULT_N_QUANTILES = 5

STATIC_SIGNAL_COLUMNS = {
    "static_embedding": "alpha_embedding",
    "static_tfidf": "alpha_tfidf",
}

SEC_PANEL_KEYS = {
    "sec_summary_similarity": "summary_theme_similarity_alpha",
    "sec_exposure_score": "exposure_label_score_alpha",
    "sec_direction_score": "direction_score_alpha",
    "sec_evidence_strength": "evidence_strength_score_alpha",
    "sec_similarity_x_exposure": "similarity_x_exposure_score_alpha",
    "sec_similarity_x_evidence": "similarity_x_evidence_strength_score_alpha",
    "sec_similarity_x_direction": "similarity_x_direction_score_alpha",
    "sec_direct_or_indirect": "direct_or_indirect_similarity_alpha",
}


def normalize_ticker(value: object) -> str:
    """Normalize ticker strings to the project convention."""
    return str(value).strip().upper()


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a DatetimeIndex sorted increasingly."""
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def load_pickle(path: Union[str, Path]):
    """Load a pickle file."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_dataframe(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Save a DataFrame to CSV and pickle using the same stem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=True)
    df.to_pickle(path.with_suffix(".pkl"))


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Clean a wide adjusted-price matrix indexed by date and with tickers as columns."""
    out = ensure_datetime_index(prices)
    out.columns = [normalize_ticker(c) for c in out.columns]
    out = out.loc[:, ~out.columns.duplicated()]
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna(how="all")
    return out


def load_prices(path: Union[str, Path]) -> pd.DataFrame:
    """Load and clean a price matrix."""
    return clean_prices(pd.read_pickle(path))


def load_universe(path: Optional[Union[str, Path]], sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load a universe file from Excel, CSV, or pickle."""
    if path is None:
        return pd.DataFrame()

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)

    raise ValueError(f"Unsupported universe file type: {path}")


def build_sector_map(universe_df: pd.DataFrame) -> Dict[str, str]:
    """Build ticker -> sector mapping from a universe DataFrame."""
    if universe_df.empty:
        return {}

    ticker_col = None
    for candidate in ("Ticker", "ticker", "Symbol", "symbol"):
        if candidate in universe_df.columns:
            ticker_col = candidate
            break

    sector_col = None
    for candidate in ("Sector", "sector", "GICS Sector", "gics_sector"):
        if candidate in universe_df.columns:
            sector_col = candidate
            break

    if ticker_col is None or sector_col is None:
        return {}

    sector_df = universe_df[[ticker_col, sector_col]].dropna()
    return {
        normalize_ticker(row[ticker_col]): str(row[sector_col])
        for _, row in sector_df.iterrows()
    }


def load_static_signals(scored_universe_path: Optional[Union[str, Path]]) -> Dict[str, pd.Series]:
    """Load static alpha Series from scored_universe.pkl."""
    if scored_universe_path is None:
        return {}

    scored = pd.read_pickle(scored_universe_path).copy()
    if "Ticker" not in scored.columns:
        raise ValueError("scored_universe file must contain a 'Ticker' column.")

    scored["Ticker"] = scored["Ticker"].map(normalize_ticker)
    scored = scored.drop_duplicates("Ticker").set_index("Ticker")

    signals: Dict[str, pd.Series] = {}

    for signal_name, column in STATIC_SIGNAL_COLUMNS.items():
        if column in scored.columns:
            signal = pd.to_numeric(scored[column], errors="coerce").dropna()
            signal.index = signal.index.map(normalize_ticker)
            signals[signal_name] = signal

    return signals


def load_sec_alpha_panels(sec_alpha_panels_path: Optional[Union[str, Path]]) -> Dict[str, pd.DataFrame]:
    """Load SEC PIT alpha panels saved by sec_panel.py."""
    if sec_alpha_panels_path is None:
        return {}

    panels = load_pickle(sec_alpha_panels_path)
    if not isinstance(panels, Mapping):
        raise ValueError("SEC alpha panels pickle must contain a dict-like object.")

    signals: Dict[str, pd.DataFrame] = {}

    for signal_name, panel_key in SEC_PANEL_KEYS.items():
        if panel_key in panels:
            panel = panels[panel_key].copy()
            panel = ensure_datetime_index(panel)
            panel.columns = [normalize_ticker(c) for c in panel.columns]
            panel = panel.loc[:, ~panel.columns.duplicated()]
            signals[signal_name] = panel

    return signals


def month_end_trading_dates(
    prices: pd.DataFrame,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DatetimeIndex:
    """Return actual month-end trading dates from the price index."""
    px = prices.copy()

    if start_date is not None:
        px = px.loc[px.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        px = px.loc[px.index <= pd.Timestamp(end_date)]

    if px.empty:
        raise ValueError("No price data available after start/end filtering.")

    # Resample to calendar month end, then take the last observed trading date in each month.
    date_series = pd.Series(px.index, index=px.index)
    dates = date_series.resample("ME").last().dropna()
    return pd.DatetimeIndex(dates.values)


def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Simple forward returns over horizon trading rows."""
    return prices.shift(-horizon).div(prices).sub(1.0)


def signal_to_panel(
    signal: SignalLike,
    evaluation_dates: pd.DatetimeIndex,
    tickers: Sequence[str],
) -> pd.DataFrame:
    """Convert static Series or PIT DataFrame signal into date-by-ticker panel."""
    tickers = [normalize_ticker(t) for t in tickers]

    if isinstance(signal, pd.Series):
        s = signal.copy()
        s.index = s.index.map(normalize_ticker)
        row = s.reindex(tickers)
        panel = pd.DataFrame(
            np.tile(row.to_numpy(), (len(evaluation_dates), 1)),
            index=evaluation_dates,
            columns=tickers,
        )
        return panel

    if isinstance(signal, pd.DataFrame):
        panel = ensure_datetime_index(signal)
        panel.columns = [normalize_ticker(c) for c in panel.columns]
        panel = panel.loc[:, ~panel.columns.duplicated()]
        panel = panel.reindex(columns=tickers)
        # Reindex to actual evaluation dates and carry the last known PIT signal forward.
        return panel.reindex(evaluation_dates).ffill()

    raise TypeError(f"Unsupported signal type: {type(signal)}")


def _valid_cross_section(
    signal_row: pd.Series,
    forward_return_row: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """Align signal and return rows and drop missing observations."""
    df = pd.concat(
        [
            pd.to_numeric(signal_row, errors="coerce").rename("signal"),
            pd.to_numeric(forward_return_row, errors="coerce").rename("forward_return"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()

    return df["signal"], df["forward_return"]


def compute_rank_ic(
    signal_panel: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int,
    method: str = "spearman",
) -> pd.Series:
    """Compute monthly cross-sectional IC time series."""
    fwd = forward_returns(prices, horizon)
    values = {}

    for date in signal_panel.index:
        if date not in fwd.index:
            continue
        x, y = _valid_cross_section(signal_panel.loc[date], fwd.loc[date])
        if len(x) < 5 or x.nunique() < 2 or y.nunique() < 2:
            values[date] = np.nan
        else:
            values[date] = x.corr(y, method=method)

    return pd.Series(values, name=f"ic_{horizon}d").sort_index()


def assign_quantiles(scores: pd.Series, n_quantiles: int) -> pd.Series:
    """Assign quantiles 1..n using ranks to avoid duplicate-edge qcut failures."""
    scores = pd.to_numeric(scores, errors="coerce").dropna()
    if scores.shape[0] < n_quantiles:
        return pd.Series(index=scores.index, dtype=float)

    ranked = scores.rank(method="first")
    try:
        q = pd.qcut(ranked, q=n_quantiles, labels=False) + 1
    except ValueError:
        return pd.Series(index=scores.index, dtype=float)

    return q.astype(float)


def compute_quantile_returns(
    signal_panel: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int,
    n_quantiles: int = DEFAULT_N_QUANTILES,
) -> pd.DataFrame:
    """Compute average forward return by signal quantile for each evaluation date."""
    fwd = forward_returns(prices, horizon)
    rows = []

    for date in signal_panel.index:
        if date not in fwd.index:
            continue

        x, y = _valid_cross_section(signal_panel.loc[date], fwd.loc[date])
        if len(x) < n_quantiles:
            continue

        q = assign_quantiles(x, n_quantiles=n_quantiles)
        df = pd.concat([q.rename("quantile"), y.rename("forward_return")], axis=1).dropna()

        if df["quantile"].nunique() < n_quantiles:
            continue

        row = df.groupby("quantile")["forward_return"].mean()
        row.name = date
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(range(1, n_quantiles + 1)), dtype=float)

    out = pd.DataFrame(rows).sort_index()
    out.columns = [int(c) for c in out.columns]
    return out.reindex(columns=list(range(1, n_quantiles + 1)))


def compute_top_minus_bottom_spread(quantile_returns: pd.DataFrame) -> pd.Series:
    """Compute top quantile minus bottom quantile return spread."""
    if quantile_returns.empty:
        return pd.Series(dtype=float, name="top_minus_bottom")

    bottom = quantile_returns.columns.min()
    top = quantile_returns.columns.max()

    spread = quantile_returns[top] - quantile_returns[bottom]
    spread.name = "top_minus_bottom"
    return spread


def _normal_two_sided_pvalue(t_stat: float) -> float:
    """Two-sided normal p-value with scipy fallback-free implementation."""
    if not np.isfinite(t_stat):
        return np.nan
    return math.erfc(abs(float(t_stat)) / math.sqrt(2.0))


def newey_west_se_of_mean(series: pd.Series, lags: int = 3) -> float:
    """Newey-West standard error for the sample mean."""
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    n = x.shape[0]

    if n <= 1:
        return np.nan

    e = x - x.mean()
    max_lag = min(int(lags), n - 1)

    gamma0 = np.mean(e * e)
    long_run_var = gamma0

    for lag in range(1, max_lag + 1):
        gamma = np.mean(e[lag:] * e[:-lag])
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_var += 2.0 * weight * gamma

    long_run_var = max(long_run_var, 0.0)
    return math.sqrt(long_run_var / n)


def summarize_series(series: pd.Series, nw_lags: int = 3) -> Dict[str, float]:
    """Summarize a monthly series with HAC inference for its mean."""
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = int(x.shape[0])

    if n == 0:
        return {
            "n_months": 0,
            "mean": np.nan,
            "std": np.nan,
            "hac_se": np.nan,
            "hac_t": np.nan,
            "hac_p": np.nan,
            "positive_rate": np.nan,
        }

    mean = float(x.mean())
    se = newey_west_se_of_mean(x, lags=nw_lags)
    t_stat = mean / se if se and np.isfinite(se) and se > 0 else np.nan

    return {
        "n_months": n,
        "mean": mean,
        "std": float(x.std(ddof=1)) if n > 1 else np.nan,
        "hac_se": se,
        "hac_t": t_stat,
        "hac_p": _normal_two_sided_pvalue(t_stat),
        "positive_rate": float((x > 0).mean()),
    }


def compute_sector_abs_alpha_share(
    signal_panel: pd.DataFrame,
    sector_map: Mapping[str, str],
) -> pd.Series:
    """Average sector share of total absolute alpha over time."""
    if not sector_map:
        return pd.Series(dtype=float, name="abs_alpha_share")

    shares_by_date = []

    for date, row in signal_panel.iterrows():
        x = pd.to_numeric(row, errors="coerce").dropna().abs()
        if x.empty or x.sum() == 0:
            continue

        df = pd.DataFrame({"abs_alpha": x})
        df["sector"] = [sector_map.get(ticker) for ticker in df.index]
        df = df.dropna(subset=["sector"])

        if df.empty:
            continue

        share = df.groupby("sector")["abs_alpha"].sum() / df["abs_alpha"].sum()
        share.name = date
        shares_by_date.append(share)

    if not shares_by_date:
        return pd.Series(dtype=float, name="abs_alpha_share")

    return pd.DataFrame(shares_by_date).mean(axis=0).sort_values(ascending=False).rename("abs_alpha_share")


def compute_coverage(signal_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute non-null signal coverage by date."""
    coverage = pd.DataFrame(index=signal_panel.index)
    coverage["n_non_null"] = signal_panel.notna().sum(axis=1)
    coverage["n_columns"] = signal_panel.shape[1]
    coverage["coverage_rate"] = coverage["n_non_null"] / max(signal_panel.shape[1], 1)
    return coverage


def plot_rank_ic(
    ic_by_horizon: Mapping[int, pd.Series],
    signal_name: str,
    out_path: Union[str, Path],
) -> None:
    """Line plot of monthly rank IC by horizon."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for horizon, series in ic_by_horizon.items():
        ax.plot(series.index, series.values, label=f"{horizon}D")

    ax.axhline(0.0, linewidth=1)
    ax.set_title(f"Monthly Rank IC — {signal_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rank IC")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_quantile_returns(
    mean_quantile_returns: pd.Series,
    signal_name: str,
    horizon: int,
    out_path: Union[str, Path],
) -> None:
    """Bar plot of average forward return by quantile."""
    fig, ax = plt.subplots(figsize=(7, 5))

    mean_quantile_returns.plot(kind="bar", ax=ax)

    ax.axhline(0.0, linewidth=1)
    ax.set_title(f"{signal_name} — Average Forward Return by Quantile ({horizon}D)")
    ax.set_xlabel("Alpha quantile")
    ax.set_ylabel("Mean forward return")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_spread(
    spread_by_horizon: Mapping[int, pd.Series],
    signal_name: str,
    out_path: Union[str, Path],
) -> None:
    """Line plot of top-minus-bottom quantile spread."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for horizon, series in spread_by_horizon.items():
        ax.plot(series.index, series.values, label=f"{horizon}D")

    ax.axhline(0.0, linewidth=1)
    ax.set_title(f"Top-minus-Bottom Spread — {signal_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Q top minus Q bottom forward return")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_sector_abs_share(
    sector_share: pd.Series,
    signal_name: str,
    out_path: Union[str, Path],
) -> None:
    """Horizontal bar plot of average absolute-alpha share by sector."""
    if sector_share.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    sector_share.sort_values().plot(kind="barh", ax=ax)

    ax.set_title(f"{signal_name} — Average Absolute Alpha Share by Sector")
    ax.set_xlabel("Share of total absolute alpha")
    ax.set_ylabel("Sector")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_coverage(
    coverage: pd.DataFrame,
    signal_name: str,
    out_path: Union[str, Path],
) -> None:
    """Line plot of PIT signal coverage."""
    if coverage.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(coverage.index, coverage["n_non_null"])
    ax.set_title(f"Signal Coverage — {signal_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Non-null tickers")
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_combined_metric_bars(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    out_path: Union[str, Path],
) -> None:
    """Bar plot for one metric across signal/horizon pairs."""
    if summary.empty or metric not in summary.columns:
        return

    plot_df = summary.copy()
    plot_df["label"] = plot_df["signal"] + " " + plot_df["horizon"].astype(str) + "D"
    plot_df = plot_df.set_index("label")

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.45), 5))
    plot_df[metric].plot(kind="bar", ax=ax)

    ax.axhline(0.0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_diagnostics_for_signal(
    signal_name: str,
    signal: SignalLike,
    prices: pd.DataFrame,
    evaluation_dates: pd.DatetimeIndex,
    sector_map: Mapping[str, str],
    output_dir: Union[str, Path],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    ic_method: str = "spearman",
    nw_lags: int = 3,
) -> Dict[str, object]:
    """Run diagnostics and save CSV/PNG outputs for one signal."""
    output_dir = Path(output_dir)
    csv_dir = output_dir / "csv"
    plot_dir = output_dir / "plots"

    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    signal_panel = signal_to_panel(signal, evaluation_dates=evaluation_dates, tickers=prices.columns)

    ic_by_horizon: Dict[int, pd.Series] = {}
    quantile_returns_by_horizon: Dict[int, pd.DataFrame] = {}
    spread_by_horizon: Dict[int, pd.Series] = {}
    summary_rows = []

    for horizon in horizons:
        ic = compute_rank_ic(
            signal_panel=signal_panel,
            prices=prices,
            horizon=horizon,
            method=ic_method,
        )

        qret = compute_quantile_returns(
            signal_panel=signal_panel,
            prices=prices,
            horizon=horizon,
            n_quantiles=n_quantiles,
        )

        spread = compute_top_minus_bottom_spread(qret)

        ic_by_horizon[horizon] = ic
        quantile_returns_by_horizon[horizon] = qret
        spread_by_horizon[horizon] = spread

        ic_summary = summarize_series(ic, nw_lags=nw_lags)
        spread_summary = summarize_series(spread, nw_lags=nw_lags)

        summary_rows.append(
            {
                "signal": signal_name,
                "horizon": horizon,
                "ic_n_months": ic_summary["n_months"],
                "mean_ic": ic_summary["mean"],
                "ic_std": ic_summary["std"],
                "ic_hac_se": ic_summary["hac_se"],
                "ic_hac_t": ic_summary["hac_t"],
                "ic_hac_p": ic_summary["hac_p"],
                "positive_ic_rate": ic_summary["positive_rate"],
                "spread_n_months": spread_summary["n_months"],
                "mean_qtop_minus_qbottom_spread": spread_summary["mean"],
                "spread_std": spread_summary["std"],
                "spread_hac_se": spread_summary["hac_se"],
                "spread_hac_t": spread_summary["hac_t"],
                "spread_hac_p": spread_summary["hac_p"],
                "positive_spread_rate": spread_summary["positive_rate"],
            }
        )

        ic.to_frame("rank_ic").to_csv(csv_dir / f"{signal_name}_rank_ic_{horizon}d.csv")
        qret.to_csv(csv_dir / f"{signal_name}_quantile_returns_{horizon}d.csv")
        spread.to_frame("top_minus_bottom_spread").to_csv(
            csv_dir / f"{signal_name}_top_minus_bottom_spread_{horizon}d.csv"
        )

        if not qret.empty:
            mean_qret = qret.mean(axis=0)
            mean_qret.to_frame("mean_forward_return").to_csv(
                csv_dir / f"{signal_name}_mean_quantile_returns_{horizon}d.csv"
            )

            plot_quantile_returns(
                mean_quantile_returns=mean_qret,
                signal_name=signal_name,
                horizon=horizon,
                out_path=plot_dir / f"{signal_name}_quantile_returns_{horizon}d.png",
            )

    plot_rank_ic(
        ic_by_horizon=ic_by_horizon,
        signal_name=signal_name,
        out_path=plot_dir / f"{signal_name}_monthly_rank_ic.png",
    )

    plot_spread(
        spread_by_horizon=spread_by_horizon,
        signal_name=signal_name,
        out_path=plot_dir / f"{signal_name}_top_minus_bottom_spread.png",
    )

    sector_share = compute_sector_abs_alpha_share(signal_panel, sector_map=sector_map)
    sector_share.to_frame("abs_alpha_share").to_csv(csv_dir / f"{signal_name}_sector_abs_alpha_share.csv")
    plot_sector_abs_share(
        sector_share=sector_share,
        signal_name=signal_name,
        out_path=plot_dir / f"{signal_name}_sector_abs_alpha_share.png",
    )

    coverage = compute_coverage(signal_panel)
    coverage.to_csv(csv_dir / f"{signal_name}_coverage.csv")
    plot_coverage(
        coverage=coverage,
        signal_name=signal_name,
        out_path=plot_dir / f"{signal_name}_coverage.png",
    )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(csv_dir / f"{signal_name}_summary.csv", index=False)

    return {
        "signal_panel": signal_panel,
        "ic_by_horizon": ic_by_horizon,
        "quantile_returns_by_horizon": quantile_returns_by_horizon,
        "spread_by_horizon": spread_by_horizon,
        "sector_share": sector_share,
        "coverage": coverage,
        "summary": summary,
    }


def plot_combined_spread_timeseries(
    diagnostics: Mapping[str, Mapping[str, object]],
    horizon: int,
    out_path: Union[str, Path],
) -> None:
    """Combined top-minus-bottom spread plot for all signals at one horizon."""
    fig, ax = plt.subplots(figsize=(11, 5))

    for signal_name, result in diagnostics.items():
        spread_by_horizon = result["spread_by_horizon"]
        if horizon in spread_by_horizon:
            series = spread_by_horizon[horizon]
            ax.plot(series.index, series.values, label=signal_name)

    ax.axhline(0.0, linewidth=1)
    ax.set_title(f"Top-minus-Bottom Spread Comparison ({horizon}D)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Q top minus Q bottom forward return")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_all_diagnostics(
    signals: Mapping[str, SignalLike],
    prices: pd.DataFrame,
    sector_map: Mapping[str, str],
    output_dir: Union[str, Path],
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    selected_signal_names: Optional[Sequence[str]] = None,
    ic_method: str = "spearman",
    nw_lags: int = 3,
) -> Dict[str, Dict[str, object]]:
    """Run diagnostics for all requested signals and save combined outputs."""
    output_dir = Path(output_dir)
    csv_dir = output_dir / "csv"
    plot_dir = output_dir / "plots"

    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if selected_signal_names:
        selected = [name for name in selected_signal_names if name in signals]
    else:
        selected = list(signals.keys())

    if not selected:
        raise ValueError("No signals selected or available.")

    eval_dates = month_end_trading_dates(prices, start_date=start_date, end_date=end_date)

    diagnostics: Dict[str, Dict[str, object]] = {}

    for signal_name in selected:
        print(f"Running diagnostics for {signal_name}...")
        diagnostics[signal_name] = run_diagnostics_for_signal(
            signal_name=signal_name,
            signal=signals[signal_name],
            prices=prices,
            evaluation_dates=eval_dates,
            sector_map=sector_map,
            output_dir=output_dir,
            horizons=horizons,
            n_quantiles=n_quantiles,
            ic_method=ic_method,
            nw_lags=nw_lags,
        )

    combined_summary = pd.concat(
        [diagnostics[name]["summary"] for name in diagnostics],
        ignore_index=True,
    )
    combined_summary.to_csv(csv_dir / "combined_summary.csv", index=False)
    combined_summary.to_pickle(csv_dir / "combined_summary.pkl")

    plot_combined_metric_bars(
        summary=combined_summary,
        metric="mean_ic",
        title="Mean Rank IC Comparison",
        out_path=plot_dir / "combined_mean_rank_ic.png",
    )

    plot_combined_metric_bars(
        summary=combined_summary,
        metric="mean_qtop_minus_qbottom_spread",
        title="Mean Top-minus-Bottom Spread Comparison",
        out_path=plot_dir / "combined_mean_top_minus_bottom_spread.png",
    )

    for horizon in horizons:
        plot_combined_spread_timeseries(
            diagnostics=diagnostics,
            horizon=horizon,
            out_path=plot_dir / f"combined_top_minus_bottom_spread_{horizon}d.png",
        )

    return diagnostics


def parse_horizons(raw: str) -> Tuple[int, ...]:
    """Parse comma-separated horizons."""
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate signal diagnostic plots for static Yahoo and SEC PIT alpha signals."
    )

    parser.add_argument("--scored-universe", type=str, default=None)
    parser.add_argument("--prices", type=str, required=True)
    parser.add_argument("--sec-alpha-panels", type=str, default=None)
    parser.add_argument("--universe", type=str, default=None)
    parser.add_argument("--universe-sheet", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--start-date", type=str, default="2023-08-31")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--horizons", type=str, default="21,63")
    parser.add_argument("--n-quantiles", type=int, default=DEFAULT_N_QUANTILES)
    parser.add_argument("--signals", nargs="*", default=None)
    parser.add_argument("--ic-method", type=str, default="spearman")
    parser.add_argument("--nw-lags", type=int, default=3)

    args = parser.parse_args()

    prices = load_prices(args.prices)

    universe_df = load_universe(args.universe, sheet_name=args.universe_sheet)
    sector_map = build_sector_map(universe_df)

    signals: Dict[str, SignalLike] = {}
    signals.update(load_static_signals(args.scored_universe))
    signals.update(load_sec_alpha_panels(args.sec_alpha_panels))

    print("Available signals:")
    for name in signals:
        print(f"  - {name}")

    run_all_diagnostics(
        signals=signals,
        prices=prices,
        sector_map=sector_map,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        horizons=parse_horizons(args.horizons),
        n_quantiles=args.n_quantiles,
        selected_signal_names=args.signals,
        ic_method=args.ic_method,
        nw_lags=args.nw_lags,
    )

    print(f"Done. Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
