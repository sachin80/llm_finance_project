# Import numpy for NaN checks and simple fake data creation.
import numpy as np

# Import pandas for Series, DataFrame, and dates.
import pandas as pd

# Import pytest for approximate numeric comparisons and exceptions.
import pytest

# Import the public diagnostics functions from the module being tested.
from src.signal_diagnostics import (
    build_static_alpha_from_scores,
    load_prices_for_alpha,
    prepare_static_signal_inputs,
    build_sector_map_from_universe,
    compute_forward_returns,
    compute_monthly_ic,
    compute_quantile_returns,
    compute_top_bottom_spread,
    compute_signal_turnover,
    compute_sector_exposure,
    summarize_ic,
    summarize_top_bottom_spread,
    summarize_signal_turnover,
    summarize_alpha_coverage,
    compute_alpha_diagnostic_bundle,
    compare_alpha_panels,
)


# Define a simple fake alpha builder so tests do not depend on OpenAI or project imports.
def simple_alpha_builder(series):
    """Subtract mean from raw score series."""

    # Convert the incoming scores to floats.
    series = series.astype(float)

    # Return mean-centered scores.
    return series - series.mean()


# Define a fake downloader so tests do not call Yahoo Finance.
def fake_download_prices(tickers, start_date, end_date):
    """Return deterministic fake prices for requested tickers."""

    # Create five business dates.
    dates = pd.bdate_range(start_date, periods=5)

    # Create simple linearly increasing prices for each ticker.
    prices = pd.DataFrame(
        {
            ticker: np.linspace(100.0, 104.0, len(dates))
            for ticker in tickers
        },
        index=dates,
    )

    # Return fake adjusted-close prices.
    return prices


def test_build_static_alpha_from_scores_uses_alpha_builder():
    """Alpha helper should use the provided alpha builder."""

    # Build a small score DataFrame like the project output.
    df = pd.DataFrame(
        {
            "Ticker": ["a", "b", "c"],
            "embedding_similarity": [1.0, 2.0, 3.0],
        }
    )

    # Build alpha using the fake builder.
    alpha = build_static_alpha_from_scores(
        df=df,
        score_col="embedding_similarity",
        alpha_builder=simple_alpha_builder,
    )

    # Check ticker normalization.
    assert list(alpha.index) == ["A", "B", "C"]

    # Check mean-centered scores.
    assert alpha.loc["A"] == pytest.approx(-1.0)
    assert alpha.loc["B"] == pytest.approx(0.0)
    assert alpha.loc["C"] == pytest.approx(1.0)


def test_load_prices_for_alpha_uses_download_func():
    """Price helper should use the provided downloader."""

    # Create a static alpha Series.
    alpha = pd.Series({"A": 1.0, "B": 2.0})

    # Load prices using the fake downloader.
    prices = load_prices_for_alpha(
        alpha=alpha,
        start_date="2024-01-01",
        end_date="2024-01-10",
        download_func=fake_download_prices,
    )

    # Check ticker columns.
    assert list(prices.columns) == ["A", "B"]

    # Check number of fake price rows.
    assert len(prices) == 5


def test_prepare_static_signal_inputs_uses_project_style_hooks():
    """Preparation helper should build alpha and align prices."""

    # Build small project-style score DataFrame.
    df = pd.DataFrame(
        {
            "Ticker": ["a", "b", "c"],
            "embedding_similarity": [1.0, 2.0, 3.0],
        }
    )

    # Build small price matrix.
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0],
            "B": [100.0, 102.0],
            "C": [100.0, 103.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    # Prepare aligned alpha and prices.
    alpha, prices = prepare_static_signal_inputs(
        df=df,
        score_col="embedding_similarity",
        prices=prices,
        alpha_builder=simple_alpha_builder,
    )

    # Check alpha ticker order.
    assert list(alpha.index) == ["A", "B", "C"]

    # Check price ticker order.
    assert list(prices.columns) == ["A", "B", "C"]


def test_build_sector_map_from_universe_finds_gics_sector():
    """Sector helper should find common GICS sector column."""

    # Create small universe DataFrame.
    universe = pd.DataFrame(
        {
            "Ticker": ["A", "B"],
            "GICS Sector": ["Technology", "Health Care"],
        }
    )

    # Build sector map.
    sector_map = build_sector_map_from_universe(universe)

    # Check sector labels.
    assert sector_map.loc["A"] == "Technology"
    assert sector_map.loc["B"] == "Health Care"


def test_compute_forward_returns_one_and_two_day_horizons():
    """Forward returns should use future prices divided by today's prices."""

    # Create five business dates.
    dates = pd.bdate_range("2024-01-01", periods=5)

    # Create simple price matrix.
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0, 133.1, 146.41],
            "B": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )

    # Compute one-day and two-day forward returns.
    forward_returns = compute_forward_returns(prices, horizons=(1, 2))

    # Check A one-day return.
    assert forward_returns[1].loc[dates[0], "A"] == pytest.approx(0.10)

    # Check A two-day return.
    assert forward_returns[2].loc[dates[0], "A"] == pytest.approx(0.21)

    # Check flat B return.
    assert forward_returns[1].loc[dates[0], "B"] == pytest.approx(0.0)

    # Check unavailable final forward return.
    assert np.isnan(forward_returns[1].loc[dates[-1], "A"])


def test_compute_monthly_ic_static_alpha_spearman():
    """Monthly IC should correlate alpha with future returns."""

    # Create static alpha.
    alpha = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})

    # Create two month-end dates.
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])

    # Create forward returns that first rise then fall with alpha.
    forward_returns = {
        21: pd.DataFrame(
            {
                "A": [0.01, 0.03],
                "B": [0.02, 0.02],
                "C": [0.03, 0.01],
            },
            index=dates,
        )
    }

    # Compute rank IC.
    ic = compute_monthly_ic(alpha, forward_returns, method="spearman")

    # Check column label.
    assert list(ic.columns) == ["21D"]

    # Check positive rank IC in January.
    assert ic.loc[pd.Timestamp("2024-01-31"), "21D"] == pytest.approx(1.0)

    # Check negative rank IC in February.
    assert ic.loc[pd.Timestamp("2024-02-29"), "21D"] == pytest.approx(-1.0)


def test_compute_quantile_returns_static_alpha():
    """Quantile returns should average returns inside alpha buckets."""

    # Create five-name alpha.
    alpha = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0})

    # Create one evaluation date.
    date = pd.Timestamp("2024-01-31")

    # Create forward returns increasing with alpha.
    forward_returns = {
        21: pd.DataFrame(
            {
                "A": [0.01],
                "B": [0.02],
                "C": [0.03],
                "D": [0.04],
                "E": [0.05],
            },
            index=[date],
        )
    }

    # Compute five quantile returns.
    quantile_returns = compute_quantile_returns(
        alpha=alpha,
        forward_returns=forward_returns,
        n_quantiles=5,
    )

    # Check one row per quantile.
    assert len(quantile_returns) == 5

    # Extract bottom quantile return.
    q1 = quantile_returns.loc[
        quantile_returns["quantile"] == 1,
        "mean_forward_return",
    ].iloc[0]

    # Extract top quantile return.
    q5 = quantile_returns.loc[
        quantile_returns["quantile"] == 5,
        "mean_forward_return",
    ].iloc[0]

    # Check bottom quantile return.
    assert q1 == pytest.approx(0.01)

    # Check top quantile return.
    assert q5 == pytest.approx(0.05)


def test_compute_top_bottom_spread():
    """Top-bottom spread should subtract bottom quantile from top quantile."""

    # Create two evaluation dates.
    date_1 = pd.Timestamp("2024-01-31")
    date_2 = pd.Timestamp("2024-02-29")

    # Create quantile-return table.
    quantile_returns = pd.DataFrame(
        {
            "date": [date_1, date_1, date_2, date_2],
            "horizon": [21, 21, 21, 21],
            "quantile": [1, 5, 1, 5],
            "mean_forward_return": [0.01, 0.05, 0.02, 0.01],
            "n_assets": [10, 10, 10, 10],
        }
    )

    # Compute spread.
    spread = compute_top_bottom_spread(quantile_returns)

    # Check spread column.
    assert list(spread.columns) == ["21D"]

    # Check positive spread.
    assert spread.loc[date_1, "21D"] == pytest.approx(0.04)

    # Check negative spread.
    assert spread.loc[date_2, "21D"] == pytest.approx(-0.01)


def test_compute_signal_turnover_static_signal_is_empty():
    """Static alpha should have no time-series turnover."""

    # Create static alpha.
    alpha = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})

    # Compute turnover.
    turnover = compute_signal_turnover(alpha)

    # Check output type.
    assert isinstance(turnover, pd.Series)

    # Check that static turnover is empty.
    assert turnover.empty


def test_compute_signal_turnover_time_varying_signal():
    """Time-varying turnover should average absolute rank changes."""

    # Create alpha history with rank reversal.
    alpha_history = pd.DataFrame(
        {
            "A": [1.0, 3.0],
            "B": [2.0, 2.0],
            "C": [3.0, 1.0],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )

    # Compute turnover.
    turnover = compute_signal_turnover(alpha_history)

    # Expected average absolute percentile-rank change.
    expected_turnover = ((2.0 / 3.0) + 0.0 + (2.0 / 3.0)) / 3.0

    # Check one turnover value after the first date.
    assert len(turnover) == 1

    # Check turnover value.
    assert turnover.loc[pd.Timestamp("2024-02-29")] == pytest.approx(expected_turnover)


def test_compute_sector_exposure_static_signal():
    """Sector exposure should summarize alpha concentration."""

    # Create static alpha.
    alpha = pd.Series({"A": 1.0, "B": -2.0, "C": 3.0})

    # Create sector map.
    sector_map = {"A": "Technology", "B": "Technology", "C": "Health Care"}

    # Compute sector exposure.
    exposure = compute_sector_exposure(alpha, sector_map)

    # Check sector counts.
    assert exposure.loc["Technology", "n_assets"] == 2
    assert exposure.loc["Health Care", "n_assets"] == 1

    # Check absolute alpha shares.
    assert exposure.loc["Technology", "abs_alpha_share"] == pytest.approx(0.5)
    assert exposure.loc["Health Care", "abs_alpha_share"] == pytest.approx(0.5)

    # Check Technology mean alpha.
    assert exposure.loc["Technology", "mean_alpha"] == pytest.approx(-0.5)


def test_summarize_ic_returns_one_row_per_horizon():
    """IC summary should return one row per horizon."""

    # Create small monthly IC table.
    monthly_ic = pd.DataFrame(
        {
            "21D": [0.10, 0.20, -0.10, 0.30],
            "63D": [0.05, 0.00, 0.10, 0.20],
        },
        index=pd.date_range("2024-01-31", periods=4, freq="ME"),
    )

    # Summarize IC.
    summary = summarize_ic(monthly_ic)

    # Check horizons.
    assert set(summary["horizon"]) == {"21D", "63D"}

    # Check expected columns.
    assert "mean_ic" in summary.columns
    assert "t_stat" in summary.columns
    assert "method" in summary.columns



def make_monthly_test_prices():
    """Create deterministic business-day prices for diagnostic comparison tests."""

    # Create enough business dates to have several monthly evaluation points.
    dates = pd.bdate_range("2024-01-01", "2024-04-30")

    # Create a time counter.
    t = np.arange(len(dates), dtype=float)

    # Build three simple positive price paths.
    prices = pd.DataFrame(
        {
            "A": 100.0 + 0.30 * t,
            "B": 100.0 + 0.15 * t,
            "C": 100.0 + 0.05 * t,
        },
        index=dates,
    )

    # Return deterministic prices.
    return prices


def test_summarize_top_bottom_spread_returns_one_row_per_horizon():
    """Spread summary should report mean spread and positive spread rate."""

    # Create small top-minus-bottom spread table.
    spread = pd.DataFrame(
        {
            "21D": [0.10, -0.05, 0.20],
            "63D": [0.02, 0.03, 0.04],
        },
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )

    # Summarize spread.
    summary = summarize_top_bottom_spread(spread)

    # Check horizons.
    assert set(summary["horizon"]) == {"21D", "63D"}

    # Extract 21D row.
    row_21d = summary.loc[summary["horizon"] == "21D"].iloc[0]

    # Check mean spread.
    assert row_21d["mean_spread"] == pytest.approx((0.10 - 0.05 + 0.20) / 3.0)

    # Check positive spread rate.
    assert row_21d["positive_spread_rate"] == pytest.approx(2.0 / 3.0)


def test_summarize_signal_turnover_handles_empty_static_turnover():
    """Empty turnover Series should produce an explicit empty-style summary."""

    # Create empty turnover Series like static alpha produces.
    turnover = pd.Series(dtype=float, name="signal_turnover")

    # Summarize turnover.
    summary = summarize_signal_turnover(turnover)

    # Check empty summary fields.
    assert summary.loc[0, "n_turnover_periods"] == 0
    assert np.isnan(summary.loc[0, "mean_turnover"])


def test_summarize_alpha_coverage_static_and_time_varying():
    """Coverage summary should support both Series and DataFrame alpha inputs."""

    # Create static alpha.
    static_alpha = pd.Series({"a": 1.0, "b": 2.0, "c": np.nan})

    # Summarize static coverage.
    static_summary = summarize_alpha_coverage(static_alpha)

    # Check static coverage.
    assert static_summary.loc[0, "signal_type"] == "static"
    assert static_summary.loc[0, "n_tickers"] == 2
    assert static_summary.loc[0, "mean_coverage_rate"] == pytest.approx(1.0)

    # Create time-varying alpha.
    alpha_panel = pd.DataFrame(
        {
            "a": [1.0, 1.0],
            "b": [np.nan, 2.0],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )

    # Summarize time-varying coverage.
    panel_summary = summarize_alpha_coverage(alpha_panel)

    # Check time-varying coverage.
    assert panel_summary.loc[0, "signal_type"] == "time_varying"
    assert panel_summary.loc[0, "n_dates"] == 2
    assert panel_summary.loc[0, "n_tickers"] == 2
    assert panel_summary.loc[0, "mean_coverage_rate"] == pytest.approx(0.75)


def test_compute_alpha_diagnostic_bundle_returns_expected_tables():
    """Single-signal bundle should expose all diagnostic components."""

    # Create alpha that ranks C highest, then B, then A.
    alpha = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})

    # Create deterministic prices.
    prices = make_monthly_test_prices()

    # Compute one diagnostic bundle using a short test horizon.
    bundle = compute_alpha_diagnostic_bundle(
        alpha=alpha,
        prices=prices,
        horizons=(1,),
        n_quantiles=3,
    )

    # Check expected bundle keys.
    assert "monthly_ic" in bundle
    assert "ic_summary" in bundle
    assert "quantile_returns" in bundle
    assert "top_bottom_spread" in bundle
    assert "spread_summary" in bundle
    assert "turnover_summary" in bundle
    assert "coverage_summary" in bundle

    # Check one horizon label.
    assert list(bundle["monthly_ic"].columns) == ["1D"]

    # Static alpha should have empty-style turnover summary.
    assert bundle["turnover_summary"].loc[0, "n_turnover_periods"] == 0


def test_compare_alpha_panels_compares_static_and_time_varying_signals():
    """Comparison helper should summarize several signals side by side."""

    # Create deterministic prices.
    prices = make_monthly_test_prices()

    # Create static alpha.
    static_alpha = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})

    # Create time-varying alpha panel on month-end dates.
    panel_alpha = pd.DataFrame(
        {
            "A": [1.0, 1.5],
            "B": [2.0, 2.0],
            "C": [3.0, 2.5],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )

    # Create sector map.
    sector_map = {
        "A": "Technology",
        "B": "Technology",
        "C": "Health Care",
    }

    # Compare the two alpha inputs.
    result = compare_alpha_panels(
        signal_panels={
            "static_signal": static_alpha,
            "sec_panel_signal": panel_alpha,
        },
        prices=prices,
        sector_map=sector_map,
        horizons=(1,),
        n_quantiles=3,
    )

    # Check result keys.
    assert "summary" in result
    assert "bundles" in result
    assert "ic_summary" in result
    assert "spread_summary" in result
    assert "turnover_summary" in result
    assert "coverage_summary" in result

    # Check both signals are present in the summary.
    assert set(result["summary"]["signal_name"]) == {
        "static_signal",
        "sec_panel_signal",
    }

    # Check comparison columns needed for inspection.
    assert "mean_ic" in result["summary"].columns
    assert "mean_spread" in result["summary"].columns
    assert "mean_turnover" in result["summary"].columns
    assert "mean_coverage_rate" in result["summary"].columns

    # Check full bundles are preserved.
    assert set(result["bundles"].keys()) == {
        "static_signal",
        "sec_panel_signal",
    }

    # Check sector exposure exists for the time-varying signal.
    assert not result["bundles"]["sec_panel_signal"]["sector_exposure"].empty
