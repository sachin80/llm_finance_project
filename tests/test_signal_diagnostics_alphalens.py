# Import numpy for deterministic fake data and approximate calculations.
import numpy as np

# Import pandas for Series, DataFrame, and date indexes.
import pandas as pd

# Import pytest for approximate checks and optional package skipping.
import pytest

# Import the Alphalens project adapter functions.
from src.signal_diagnostics_alphalens import (
    ALPHALENS_AVAILABLE,
    build_static_alpha_from_scores,
    load_prices_for_alpha,
    prepare_static_signal_inputs,
    build_sector_map_from_universe,
    make_static_factor_for_alphalens,
    build_factor_data_from_forward_returns,
    compute_forward_returns,
    compute_monthly_ic,
    compute_quantile_returns,
    compute_top_bottom_spread,
    compute_signal_turnover,
    compute_sector_exposure,
)


def simple_alpha_builder(series):
    """Fake alpha builder used to avoid importing the full project in tests."""

    # Convert raw scores to floats.
    series = series.astype(float)

    # Return mean-centered scores.
    return series - series.mean()


def fake_download_prices(tickers, start_date, end_date):
    """Fake downloader used to avoid Yahoo Finance in tests."""

    # Create five business dates.
    dates = pd.bdate_range(start_date, periods=5)

    # Build deterministic price paths.
    prices = pd.DataFrame(
        {
            ticker: np.linspace(100.0, 104.0, len(dates))
            for ticker in tickers
        },
        index=dates,
    )

    # Return fake price matrix.
    return prices


def test_build_static_alpha_from_scores_uses_alpha_builder():
    """Static alpha helper should use the project-style alpha builder hook."""

    # Create small score DataFrame.
    df = pd.DataFrame(
        {
            "Ticker": ["a", "b", "c"],
            "embedding_similarity": [1.0, 2.0, 3.0],
        }
    )

    # Build alpha.
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


def test_load_prices_for_alpha_uses_download_function():
    """Price helper should use the provided downloader."""

    # Create static alpha.
    alpha = pd.Series({"A": 1.0, "B": 2.0})

    # Load fake prices.
    prices = load_prices_for_alpha(
        alpha=alpha,
        start_date="2024-01-01",
        end_date="2024-01-10",
        download_func=fake_download_prices,
    )

    # Check columns.
    assert list(prices.columns) == ["A", "B"]

    # Check row count.
    assert len(prices) == 5


def test_prepare_static_signal_inputs_aligns_alpha_and_prices():
    """Preparation helper should build alpha and align prices."""

    # Create score DataFrame.
    df = pd.DataFrame(
        {
            "Ticker": ["a", "b", "c"],
            "tfidf_similarity": [1.0, 2.0, 3.0],
        }
    )

    # Create price matrix.
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0],
            "B": [100.0, 102.0],
            "C": [100.0, 103.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    # Prepare inputs.
    alpha, prices = prepare_static_signal_inputs(
        df=df,
        score_col="tfidf_similarity",
        prices=prices,
        alpha_builder=simple_alpha_builder,
    )

    # Check alpha tickers.
    assert list(alpha.index) == ["A", "B", "C"]

    # Check price columns.
    assert list(prices.columns) == ["A", "B", "C"]


def test_build_sector_map_from_universe():
    """Sector helper should detect common GICS sector column."""

    # Create universe DataFrame.
    universe = pd.DataFrame(
        {
            "Ticker": ["A", "B"],
            "GICS Sector": ["Technology", "Health Care"],
        }
    )

    # Build sector map.
    sector_map = build_sector_map_from_universe(universe)

    # Check sector values.
    assert sector_map.loc["A"] == "Technology"
    assert sector_map.loc["B"] == "Health Care"


def test_make_static_factor_for_alphalens_has_multiindex():
    """Static alpha should be repeated across monthly evaluation dates."""

    # Create business dates.
    dates = pd.bdate_range("2024-01-01", periods=90)

    # Create prices.
    prices = pd.DataFrame(
        {
            "A": np.linspace(100, 120, len(dates)),
            "B": np.linspace(100, 110, len(dates)),
        },
        index=dates,
    )

    # Create static alpha.
    alpha = pd.Series({"A": 1.0, "B": 2.0})

    # Build Alphalens factor.
    factor = make_static_factor_for_alphalens(
        alpha=alpha,
        prices=prices,
        periods=(21,),
    )

    # Check MultiIndex.
    assert isinstance(factor.index, pd.MultiIndex)

    # Check index names.
    assert factor.index.names == ["date", "asset"]

    # Check factor name.
    assert factor.name == "factor"


def test_compute_forward_returns_basic_values():
    """Forward returns should match future price arithmetic."""

    # Create dates.
    dates = pd.bdate_range("2024-01-01", periods=5)

    # Create price matrix.
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0, 133.1, 146.41],
            "B": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )

    # Compute forward returns.
    forward_returns = compute_forward_returns(prices, horizons=(1, 2))

    # Check one-day return.
    assert forward_returns[1].loc[dates[0], "A"] == pytest.approx(0.10)

    # Check two-day return.
    assert forward_returns[2].loc[dates[0], "A"] == pytest.approx(0.21)

    # Check flat asset.
    assert forward_returns[1].loc[dates[0], "B"] == pytest.approx(0.0)


def test_build_factor_data_from_forward_returns():
    """Factor data should have Alphalens-style columns and MultiIndex."""

    # Create alpha with five assets.
    alpha = pd.Series({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}, dtype=float)

    # Create date.
    date = pd.Timestamp("2024-01-31")

    # Create forward returns.
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

    # Build factor_data.
    factor_data = build_factor_data_from_forward_returns(
        alpha=alpha,
        forward_returns=forward_returns,
        n_quantiles=5,
    )

    # Check MultiIndex.
    assert isinstance(factor_data.index, pd.MultiIndex)

    # Check columns.
    assert "factor" in factor_data.columns
    assert "factor_quantile" in factor_data.columns
    assert "21D" in factor_data.columns


def test_compute_monthly_ic_static_alpha():
    """Monthly IC should be positive when returns rise with alpha."""

    # Create alpha.
    alpha = pd.Series({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}, dtype=float)

    # Create date.
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

    # Compute IC.
    ic = compute_monthly_ic(alpha, forward_returns, method="spearman")

    # Check perfect rank IC.
    assert ic.loc[date, "21D"] == pytest.approx(1.0)


def test_compute_quantile_returns_static_alpha():
    """Quantile return table should put low alpha in low quantile."""

    # Create alpha.
    alpha = pd.Series({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}, dtype=float)

    # Create date.
    date = pd.Timestamp("2024-01-31")

    # Create returns.
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

    # Compute quantile returns.
    qret = compute_quantile_returns(alpha, forward_returns, n_quantiles=5)

    # Check row count.
    assert len(qret) == 5

    # Check bottom quantile.
    assert qret.loc[qret["quantile"] == 1, "mean_forward_return"].iloc[0] == pytest.approx(0.01)

    # Check top quantile.
    assert qret.loc[qret["quantile"] == 5, "mean_forward_return"].iloc[0] == pytest.approx(0.05)


def test_compute_top_bottom_spread():
    """Top-minus-bottom spread should subtract quantile 1 from quantile 5."""

    # Create date.
    date = pd.Timestamp("2024-01-31")

    # Create quantile return table.
    qret = pd.DataFrame(
        {
            "date": [date, date],
            "horizon": [21, 21],
            "quantile": [1, 5],
            "mean_forward_return": [0.01, 0.05],
            "n_assets": [10, 10],
        }
    )

    # Compute spread.
    spread = compute_top_bottom_spread(qret)

    # Check spread.
    assert spread.loc[date, "21D"] == pytest.approx(0.04)


def test_compute_signal_turnover_static_is_empty():
    """Static alpha should have no turnover."""

    # Create static alpha.
    alpha = pd.Series({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}, dtype=float)

    # Compute turnover.
    turnover = compute_signal_turnover(alpha)

    # Check empty.
    assert turnover.empty


def test_compute_signal_turnover_time_varying():
    """Top quantile turnover should detect changing top names."""

    # Create alpha history with top name changing from E to A.
    alpha_history = pd.DataFrame(
        {
            "A": [1, 5],
            "B": [2, 4],
            "C": [3, 3],
            "D": [4, 2],
            "E": [5, 1],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )

    # Compute turnover.
    turnover = compute_signal_turnover(alpha_history)

    # Top quantile changed completely.
    assert turnover.iloc[0] == pytest.approx(1.0)


def test_compute_sector_exposure_static():
    """Sector exposure should summarize alpha by sector."""

    # Create alpha.
    alpha = pd.Series({"A": 1.0, "B": -2.0, "C": 3.0})

    # Create sector map.
    sector_map = {"A": "Technology", "B": "Technology", "C": "Health Care"}

    # Compute exposure.
    exposure = compute_sector_exposure(alpha, sector_map)

    # Check counts.
    assert exposure.loc["Technology", "n_assets"] == 2
    assert exposure.loc["Health Care", "n_assets"] == 1

    # Check concentration.
    assert exposure.loc["Technology", "abs_alpha_share"] == pytest.approx(0.5)


def test_real_alphalens_smoke_if_installed():
    """Optional smoke test for the real alphalens-reloaded package."""

    # Skip this test if Alphalens is not installed.
    pytest.importorskip("alphalens")

    # Check module flag.
    assert ALPHALENS_AVAILABLE is True
