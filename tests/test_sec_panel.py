"""
Unit tests for src.sec_panel.

These tests use small deterministic pandas DataFrames. They do not call
EdgarTools, OpenAI, Yahoo Finance, or any network service.

The goals are:

    1. verify filing-event signals are carried forward correctly,
    2. verify conservative availability lags are applied,
    3. verify duplicate same-day filing signals have an explicit rule,
    4. verify multiple raw panels can be built,
    5. verify cross-sectional alpha panels can be built,
    6. verify coverage and signal-age diagnostics work,
    7. verify panel pickles round trip.

Run from the repository root with:

    pytest -q tests/test_sec_panel.py
"""

import numpy as np
import pandas as pd
import pytest

from src.sec_panel import (
    add_signal_available_date,
    build_alpha_panel,
    build_alpha_panels,
    build_evaluation_dates,
    build_signal_age_panel,
    build_signal_panel,
    build_signal_panels,
    clean_signal_table,
    compute_panel_coverage,
    get_month_end_trading_dates,
    load_panel_dict_pickle,
    save_panel_dict_pickle,
    save_panel_pickle,
    summarize_signal_age,
)


# -----------------------------------------------------------------------------
# Test-data helpers
# -----------------------------------------------------------------------------


def make_signal_table():
    """Create a small filing-level signal table."""

    return pd.DataFrame(
        {
            "ticker": [
                "aaa",
                "aaa",
                "bbb",
                "bbb",
                "ccc",
            ],
            "filing_date": [
                "2024-01-10",
                "2024-03-05",
                "2024-02-15",
                "2024-04-10",
                "2024-04-20",
            ],
            "acceptance_datetime": [
                "2024-01-10T16:00:00Z",
                "2024-03-05T16:00:00Z",
                "2024-02-15T16:00:00Z",
                "2024-04-10T16:00:00Z",
                "2024-04-20T16:00:00Z",
            ],
            "accession_number": [
                "aaa-old",
                "aaa-new",
                "bbb-old",
                "bbb-new",
                "ccc-only",
            ],
            "summary_theme_similarity": [
                0.10,
                0.30,
                0.20,
                0.40,
                0.50,
            ],
            "exposure_label_score": [
                1,
                3,
                2,
                2,
                0,
            ],
        }
    )


def simple_alpha_builder(series):
    """Mean-center a cross-section for easy expected values."""

    series = series.astype(float)

    return series - series.mean()


# -----------------------------------------------------------------------------
# Cleaning and date tests
# -----------------------------------------------------------------------------


def test_clean_signal_table_normalizes_tickers_and_dates():
    """Signal-table cleaning should normalize tickers and parse dates."""

    table = make_signal_table()

    cleaned = clean_signal_table(
        table,
        required_signal_columns=["summary_theme_similarity"],
    )

    assert cleaned.loc[0, "ticker"] == "AAA"
    assert pd.api.types.is_datetime64_any_dtype(cleaned["filing_date"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["acceptance_datetime"])
    assert cleaned["summary_theme_similarity"].dtype.kind in "fi"


def test_clean_signal_table_requires_columns():
    """Missing required columns should fail clearly."""

    table = pd.DataFrame({"ticker": ["A"]})

    with pytest.raises(ValueError, match="missing required columns"):
        clean_signal_table(
            table,
            required_signal_columns=["summary_theme_similarity"],
        )


def test_add_signal_available_date_applies_lag():
    """Availability date should use acceptance date plus lag."""

    table = clean_signal_table(
        make_signal_table(),
        required_signal_columns=["summary_theme_similarity"],
    )

    table = add_signal_available_date(
        table,
        availability_lag_days=1,
    )

    assert table.loc[0, "signal_available_date"] == pd.Timestamp("2024-01-11")


def test_get_month_end_trading_dates_uses_actual_last_available_date():
    """Month-end dates should come from available trading dates."""

    dates = pd.to_datetime(
        [
            "2024-01-30",
            "2024-01-31",
            "2024-02-27",
            "2024-02-28",
        ]
    )

    result = get_month_end_trading_dates(dates)

    assert list(result) == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-28"),
    ]


def test_build_evaluation_dates_from_prices():
    """Evaluation dates should be actual month-end price dates."""

    prices = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0, 103.0],
        },
        index=pd.to_datetime(
            [
                "2024-01-30",
                "2024-01-31",
                "2024-02-27",
                "2024-02-28",
            ]
        ),
    )

    dates = build_evaluation_dates(prices=prices)

    assert list(dates) == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-28"),
    ]


# -----------------------------------------------------------------------------
# Raw panel tests
# -----------------------------------------------------------------------------


def test_build_signal_panel_carries_latest_filing_forward():
    """Panel values should carry forward until a newer filing arrives."""

    table = make_signal_table()

    evaluation_dates = pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-29",
            "2024-03-31",
            "2024-04-30",
        ]
    )

    panel = build_signal_panel(
        table,
        signal_col="summary_theme_similarity",
        evaluation_dates=evaluation_dates,
        availability_lag_days=1,
    )

    assert panel.loc[pd.Timestamp("2024-01-31"), "AAA"] == pytest.approx(0.10)
    assert np.isnan(panel.loc[pd.Timestamp("2024-01-31"), "BBB"])

    assert panel.loc[pd.Timestamp("2024-02-29"), "AAA"] == pytest.approx(0.10)
    assert panel.loc[pd.Timestamp("2024-02-29"), "BBB"] == pytest.approx(0.20)

    assert panel.loc[pd.Timestamp("2024-03-31"), "AAA"] == pytest.approx(0.30)
    assert panel.loc[pd.Timestamp("2024-03-31"), "BBB"] == pytest.approx(0.20)

    assert panel.loc[pd.Timestamp("2024-04-30"), "BBB"] == pytest.approx(0.40)
    assert panel.loc[pd.Timestamp("2024-04-30"), "CCC"] == pytest.approx(0.50)


def test_build_signal_panel_keeps_latest_same_day_duplicate():
    """Duplicate ticker/date signals should keep the later sorted row."""

    table = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "filing_date": ["2024-01-10", "2024-01-10"],
            "acceptance_datetime": [
                "2024-01-10T16:00:00Z",
                "2024-01-10T16:00:00Z",
            ],
            "accession_number": ["0001", "0002"],
            "summary_theme_similarity": [0.10, 0.90],
        }
    )

    panel = build_signal_panel(
        table,
        signal_col="summary_theme_similarity",
        evaluation_dates=["2024-01-31"],
        availability_lag_days=1,
    )

    assert panel.loc[pd.Timestamp("2024-01-31"), "AAA"] == pytest.approx(0.90)


def test_build_signal_panels_builds_multiple_columns():
    """Several raw signal panels should be built from one signal table."""

    table = make_signal_table()

    panels = build_signal_panels(
        table,
        signal_cols=[
            "summary_theme_similarity",
            "exposure_label_score",
        ],
        evaluation_dates=["2024-04-30"],
        availability_lag_days=1,
    )

    assert set(panels.keys()) == {
        "summary_theme_similarity",
        "exposure_label_score",
    }

    assert panels["summary_theme_similarity"].loc[
        pd.Timestamp("2024-04-30"),
        "AAA",
    ] == pytest.approx(0.30)

    assert panels["exposure_label_score"].loc[
        pd.Timestamp("2024-04-30"),
        "AAA",
    ] == pytest.approx(3.0)


# -----------------------------------------------------------------------------
# Alpha panel tests
# -----------------------------------------------------------------------------


def test_build_alpha_panel_normalizes_each_date():
    """Alpha panel should normalize each cross-section separately."""

    raw_panel = pd.DataFrame(
        {
            "AAA": [1.0, 2.0],
            "BBB": [2.0, 4.0],
            "CCC": [3.0, 6.0],
        },
        index=pd.to_datetime(
            [
                "2024-01-31",
                "2024-02-29",
            ]
        ),
    )

    alpha_panel = build_alpha_panel(
        raw_panel,
        alpha_builder=simple_alpha_builder,
    )

    assert alpha_panel.loc[pd.Timestamp("2024-01-31"), "AAA"] == pytest.approx(-1.0)
    assert alpha_panel.loc[pd.Timestamp("2024-01-31"), "BBB"] == pytest.approx(0.0)
    assert alpha_panel.loc[pd.Timestamp("2024-01-31"), "CCC"] == pytest.approx(1.0)

    assert alpha_panel.loc[pd.Timestamp("2024-02-29"), "AAA"] == pytest.approx(-2.0)
    assert alpha_panel.loc[pd.Timestamp("2024-02-29"), "BBB"] == pytest.approx(0.0)
    assert alpha_panel.loc[pd.Timestamp("2024-02-29"), "CCC"] == pytest.approx(2.0)


def test_build_alpha_panel_respects_min_non_null():
    """Dates with too few names should remain missing."""

    raw_panel = pd.DataFrame(
        {
            "AAA": [1.0],
            "BBB": [np.nan],
        },
        index=pd.to_datetime(["2024-01-31"]),
    )

    alpha_panel = build_alpha_panel(
        raw_panel,
        alpha_builder=simple_alpha_builder,
        min_non_null=2,
    )

    assert alpha_panel.isna().all(axis=None)


def test_build_alpha_panels_appends_suffix():
    """Alpha-panel dictionary should append a readable suffix."""

    raw_panel = pd.DataFrame(
        {
            "AAA": [1.0],
            "BBB": [2.0],
        },
        index=pd.to_datetime(["2024-01-31"]),
    )

    alpha_panels = build_alpha_panels(
        {"raw_signal": raw_panel},
        alpha_builder=simple_alpha_builder,
    )

    assert "raw_signal_alpha" in alpha_panels


# -----------------------------------------------------------------------------
# Coverage and age tests
# -----------------------------------------------------------------------------


def test_compute_panel_coverage_counts_available_names():
    """Coverage should count non-missing signals per date."""

    panel = pd.DataFrame(
        {
            "AAA": [1.0, 1.0],
            "BBB": [np.nan, 2.0],
        },
        index=pd.to_datetime(
            [
                "2024-01-31",
                "2024-02-29",
            ]
        ),
    )

    coverage = compute_panel_coverage(panel)

    assert coverage.loc[0, "n_available"] == 1
    assert coverage.loc[0, "n_total"] == 2
    assert coverage.loc[0, "coverage_rate"] == pytest.approx(0.5)

    assert coverage.loc[1, "n_available"] == 2
    assert coverage.loc[1, "coverage_rate"] == pytest.approx(1.0)


def test_build_signal_age_panel_tracks_carried_forward_age():
    """Signal age should increase as a filing signal is carried forward."""

    table = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "filing_date": ["2024-01-10"],
            "acceptance_datetime": ["2024-01-10T16:00:00Z"],
            "accession_number": ["aaa-old"],
            "summary_theme_similarity": [0.10],
        }
    )

    age_panel = build_signal_age_panel(
        table,
        evaluation_dates=[
            "2024-01-31",
            "2024-02-29",
        ],
        availability_lag_days=1,
    )

    assert age_panel.loc[pd.Timestamp("2024-01-31"), "AAA"] == 20
    assert age_panel.loc[pd.Timestamp("2024-02-29"), "AAA"] == 49


def test_summarize_signal_age_returns_date_level_stats():
    """Age summary should aggregate across tickers by date."""

    age_panel = pd.DataFrame(
        {
            "AAA": [10.0, 20.0],
            "BBB": [20.0, 40.0],
        },
        index=pd.to_datetime(
            [
                "2024-01-31",
                "2024-02-29",
            ]
        ),
    )

    summary = summarize_signal_age(age_panel)

    assert summary.loc[0, "mean_age_days"] == pytest.approx(15.0)
    assert summary.loc[1, "median_age_days"] == pytest.approx(30.0)
    assert summary.loc[1, "max_age_days"] == pytest.approx(40.0)


# -----------------------------------------------------------------------------
# Pickle tests
# -----------------------------------------------------------------------------


def test_save_panel_pickle_writes_dataframe(tmp_path):
    """Single panel pickle should be readable by pandas."""

    panel = pd.DataFrame(
        {"AAA": [1.0]},
        index=pd.to_datetime(["2024-01-31"]),
    )

    path = tmp_path / "panel.pkl"

    save_panel_pickle(panel, path)

    loaded = pd.read_pickle(path)

    assert loaded.loc[pd.Timestamp("2024-01-31"), "AAA"] == pytest.approx(1.0)


def test_save_and_load_panel_dict_pickle_round_trip(tmp_path):
    """Panel dictionaries should round trip through pickle."""

    panels = {
        "summary_theme_similarity": pd.DataFrame(
            {"AAA": [1.0]},
            index=pd.to_datetime(["2024-01-31"]),
        )
    }

    path = tmp_path / "panels.pkl"

    save_panel_dict_pickle(panels, path)

    loaded = load_panel_dict_pickle(path)

    assert set(loaded.keys()) == {"summary_theme_similarity"}
    assert loaded["summary_theme_similarity"].loc[
        pd.Timestamp("2024-01-31"),
        "AAA",
    ] == pytest.approx(1.0)
