"""Tests for the saved-output factor-selection audit."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.factor_selection_audit import (
    OUTPUT_COLUMNS,
    build_factor_selection_audit,
    confirm_factor_names,
    generate_factor_selection_audit,
)


TEST_FACTOR_METADATA = {
    "sec_test_factor": {
        "factor_family": "SEC PIT",
        "signal_definition_short": "Synthetic SEC factor.",
        "point_in_time": True,
        "interpretation": "upper-tail selector",
        "recommended_role": "primary EB factor",
    },
    "sec_test_factor_sector_neutral": {
        "factor_family": "SEC PIT sector-neutral",
        "signal_definition_short": "Synthetic within-sector SEC factor.",
        "point_in_time": True,
        "interpretation": "within-sector signal",
        "recommended_role": "robustness factor",
    },
    "static_tfidf": {
        "factor_family": "static baseline",
        "signal_definition_short": "Synthetic static baseline.",
        "point_in_time": False,
        "interpretation": "sector/thematic exposure",
        "recommended_role": "static baseline",
    },
    "static_embedding": {
        "factor_family": "static baseline",
        "signal_definition_short": "Second synthetic static baseline.",
        "point_in_time": False,
        "interpretation": "sector/thematic exposure",
        "recommended_role": "static baseline",
    },
}


def make_summary_rows(signal_names):
    """Create project-shaped 21D and 63D summary rows."""

    rows = []
    for signal_index, signal_name in enumerate(signal_names):
        for horizon in ("21D", "63D"):
            horizon_scale = 1.0 if horizon == "21D" else 2.0
            rows.append(
                {
                    "signal_name": signal_name,
                    "horizon": horizon,
                    "mean_ic": 0.01 * (signal_index + 1) * horizon_scale,
                    "mean_spread": 0.02 * (signal_index + 1) * horizon_scale,
                    "positive_spread_rate": 0.60 + 0.05 * signal_index,
                    "spread_p_value": 0.04 / (signal_index + 1),
                    "mean_turnover": 0.10 + 0.01 * signal_index,
                    "n_dates": 36,
                    "n_tickers": 500,
                    "mean_n_available": 490.0,
                    "mean_coverage_rate": 0.98,
                }
            )
    columns = [
        "signal_name",
        "horizon",
        "mean_ic",
        "mean_spread",
        "positive_spread_rate",
        "spread_p_value",
        "mean_turnover",
        "n_dates",
        "n_tickers",
        "mean_n_available",
        "mean_coverage_rate",
    ]
    return pd.DataFrame(rows, columns=columns)


def make_paired_delta_rows():
    """Create project-shaped paired delta rows including irrelevant rows."""

    rows = []
    for baseline, delta in (("static_tfidf", 0.01), ("static_embedding", 0.02)):
        rows.extend(
            [
                {
                    "candidate_signal": "sec_test_factor",
                    "baseline_signal": baseline,
                    "horizon": "63D",
                    "kind": "spread",
                    "mean_delta": delta,
                    "p_value": 0.01,
                    "candidate_win_rate": 0.75,
                },
                {
                    "candidate_signal": "sec_test_factor",
                    "baseline_signal": baseline,
                    "horizon": "63D",
                    "kind": "ic",
                    "mean_delta": 0.001,
                    "p_value": 0.50,
                    "candidate_win_rate": 0.55,
                },
            ]
        )
    return pd.DataFrame(rows)


def make_sector_share_rows():
    """Create latest-date sector absolute-alpha shares."""

    return pd.DataFrame(
        {
            "sector": ["Technology", "Industrials"],
            "sec_test_factor": [0.60, 0.40],
        }
    )


def make_quantile_return_rows():
    """Create focused mean quantile returns for the synthetic SEC factor."""

    return pd.DataFrame(
        [
            {
                "signal_name": "sec_test_factor",
                "horizon": 21,
                "1": 0.01,
                "2": 0.02,
                "3": 0.03,
                "4": 0.04,
                "5": 0.05,
            },
            {
                "signal_name": "sec_test_factor",
                "horizon": 63,
                "1": 0.02,
                "2": 0.03,
                "3": 0.04,
                "4": 0.05,
                "5": 0.08,
            },
        ]
    )


def test_confirm_factor_names_returns_exact_names_from_both_files():
    """The audit should report exact names rather than infer aliases."""

    full_summary = make_summary_rows(
        ["sec_test_factor", "static_tfidf", "static_embedding"]
    )
    sector_neutral_summary = make_summary_rows(
        ["sec_test_factor_sector_neutral"]
    )

    names = confirm_factor_names(
        full_summary,
        sector_neutral_summary,
        factor_metadata=TEST_FACTOR_METADATA,
    )

    assert names["full_summary"] == [
        "sec_test_factor",
        "static_embedding",
        "static_tfidf",
    ]
    assert names["sector_neutral_summary"] == [
        "sec_test_factor_sector_neutral"
    ]
    assert names["all_factors"] == [
        "sec_test_factor",
        "sec_test_factor_sector_neutral",
        "static_embedding",
        "static_tfidf",
    ]


def test_build_audit_pivots_horizons_and_attaches_saved_metrics():
    """One row should contain both horizons and all available robustness data."""

    audit = build_factor_selection_audit(
        full_summary=make_summary_rows(
            ["sec_test_factor", "static_tfidf", "static_embedding"]
        ),
        sector_neutral_summary=make_summary_rows(
            ["sec_test_factor_sector_neutral"]
        ),
        paired_delta_tests=make_paired_delta_rows(),
        sector_abs_alpha_share=make_sector_share_rows(),
        mean_quantile_returns=make_quantile_return_rows(),
        factor_metadata=TEST_FACTOR_METADATA,
    )

    row = audit.set_index("factor_name").loc["sec_test_factor"]

    assert list(audit.columns) == OUTPUT_COLUMNS
    assert row["mean_rank_ic_21d"] == pytest.approx(0.01)
    assert row["mean_rank_ic_63d"] == pytest.approx(0.02)
    assert row["q5_q1_spread_21d"] == pytest.approx(0.02)
    assert row["q5_q1_spread_63d"] == pytest.approx(0.04)
    assert row["sector_neutral_ic_63d"] == pytest.approx(0.02)
    assert row["sector_neutral_q5_q1_63d"] == pytest.approx(0.04)
    assert row["paired_spread_delta_vs_static_tfidf_63d"] == pytest.approx(
        0.01
    )
    assert row["paired_spread_delta_vs_static_embedding_63d"] == pytest.approx(
        0.02
    )
    assert row["latest_sector_abs_alpha_largest_sector"] == "Technology"
    assert row["latest_sector_abs_alpha_max_share"] == pytest.approx(0.60)
    assert row["mean_quantile_return_q5_63d"] == pytest.approx(0.08)
    assert row["recommended_role"] == "primary EB factor"


def test_unavailable_optional_metrics_remain_missing():
    """Static and sector-neutral rows should not receive invented diagnostics."""

    audit = build_factor_selection_audit(
        full_summary=make_summary_rows(
            ["sec_test_factor", "static_tfidf", "static_embedding"]
        ),
        sector_neutral_summary=make_summary_rows(
            ["sec_test_factor_sector_neutral"]
        ),
        paired_delta_tests=make_paired_delta_rows(),
        sector_abs_alpha_share=make_sector_share_rows(),
        mean_quantile_returns=make_quantile_return_rows(),
        factor_metadata=TEST_FACTOR_METADATA,
    ).set_index("factor_name")

    static_row = audit.loc["static_tfidf"]
    sector_neutral_row = audit.loc["sec_test_factor_sector_neutral"]

    assert np.isnan(static_row["paired_spread_delta_vs_static_tfidf_63d"])
    assert pd.isna(static_row["latest_sector_abs_alpha_largest_sector"])
    assert np.isnan(static_row["mean_quantile_return_q5_63d"])
    assert np.isnan(sector_neutral_row["sector_neutral_ic_63d"])


def test_missing_declared_factor_metadata_raises_clear_error():
    """Every factor found in the files must have explicit metadata."""

    full_summary = make_summary_rows(["unknown_factor"])
    sector_neutral_summary = make_summary_rows([])

    with pytest.raises(ValueError, match="unknown_factor"):
        confirm_factor_names(
            full_summary,
            sector_neutral_summary,
            factor_metadata=TEST_FACTOR_METADATA,
        )


def test_generate_audit_writes_csv_and_metadata_json(tmp_path):
    """The file-based entry point should write reproducible reporting artifacts."""

    full_path = tmp_path / "full.csv"
    sector_neutral_path = tmp_path / "sector_neutral.csv"
    paired_path = tmp_path / "paired.csv"
    sector_share_path = tmp_path / "sector_share.csv"
    quantile_path = tmp_path / "quantiles.csv"
    output_csv = tmp_path / "factor_selection_audit.csv"
    output_json = tmp_path / "factor_selection_audit_metadata.json"

    make_summary_rows(
        ["sec_test_factor", "static_tfidf", "static_embedding"]
    ).to_csv(full_path, index=False)
    make_summary_rows(["sec_test_factor_sector_neutral"]).to_csv(
        sector_neutral_path,
        index=False,
    )
    make_paired_delta_rows().to_csv(paired_path, index=False)
    make_sector_share_rows().to_csv(sector_share_path, index=False)
    make_quantile_return_rows().to_csv(quantile_path, index=False)

    audit, metadata = generate_factor_selection_audit(
        full_summary_path=full_path,
        sector_neutral_summary_path=sector_neutral_path,
        paired_delta_path=paired_path,
        sector_share_path=sector_share_path,
        quantile_returns_path=quantile_path,
        output_csv_path=output_csv,
        output_metadata_path=output_json,
        factor_metadata=TEST_FACTOR_METADATA,
    )

    saved_audit = pd.read_csv(output_csv)
    saved_metadata = json.loads(output_json.read_text(encoding="utf-8"))

    assert output_csv.is_file()
    assert output_json.is_file()
    assert len(saved_audit) == len(audit) == 4
    assert saved_metadata["horizons"] == ["21D", "63D"]
    assert saved_metadata["ic_method"].startswith("Spearman")
    assert saved_metadata["composite_score"] is None
    assert saved_metadata["ranking_rule"] is None
    assert metadata["factor_names_from_files"]["all_factors"] == [
        "sec_test_factor",
        "sec_test_factor_sector_neutral",
        "static_embedding",
        "static_tfidf",
    ]


def test_duplicate_factor_horizon_rows_are_rejected():
    """Ambiguous source summaries should fail rather than silently aggregate."""

    full_summary = make_summary_rows(
        ["sec_test_factor", "static_tfidf", "static_embedding"]
    )
    full_summary = pd.concat([full_summary, full_summary.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate factor/horizon"):
        build_factor_selection_audit(
            full_summary=full_summary,
            sector_neutral_summary=make_summary_rows(
                ["sec_test_factor_sector_neutral"]
            ),
            paired_delta_tests=make_paired_delta_rows(),
            sector_abs_alpha_share=make_sector_share_rows(),
            mean_quantile_returns=make_quantile_return_rows(),
            factor_metadata=TEST_FACTOR_METADATA,
        )
