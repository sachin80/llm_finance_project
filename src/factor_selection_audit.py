"""Build a transparent factor-selection audit from saved diagnostic tables.

This reporting module only reads already-generated CSV outputs. It does not
retrieve SEC filings, call OpenAI, rebuild signals, recompute returns, rerun
signal diagnostics, or invoke the portfolio optimizer.

Run from the project root with the default project paths::

    python -m src.factor_selection_audit

The default output is::

    outputs/signal_diagnostics/static_vs_sec_pit/tables/
        factor_selection_audit.csv

A companion JSON file records the source files and methodological metadata
used to construct the table.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_FULL_SUMMARY_PATH = Path(
    "outputs/sec_pit_openai/diagnostics/full_static_vs_sec_summary.csv"
)
DEFAULT_SECTOR_NEUTRAL_SUMMARY_PATH = Path(
    "outputs/signal_diagnostics/static_vs_sec_pit/tables/sector_neutral_summary.csv"
)
DEFAULT_PAIRED_DELTA_PATH = Path(
    "outputs/signal_diagnostics/static_vs_sec_pit/tables/paired_delta_tests.csv"
)
DEFAULT_SECTOR_SHARE_PATH = Path(
    "outputs/signal_diagnostics/static_vs_sec_pit/tables/sec_sector_abs_alpha_share.csv"
)
DEFAULT_QUANTILE_RETURNS_PATH = Path(
    "outputs/signal_diagnostics/static_vs_sec_pit/tables/mean_quantile_returns.csv"
)
DEFAULT_OUTPUT_CSV_PATH = Path(
    "outputs/signal_diagnostics/static_vs_sec_pit/tables/factor_selection_audit.csv"
)
DEFAULT_OUTPUT_METADATA_PATH = Path(
    "outputs/signal_diagnostics/static_vs_sec_pit/tables/"
    "factor_selection_audit_metadata.json"
)

HORIZONS = ("21D", "63D")
QUANTILES = (1, 2, 3, 4, 5)


# The order is deliberate: SEC PIT discovery factors, corresponding
# sector-neutral robustness factors, and then static baselines.
FACTOR_METADATA: dict[str, dict[str, Any]] = {
    "sec_summary_similarity": {
        "factor_family": "SEC PIT",
        "signal_definition_short": (
            "Cosine similarity between the PIT SEC filing summary embedding "
            "and the AI-infrastructure theme embedding."
        ),
        "point_in_time": True,
        "interpretation": "sector/thematic exposure",
        "recommended_role": "robustness factor",
    },
    "sec_exposure_score": {
        "factor_family": "SEC PIT",
        "signal_definition_short": (
            "LLM-assessed filing exposure materiality: Direct=3, Indirect=2, "
            "Minimal=1, None=0."
        ),
        "point_in_time": True,
        "interpretation": "upper-tail selector",
        "recommended_role": "robustness factor",
    },
    "sec_evidence_strength": {
        "factor_family": "SEC PIT",
        "signal_definition_short": (
            "LLM-assessed strength of the filing evidence supporting the "
            "structured exposure judgment."
        ),
        "point_in_time": True,
        "interpretation": "weak/noisy",
        "recommended_role": "drop",
    },
    "sec_similarity_x_exposure": {
        "factor_family": "SEC PIT",
        "signal_definition_short": (
            "PIT filing-summary theme similarity multiplied by LLM-assessed "
            "exposure materiality."
        ),
        "point_in_time": True,
        "interpretation": "upper-tail selector",
        "recommended_role": "primary EB factor",
    },
    "sec_similarity_x_evidence": {
        "factor_family": "SEC PIT",
        "signal_definition_short": (
            "PIT filing-summary theme similarity multiplied by evidence "
            "strength."
        ),
        "point_in_time": True,
        "interpretation": "weak/noisy",
        "recommended_role": "appendix only",
    },
    "sec_similarity_x_direction": {
        "factor_family": "SEC PIT",
        "signal_definition_short": (
            "PIT filing-summary theme similarity multiplied by the ordinal "
            "exposure-direction score."
        ),
        "point_in_time": True,
        "interpretation": "sector/thematic exposure",
        "recommended_role": "appendix only",
    },
    "sec_direct_or_indirect": {
        "factor_family": "SEC PIT",
        "signal_definition_short": (
            "PIT filing-summary theme similarity retained only when exposure "
            "is classified as Direct or Indirect."
        ),
        "point_in_time": True,
        "interpretation": "sector/thematic exposure",
        "recommended_role": "appendix only",
    },
    "sec_summary_similarity_sector_neutral": {
        "factor_family": "SEC PIT sector-neutral",
        "signal_definition_short": (
            "Within-date, within-sector standardized version of SEC summary "
            "similarity."
        ),
        "point_in_time": True,
        "interpretation": "within-sector signal",
        "recommended_role": "appendix only",
    },
    "sec_exposure_score_sector_neutral": {
        "factor_family": "SEC PIT sector-neutral",
        "signal_definition_short": (
            "Within-date, within-sector standardized version of SEC exposure "
            "materiality."
        ),
        "point_in_time": True,
        "interpretation": "within-sector signal",
        "recommended_role": "appendix only",
    },
    "sec_similarity_x_exposure_sector_neutral": {
        "factor_family": "SEC PIT sector-neutral",
        "signal_definition_short": (
            "Within-date, within-sector standardized version of SEC "
            "similarity multiplied by exposure materiality."
        ),
        "point_in_time": True,
        "interpretation": "within-sector signal",
        "recommended_role": "robustness factor",
    },
    "sec_direct_or_indirect_sector_neutral": {
        "factor_family": "SEC PIT sector-neutral",
        "signal_definition_short": (
            "Within-date, within-sector standardized version of the Direct- "
            "or Indirect-filtered SEC similarity factor."
        ),
        "point_in_time": True,
        "interpretation": "weak/noisy",
        "recommended_role": "drop",
    },
    "static_tfidf": {
        "factor_family": "static baseline",
        "signal_definition_short": (
            "TF-IDF cosine similarity between one static Yahoo company "
            "description and the AI-infrastructure theme."
        ),
        "point_in_time": False,
        "interpretation": "sector/thematic exposure",
        "recommended_role": "static baseline",
    },
    "static_embedding": {
        "factor_family": "static baseline",
        "signal_definition_short": (
            "Embedding cosine similarity between one static Yahoo company "
            "description and the AI-infrastructure theme."
        ),
        "point_in_time": False,
        "interpretation": "sector/thematic exposure",
        "recommended_role": "static baseline",
    },
    "static_tfidf_sector_neutral": {
        "factor_family": "static baseline sector-neutral",
        "signal_definition_short": (
            "Within-date, within-sector standardized version of the static "
            "TF-IDF baseline."
        ),
        "point_in_time": False,
        "interpretation": "within-sector signal",
        "recommended_role": "static baseline",
    },
    "static_embedding_sector_neutral": {
        "factor_family": "static baseline sector-neutral",
        "signal_definition_short": (
            "Within-date, within-sector standardized version of the static "
            "embedding baseline."
        ),
        "point_in_time": False,
        "interpretation": "within-sector signal",
        "recommended_role": "static baseline",
    },
}


OUTPUT_COLUMNS = [
    "factor_name",
    "factor_family",
    "signal_definition_short",
    "point_in_time",
    "coverage",
    "n_dates",
    "n_tickers",
    "mean_n_available",
    "mean_rank_ic_21d",
    "mean_rank_ic_63d",
    "q5_q1_spread_21d",
    "q5_q1_spread_63d",
    "p_value_63d_spread",
    "positive_spread_rate_63d",
    "mean_turnover",
    "latest_sector_abs_alpha_largest_sector",
    "latest_sector_abs_alpha_max_share",
    "sector_neutral_ic_63d",
    "sector_neutral_q5_q1_63d",
    "sector_neutral_p_value_63d_spread",
    "sector_neutral_positive_spread_rate_63d",
    "paired_spread_delta_vs_static_tfidf_63d",
    "paired_spread_delta_p_value_vs_static_tfidf_63d",
    "paired_spread_delta_win_rate_vs_static_tfidf_63d",
    "paired_spread_delta_vs_static_embedding_63d",
    "paired_spread_delta_p_value_vs_static_embedding_63d",
    "paired_spread_delta_win_rate_vs_static_embedding_63d",
    *[
        f"mean_quantile_return_q{quantile}_{horizon.lower()}"
        for horizon in HORIZONS
        for quantile in QUANTILES
    ],
    "interpretation",
    "recommended_role",
]


SUMMARY_REQUIRED_COLUMNS = {
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
}
PAIRED_REQUIRED_COLUMNS = {
    "candidate_signal",
    "baseline_signal",
    "horizon",
    "kind",
    "mean_delta",
    "p_value",
    "candidate_win_rate",
}
QUANTILE_REQUIRED_COLUMNS = {"signal_name", "horizon", "1", "2", "3", "4", "5"}
SECTOR_SHARE_REQUIRED_COLUMNS = {"sector"}


def load_factor_selection_inputs(
    full_summary_path: str | Path = DEFAULT_FULL_SUMMARY_PATH,
    sector_neutral_summary_path: str | Path = DEFAULT_SECTOR_NEUTRAL_SUMMARY_PATH,
    paired_delta_path: str | Path = DEFAULT_PAIRED_DELTA_PATH,
    sector_share_path: str | Path = DEFAULT_SECTOR_SHARE_PATH,
    quantile_returns_path: str | Path = DEFAULT_QUANTILE_RETURNS_PATH,
) -> dict[str, pd.DataFrame]:
    """Load the five saved CSV inputs used by the factor-selection audit."""

    paths = {
        "full_summary": Path(full_summary_path),
        "sector_neutral_summary": Path(sector_neutral_summary_path),
        "paired_delta_tests": Path(paired_delta_path),
        "sector_abs_alpha_share": Path(sector_share_path),
        "mean_quantile_returns": Path(quantile_returns_path),
    }

    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing factor-selection audit input file(s): " + ", ".join(missing)
        )

    return {name: pd.read_csv(path) for name, path in paths.items()}


def confirm_factor_names(
    full_summary: pd.DataFrame,
    sector_neutral_summary: pd.DataFrame,
    factor_metadata: Mapping[str, Mapping[str, Any]] = FACTOR_METADATA,
) -> dict[str, list[str]]:
    """Return exact factor names and require declared metadata for every factor."""

    _require_columns(full_summary, {"signal_name"}, "full_summary")
    _require_columns(
        sector_neutral_summary,
        {"signal_name"},
        "sector_neutral_summary",
    )

    full_names = sorted(full_summary["signal_name"].dropna().astype(str).unique())
    sector_neutral_names = sorted(
        sector_neutral_summary["signal_name"].dropna().astype(str).unique()
    )
    all_names = sorted(set(full_names).union(sector_neutral_names))

    missing_metadata = [name for name in all_names if name not in factor_metadata]
    if missing_metadata:
        raise ValueError(
            "Missing declared factor metadata for: " + ", ".join(missing_metadata)
        )

    return {
        "full_summary": full_names,
        "sector_neutral_summary": sector_neutral_names,
        "all_factors": all_names,
    }


def build_factor_selection_audit(
    full_summary: pd.DataFrame,
    sector_neutral_summary: pd.DataFrame,
    paired_delta_tests: pd.DataFrame,
    sector_abs_alpha_share: pd.DataFrame,
    mean_quantile_returns: pd.DataFrame,
    factor_metadata: Mapping[str, Mapping[str, Any]] = FACTOR_METADATA,
) -> pd.DataFrame:
    """Merge saved diagnostics into one transparent row per evaluated factor."""

    _require_columns(full_summary, SUMMARY_REQUIRED_COLUMNS, "full_summary")
    _require_columns(
        sector_neutral_summary,
        SUMMARY_REQUIRED_COLUMNS,
        "sector_neutral_summary",
    )
    _require_columns(
        paired_delta_tests,
        PAIRED_REQUIRED_COLUMNS,
        "paired_delta_tests",
    )
    _require_columns(
        sector_abs_alpha_share,
        SECTOR_SHARE_REQUIRED_COLUMNS,
        "sector_abs_alpha_share",
    )
    _require_columns(
        mean_quantile_returns,
        QUANTILE_REQUIRED_COLUMNS,
        "mean_quantile_returns",
    )

    confirmed_names = confirm_factor_names(
        full_summary=full_summary,
        sector_neutral_summary=sector_neutral_summary,
        factor_metadata=factor_metadata,
    )

    original_rows = _pivot_summary_metrics(full_summary)
    sector_neutral_rows = _pivot_summary_metrics(sector_neutral_summary)
    audit = pd.concat([original_rows, sector_neutral_rows], ignore_index=True)
    audit = audit.drop_duplicates(subset="factor_name", keep="first")

    audit = _attach_declared_factor_metadata(audit, factor_metadata)
    audit = _attach_sector_neutral_metrics(audit, sector_neutral_rows)
    audit = _attach_paired_spread_deltas(audit, paired_delta_tests)
    audit = _attach_sector_concentration(audit, sector_abs_alpha_share)
    audit = _attach_mean_quantile_returns(audit, mean_quantile_returns)

    # Include every confirmed factor exactly once. This protects against accidental
    # row loss during pivots and merges.
    observed_names = set(audit["factor_name"])
    expected_names = set(confirmed_names["all_factors"])
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise ValueError(
            "Audit factor mismatch after merge. "
            f"Missing={missing}; unexpected={unexpected}."
        )

    audit = _order_factors(audit, factor_metadata)

    # Reindex creates explicit NaN columns when an optional diagnostic is not
    # available. Missing values are intentionally preserved rather than filled.
    return audit.reindex(columns=OUTPUT_COLUMNS)


def build_factor_selection_audit_metadata(
    *,
    input_paths: Mapping[str, str | Path],
    output_csv_path: str | Path,
    factor_names: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Create the companion JSON metadata for the audit table."""

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_csv": str(output_csv_path),
        "input_files": {name: str(path) for name, path in input_paths.items()},
        "factor_names_from_files": {
            name: list(values) for name, values in factor_names.items()
        },
        "horizons": list(HORIZONS),
        "ic_method": "Spearman rank correlation; mean monthly rank IC",
        "spread_definition": "Mean monthly Q5 minus Q1 forward return",
        "sector_neutralization_method": (
            "Within each rebalance date and sector, subtract the sector mean "
            "and divide by the sector sample standard deviation; sectors require "
            "at least three available stocks."
        ),
        "sector_concentration_method": (
            "Latest-date share of total absolute alpha by sector; the audit "
            "records the largest sector and its share when available."
        ),
        "selection_protocol": [
            "Prefer point-in-time SEC signals over static baselines.",
            "Emphasize 63D performance only when consistent across rank IC, "
            "Q5-Q1 spread, positive-spread rate, and robustness checks.",
            "Prefer factors with strong Q5-Q1 spread and positive spread rate.",
            "Prefer factors that remain positive after sector neutralization.",
            "Prefer economically interpretable engineered factors.",
            "Do not select a factor from a single p-value.",
            "Do not add an opaque composite score or expand the factor search.",
        ],
        "missing_value_policy": (
            "Unavailable diagnostics remain null/NaN; no metric is imputed or "
            "invented."
        ),
        "composite_score": None,
        "ranking_rule": None,
    }


def generate_factor_selection_audit(
    full_summary_path: str | Path = DEFAULT_FULL_SUMMARY_PATH,
    sector_neutral_summary_path: str | Path = DEFAULT_SECTOR_NEUTRAL_SUMMARY_PATH,
    paired_delta_path: str | Path = DEFAULT_PAIRED_DELTA_PATH,
    sector_share_path: str | Path = DEFAULT_SECTOR_SHARE_PATH,
    quantile_returns_path: str | Path = DEFAULT_QUANTILE_RETURNS_PATH,
    output_csv_path: str | Path = DEFAULT_OUTPUT_CSV_PATH,
    output_metadata_path: str | Path | None = DEFAULT_OUTPUT_METADATA_PATH,
    factor_metadata: Mapping[str, Mapping[str, Any]] = FACTOR_METADATA,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load saved diagnostics, build the audit, and write CSV and JSON outputs."""

    input_paths = {
        "full_summary": Path(full_summary_path),
        "sector_neutral_summary": Path(sector_neutral_summary_path),
        "paired_delta_tests": Path(paired_delta_path),
        "sector_abs_alpha_share": Path(sector_share_path),
        "mean_quantile_returns": Path(quantile_returns_path),
    }

    inputs = load_factor_selection_inputs(
        full_summary_path=input_paths["full_summary"],
        sector_neutral_summary_path=input_paths["sector_neutral_summary"],
        paired_delta_path=input_paths["paired_delta_tests"],
        sector_share_path=input_paths["sector_abs_alpha_share"],
        quantile_returns_path=input_paths["mean_quantile_returns"],
    )

    factor_names = confirm_factor_names(
        full_summary=inputs["full_summary"],
        sector_neutral_summary=inputs["sector_neutral_summary"],
        factor_metadata=factor_metadata,
    )
    audit = build_factor_selection_audit(
        full_summary=inputs["full_summary"],
        sector_neutral_summary=inputs["sector_neutral_summary"],
        paired_delta_tests=inputs["paired_delta_tests"],
        sector_abs_alpha_share=inputs["sector_abs_alpha_share"],
        mean_quantile_returns=inputs["mean_quantile_returns"],
        factor_metadata=factor_metadata,
    )

    output_csv = Path(output_csv_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_csv, index=False)

    metadata = build_factor_selection_audit_metadata(
        input_paths=input_paths,
        output_csv_path=output_csv,
        factor_names=factor_names,
    )

    if output_metadata_path is not None:
        metadata_path = Path(output_metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return audit, metadata


def _require_columns(df: pd.DataFrame, required: set[str], table_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _normalize_horizon(value: object) -> str:
    text = str(value).strip().upper()
    if text.endswith(".0"):
        text = text[:-2]
    if text.endswith("D"):
        text = text[:-1]
    try:
        horizon = int(text)
    except ValueError as exc:
        raise ValueError(f"Unsupported horizon value: {value!r}") from exc
    return f"{horizon}D"


def _pivot_summary_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    work = summary.copy()
    work["signal_name"] = work["signal_name"].astype(str)
    work["horizon"] = work["horizon"].map(_normalize_horizon)

    duplicates = work.duplicated(subset=["signal_name", "horizon"], keep=False)
    if duplicates.any():
        duplicate_rows = work.loc[duplicates, ["signal_name", "horizon"]]
        raise ValueError(
            "Summary contains duplicate factor/horizon rows: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )

    rows: list[dict[str, Any]] = []
    for factor_name, factor_rows in work.groupby("signal_name", sort=False):
        by_horizon = factor_rows.set_index("horizon")
        reference = (
            by_horizon.loc["63D"]
            if "63D" in by_horizon.index
            else by_horizon.iloc[0]
        )

        row: dict[str, Any] = {
            "factor_name": factor_name,
            "coverage": reference.get("mean_coverage_rate", np.nan),
            "n_dates": reference.get("n_dates", np.nan),
            "n_tickers": reference.get("n_tickers", np.nan),
            "mean_n_available": reference.get("mean_n_available", np.nan),
            "mean_turnover": reference.get("mean_turnover", np.nan),
        }

        for horizon in HORIZONS:
            suffix = horizon.lower()
            if horizon not in by_horizon.index:
                row[f"mean_rank_ic_{suffix}"] = np.nan
                row[f"q5_q1_spread_{suffix}"] = np.nan
                continue

            horizon_row = by_horizon.loc[horizon]
            row[f"mean_rank_ic_{suffix}"] = horizon_row.get("mean_ic", np.nan)
            row[f"q5_q1_spread_{suffix}"] = horizon_row.get(
                "mean_spread",
                np.nan,
            )

            if horizon == "63D":
                row["p_value_63d_spread"] = horizon_row.get(
                    "spread_p_value",
                    np.nan,
                )
                row["positive_spread_rate_63d"] = horizon_row.get(
                    "positive_spread_rate",
                    np.nan,
                )

        rows.append(row)

    return pd.DataFrame(rows)


def _attach_declared_factor_metadata(
    audit: pd.DataFrame,
    factor_metadata: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    metadata_df = pd.DataFrame.from_dict(factor_metadata, orient="index")
    metadata_df.index.name = "factor_name"
    metadata_df = metadata_df.reset_index()
    return audit.merge(metadata_df, on="factor_name", how="left", validate="one_to_one")


def _attach_sector_neutral_metrics(
    audit: pd.DataFrame,
    sector_neutral_rows: pd.DataFrame,
) -> pd.DataFrame:
    available_names = set(sector_neutral_rows["factor_name"])
    lookup = sector_neutral_rows.set_index("factor_name")

    columns = {
        "mean_rank_ic_63d": "sector_neutral_ic_63d",
        "q5_q1_spread_63d": "sector_neutral_q5_q1_63d",
        "p_value_63d_spread": "sector_neutral_p_value_63d_spread",
        "positive_spread_rate_63d": "sector_neutral_positive_spread_rate_63d",
    }

    out = audit.copy()
    for source_column, output_column in columns.items():
        values = []
        for factor_name in out["factor_name"]:
            counterpart = f"{factor_name}_sector_neutral"
            if counterpart in available_names:
                values.append(lookup.at[counterpart, source_column])
            else:
                values.append(np.nan)
        out[output_column] = values

    return out


def _attach_paired_spread_deltas(
    audit: pd.DataFrame,
    paired_delta_tests: pd.DataFrame,
) -> pd.DataFrame:
    paired = paired_delta_tests.copy()
    paired["horizon"] = paired["horizon"].map(_normalize_horizon)
    paired["kind"] = paired["kind"].astype(str).str.lower()
    paired = paired.loc[
        (paired["horizon"] == "63D") & (paired["kind"] == "spread")
    ]

    duplicates = paired.duplicated(
        subset=["candidate_signal", "baseline_signal"],
        keep=False,
    )
    if duplicates.any():
        duplicate_rows = paired.loc[
            duplicates,
            ["candidate_signal", "baseline_signal"],
        ]
        raise ValueError(
            "Paired delta input contains duplicate 63D spread rows: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )

    out = audit.copy()
    for baseline in ("static_tfidf", "static_embedding"):
        baseline_rows = paired.loc[paired["baseline_signal"] == baseline].set_index(
            "candidate_signal"
        )
        out[f"paired_spread_delta_vs_{baseline}_63d"] = out["factor_name"].map(
            baseline_rows["mean_delta"]
        )
        out[f"paired_spread_delta_p_value_vs_{baseline}_63d"] = out[
            "factor_name"
        ].map(baseline_rows["p_value"])
        out[f"paired_spread_delta_win_rate_vs_{baseline}_63d"] = out[
            "factor_name"
        ].map(baseline_rows["candidate_win_rate"])

    return out


def _attach_sector_concentration(
    audit: pd.DataFrame,
    sector_abs_alpha_share: pd.DataFrame,
) -> pd.DataFrame:
    sector_share = sector_abs_alpha_share.copy()
    sector_share["sector"] = sector_share["sector"].astype(str)

    largest_sector: dict[str, str] = {}
    largest_share: dict[str, float] = {}

    for factor_name in sector_share.columns:
        if factor_name == "sector":
            continue
        values = pd.to_numeric(sector_share[factor_name], errors="coerce")
        if not values.notna().any():
            continue
        index = values.idxmax()
        largest_sector[factor_name] = sector_share.at[index, "sector"]
        largest_share[factor_name] = float(values.at[index])

    out = audit.copy()
    out["latest_sector_abs_alpha_largest_sector"] = out["factor_name"].map(
        largest_sector
    )
    out["latest_sector_abs_alpha_max_share"] = out["factor_name"].map(
        largest_share
    )
    return out


def _attach_mean_quantile_returns(
    audit: pd.DataFrame,
    mean_quantile_returns: pd.DataFrame,
) -> pd.DataFrame:
    quantiles = mean_quantile_returns.copy()
    quantiles["signal_name"] = quantiles["signal_name"].astype(str)
    quantiles["horizon"] = quantiles["horizon"].map(_normalize_horizon)

    duplicates = quantiles.duplicated(
        subset=["signal_name", "horizon"],
        keep=False,
    )
    if duplicates.any():
        duplicate_rows = quantiles.loc[duplicates, ["signal_name", "horizon"]]
        raise ValueError(
            "Mean quantile return input contains duplicate factor/horizon rows: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )

    lookup = quantiles.set_index(["signal_name", "horizon"])
    out = audit.copy()

    for horizon in HORIZONS:
        for quantile in QUANTILES:
            output_column = (
                f"mean_quantile_return_q{quantile}_{horizon.lower()}"
            )
            values = []
            for factor_name in out["factor_name"]:
                key = (factor_name, horizon)
                if key in lookup.index:
                    values.append(lookup.at[key, str(quantile)])
                else:
                    values.append(np.nan)
            out[output_column] = values

    return out


def _order_factors(
    audit: pd.DataFrame,
    factor_metadata: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    declared_order = {name: index for index, name in enumerate(factor_metadata)}
    out = audit.copy()
    out["_factor_order"] = out["factor_name"].map(declared_order)
    out = out.sort_values(["_factor_order", "factor_name"], na_position="last")
    return out.drop(columns="_factor_order").reset_index(drop=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the factor-selection audit from saved diagnostic CSV files."
        )
    )
    parser.add_argument("--full-summary", default=DEFAULT_FULL_SUMMARY_PATH)
    parser.add_argument(
        "--sector-neutral-summary",
        default=DEFAULT_SECTOR_NEUTRAL_SUMMARY_PATH,
    )
    parser.add_argument("--paired-deltas", default=DEFAULT_PAIRED_DELTA_PATH)
    parser.add_argument("--sector-share", default=DEFAULT_SECTOR_SHARE_PATH)
    parser.add_argument(
        "--quantile-returns",
        default=DEFAULT_QUANTILE_RETURNS_PATH,
    )
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV_PATH)
    parser.add_argument(
        "--output-metadata",
        default=DEFAULT_OUTPUT_METADATA_PATH,
        help="Pass an empty string to skip the metadata JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    metadata_path = args.output_metadata or None

    audit, _ = generate_factor_selection_audit(
        full_summary_path=args.full_summary,
        sector_neutral_summary_path=args.sector_neutral_summary,
        paired_delta_path=args.paired_deltas,
        sector_share_path=args.sector_share,
        quantile_returns_path=args.quantile_returns,
        output_csv_path=args.output_csv,
        output_metadata_path=metadata_path,
    )

    print(f"Wrote {len(audit)} factor rows to {args.output_csv}")
    if metadata_path is not None:
        print(f"Wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
