"""
Project runner for the thematic signal comparison project.

This file keeps the original static Yahoo-description project path and adds a
separate SEC/OpenAI mode for the new point-in-time filing pipeline.

Usage examples
--------------

Run the original static Yahoo description project:

    python run_project.py

or explicitly:

    python run_project.py static

Run a dry-run SEC/OpenAI pipeline on the first 20 tickers:

    python run_project.py sec-openai \
        --limit 20 \
        --max-filings 8 \
        --dry-run \
        --edgar-identity "Your Name your.email@example.com"

Run a live SEC/OpenAI pipeline on the first 50 tickers:

    python run_project.py sec-openai \
        --limit 50 \
        --batch-size 25 \
        --max-filings 8 \
        --edgar-identity "Your Name your.email@example.com"

Run one selected batch from the S&P 500 universe:

    python run_project.py sec-openai \
        --batch-size 25 \
        --start-batch 0 \
        --end-batch 0 \
        --max-filings 8 \
        --edgar-identity "Your Name your.email@example.com"

Design notes
------------

The original static mode remains the baseline:

    Yahoo company descriptions
        -> TF-IDF / embedding theme similarity
        -> normalized alpha
        -> optimizer / backtest

The new SEC/OpenAI mode only creates the measurement layer:

    EdgarTools filings
        -> selected 10-K / 10-Q text
        -> OpenAI structured filing summaries
        -> cached summary table

It deliberately does not build alpha, run diagnostics, or optimize a portfolio.
Those steps will be added later through separate modules such as sec_signal.py,
sec_panel.py, and signal_diagnostics.py.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from time import perf_counter
from typing import Iterable

import pandas as pd

from src.config import (
    END_DATE,
    OUT_DIR,
    PATH_IN,
    SHEET,
    START_DATE,
    THEME_TEXT,
)


# -----------------------------------------------------------------------------
# Small command-line helpers
# -----------------------------------------------------------------------------


SUPPORTED_FORMS_FOR_SEC_OPENAI = {"10-K", "10-Q"}


# Yahoo-style class-share tickers and EdgarTools-style class-share tickers are
# not always the same. The project keeps the original project ticker in outputs,
# but uses the EdgarTools ticker only for lookup.
EDGAR_TICKER_OVERRIDES = {
    "BRKB": "BRK-B",
    "BRK.B": "BRK-B",
    "BFB": "BF-B",
    "BF.B": "BF-B",
}


def parse_comma_separated_tickers(value: str | None) -> list[str] | None:
    """Parse an optional comma-separated ticker list.

    Returning None means the caller should load the universe from the standard
    project spreadsheet. Returning a list means the caller should use only those
    explicit tickers.
    """

    if value is None:
        return None

    tickers = []

    for item in value.split(","):
        ticker = item.strip().upper()

        if ticker:
            tickers.append(ticker)

    if not tickers:
        raise ValueError("--tickers was provided but no tickers were parsed")

    return tickers


def normalize_project_ticker(ticker: str) -> str:
    """Normalize a ticker for project outputs and joins."""

    cleaned = str(ticker).strip().upper()

    if not cleaned:
        raise ValueError("ticker must not be blank")

    return cleaned


def to_edgar_ticker(project_ticker: str) -> str:
    """Map a project ticker to the ticker expected by EdgarTools."""

    project_ticker = normalize_project_ticker(project_ticker)

    return EDGAR_TICKER_OVERRIDES.get(project_ticker, project_ticker)


def get_filing_form(filing) -> str:
    """Read and normalize the filing form from an EdgarTools filing object."""

    return str(getattr(filing, "form", "")).strip().upper()


def get_filing_accession(filing) -> str | None:
    """Read an accession number from a filing object when available."""

    for attribute_name in ("accession_no", "accession_number"):
        if hasattr(filing, attribute_name):
            value = getattr(filing, attribute_name)

            if value is not None:
                return str(value).strip()

    return None


def build_batches(items: list[str], batch_size: int) -> list[list[str]]:
    """Split a list into simple consecutive batches."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")

    return [
        items[start : start + int(batch_size)]
        for start in range(0, len(items), int(batch_size))
    ]


def select_batches(
    batches: list[list[str]],
    *,
    start_batch: int,
    end_batch: int | None,
) -> list[tuple[int, list[str]]]:
    """Select a range of numbered batches for this run."""

    if int(start_batch) < 0:
        raise ValueError("start_batch must be non-negative")

    if end_batch is None:
        end_batch = len(batches) - 1

    if int(end_batch) < int(start_batch):
        raise ValueError("end_batch must be greater than or equal to start_batch")

    selected = []

    for batch_index, batch in enumerate(batches):
        if batch_index < int(start_batch):
            continue

        if batch_index > int(end_batch):
            continue

        selected.append((batch_index, batch))

    return selected


def write_pickle_dataframe(rows: list[dict], path: Path) -> pd.DataFrame:
    """Write a list of dictionaries as a pickle DataFrame and return it."""

    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows)
    dataframe.to_pickle(path)

    return dataframe


# -----------------------------------------------------------------------------
# Static Yahoo-description baseline
# -----------------------------------------------------------------------------


def run_static_project() -> None:
    """Run the original static Yahoo-description project pipeline."""

    # These imports are intentionally local. The static pipeline uses the
    # OpenAI embedding client through signal_methods.py, while the sec-openai
    # command uses sec_openai.py. Keeping imports local makes each mode clearer.
    from src.alpha_model import build_alpha
    from src.backtest import build_all_weight_schedules, run_all_backtests
    from src.data_ingest import (
        add_company_descriptions,
        download_prices,
        load_universe,
    )
    from src.metrics import compute_perf_stats
    from src.signal_methods import (
        add_embedding_similarity,
        add_embeddings,
        add_tfidf_similarity,
    )

    # Load text data.
    df = load_universe(PATH_IN, SHEET)
    df = add_company_descriptions(df)
    df = df[df["Company Description"].astype(str).str.strip() != ""].copy()

    # Build TF-IDF signal and normalize it.
    df = add_tfidf_similarity(df, THEME_TEXT)
    df["alpha_tfidf"] = build_alpha(df["tfidf_similarity"])

    # Build embedding signal and normalize it.
    df = add_embeddings(df)
    df = add_embedding_similarity(df, THEME_TEXT)
    df["alpha_embedding"] = build_alpha(df["embedding_similarity"])

    df.to_pickle(OUT_DIR / "scored_universe.pkl")

    # Download prices and create target-weight schedules.
    ticker_list = df["Ticker"].tolist()
    prices = download_prices(ticker_list, START_DATE, END_DATE)

    # Create alpha Series indexed by ticker name.
    alpha_tfidf = df.set_index("Ticker")["alpha_tfidf"].dropna()
    alpha_embedding = df.set_index("Ticker")["alpha_embedding"].dropna()

    # Run weight schedules.
    schedules = build_all_weight_schedules(prices, alpha_tfidf, alpha_embedding)

    for name, schedule in schedules.items():
        schedule.to_csv(OUT_DIR / f"weights_{name}.csv")

    # Run backtests and save results.
    returns_df, values_df = run_all_backtests(prices, schedules)
    returns_df.to_csv(OUT_DIR / "returns_all_strategies.csv")
    values_df.to_csv(OUT_DIR / "portfolio_values_all_strategies.csv")

    # Print simple summaries.
    print("\nTop 10 names by TF-IDF:")
    print(
        df[["Ticker", "tfidf_similarity", "alpha_tfidf"]]
        .sort_values("tfidf_similarity", ascending=False)
        .head(10)
    )

    print("\nTop 10 names by embeddings:")
    print(
        df[["Ticker", "embedding_similarity", "alpha_embedding"]]
        .sort_values("embedding_similarity", ascending=False)
        .head(10)
    )

    print("\nPerformance summary:")
    for strategy_name in returns_df.columns:
        stats = compute_perf_stats(returns_df[strategy_name].dropna())
        print(strategy_name, stats)


# -----------------------------------------------------------------------------
# SEC/OpenAI summary-generation mode
# -----------------------------------------------------------------------------


def load_tickers_for_sec_openai(args: argparse.Namespace) -> list[str]:
    """Load the ticker list for the SEC/OpenAI run."""

    explicit_tickers = parse_comma_separated_tickers(args.tickers)

    if explicit_tickers is not None:
        tickers = explicit_tickers
    else:
        from src.data_ingest import load_universe

        universe_df = load_universe(PATH_IN, SHEET)

        if "Ticker" not in universe_df.columns:
            raise ValueError("Expected a Ticker column in the universe file")

        tickers = (
            universe_df["Ticker"]
            .astype(str)
            .str.strip()
            .str.upper()
            .dropna()
            .drop_duplicates()
            .tolist()
        )

    tickers = [normalize_project_ticker(ticker) for ticker in tickers]

    if args.offset:
        tickers = tickers[int(args.offset) :]

    if args.limit is not None:
        tickers = tickers[: int(args.limit)]

    if not tickers:
        raise ValueError("No tickers selected for SEC/OpenAI run")

    return tickers


def build_sec_openai_theme(args: argparse.Namespace):
    """Build the ThemeConfig used by the OpenAI SEC summary prompt."""

    from src.sec_models import ThemeConfig

    return ThemeConfig(
        theme_name=args.theme_name,
        theme_description=args.theme_text,
        prompt_version=args.prompt_version,
    )


def build_selected_inputs_for_tickers(
    *,
    tickers: Iterable[str],
    args: argparse.Namespace,
    manifest_dir: Path,
    selected_text_dir: Path,
    raw_text_dir: Path,
) -> tuple[list, list[dict], list[dict], list[dict], list]:
    """Retrieve filings and build SelectedFilingInput objects.

    Returns
    -------
    selected_inputs:
        Valid SelectedFilingInput objects ready for sec_openai.py.

    selected_input_rows:
        Plain dictionaries for an audit table.

    error_rows:
        Plain dictionaries describing non-fatal errors.

    skipped_rows:
        Plain dictionaries describing intentionally skipped filings.

    all_records:
        Valid FilingRecord objects for manifest/audit output.
    """

    from src import sec_edgartools as se

    selected_inputs = []
    selected_input_rows = []
    error_rows = []
    skipped_rows = []
    all_records = []

    tickers = list(tickers)

    for ticker_index, project_ticker in enumerate(tickers, start=1):
        project_ticker = normalize_project_ticker(project_ticker)
        edgar_ticker = to_edgar_ticker(project_ticker)

        print(
            f"\n[{ticker_index}/{len(tickers)}] "
            f"{project_ticker} via EdgarTools={edgar_ticker}"
        )

        try:
            filings = se.get_recent_company_filings(
                ticker=edgar_ticker,
                forms=("10-K", "10-Q"),
                max_filings=args.max_filings,
            )

            print(f"  filings retrieved: {len(filings)}")

            supported_filings = []

            for filing in filings:
                form = get_filing_form(filing)

                if form not in SUPPORTED_FORMS_FOR_SEC_OPENAI:
                    skipped_rows.append(
                        {
                            "ticker": project_ticker,
                            "edgar_ticker": edgar_ticker,
                            "form": form,
                            "filing_date": getattr(filing, "filing_date", None),
                            "accession_number": get_filing_accession(filing),
                            "reason": "unsupported_amended_or_nonstandard_form",
                        }
                    )

                    print(
                        "    skipped",
                        form,
                        getattr(filing, "filing_date", None),
                        get_filing_accession(filing),
                    )

                    continue

                supported_filings.append(filing)

            records = se.filings_to_records(
                ticker=project_ticker,
                filings=supported_filings,
                continue_on_error=True,
            )

            all_records.extend(records)

            manifest_path = manifest_dir / f"{project_ticker}_filing_manifest.pkl"
            se.save_filing_records_pickle(records, manifest_path)

            print(f"  supported filings: {len(supported_filings)}")
            print(f"  records validated: {len(records)}")

            for filing in supported_filings:
                try:
                    record = se.filing_to_record(project_ticker, filing)

                    selected_input = se.build_selected_input(
                        record=record,
                        filing=filing,
                        output_dir=selected_text_dir,
                        max_chars=args.max_chars,
                        cache_raw=args.cache_raw,
                        raw_output_dir=raw_text_dir,
                    )

                    selected_inputs.append(selected_input)

                    selected_input_rows.append(
                        {
                            "ticker": selected_input.filing.ticker,
                            "edgar_ticker": edgar_ticker,
                            "form": selected_input.filing.form,
                            "filing_date": selected_input.filing.filing_date,
                            "period_of_report": selected_input.filing.period_of_report,
                            "accession_number": selected_input.filing.accession_number,
                            "selected_sections": " | ".join(
                                selected_input.selected_sections
                            ),
                            "selected_text_path": selected_input.selected_text.path,
                            "selected_text_chars": selected_input.selected_text.char_count,
                            "selected_text_words": selected_input.selected_text.word_count,
                            "raw_filing_cached": selected_input.raw_filing_text is not None,
                        }
                    )

                    print(
                        "   ",
                        record.form,
                        record.filing_date,
                        record.accession_number,
                        selected_input.selected_text.char_count,
                        "chars",
                    )

                except Exception as exc:
                    error_rows.append(
                        {
                            "stage": "build_selected_input",
                            "ticker": project_ticker,
                            "edgar_ticker": edgar_ticker,
                            "form": get_filing_form(filing),
                            "filing_date": getattr(filing, "filing_date", None),
                            "accession_number": get_filing_accession(filing),
                            "error": repr(exc),
                        }
                    )

                    print("    selected input error:", repr(exc))

        except Exception as exc:
            error_rows.append(
                {
                    "stage": "ticker_retrieval",
                    "ticker": project_ticker,
                    "edgar_ticker": edgar_ticker,
                    "error": repr(exc),
                }
            )

            print("  ticker error:", repr(exc))

        if args.sleep_sec_between_tickers > 0:
            time.sleep(float(args.sleep_sec_between_tickers))

    return (
        selected_inputs,
        selected_input_rows,
        error_rows,
        skipped_rows,
        all_records,
    )


def run_sec_openai(args: argparse.Namespace) -> None:
    """Run the SEC/OpenAI summary-generation pipeline."""

    from src import sec_edgartools as se
    from src import sec_openai as so

    sec_out_dir = Path(args.output_dir)
    manifest_dir = sec_out_dir / "manifests"
    selected_text_dir = sec_out_dir / "selected_text"
    raw_text_dir = sec_out_dir / "raw_text"
    summary_dir = sec_out_dir / "summaries"

    for path in [manifest_dir, selected_text_dir, raw_text_dir, summary_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # Use separate dry-run and live caches so placeholder records do not mix
    # with actual OpenAI outputs.
    if args.summary_cache_path is not None:
        summary_cache_path = Path(args.summary_cache_path)
    elif args.dry_run:
        summary_cache_path = summary_dir / "dry_openai_summary_cache.pkl"
    else:
        summary_cache_path = summary_dir / "openai_summary_cache.pkl"

    summary_table_path = Path(args.summary_table_path) if args.summary_table_path else (
        summary_dir / "openai_summary_table.pkl"
    )

    selected_input_audit_path = summary_dir / "selected_input_audit.pkl"
    error_log_path = summary_dir / "errors.pkl"
    skipped_log_path = summary_dir / "skipped_filings.pkl"
    filing_records_path = manifest_dir / "filing_records.pkl"

    edgar_identity = args.edgar_identity or os.getenv("EDGAR_IDENTITY")

    if not edgar_identity:
        raise ValueError(
            "Provide --edgar-identity or set EDGAR_IDENTITY in the environment"
        )

    se.configure_edgar_identity(edgar_identity)

    tickers = load_tickers_for_sec_openai(args)
    batches = build_batches(tickers, args.batch_size)
    selected_batches = select_batches(
        batches,
        start_batch=args.start_batch,
        end_batch=args.end_batch,
    )

    print("\nSEC/OpenAI run configuration")
    print("----------------------------")
    print("tickers selected:", len(tickers))
    print("batch size:", args.batch_size)
    print("total batches:", len(batches))
    print("batches in this run:", [index for index, _ in selected_batches])
    print("max filings per ticker:", args.max_filings)
    print("max chars per filing:", args.max_chars)
    print("dry run:", args.dry_run)
    print("force refresh:", args.force_refresh)
    print("cache raw filings:", args.cache_raw)
    print("output directory:", sec_out_dir)
    print("summary cache:", summary_cache_path)
    print("summary table:", summary_table_path)

    theme = build_sec_openai_theme(args)

    client = None

    if not args.dry_run:
        client = so.create_openai_client()

    run_selected_inputs = []
    selected_input_rows = []
    error_rows = []
    skipped_rows = []
    all_records = []

    t0 = perf_counter()

    for batch_index, batch_tickers in selected_batches:
        print("\n" + "=" * 80)
        print(f"Starting SEC/OpenAI batch {batch_index}")
        print("Tickers:", ", ".join(batch_tickers))
        print("=" * 80)

        (
            batch_selected_inputs,
            batch_selected_input_rows,
            batch_error_rows,
            batch_skipped_rows,
            batch_records,
        ) = build_selected_inputs_for_tickers(
            tickers=batch_tickers,
            args=args,
            manifest_dir=manifest_dir,
            selected_text_dir=selected_text_dir,
            raw_text_dir=raw_text_dir,
        )

        run_selected_inputs.extend(batch_selected_inputs)
        selected_input_rows.extend(batch_selected_input_rows)
        error_rows.extend(batch_error_rows)
        skipped_rows.extend(batch_skipped_rows)
        all_records.extend(batch_records)

        print(
            f"\nBatch {batch_index} selected inputs: {len(batch_selected_inputs)}"
        )

        if not batch_selected_inputs:
            print(f"Batch {batch_index} has no selected inputs. Skipping OpenAI stage.")
            continue

        def progress_callback(index, total, selected_input):
            filing = selected_input.filing
            print(
                f"Summary {index}/{total}: "
                f"{filing.ticker} {filing.form} {filing.filing_date}"
            )

        batch_records_summary = so.summarize_selected_inputs(
            selected_inputs=batch_selected_inputs,
            theme=theme,
            summary_model=args.summary_model,
            embedding_model=so.DEFAULT_EMBEDDING_MODEL,
            cache_path=summary_cache_path,
            client=client,
            dry_run=args.dry_run,
            force_refresh=args.force_refresh,
            max_attempts=args.max_attempts,
            retry_wait_seconds=args.retry_wait_seconds,
            progress_callback=progress_callback,
        )

        batch_summary_table_path = summary_dir / f"batch_{batch_index:03d}_summary_table.pkl"
        so.save_summary_table_pickle(batch_records_summary, batch_summary_table_path)

        print(f"Saved batch summary table: {batch_summary_table_path}")

    # Save run-level audit artifacts. These describe the selected batches in
    # this invocation, not necessarily every historical batch ever run.
    write_pickle_dataframe(selected_input_rows, selected_input_audit_path)
    write_pickle_dataframe(error_rows, error_log_path)
    write_pickle_dataframe(skipped_rows, skipped_log_path)

    if all_records:
        se.save_filing_records_pickle(all_records, filing_records_path)

    # Save a cumulative summary table from the full cache. This means repeated
    # runs with force_refresh=False build up one reusable project output table.
    if summary_cache_path.exists():
        cumulative_cache = so.load_summary_cache(summary_cache_path)
        cumulative_records = list(cumulative_cache.values())
        so.save_summary_table_pickle(cumulative_records, summary_table_path)

        print(f"\nCumulative summary records: {len(cumulative_records)}")
        print(f"Saved cumulative summary table: {summary_table_path}")

    elapsed = perf_counter() - t0

    print("\nSEC/OpenAI run complete")
    print("-----------------------")
    print("selected inputs this run:", len(run_selected_inputs))
    print("errors this run:", len(error_rows))
    print("skipped filings this run:", len(skipped_rows))
    print("elapsed minutes:", elapsed / 60.0)

    if run_selected_inputs:
        print("minutes per selected filing:", (elapsed / 60.0) / len(run_selected_inputs))

    print("\nOutput files")
    print("------------")
    print("selected input audit:", selected_input_audit_path)
    print("error log:", error_log_path)
    print("skipped filing log:", skipped_log_path)
    print("summary cache:", summary_cache_path)
    print("summary table:", summary_table_path)


# -----------------------------------------------------------------------------
# Command-line parser
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run the thematic signal comparison project."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=False,
    )

    # Existing static Yahoo-description project path.
    subparsers.add_parser(
        "static",
        help="Run the original Yahoo-description TF-IDF / embedding project.",
    )

    # New SEC/OpenAI summary-generation path.
    sec_parser = subparsers.add_parser(
        "sec-openai",
        help="Build SEC selected text and OpenAI structured summary cache.",
    )

    sec_parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Optional comma-separated ticker list, for example NVDA,MSFT,AAPL.",
    )
    sec_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of tickers to skip before applying --limit.",
    )
    sec_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tickers to process after --offset.",
    )
    sec_parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of tickers in each batch.",
    )
    sec_parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="First batch index to run, using zero-based numbering.",
    )
    sec_parser.add_argument(
        "--end-batch",
        type=int,
        default=None,
        help="Last batch index to run, using zero-based numbering.",
    )
    sec_parser.add_argument(
        "--max-filings",
        type=int,
        default=8,
        help="Maximum recent 10-K / 10-Q filings to retrieve per ticker.",
    )
    sec_parser.add_argument(
        "--max-chars",
        type=int,
        default=40_000,
        help="Maximum selected-text characters per filing.",
    )
    sec_parser.add_argument(
        "--cache-raw",
        action="store_true",
        help="Also save raw filing text for debugging. Off by default.",
    )
    sec_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create schema-valid placeholder summaries without OpenAI calls.",
    )
    sec_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached summaries and recreate them. Use carefully.",
    )
    sec_parser.add_argument(
        "--edgar-identity",
        type=str,
        default=None,
        help=(
            "SEC EdgarTools identity string. You can also set EDGAR_IDENTITY "
            "in the environment."
        ),
    )
    sec_parser.add_argument(
        "--summary-model",
        type=str,
        default=None,
        help=(
            "OpenAI summary model. Defaults to src.sec_openai.DEFAULT_SUMMARY_MODEL."
        ),
    )
    sec_parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum OpenAI attempts per filing.",
    )
    sec_parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=2.0,
        help="Base wait time for OpenAI retry backoff.",
    )
    sec_parser.add_argument(
        "--sleep-sec-between-tickers",
        type=float,
        default=0.0,
        help="Optional pause between ticker retrievals.",
    )
    sec_parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR / "sec_pit_openai",
        help="Output directory for SEC/OpenAI artifacts.",
    )
    sec_parser.add_argument(
        "--summary-cache-path",
        type=Path,
        default=None,
        help="Optional explicit summary cache path.",
    )
    sec_parser.add_argument(
        "--summary-table-path",
        type=Path,
        default=None,
        help="Optional explicit cumulative summary table path.",
    )
    sec_parser.add_argument(
        "--theme-name",
        type=str,
        default="AI infrastructure",
        help="Research theme name stored in summary records.",
    )
    sec_parser.add_argument(
        "--theme-text",
        type=str,
        default=THEME_TEXT,
        help="Research theme definition used in the summary prompt.",
    )
    sec_parser.add_argument(
        "--prompt-version",
        type=str,
        default="sec_theme_summary_v1",
        help="Prompt version stored in ThemeConfig.",
    )

    return parser


def main() -> None:
    """Parse command-line arguments and run the selected project mode."""

    parser = build_parser()
    args = parser.parse_args()

    # Preserve old behavior: `python run_project.py` runs the original static
    # Yahoo-description project.
    if args.command is None or args.command == "static":
        run_static_project()
        return

    if args.command == "sec-openai":
        # Import here so parser help and static runs do not depend on sec_openai.
        from src import sec_openai as so

        if args.summary_model is None:
            args.summary_model = so.DEFAULT_SUMMARY_MODEL

        run_sec_openai(args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
