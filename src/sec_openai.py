"""
OpenAI structured-summary layer for point-in-time SEC filing research.

This module begins where `sec_edgartools.py` ends.

Input:
    SelectedFilingInput

Output:
    FilingSummaryRecord

The module has five clear responsibilities:

    1. Read and revalidate the exact focused filing text saved on disk.
    2. Build an explicit theme-specific prompt.
    3. Request a Pydantic-validated FilingThemeSummary from OpenAI.
    4. Cache completed summary records so API calls are not repeated.
    5. Convert summary records into a flat pandas table for inspection.

The module does not:

    - download SEC filings,
    - extract filing sections,
    - create embeddings,
    - calculate theme similarity,
    - normalize alpha,
    - run signal diagnostics,
    - construct portfolios.

Those tasks remain in their own project modules.

Project integration and design principles
-----------------------------------------
This module follows the conventions already used by the project:

    - python-dotenv loads variables from the local .env file,
    - OPENAI_API_KEY is read from the environment,
    - the embedding model comes from config.EMBED_MODEL,
    - embedding creation and cosine similarity remain in signal_methods.py.

The module deliberately avoids creating an OpenAI client at import time. This
means researchers can still:

    - import the module without making an API request,
    - run unit tests without network access,
    - inspect prompts in dry-run mode,
    - inject a fake client during testing.

The exact focused text in `selected_input.selected_text` is used for the model
request. The optional full filing in `selected_input.raw_filing_text` is an
archive/debug artifact and is not sent to OpenAI by this module.

Cache format
------------
The project uses trusted local `.pkl` files for working outputs. The summary
cache therefore stores plain Python dictionaries in a pickle file. Each
dictionary is revalidated as a FilingSummaryRecord when loaded.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from .config import EMBED_MODEL
from .sec_models import (
    FilingSummaryRecord,
    FilingThemeSummary,
    SelectedFilingInput,
    ThemeConfig,
    sha256_text,
)


# -----------------------------------------------------------------------------
# Project environment and default configuration
# -----------------------------------------------------------------------------


# Load variables from the project's local .env file when one is present.
#
# This is the same convention already used in signal_methods.py. The call does
# not create an OpenAI client and does not make a network request. It simply
# makes values such as OPENAI_API_KEY and OPENAI_SUMMARY_MODEL available through
# os.getenv(). Variables already exported by the shell remain supported.
load_dotenv()


# The summary model is specific to the SEC summarization stage. Keeping it in an
# environment variable lets the researcher change models without editing code.
DEFAULT_SUMMARY_MODEL = os.getenv(
    "OPENAI_SUMMARY_MODEL",
    "gpt-4.1-mini",
)


# Reuse the embedding model already defined in the project's config.py. This
# field is stored in FilingSummaryRecord for reproducibility, but this module
# does not calculate embeddings. The later sec_signal.py module will reuse
# get_embedding() and cosine_similarity() from signal_methods.py.
DEFAULT_EMBEDDING_MODEL = EMBED_MODEL


DEFAULT_SYSTEM_PROMPT = """
You are analyzing selected sections from an SEC 10-K or 10-Q filing for a
research project.

Use only the filing text supplied by the user. Do not use outside knowledge,
recent news, market prices, or assumptions about the company.

Assess the company's exposure to the specified research theme. Distinguish
direct operating exposure from indirect, minimal, absent, or unclear exposure.

Support every conclusion with evidence from the supplied filing text. If the
text is incomplete, truncated, repetitive, or does not support a conclusion,
state that uncertainty rather than guessing.

Do not recommend buying, selling, or holding the security. Return only the
structured response requested by the schema.
""".strip()


# -----------------------------------------------------------------------------
# OpenAI client creation
# -----------------------------------------------------------------------------


def create_openai_client(
    *,
    api_key: str | None = None,
    client_factory: Callable[..., Any] | None = None,
) -> Any:
    """Create the OpenAI client used for structured SEC summaries.

    The project calls load_dotenv() above, so an OPENAI_API_KEY stored in the
    local .env file is available through os.getenv(). An explicit key can still
    be supplied for a controlled test or a different execution environment.

    The client is created only when this function is called. Importing this
    module therefore does not make a network request and still supports dry-run
    and unit-test workflows.

    Args:
        api_key:
            Optional explicit API key. When omitted, OPENAI_API_KEY is read
            from the environment, including values loaded from .env.

        client_factory:
            Optional constructor used by unit tests. Normal production code
            leaves this as None, which uses openai.OpenAI.

    Returns:
        A configured OpenAI client object.

    Raises:
        ValueError:
            If no API key is available.
    """

    # Prefer an explicitly supplied key, then use the project environment.
    resolved_api_key = (
        api_key
        or os.getenv("OPENAI_API_KEY")
    )

    if not resolved_api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in .env or the current environment"
        )

    # The injectable constructor keeps the function easy to test without
    # importing or creating a real SDK client.
    if client_factory is None:
        try:
            from openai import OpenAI

        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI Python package is required for live summary "
                "calls. Install it with: python -m pip install openai"
            ) from exc

        client_factory = OpenAI

    return client_factory(
        api_key=resolved_api_key
    )


# -----------------------------------------------------------------------------
# Selected-text validation and prompt construction
# -----------------------------------------------------------------------------


def read_selected_text(
    selected_input: SelectedFilingInput,
    *,
    max_chars: int | None = None,
) -> str:
    """Read and revalidate the exact focused filing text."""

    path = Path(selected_input.selected_text.path)

    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_file():
        raise ValueError(
            f"selected text path is not a file: {path}"
        )

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if len(text) != selected_input.selected_text.char_count:
        raise ValueError(
            "selected text character count no longer matches TextArtifact"
        )

    if len(text.split()) != selected_input.selected_text.word_count:
        raise ValueError(
            "selected text word count no longer matches TextArtifact"
        )

    if sha256_text(text) != selected_input.selected_text.sha256:
        raise ValueError(
            "selected text SHA-256 no longer matches TextArtifact"
        )

    if max_chars is None:
        return text

    if int(max_chars) <= 0:
        raise ValueError("max_chars must be positive when provided")

    return text[: int(max_chars)]


def build_section_audit_text(
    selected_input: SelectedFilingInput,
) -> str:
    """Create a compact human-readable section audit for the prompt."""

    if selected_input.section_extraction_details:
        lines: list[str] = []

        for detail in selected_input.section_extraction_details:
            line = (
                f"- {detail.section_name}: "
                f"requested={detail.requested}, "
                f"found={detail.found}, "
                f"original_chars={detail.original_char_count}, "
                f"retained_chars={detail.retained_char_count}, "
                f"char_limit={detail.char_limit}, "
                f"truncated={detail.truncated}"
            )

            lines.append(line)

        return "\n".join(lines)

    return "\n".join(
        f"- {section_name}: selected"
        for section_name in selected_input.selected_sections
    )


def build_user_prompt(
    selected_input: SelectedFilingInput,
    theme: ThemeConfig,
    *,
    selected_text: str | None = None,
) -> str:
    """Build the explicit user prompt for one filing summary request."""

    filing = selected_input.filing

    if selected_text is None:
        selected_text = read_selected_text(selected_input)

    section_names = ", ".join(
        selected_input.selected_sections
    )

    section_audit = build_section_audit_text(
        selected_input
    )

    return f"""
Research theme
--------------
Theme name: {theme.theme_name}
Theme definition: {theme.theme_description}
Prompt version: {theme.prompt_version}

Filing metadata
---------------
Ticker: {filing.ticker}
Company: {filing.company_name}
Form: {filing.form}
Filing date: {filing.filing_date}
Period of report: {filing.period_of_report}
Accession number: {filing.accession_number}
Selected sections: {section_names}

Section extraction audit
------------------------
{section_audit}

Research task
-------------
Using only the selected filing text below, assess:

1. Whether the company has Direct, Indirect, Minimal, None, or Unclear
   exposure to the research theme.
2. Whether the exposure appears Increasing, Stable, Decreasing, or Unclear.
3. The strength of the filing evidence: High, Medium, Low, or Unclear.
4. The strongest filing-specific evidence supporting the assessment.
5. Risks, limitations, or uncertainties that weaken the assessment.

Interpretation rules
--------------------
- Use only the selected filing text below.
- Do not use outside knowledge.
- Do not infer facts that are absent from the filing text.
- Treat boilerplate language cautiously.
- Distinguish current operating evidence from generic statements.
- A marker reading [SECTION MIDDLE TRUNCATED] means the section was sampled
  from its beginning and end because it exceeded the research text budget.
- Keep the main summary approximately 150 to 220 words.
- If the evidence is insufficient or contradictory, use Unclear.
- Do not provide investment advice.

Selected filing text
--------------------
{selected_text}
""".strip()


# -----------------------------------------------------------------------------
# Dry-run summary
# -----------------------------------------------------------------------------


def make_dry_run_summary(
    selected_input: SelectedFilingInput,
) -> FilingThemeSummary:
    """Create a schema-valid placeholder without calling OpenAI."""

    filing = selected_input.filing

    return FilingThemeSummary(
        theme_exposure_summary=(
            f"Dry-run summary for {filing.ticker}'s {filing.form} filed on "
            f"{filing.filing_date}. This placeholder confirms that the "
            "structured SEC summary pipeline can create, validate, cache, and "
            "reload a FilingThemeSummary. Dry-run mode does not interpret the "
            "selected filing text, estimate actual thematic exposure, or make "
            "claims about the company's operations, direction, evidence, or "
            "investment characteristics."
        ),
        exposure_label="Unclear",
        exposure_direction="Unclear",
        evidence_strength="Low",
        key_evidence=[
            (
                "Dry-run placeholder evidence. Replace this record with "
                "model-extracted filing evidence during a live run."
            )
        ],
        risks_or_uncertainties=[
            (
                "Dry-run mode does not inspect or interpret the selected "
                "filing text."
            )
        ],
    )


# -----------------------------------------------------------------------------
# OpenAI structured-summary call
# -----------------------------------------------------------------------------


def _parse_structured_response(
    response: Any,
) -> FilingThemeSummary:
    """Convert an OpenAI response wrapper into FilingThemeSummary."""

    output_parsed = getattr(
        response,
        "output_parsed",
        None,
    )

    if isinstance(output_parsed, FilingThemeSummary):
        return output_parsed

    if output_parsed is not None:
        return FilingThemeSummary.model_validate(
            output_parsed
        )

    output_text = getattr(
        response,
        "output_text",
        None,
    )

    if output_text:
        return FilingThemeSummary.model_validate_json(
            output_text
        )

    raise RuntimeError(
        "OpenAI response contained neither output_parsed nor output_text"
    )


def _summarize_once(
    *,
    client: Any,
    selected_input: SelectedFilingInput,
    theme: ThemeConfig,
    summary_model: str,
    system_prompt: str,
) -> FilingThemeSummary:
    """Perform one structured summary request without retry logic."""

    if client is None:
        raise ValueError(
            "client must be provided for a live OpenAI summary call"
        )

    summary_model = str(summary_model).strip()

    if not summary_model:
        raise ValueError("summary_model must not be blank")

    selected_text = read_selected_text(
        selected_input
    )

    user_prompt = build_user_prompt(
        selected_input=selected_input,
        theme=theme,
        selected_text=selected_text,
    )

    responses_api = getattr(
        client,
        "responses",
        None,
    )

    if responses_api is None:
        raise AttributeError(
            "client does not expose a responses API"
        )

    parse_method = getattr(
        responses_api,
        "parse",
        None,
    )

    if callable(parse_method):
        response = parse_method(
            model=summary_model,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            text_format=FilingThemeSummary,
        )

        return _parse_structured_response(
            response
        )

    create_method = getattr(
        responses_api,
        "create",
        None,
    )

    if not callable(create_method):
        raise AttributeError(
            "client.responses exposes neither parse nor create"
        )

    response = create_method(
        model=summary_model,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "filing_theme_summary",
                "schema": (
                    FilingThemeSummary
                    .model_json_schema()
                ),
                "strict": True,
            }
        },
    )

    return _parse_structured_response(
        response
    )


def summarize_with_openai(
    *,
    client: Any,
    selected_input: SelectedFilingInput,
    theme: ThemeConfig,
    summary_model: str = DEFAULT_SUMMARY_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_attempts: int = 3,
    retry_wait_seconds: float = 2.0,
    sleep_func: Callable[[float], None] = time.sleep,
) -> FilingThemeSummary:
    """Call OpenAI with simple, explicit retry behavior."""

    if int(max_attempts) <= 0:
        raise ValueError("max_attempts must be positive")

    if float(retry_wait_seconds) < 0:
        raise ValueError(
            "retry_wait_seconds must be non-negative"
        )

    last_exception: Exception | None = None

    for attempt_number in range(
        1,
        int(max_attempts) + 1,
    ):
        try:
            return _summarize_once(
                client=client,
                selected_input=selected_input,
                theme=theme,
                summary_model=summary_model,
                system_prompt=system_prompt,
            )

        except Exception as exc:
            last_exception = exc

            if attempt_number >= int(max_attempts):
                raise

            wait_seconds = (
                float(retry_wait_seconds)
                * attempt_number
            )

            sleep_func(wait_seconds)

    if last_exception is not None:
        raise last_exception

    raise RuntimeError(
        "summary retry loop exited unexpectedly"
    )


# -----------------------------------------------------------------------------
# Summary cache
# -----------------------------------------------------------------------------


def make_summary_cache_key(
    *,
    selected_input: SelectedFilingInput,
    theme: ThemeConfig,
    summary_model: str = DEFAULT_SUMMARY_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Create a stable cache key for one summary request."""

    filing = selected_input.filing

    selected_sections = ",".join(
        selected_input.selected_sections
    )

    raw_key = "|".join(
        [
            filing.ticker,
            filing.accession_number,
            selected_input.selected_text.sha256,
            selected_sections,
            theme.theme_name,
            theme.theme_description,
            theme.prompt_version,
            str(summary_model).strip(),
            sha256_text(system_prompt),
        ]
    )

    return hashlib.sha256(
        raw_key.encode("utf-8")
    ).hexdigest()


def load_summary_cache(
    path: str | Path,
) -> dict[str, FilingSummaryRecord]:
    """Load and revalidate the trusted local `.pkl` summary cache."""

    path = Path(path)

    if not path.exists():
        return {}

    raw_cache = pd.read_pickle(path)

    if not isinstance(raw_cache, dict):
        raise TypeError(
            "summary cache pickle must contain a dictionary"
        )

    validated_cache: dict[
        str,
        FilingSummaryRecord,
    ] = {}

    for cache_key, raw_record in raw_cache.items():
        validated_cache[str(cache_key)] = (
            FilingSummaryRecord.model_validate(
                raw_record
            )
        )

    return validated_cache


def save_summary_cache(
    cache: dict[str, FilingSummaryRecord],
    path: str | Path,
) -> Path:
    """Save summary records as plain dictionaries in a trusted local pickle."""

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_cache = {
        cache_key: record.model_dump(
            mode="python"
        )
        for cache_key, record in cache.items()
    }

    pd.to_pickle(
        serializable_cache,
        path,
    )

    return path


# -----------------------------------------------------------------------------
# FilingSummaryRecord construction and orchestration
# -----------------------------------------------------------------------------


def build_summary_record(
    *,
    selected_input: SelectedFilingInput,
    theme: ThemeConfig,
    summary: FilingThemeSummary,
    summary_model: str = DEFAULT_SUMMARY_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    created_at_utc: datetime | None = None,
) -> FilingSummaryRecord:
    """Combine the input, theme, model names, and structured summary."""

    if created_at_utc is None:
        created_at_utc = datetime.now(
            timezone.utc
        )

    return FilingSummaryRecord(
        selected_input=selected_input,
        theme=theme,
        summary_model=summary_model,
        embedding_model=embedding_model,
        created_at_utc=created_at_utc,
        summary=summary,
    )


def get_or_create_summary_record(
    *,
    selected_input: SelectedFilingInput,
    theme: ThemeConfig,
    summary_model: str = DEFAULT_SUMMARY_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    cache: dict[str, FilingSummaryRecord] | None = None,
    cache_path: str | Path | None = None,
    client: Any | None = None,
    dry_run: bool = True,
    force_refresh: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_attempts: int = 3,
    retry_wait_seconds: float = 2.0,
    sleep_func: Callable[[float], None] = time.sleep,
    created_at_utc: datetime | None = None,
) -> FilingSummaryRecord:
    """Load a cached summary or create and cache a new one."""

    if cache is None:
        if cache_path is None:
            cache = {}
        else:
            cache = load_summary_cache(
                cache_path
            )

    cache_key = make_summary_cache_key(
        selected_input=selected_input,
        theme=theme,
        summary_model=summary_model,
        system_prompt=system_prompt,
    )

    if (
        not force_refresh
        and cache_key in cache
    ):
        return cache[cache_key]

    if dry_run:
        summary = make_dry_run_summary(
            selected_input
        )

    else:
        if client is None:
            raise ValueError(
                "client must be provided when dry_run=False"
            )

        summary = summarize_with_openai(
            client=client,
            selected_input=selected_input,
            theme=theme,
            summary_model=summary_model,
            system_prompt=system_prompt,
            max_attempts=max_attempts,
            retry_wait_seconds=retry_wait_seconds,
            sleep_func=sleep_func,
        )

    record = build_summary_record(
        selected_input=selected_input,
        theme=theme,
        summary=summary,
        summary_model=summary_model,
        embedding_model=embedding_model,
        created_at_utc=created_at_utc,
    )

    cache[cache_key] = record

    if cache_path is not None:
        save_summary_cache(
            cache,
            cache_path,
        )

    return record


def summarize_selected_inputs(
    selected_inputs: Sequence[
        SelectedFilingInput
    ],
    *,
    theme: ThemeConfig,
    summary_model: str = DEFAULT_SUMMARY_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    cache_path: str | Path | None = None,
    client: Any | None = None,
    dry_run: bool = True,
    force_refresh: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_attempts: int = 3,
    retry_wait_seconds: float = 2.0,
    sleep_func: Callable[[float], None] = time.sleep,
    progress_callback: Callable[
        [int, int, SelectedFilingInput],
        None,
    ]
    | None = None,
) -> list[FilingSummaryRecord]:
    """Summarize a sequence of selected filing inputs."""

    cache = (
        load_summary_cache(cache_path)
        if cache_path is not None
        else {}
    )

    records: list[FilingSummaryRecord] = []
    total = len(selected_inputs)

    for index, selected_input in enumerate(
        selected_inputs,
        start=1,
    ):
        record = get_or_create_summary_record(
            selected_input=selected_input,
            theme=theme,
            summary_model=summary_model,
            embedding_model=embedding_model,
            cache=cache,
            cache_path=cache_path,
            client=client,
            dry_run=dry_run,
            force_refresh=force_refresh,
            system_prompt=system_prompt,
            max_attempts=max_attempts,
            retry_wait_seconds=retry_wait_seconds,
            sleep_func=sleep_func,
        )

        records.append(record)

        if progress_callback is not None:
            progress_callback(
                index,
                total,
                selected_input,
            )

    return records


# -----------------------------------------------------------------------------
# Flat inspection table
# -----------------------------------------------------------------------------


def flatten_summary_record(
    record: FilingSummaryRecord,
) -> dict[str, Any]:
    """Flatten one nested summary record into an analysis-friendly row."""

    filing = record.selected_input.filing
    summary = record.summary

    return {
        "ticker": filing.ticker,
        "cik": filing.cik,
        "company_name": filing.company_name,
        "form": filing.form,
        "filing_date": filing.filing_date,
        "period_of_report": filing.period_of_report,
        "accession_number": filing.accession_number,
        "acceptance_datetime": filing.acceptance_datetime,
        "selected_sections": " | ".join(
            record.selected_input.selected_sections
        ),
        "selected_text_path": (
            record.selected_input.selected_text.path
        ),
        "selected_text_sha256": (
            record.selected_input.selected_text.sha256
        ),
        "selected_text_chars": (
            record.selected_input.selected_text.char_count
        ),
        "selected_text_words": (
            record.selected_input.selected_text.word_count
        ),
        "raw_filing_cached": (
            record.selected_input.raw_filing_text
            is not None
        ),
        "theme_name": record.theme.theme_name,
        "theme_description": (
            record.theme.theme_description
        ),
        "prompt_version": (
            record.theme.prompt_version
        ),
        "summary_model": record.summary_model,
        "embedding_model": record.embedding_model,
        "created_at_utc": record.created_at_utc,
        "theme_exposure_summary": (
            summary.theme_exposure_summary
        ),
        "exposure_label": (
            summary.exposure_label
        ),
        "exposure_direction": (
            summary.exposure_direction
        ),
        "evidence_strength": (
            summary.evidence_strength
        ),
        "key_evidence": " | ".join(
            summary.key_evidence
        ),
        "risks_or_uncertainties": " | ".join(
            summary.risks_or_uncertainties
        ),
    }


def summary_records_to_dataframe(
    records: Sequence[FilingSummaryRecord],
) -> pd.DataFrame:
    """Convert validated summary records into a flat pandas DataFrame."""

    rows = [
        flatten_summary_record(record)
        for record in records
    ]

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        return dataframe

    return dataframe.sort_values(
        [
            "ticker",
            "filing_date",
            "accession_number",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)


def save_summary_table_pickle(
    records: Sequence[FilingSummaryRecord],
    path: str | Path,
) -> Path:
    """Save the flat summary inspection table as a `.pkl` file."""

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = summary_records_to_dataframe(
        records
    )

    dataframe.to_pickle(path)

    return path


__all__ = [
    "DEFAULT_SUMMARY_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_SYSTEM_PROMPT",
    "create_openai_client",
    "read_selected_text",
    "build_section_audit_text",
    "build_user_prompt",
    "make_dry_run_summary",
    "summarize_with_openai",
    "make_summary_cache_key",
    "load_summary_cache",
    "save_summary_cache",
    "build_summary_record",
    "get_or_create_summary_record",
    "summarize_selected_inputs",
    "flatten_summary_record",
    "summary_records_to_dataframe",
    "save_summary_table_pickle",
]
