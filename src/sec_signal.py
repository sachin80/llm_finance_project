"""
Filing-level numeric signal construction for point-in-time SEC summaries.

This module begins where `sec_openai.py` ends.

Input:
    FilingSummaryRecord

Output:
    FilingSignalRecord

The module has five clear responsibilities:

    1. Convert one OpenAI filing summary into text for embedding.
    2. Embed the filing summary and the research theme.
    3. Compute summary-to-theme cosine similarity.
    4. Build validated FilingSignalRecord objects.
    5. Convert signal records into flat pandas tables for diagnostics.

The module does not:

    - retrieve SEC filings,
    - extract filing sections,
    - call the OpenAI Responses API,
    - normalize alpha,
    - build point-in-time monthly panels,
    - run signal diagnostics,
    - construct portfolios.

Those tasks remain in their own project modules.

Design principles
-----------------
The OpenAI structured summary is the measurement layer. This module is the
first numerical signal layer.

The first version intentionally keeps the signal components separate:

    - summary-to-theme similarity,
    - exposure label score,
    - exposure direction score,
    - evidence strength score.

Later modules can test combinations, neutralization, and rolling alpha models.
This module should not silently collapse everything into one opaque score.

The embedding and cosine helpers are reused from `signal_methods.py`, but they
are imported lazily. This keeps unit tests from creating an OpenAI client at
import time and lets tests inject fake embedding functions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import EMBED_MODEL
from .sec_models import (
    FilingSignalRecord,
    FilingSummaryRecord,
    ThemeConfig,
)


# -----------------------------------------------------------------------------
# Default configuration
# -----------------------------------------------------------------------------


# Version identifier for the first SEC filing signal experiment.
#
# This is deliberately separate from the OpenAI prompt version. The prompt
# version describes how the structured summary was created. The signal version
# describes how the numeric filing-level signal was created from that summary.
DEFAULT_SIGNAL_VERSION = "sec_summary_similarity_v1"


# -----------------------------------------------------------------------------
# Dependency helpers
# -----------------------------------------------------------------------------


def _resolve_embedding_func(
    embedding_func: Callable[..., Any] | None,
) -> Callable[..., Any]:
    """Return an injected embedding function or the project default."""

    if embedding_func is not None:
        return embedding_func

    # Import lazily so tests can import this module without requiring a live
    # OpenAI key. The existing project function remains the production path.
    from .signal_methods import get_embedding

    return get_embedding


def _resolve_similarity_func(
    similarity_func: Callable[..., float] | None,
) -> Callable[..., float]:
    """Return an injected similarity function or the project default."""

    if similarity_func is not None:
        return similarity_func

    # Import lazily for the same reason as _resolve_embedding_func.
    from .signal_methods import cosine_similarity

    return cosine_similarity


def _call_embedding_func(
    embedding_func: Callable[..., Any],
    text: str,
    *,
    model: str,
) -> Any:
    """Call an embedding function while supporting simple test doubles."""

    try:
        return embedding_func(text, model=model)

    except TypeError:
        # Unit tests sometimes use a minimal fake function that only accepts
        # text. Production get_embedding accepts both text and model.
        return embedding_func(text)


def _validate_embedding_vector(
    vector: Any,
    *,
    label: str,
) -> Any:
    """Reject empty embedding vectors before cosine similarity."""

    if vector is None:
        raise ValueError(f"{label} embedding is None")

    try:
        vector_length = len(vector)

    except TypeError as exc:
        raise TypeError(
            f"{label} embedding must be a sequence of numbers"
        ) from exc

    if vector_length == 0:
        raise ValueError(f"{label} embedding is empty")

    return vector


# -----------------------------------------------------------------------------
# Text construction for embeddings
# -----------------------------------------------------------------------------


def build_theme_embedding_text(
    theme: ThemeConfig,
) -> str:
    """Build the text embedded for the research theme."""

    return (
        f"Theme name: {theme.theme_name}\n"
        f"Theme definition: {theme.theme_description}\n"
        f"Prompt version: {theme.prompt_version}"
    )


def build_summary_embedding_text(
    record: FilingSummaryRecord,
    *,
    include_key_evidence: bool = True,
    include_risks: bool = True,
) -> str:
    """Build the text embedded for one filing summary.

    The main text is the model's theme_exposure_summary. Evidence and risks are
    optionally included because they can contain the filing-specific language
    that justifies the classification.

    This text is not sent back to the OpenAI Responses API. It is only used for
    the embedding-based similarity calculation.
    """

    summary = record.summary

    pieces: list[str] = [
        f"Filing theme exposure summary: {summary.theme_exposure_summary}",
    ]

    if include_key_evidence and summary.key_evidence:
        evidence_text = "\n".join(
            f"- {item}"
            for item in summary.key_evidence
        )

        pieces.append(
            "Key filing evidence:\n"
            f"{evidence_text}"
        )

    if include_risks and summary.risks_or_uncertainties:
        risk_text = "\n".join(
            f"- {item}"
            for item in summary.risks_or_uncertainties
        )

        pieces.append(
            "Risks or uncertainties:\n"
            f"{risk_text}"
        )

    return "\n\n".join(pieces)


def make_theme_embedding_cache_key(
    *,
    theme: ThemeConfig,
    embedding_model: str,
) -> tuple[str, str, str, str]:
    """Create a stable key for reusing one theme embedding."""

    return (
        str(embedding_model),
        theme.theme_name,
        theme.theme_description,
        theme.prompt_version,
    )


# -----------------------------------------------------------------------------
# Similarity calculation
# -----------------------------------------------------------------------------


def compute_summary_theme_similarity(
    record: FilingSummaryRecord,
    *,
    embedding_model: str | None = None,
    embedding_func: Callable[..., Any] | None = None,
    similarity_func: Callable[..., float] | None = None,
    theme_embedding_cache: dict[tuple[str, str, str, str], Any] | None = None,
) -> float:
    """Compute cosine similarity between one filing summary and its theme.

    Args:
        record:
            Validated OpenAI summary record.

        embedding_model:
            Embedding model name. If omitted, use the model stored on the
            summary record. If that is blank, use config.EMBED_MODEL.

        embedding_func:
            Optional injected embedding function for tests. Production uses
            signal_methods.get_embedding.

        similarity_func:
            Optional injected similarity function for tests. Production uses
            signal_methods.cosine_similarity.

        theme_embedding_cache:
            Optional dictionary used to avoid recomputing the same theme
            embedding for every filing.

    Returns:
        A cosine similarity value between -1 and 1.
    """

    if embedding_model is None:
        embedding_model = record.embedding_model or EMBED_MODEL

    embedding_model = str(embedding_model).strip()

    if not embedding_model:
        raise ValueError("embedding_model must not be blank")

    embedding_func = _resolve_embedding_func(embedding_func)
    similarity_func = _resolve_similarity_func(similarity_func)

    if theme_embedding_cache is None:
        theme_embedding_cache = {}

    theme_key = make_theme_embedding_cache_key(
        theme=record.theme,
        embedding_model=embedding_model,
    )

    if theme_key in theme_embedding_cache:
        theme_embedding = theme_embedding_cache[theme_key]

    else:
        theme_text = build_theme_embedding_text(record.theme)

        theme_embedding = _call_embedding_func(
            embedding_func,
            theme_text,
            model=embedding_model,
        )

        theme_embedding = _validate_embedding_vector(
            theme_embedding,
            label="theme",
        )

        theme_embedding_cache[theme_key] = theme_embedding

    summary_text = build_summary_embedding_text(record)

    summary_embedding = _call_embedding_func(
        embedding_func,
        summary_text,
        model=embedding_model,
    )

    summary_embedding = _validate_embedding_vector(
        summary_embedding,
        label="summary",
    )

    similarity = similarity_func(
        summary_embedding,
        theme_embedding,
    )

    similarity = float(similarity)

    if similarity < -1.000001 or similarity > 1.000001:
        raise ValueError(
            "summary-theme similarity must be between -1 and 1"
        )

    # The tiny tolerance above avoids failing because of numerical roundoff.
    # Store the cleaned value inside the strict model range.
    if similarity < -1.0:
        similarity = -1.0

    if similarity > 1.0:
        similarity = 1.0

    return similarity


# -----------------------------------------------------------------------------
# Signal-record construction
# -----------------------------------------------------------------------------


def build_signal_record(
    record: FilingSummaryRecord,
    *,
    embedding_model: str | None = None,
    embedding_func: Callable[..., Any] | None = None,
    similarity_func: Callable[..., float] | None = None,
    theme_embedding_cache: dict[tuple[str, str, str, str], Any] | None = None,
    signal_version: str = DEFAULT_SIGNAL_VERSION,
    created_at_utc: datetime | None = None,
) -> FilingSignalRecord:
    """Build one validated filing-level signal record."""

    if created_at_utc is None:
        created_at_utc = datetime.now(timezone.utc)

    similarity = compute_summary_theme_similarity(
        record,
        embedding_model=embedding_model,
        embedding_func=embedding_func,
        similarity_func=similarity_func,
        theme_embedding_cache=theme_embedding_cache,
    )

    return FilingSignalRecord.from_summary_record(
        summary_record=record,
        summary_theme_similarity=similarity,
        signal_version=signal_version,
        created_at_utc=created_at_utc,
    )


def build_signal_records(
    records: Sequence[FilingSummaryRecord],
    *,
    embedding_model: str | None = None,
    embedding_func: Callable[..., Any] | None = None,
    similarity_func: Callable[..., float] | None = None,
    signal_version: str = DEFAULT_SIGNAL_VERSION,
    created_at_utc: datetime | None = None,
    progress_callback: Callable[[int, int, FilingSummaryRecord], None] | None = None,
) -> list[FilingSignalRecord]:
    """Build signal records for a sequence of filing summaries."""

    signal_records: list[FilingSignalRecord] = []
    theme_embedding_cache: dict[tuple[str, str, str, str], Any] = {}

    total = len(records)

    for index, record in enumerate(records, start=1):
        signal_record = build_signal_record(
            record,
            embedding_model=embedding_model,
            embedding_func=embedding_func,
            similarity_func=similarity_func,
            theme_embedding_cache=theme_embedding_cache,
            signal_version=signal_version,
            created_at_utc=created_at_utc,
        )

        signal_records.append(signal_record)

        if progress_callback is not None:
            progress_callback(index, total, record)

    return signal_records


# -----------------------------------------------------------------------------
# Loading and saving records
# -----------------------------------------------------------------------------


def load_summary_records_pickle(
    path: str | Path,
) -> list[FilingSummaryRecord]:
    """Load nested FilingSummaryRecord objects from a pickle file.

    This function is intended for the OpenAI summary cache created by
    sec_openai.py. The flat openai_summary_table.pkl is useful for inspection,
    but it does not preserve the full nested Pydantic objects as directly.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    payload = pd.read_pickle(path)

    if isinstance(payload, pd.DataFrame):
        raise TypeError(
            "Expected a summary cache or list of FilingSummaryRecord objects. "
            "The flat summary table does not preserve nested records directly. "
            "Use outputs/sec_pit_openai/summaries/openai_summary_cache.pkl."
        )

    if isinstance(payload, FilingSummaryRecord):
        raw_records = [payload]

    elif isinstance(payload, dict):
        raw_records = list(payload.values())

    elif isinstance(payload, list | tuple):
        raw_records = list(payload)

    else:
        raise TypeError(
            "Unsupported summary pickle format. Expected dict, list, tuple, "
            "or FilingSummaryRecord."
        )

    records: list[FilingSummaryRecord] = []

    for item in raw_records:
        records.append(
            FilingSummaryRecord.model_validate(item)
        )

    return records


def save_signal_records_pickle(
    records: Sequence[FilingSignalRecord],
    path: str | Path,
) -> Path:
    """Save validated signal records as revalidatable dictionaries."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        record.model_dump(mode="python")
        for record in records
    ]

    pd.to_pickle(payload, path)

    return path


def load_signal_records_pickle(
    path: str | Path,
) -> list[FilingSignalRecord]:
    """Load and revalidate signal records saved by this module."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    payload = pd.read_pickle(path)

    if isinstance(payload, pd.DataFrame):
        raise TypeError(
            "Expected saved signal records, not a flat signal table."
        )

    if isinstance(payload, FilingSignalRecord):
        raw_records = [payload]

    elif isinstance(payload, dict):
        raw_records = list(payload.values())

    elif isinstance(payload, list | tuple):
        raw_records = list(payload)

    else:
        raise TypeError(
            "Unsupported signal pickle format. Expected dict, list, tuple, "
            "or FilingSignalRecord."
        )

    records: list[FilingSignalRecord] = []

    for item in raw_records:
        records.append(
            FilingSignalRecord.model_validate(item)
        )

    return records


# -----------------------------------------------------------------------------
# Flat inspection table
# -----------------------------------------------------------------------------


def _optional_product(
    left: float,
    right: int | float | None,
) -> float | None:
    """Multiply a signal by an optional score."""

    if right is None:
        return None

    return float(left) * float(right)


def flatten_signal_record(
    record: FilingSignalRecord,
) -> dict[str, Any]:
    """Flatten one nested signal record into an analysis-friendly row."""

    summary_record = record.summary_record
    selected_input = summary_record.selected_input
    filing = selected_input.filing
    theme = summary_record.theme
    summary = summary_record.summary

    direct_or_indirect_flag = (
        summary.exposure_label in ["Direct", "Indirect"]
    )

    if direct_or_indirect_flag:
        direct_or_indirect_similarity = record.summary_theme_similarity

    else:
        direct_or_indirect_similarity = 0.0

    return {
        # Filing metadata.
        "ticker": filing.ticker,
        "cik": filing.cik,
        "company_name": filing.company_name,
        "form": filing.form,
        "filing_date": filing.filing_date,
        "period_of_report": filing.period_of_report,
        "accession_number": filing.accession_number,
        "acceptance_datetime": filing.acceptance_datetime,
        # Selected-text metadata.
        "selected_sections": " | ".join(
            selected_input.selected_sections
        ),
        "selected_text_path": selected_input.selected_text.path,
        "selected_text_sha256": selected_input.selected_text.sha256,
        "selected_text_chars": selected_input.selected_text.char_count,
        "selected_text_words": selected_input.selected_text.word_count,
        # Theme and model metadata.
        "theme_name": theme.theme_name,
        "theme_description": theme.theme_description,
        "prompt_version": theme.prompt_version,
        "summary_model": summary_record.summary_model,
        "embedding_model": summary_record.embedding_model,
        "summary_created_at_utc": summary_record.created_at_utc,
        "signal_created_at_utc": record.created_at_utc,
        "signal_version": record.signal_version,
        # OpenAI structured labels.
        "exposure_label": summary.exposure_label,
        "exposure_direction": summary.exposure_direction,
        "evidence_strength": summary.evidence_strength,
        # Numeric signal components.
        "summary_theme_similarity": record.summary_theme_similarity,
        "exposure_label_score": record.exposure_label_score,
        "direction_score": record.direction_score,
        "evidence_strength_score": record.evidence_strength_score,
        # Simple first-pass variants for diagnostics. These are table columns,
        # not new source-of-truth fields in the Pydantic model.
        "similarity_x_exposure_score": _optional_product(
            record.summary_theme_similarity,
            record.exposure_label_score,
        ),
        "similarity_x_evidence_strength_score": _optional_product(
            record.summary_theme_similarity,
            record.evidence_strength_score,
        ),
        "similarity_x_direction_score": _optional_product(
            record.summary_theme_similarity,
            record.direction_score,
        ),
        "direct_or_indirect_flag": direct_or_indirect_flag,
        "direct_or_indirect_similarity": direct_or_indirect_similarity,
        # Text fields for audit.
        "theme_exposure_summary": summary.theme_exposure_summary,
        "key_evidence": " | ".join(summary.key_evidence),
        "risks_or_uncertainties": " | ".join(
            summary.risks_or_uncertainties
        ),
    }


def signal_records_to_dataframe(
    records: Sequence[FilingSignalRecord],
) -> pd.DataFrame:
    """Convert validated signal records into a flat pandas DataFrame."""

    rows = [
        flatten_signal_record(record)
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


def save_signal_table_pickle(
    records: Sequence[FilingSignalRecord],
    path: str | Path,
) -> Path:
    """Save the flat signal inspection table as a `.pkl` file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = signal_records_to_dataframe(records)

    dataframe.to_pickle(path)

    return path


__all__ = [
    "DEFAULT_SIGNAL_VERSION",
    "build_theme_embedding_text",
    "build_summary_embedding_text",
    "make_theme_embedding_cache_key",
    "compute_summary_theme_similarity",
    "build_signal_record",
    "build_signal_records",
    "load_summary_records_pickle",
    "save_signal_records_pickle",
    "load_signal_records_pickle",
    "flatten_signal_record",
    "signal_records_to_dataframe",
    "save_signal_table_pickle",
]
