"""
Unit tests for src.sec_signal.

These tests are intentionally explicit and use only temporary local files.
They do not call OpenAI, EdgarTools, the SEC website, Yahoo Finance, or any
other network service.

The goals are:

    1. verify filing-summary text is converted into embedding text,
    2. verify summary-to-theme similarity can be computed with fake embeddings,
    3. verify FilingSignalRecord objects use the canonical label-score maps,
    4. verify batch construction reuses the theme embedding,
    5. verify flat tables contain useful diagnostic columns,
    6. verify pickle round trips revalidate records,
    7. verify flat summary tables are rejected when nested records are needed.

Run from the repository root with:

    pytest -q tests/test_sec_signal.py
"""

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.sec_models import (
    FilingRecord,
    FilingSummaryRecord,
    FilingThemeSummary,
    SelectedFilingInput,
    TextArtifact,
    ThemeConfig,
    sha256_text,
)
from src.sec_signal import (
    DEFAULT_SIGNAL_VERSION,
    build_signal_record,
    build_signal_records,
    build_summary_embedding_text,
    build_theme_embedding_text,
    compute_summary_theme_similarity,
    flatten_signal_record,
    load_signal_records_pickle,
    load_summary_records_pickle,
    save_signal_records_pickle,
    save_signal_table_pickle,
    signal_records_to_dataframe,
)


# -----------------------------------------------------------------------------
# Test-data helpers
# -----------------------------------------------------------------------------


def write_text_file(path: Path, text: str) -> Path:
    """Write one deterministic UTF-8 text file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

    return path


def make_text_artifact(
    tmp_path: Path,
    *,
    filename: str = "selected.txt",
    text: str | None = None,
) -> TextArtifact:
    """Create a valid TextArtifact for tests."""

    if text is None:
        text = (
            "Management discussion describes AI infrastructure, accelerated "
            "computing demand, data-center products, and supply constraints."
        )

    path = write_text_file(tmp_path / filename, text)

    return TextArtifact(
        path=str(path),
        extraction_method="unit_test_selected_text",
        sha256=sha256_text(text),
        char_count=len(text),
        word_count=len(text.split()),
        created_at_utc=datetime(2026, 6, 17, 12, 0),
    )


def make_filing_record(**overrides) -> FilingRecord:
    """Create one valid FilingRecord."""

    data = {
        "ticker": "NVDA",
        "cik": 1045810,
        "company_name": "NVIDIA Corporation",
        "form": "10-Q",
        "filing_date": date(2026, 5, 20),
        "period_of_report": date(2026, 4, 27),
        "accession_number": "0001045810-26-000052",
        "primary_document": "nvda-20260427.htm",
        "acceptance_datetime": datetime(
            2026,
            5,
            20,
            16,
            22,
            tzinfo=timezone.utc,
        ),
    }

    data.update(overrides)

    return FilingRecord(**data)


def make_theme() -> ThemeConfig:
    """Create one valid theme definition."""

    return ThemeConfig(
        theme_name="AI infrastructure",
        theme_description=(
            "Companies with economically meaningful exposure to artificial "
            "intelligence infrastructure, including accelerated computing, "
            "GPUs, semiconductors, memory, networking, data centers, cloud "
            "compute capacity, power, cooling, and AI software infrastructure."
        ),
        prompt_version="sec_theme_summary_v1",
    )


def make_summary_record(
    tmp_path: Path,
    *,
    ticker: str = "NVDA",
    filing_date: date = date(2026, 5, 20),
    accession_number: str = "0001045810-26-000052",
    exposure_label: str = "Direct",
    exposure_direction: str = "Increasing",
    evidence_strength: str = "High",
) -> FilingSummaryRecord:
    """Create one valid FilingSummaryRecord."""

    filing = make_filing_record(
        ticker=ticker,
        filing_date=filing_date,
        period_of_report=filing_date,
        accession_number=accession_number,
    )

    text_artifact = make_text_artifact(
        tmp_path,
        filename=f"{ticker}_{accession_number}_selected.txt",
    )

    selected_input = SelectedFilingInput(
        filing=filing,
        selected_sections=["management_discussion"],
        selected_text=text_artifact,
    )

    summary = FilingThemeSummary(
        theme_exposure_summary=(
            "The selected filing text indicates economically meaningful "
            "exposure to AI infrastructure through accelerated computing, "
            "data-center demand, specialized hardware, and related customer "
            "adoption. The assessment is limited to the selected filing text "
            "and does not rely on market prices, outside news, or investment "
            "views."
        ),
        exposure_label=exposure_label,
        exposure_direction=exposure_direction,
        evidence_strength=evidence_strength,
        key_evidence=[
            (
                "The filing discusses accelerated computing and data-center "
                "demand as important operating drivers."
            )
        ],
        risks_or_uncertainties=[
            (
                "The selected sections may omit some detailed financial "
                "statement context."
            )
        ],
    )

    return FilingSummaryRecord(
        selected_input=selected_input,
        theme=make_theme(),
        summary_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-large",
        created_at_utc=datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc),
        summary=summary,
    )


def fake_embedding(text, model=None):
    """Return a simple deterministic embedding vector."""

    normalized = text.lower()

    if "ai infrastructure" in normalized:
        return [1.0, 0.0, 0.0]

    if "accelerated computing" in normalized or "data-center" in normalized:
        return [1.0, 0.0, 0.0]

    if "retail" in normalized:
        return [0.0, 1.0, 0.0]

    return [0.0, 0.0, 1.0]


def fake_similarity(left, right):
    """Return one for equal fake vectors and zero otherwise."""

    return 1.0 if list(left) == list(right) else 0.0


# -----------------------------------------------------------------------------
# Text construction tests
# -----------------------------------------------------------------------------


def test_build_theme_embedding_text_includes_theme_fields():
    """Theme embedding text should include name, definition, and version."""

    theme = make_theme()

    text = build_theme_embedding_text(theme)

    assert "AI infrastructure" in text
    assert "accelerated computing" in text
    assert "sec_theme_summary_v1" in text


def test_build_summary_embedding_text_includes_summary_evidence_and_risks(tmp_path):
    """Summary embedding text should include auditable filing evidence."""

    record = make_summary_record(tmp_path)

    text = build_summary_embedding_text(record)

    assert "Filing theme exposure summary" in text
    assert "Key filing evidence" in text
    assert "Risks or uncertainties" in text
    assert "accelerated computing" in text


# -----------------------------------------------------------------------------
# Similarity and signal-record tests
# -----------------------------------------------------------------------------


def test_compute_summary_theme_similarity_with_fake_embeddings(tmp_path):
    """Similarity calculation should work with injected fake functions."""

    record = make_summary_record(tmp_path)

    similarity = compute_summary_theme_similarity(
        record,
        embedding_func=fake_embedding,
        similarity_func=fake_similarity,
    )

    assert similarity == pytest.approx(1.0)


def test_compute_summary_theme_similarity_rejects_empty_embedding(tmp_path):
    """Empty embeddings should fail loudly."""

    record = make_summary_record(tmp_path)

    def empty_embedding(text, model=None):
        return []

    with pytest.raises(ValueError, match="embedding is empty"):
        compute_summary_theme_similarity(
            record,
            embedding_func=empty_embedding,
            similarity_func=fake_similarity,
        )


def test_build_signal_record_sets_scores_from_summary_labels(tmp_path):
    """Signal record should use canonical numeric scores from sec_models."""

    record = make_summary_record(
        tmp_path,
        exposure_label="Direct",
        exposure_direction="Increasing",
        evidence_strength="High",
    )

    signal_record = build_signal_record(
        record,
        embedding_func=fake_embedding,
        similarity_func=fake_similarity,
        created_at_utc=datetime(2026, 7, 1, 12, 0),
    )

    assert signal_record.summary_theme_similarity == pytest.approx(1.0)
    assert signal_record.exposure_label_score == 3
    assert signal_record.direction_score == 1
    assert signal_record.evidence_strength_score == 3
    assert signal_record.signal_version == DEFAULT_SIGNAL_VERSION
    assert signal_record.created_at_utc.tzinfo is not None


def test_build_signal_record_handles_unclear_scores(tmp_path):
    """Unclear labels should map to None numeric values."""

    record = make_summary_record(
        tmp_path,
        exposure_label="Unclear",
        exposure_direction="Unclear",
        evidence_strength="Unclear",
    )

    signal_record = build_signal_record(
        record,
        embedding_func=fake_embedding,
        similarity_func=fake_similarity,
    )

    assert signal_record.exposure_label_score is None
    assert signal_record.direction_score is None
    assert signal_record.evidence_strength_score is None


def test_build_signal_records_reuses_theme_embedding(tmp_path):
    """Batch construction should not recompute the same theme embedding."""

    records = [
        make_summary_record(
            tmp_path,
            ticker="NVDA",
            filing_date=date(2026, 5, 20),
            accession_number="0001045810-26-000052",
        ),
        make_summary_record(
            tmp_path,
            ticker="MSFT",
            filing_date=date(2026, 4, 29),
            accession_number="0001193125-26-191507",
        ),
    ]

    calls = []

    def counting_embedding(text, model=None):
        calls.append(text)
        return fake_embedding(text, model=model)

    signal_records = build_signal_records(
        records,
        embedding_func=counting_embedding,
        similarity_func=fake_similarity,
    )

    # Two summary embeddings plus one cached theme embedding.
    assert len(signal_records) == 2
    assert len(calls) == 3


# -----------------------------------------------------------------------------
# DataFrame and pickle tests
# -----------------------------------------------------------------------------


def test_flatten_signal_record_contains_diagnostic_columns(tmp_path):
    """Flat rows should expose raw components and simple diagnostic variants."""

    record = make_summary_record(tmp_path)

    signal_record = build_signal_record(
        record,
        embedding_func=fake_embedding,
        similarity_func=fake_similarity,
    )

    row = flatten_signal_record(signal_record)

    assert row["ticker"] == "NVDA"
    assert row["form"] == "10-Q"
    assert row["summary_theme_similarity"] == pytest.approx(1.0)
    assert row["exposure_label_score"] == 3
    assert row["similarity_x_exposure_score"] == pytest.approx(3.0)
    assert row["direct_or_indirect_flag"] is True
    assert row["direct_or_indirect_similarity"] == pytest.approx(1.0)


def test_signal_records_to_dataframe_sorts_records(tmp_path):
    """Signal table should sort by ticker and newest filing first."""

    older = make_summary_record(
        tmp_path,
        ticker="AAPL",
        filing_date=date(2025, 1, 31),
        accession_number="old00",
        exposure_label="Indirect",
    )

    newer = make_summary_record(
        tmp_path,
        ticker="AAPL",
        filing_date=date(2026, 5, 1),
        accession_number="new00",
        exposure_label="Indirect",
    )

    signal_records = build_signal_records(
        [older, newer],
        embedding_func=fake_embedding,
        similarity_func=fake_similarity,
    )

    dataframe = signal_records_to_dataframe(signal_records)

    assert list(dataframe["accession_number"]) == ["new00", "old00"]


def test_save_and_load_signal_records_pickle_round_trip(tmp_path):
    """Saved signal records should reload as validated Pydantic records."""

    record = make_summary_record(tmp_path)

    signal_record = build_signal_record(
        record,
        embedding_func=fake_embedding,
        similarity_func=fake_similarity,
    )

    path = tmp_path / "signal_records.pkl"

    save_signal_records_pickle([signal_record], path)

    loaded = load_signal_records_pickle(path)

    assert len(loaded) == 1
    assert loaded[0].summary_record.selected_input.filing.ticker == "NVDA"
    assert loaded[0].summary_theme_similarity == pytest.approx(1.0)


def test_save_signal_table_pickle_creates_dataframe(tmp_path):
    """Flat signal table should be saved as a pandas DataFrame."""

    record = make_summary_record(tmp_path)

    signal_record = build_signal_record(
        record,
        embedding_func=fake_embedding,
        similarity_func=fake_similarity,
    )

    path = tmp_path / "signal_table.pkl"

    save_signal_table_pickle([signal_record], path)

    dataframe = pd.read_pickle(path)

    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.loc[0, "ticker"] == "NVDA"


def test_load_summary_records_pickle_accepts_cache_dict(tmp_path):
    """Summary-cache dictionaries should load into summary records."""

    record = make_summary_record(tmp_path)

    path = tmp_path / "summary_cache.pkl"

    pd.to_pickle({"cache-key": record}, path)

    loaded = load_summary_records_pickle(path)

    assert len(loaded) == 1
    assert loaded[0].selected_input.filing.ticker == "NVDA"


def test_load_summary_records_pickle_rejects_flat_dataframe(tmp_path):
    """Flat summary tables should not be mistaken for nested records."""

    path = tmp_path / "flat_summary_table.pkl"

    pd.DataFrame({"ticker": ["NVDA"]}).to_pickle(path)

    with pytest.raises(TypeError, match="flat summary table"):
        load_summary_records_pickle(path)
