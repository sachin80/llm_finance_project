"""
Unit tests for src.sec_models.

These tests are intentionally explicit and use only temporary local files.
They do not call EdgarTools, the SEC website, OpenAI, Yahoo Finance, or any
other network service.

The goals are:

    1. verify normal records are accepted,
    2. verify ticker and text normalization,
    3. verify file hashes and text metadata,
    4. verify malformed records fail loudly,
    5. verify point-in-time selections cannot use future filings,
    6. verify numeric signal scores agree with structured labels.

Run from the repository root with:

    pytest -q tests/test_sec_models.py
"""

from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.sec_models import (
    DIRECTION_SCORE,
    EVIDENCE_STRENGTH_SCORE,
    EXPOSURE_LABEL_SCORE,
    FilingRecord,
    FilingSignalRecord,
    FilingSummaryRecord,
    FilingThemeSummary,
    PointInTimeFilingSelection,
    SectionExtractionDetail,
    SelectedFilingInput,
    TextArtifact,
    ThemeConfig,
    normalize_text,
    sha256_text,
)


# -----------------------------------------------------------------------------
# Test-data helpers
# -----------------------------------------------------------------------------


def write_text_file(path: Path, text: str) -> Path:
    """Write a deterministic UTF-8 text file for a test."""

    # Create any missing parent directory.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write the exact text that the model will later validate.
    path.write_text(text, encoding="utf-8")

    return path


def make_filing_record(**overrides) -> FilingRecord:
    """Build a valid filing record, with optional field overrides."""

    # Start from one realistic 10-Q metadata record.
    data = {
        "ticker": "nvda",
        "cik": 1045810,
        "company_name": "NVIDIA Corporation",
        "form": "10-Q",
        "filing_date": date(2024, 5, 29),
        "period_of_report": date(2024, 4, 28),
        "accession_number": "0001045810-24-000124",
        "primary_document": "nvda-20240428.htm",
        "acceptance_datetime": datetime(
            2024,
            5,
            29,
            16,
            22,
            tzinfo=timezone.utc,
        ),
    }

    # Replace any fields supplied by the individual test.
    data.update(overrides)

    # Validate and return the Pydantic model.
    return FilingRecord(**data)


def make_text_artifact(
    tmp_path: Path,
    *,
    filename: str = "selected.txt",
    text: str | None = None,
) -> TextArtifact:
    """Create a valid local text artifact for a test."""

    # Use enough text to resemble a real focused filing excerpt.
    if text is None:
        text = (
            "Business and management discussion describe accelerated computing, "
            "data-center demand, networking products, supply constraints, and "
            "customer concentration. Risk factors discuss competition, regulation, "
            "manufacturing capacity, and uncertainty in future demand."
        )

    # Save the exact text.
    path = write_text_file(tmp_path / filename, text)

    # Build metadata from the same exact text.
    return TextArtifact(
        path=str(path),
        extraction_method="edgartools_filing_obj_selected_sections",
        sha256=sha256_text(text),
        char_count=len(text),
        word_count=len(text.split()),
        created_at_utc=datetime(2026, 6, 17, 12, 0),
    )


def make_selected_input(tmp_path: Path) -> SelectedFilingInput:
    """Build one valid selected filing input."""

    return SelectedFilingInput(
        filing=make_filing_record(),
        selected_sections=[
            "management_discussion",
            "risk_factors",
        ],
        selected_text=make_text_artifact(tmp_path),
    )


def make_theme() -> ThemeConfig:
    """Build the versioned theme used throughout the tests."""

    return ThemeConfig(
        theme_name="AI infrastructure",
        theme_description=(
            "Companies with direct or indirect exposure to accelerated "
            "computing, AI servers, data centers, networking, semiconductor "
            "equipment, cloud infrastructure, and related power or cooling."
        ),
        prompt_version="sec_theme_summary_v1",
    )


def make_summary() -> FilingThemeSummary:
    """Build one valid structured filing summary."""

    return FilingThemeSummary(
        theme_exposure_summary=(
            "The filing provides direct evidence of exposure to AI "
            "infrastructure through accelerated computing products, data-center "
            "networking, and customer demand for large-scale computing systems. "
            "Management also discusses manufacturing capacity, supply "
            "constraints, competition, and uncertainty that could affect the "
            "pace and durability of this exposure over future periods."
        ),
        exposure_label="Direct",
        exposure_direction="Increasing",
        evidence_strength="High",
        key_evidence=[
            "Management reports increased data-center and accelerated-computing demand.",
            "The filing discusses networking products used in large computing systems.",
        ],
        risks_or_uncertainties=[
            "Supply constraints and competition may limit future growth."
        ],
    )


def make_summary_record(tmp_path: Path) -> FilingSummaryRecord:
    """Build one complete valid summary record."""

    return FilingSummaryRecord(
        selected_input=make_selected_input(tmp_path),
        theme=make_theme(),
        summary_model="gpt-test-summary-model",
        embedding_model="text-embedding-test-model",
        created_at_utc=datetime(2026, 6, 17, 13, 0),
        summary=make_summary(),
    )


# -----------------------------------------------------------------------------
# Basic helper tests
# -----------------------------------------------------------------------------


def test_normalize_text_collapses_repeated_whitespace():
    """Text normalization should produce a stable single-space string."""

    raw = "AI   infrastructure\n\nuses\taccelerated computing."

    assert normalize_text(raw) == (
        "AI infrastructure uses accelerated computing."
    )


def test_sha256_text_returns_stable_64_character_digest():
    """The same exact text should always produce the same valid digest."""

    first = sha256_text("same text")
    second = sha256_text("same text")

    assert first == second
    assert len(first) == 64


# -----------------------------------------------------------------------------
# FilingRecord tests
# -----------------------------------------------------------------------------


def test_filing_record_normalizes_ticker_and_acceptance_time():
    """Filing metadata should be cleaned without requiring a raw text file."""

    record = make_filing_record(ticker="  nvda  ")

    # Ticker normalization is important for joins with price and alpha tables.
    assert record.ticker == "NVDA"

    # The model stores explicit UTC timestamps for auditability.
    assert record.acceptance_datetime is not None
    assert record.acceptance_datetime.tzinfo == timezone.utc


def test_filing_record_does_not_require_full_raw_filing_file():
    """EdgarTools metadata should validate before any raw file is saved."""

    record = make_filing_record()

    assert record.accession_number == "0001045810-24-000124"


def test_filing_record_rejects_report_period_after_filing_date():
    """An impossible report-period ordering should fail validation."""

    with pytest.raises(
        ValidationError,
        match="period_of_report cannot be later than filing_date",
    ):
        make_filing_record(
            period_of_report=date(2024, 6, 30),
            filing_date=date(2024, 5, 29),
        )


def test_filing_record_rejects_unexpected_fields():
    """Forbid extra fields so schema drift and typos are visible."""

    with pytest.raises(ValidationError):
        make_filing_record(unexpected_column="should fail")


# -----------------------------------------------------------------------------
# TextArtifact tests
# -----------------------------------------------------------------------------


def test_text_artifact_validates_exact_file_metadata(tmp_path):
    """A matching path, hash, character count, and word count should pass."""

    artifact = make_text_artifact(tmp_path)

    assert Path(artifact.path).exists()
    assert artifact.char_count > 0
    assert artifact.word_count > 0
    assert artifact.created_at_utc.tzinfo == timezone.utc


def test_text_artifact_rejects_missing_file(tmp_path):
    """A manifest must not claim that a nonexistent file is available."""

    missing_path = tmp_path / "missing.txt"

    with pytest.raises(
        ValidationError,
        match="text artifact does not exist",
    ):
        TextArtifact(
            path=str(missing_path),
            extraction_method="edgartools",
            sha256="0" * 64,
            char_count=0,
            word_count=0,
            created_at_utc=datetime(2026, 6, 17),
        )


def test_text_artifact_rejects_mismatched_hash(tmp_path):
    """Changing the recorded hash should make the artifact invalid."""

    text = "Exact selected filing text."
    path = write_text_file(tmp_path / "selected.txt", text)

    with pytest.raises(
        ValidationError,
        match="sha256 does not match",
    ):
        TextArtifact(
            path=str(path),
            extraction_method="edgartools",
            sha256="0" * 64,
            char_count=len(text),
            word_count=len(text.split()),
            created_at_utc=datetime(2026, 6, 17),
        )


def test_text_artifact_rejects_mismatched_character_count(tmp_path):
    """The saved-file length must match the recorded length."""

    text = "Exact selected filing text."
    path = write_text_file(tmp_path / "selected.txt", text)

    with pytest.raises(
        ValidationError,
        match="char_count does not match",
    ):
        TextArtifact(
            path=str(path),
            extraction_method="edgartools",
            sha256=sha256_text(text),
            char_count=len(text) + 1,
            word_count=len(text.split()),
            created_at_utc=datetime(2026, 6, 17),
        )


# -----------------------------------------------------------------------------
# SelectedFilingInput and theme-summary tests
# -----------------------------------------------------------------------------


def test_selected_filing_input_cleans_sections(tmp_path):
    """Blank and duplicate section labels should be removed explicitly."""

    selected_input = SelectedFilingInput(
        filing=make_filing_record(),
        selected_sections=[
            " management_discussion ",
            "",
            "risk_factors",
            "management_discussion",
        ],
        selected_text=make_text_artifact(tmp_path),
    )

    assert selected_input.selected_sections == [
        "management_discussion",
        "risk_factors",
    ]


def test_selected_filing_input_defaults_raw_filing_text_to_none(tmp_path):
    """Normal runs should not require a complete raw filing cache."""

    selected_input = make_selected_input(tmp_path)

    assert selected_input.raw_filing_text is None


def test_selected_filing_input_accepts_optional_raw_filing_text(tmp_path):
    """Debugging runs may attach a validated complete-filing artifact."""

    selected_input = SelectedFilingInput(
        filing=make_filing_record(),
        selected_sections=["management_discussion", "risk_factors"],
        selected_text=make_text_artifact(
            tmp_path,
            filename="selected.txt",
        ),
        raw_filing_text=make_text_artifact(
            tmp_path,
            filename="raw_filing.md",
            text=(
                "Complete filing markdown retained for debugging and "
                "reproducibility checks."
            ),
        ),
    )

    assert selected_input.raw_filing_text is not None
    assert Path(selected_input.raw_filing_text.path).name == "raw_filing.md"


def test_selected_filing_input_rejects_all_blank_sections(tmp_path):
    """At least one meaningful selected section is required."""

    with pytest.raises(
        ValidationError,
        match="at least one non-empty section",
    ):
        SelectedFilingInput(
            filing=make_filing_record(),
            selected_sections=["", "   "],
            selected_text=make_text_artifact(tmp_path),
        )


def test_section_extraction_detail_accepts_consistent_counts():
    """A found and truncated section should preserve coherent audit metadata."""

    detail = SectionExtractionDetail(
        section_name="management_discussion",
        requested=True,
        found=True,
        original_char_count=40_000,
        retained_char_count=22_000,
        char_limit=22_000,
        truncated=True,
    )

    assert detail.section_name == "management_discussion"
    assert detail.truncated is True


def test_section_extraction_detail_rejects_negative_count():
    """Pydantic should reject impossible negative character counts."""

    with pytest.raises(ValidationError):
        SectionExtractionDetail(
            section_name="business",
            requested=True,
            found=True,
            original_char_count=-1,
            retained_char_count=1,
            char_limit=10,
            truncated=False,
        )


def test_section_extraction_detail_rejects_missing_section_with_counts():
    """A not-found section cannot claim retained text."""

    with pytest.raises(
        ValidationError,
        match="original_char_count must be zero",
    ):
        SectionExtractionDetail(
            section_name="risk_factors",
            requested=True,
            found=False,
            original_char_count=100,
            retained_char_count=0,
            char_limit=8_000,
            truncated=False,
        )


def test_selected_filing_input_defaults_extraction_details_to_empty(tmp_path):
    """Older manual constructors remain valid without audit detail rows."""

    selected_input = make_selected_input(tmp_path)

    assert selected_input.section_extraction_details == []


def test_selected_filing_input_accepts_extraction_details(tmp_path):
    """A selected input should preserve section-level audit information."""

    detail = SectionExtractionDetail(
        section_name="management_discussion",
        requested=True,
        found=True,
        original_char_count=1_000,
        retained_char_count=1_000,
        char_limit=22_000,
        truncated=False,
    )

    selected_input = SelectedFilingInput(
        filing=make_filing_record(),
        selected_sections=["management_discussion"],
        section_extraction_details=[detail],
        selected_text=make_text_artifact(tmp_path),
    )

    assert len(selected_input.section_extraction_details) == 1
    assert (
        selected_input.section_extraction_details[0].section_name
        == "management_discussion"
    )


def test_theme_config_normalizes_whitespace():
    """Theme strings should be stable for prompts and cache keys."""

    theme = ThemeConfig(
        theme_name="  AI   infrastructure ",
        theme_description=(
            "A sufficiently detailed   description of companies exposed to "
            "accelerated computing and data-center infrastructure."
        ),
        prompt_version="  version_1 ",
    )

    assert theme.theme_name == "AI infrastructure"
    assert theme.prompt_version == "version_1"


def test_filing_theme_summary_cleans_lists():
    """The OpenAI schema should remove blank evidence and risk items."""

    summary = FilingThemeSummary(
        theme_exposure_summary=(
            "The filing contains detailed evidence of direct thematic exposure "
            "through products, customers, capital spending, and management "
            "discussion. It also identifies operational uncertainty, competition, "
            "and supply constraints that could weaken the durability of future "
            "theme-related revenue and investment."
        ),
        exposure_label="Direct",
        exposure_direction="Stable",
        evidence_strength="Medium",
        key_evidence=[
            "  Data-center demand is discussed.  ",
            "",
        ],
        risks_or_uncertainties=[
            "",
            "  Supply constraints remain material. ",
        ],
    )

    assert summary.key_evidence == [
        "Data-center demand is discussed."
    ]
    assert summary.risks_or_uncertainties == [
        "Supply constraints remain material."
    ]


def test_filing_theme_summary_rejects_blank_evidence():
    """Structured output must contain at least one useful evidence item."""

    with pytest.raises(
        ValidationError,
        match="at least one non-empty item",
    ):
        FilingThemeSummary(
            theme_exposure_summary=(
                "This summary contains more than twenty words but all supporting "
                "evidence fields are blank, which should make the structured "
                "record invalid for a reproducible research workflow."
            ),
            exposure_label="Unclear",
            exposure_direction="Unclear",
            evidence_strength="Low",
            key_evidence=["", "   "],
        )


def test_filing_summary_record_normalizes_audit_timestamp(tmp_path):
    """Naive notebook timestamps should be stored explicitly as UTC."""

    record = make_summary_record(tmp_path)

    assert record.created_at_utc.tzinfo == timezone.utc
    assert record.selected_input.filing.ticker == "NVDA"


# -----------------------------------------------------------------------------
# PointInTimeFilingSelection tests
# -----------------------------------------------------------------------------


def test_point_in_time_selection_accepts_eligible_filing():
    """A filing available before the rebalance date should be valid."""

    selection = PointInTimeFilingSelection(
        ticker="nvda",
        rebalance_date=date(2024, 5, 31),
        filing=make_filing_record(),
        effective_signal_date=date(2024, 5, 30),
    )

    assert selection.ticker == "NVDA"
    assert selection.effective_signal_date <= selection.rebalance_date


def test_point_in_time_selection_rejects_future_signal():
    """The core no-lookahead rule must fail if the signal arrives later."""

    with pytest.raises(
        ValidationError,
        match="cannot be later than rebalance_date",
    ):
        PointInTimeFilingSelection(
            ticker="NVDA",
            rebalance_date=date(2024, 5, 28),
            filing=make_filing_record(),
            effective_signal_date=date(2024, 5, 30),
        )


def test_point_in_time_selection_rejects_ticker_mismatch():
    """The selected filing must belong to the requested ticker."""

    with pytest.raises(
        ValidationError,
        match="selection ticker must match filing ticker",
    ):
        PointInTimeFilingSelection(
            ticker="AAPL",
            rebalance_date=date(2024, 5, 31),
            filing=make_filing_record(),
            effective_signal_date=date(2024, 5, 30),
        )


def test_point_in_time_selection_rejects_effective_date_before_filing():
    """A filing cannot affect a signal before its filing date."""

    with pytest.raises(
        ValidationError,
        match="cannot be earlier than filing_date",
    ):
        PointInTimeFilingSelection(
            ticker="NVDA",
            rebalance_date=date(2024, 5, 31),
            filing=make_filing_record(),
            effective_signal_date=date(2024, 5, 28),
        )


# -----------------------------------------------------------------------------
# FilingSignalRecord tests
# -----------------------------------------------------------------------------


def test_filing_signal_factory_derives_consistent_scores(tmp_path):
    """The factory should map categorical labels to canonical numeric scores."""

    summary_record = make_summary_record(tmp_path)

    signal = FilingSignalRecord.from_summary_record(
        summary_record=summary_record,
        summary_theme_similarity=0.82,
        signal_version="filing_summary_similarity_v1",
        created_at_utc=datetime(2026, 6, 17, 14, 0),
    )

    assert signal.exposure_label_score == EXPOSURE_LABEL_SCORE["Direct"]
    assert signal.direction_score == DIRECTION_SCORE["Increasing"]
    assert (
        signal.evidence_strength_score
        == EVIDENCE_STRENGTH_SCORE["High"]
    )
    assert signal.created_at_utc.tzinfo == timezone.utc


def test_filing_signal_rejects_score_label_disagreement(tmp_path):
    """Stored numeric scores must not contradict the structured summary."""

    summary_record = make_summary_record(tmp_path)

    with pytest.raises(
        ValidationError,
        match="exposure_label_score does not match",
    ):
        FilingSignalRecord(
            summary_record=summary_record,
            summary_theme_similarity=0.82,
            exposure_label_score=0,  # Incorrect: summary label is Direct.
            direction_score=1,
            evidence_strength_score=3,
            signal_version="filing_summary_similarity_v1",
            created_at_utc=datetime(2026, 6, 17, 14, 0),
        )


def test_filing_signal_rejects_similarity_outside_cosine_range(tmp_path):
    """Cosine similarity must remain between -1 and 1."""

    summary_record = make_summary_record(tmp_path)

    with pytest.raises(ValidationError):
        FilingSignalRecord.from_summary_record(
            summary_record=summary_record,
            summary_theme_similarity=1.2,
            signal_version="filing_summary_similarity_v1",
            created_at_utc=datetime(2026, 6, 17, 14, 0),
        )
