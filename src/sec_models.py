"""
Pydantic data models for the point-in-time SEC filing signal pipeline.

This module defines the data contracts shared by the SEC integration stages.

The primary production path is:

    EdgarTools filing metadata
        -> FilingRecord

    EdgarTools section extraction
        -> TextArtifact
        -> SelectedFilingInput

    OpenAI structured analysis
        -> FilingThemeSummary
        -> FilingSummaryRecord

    Filing-level numeric signal
        -> FilingSignalRecord

    Historical as-of selection
        -> PointInTimeFilingSelection

These models do not call EdgarTools, the SEC website, OpenAI, or the existing
portfolio optimizer. Their job is narrower:

    1. define the expected fields,
    2. normalize simple values such as ticker symbols,
    3. reject malformed or inconsistent records,
    4. preserve enough metadata to reproduce and audit the research pipeline.

The code intentionally favors explicit validation over compact or clever code.
That makes failures easier to understand when another researcher audits or
extends the project.

Important design choice
-----------------------
`FilingRecord` contains filing metadata only. It does not require a full raw
filing to be written to disk.

That keeps EdgarTools as the primary implementation. EdgarTools can retrieve
and parse a filing object directly, and the project only needs to save the
focused text that is actually used downstream.

Any saved raw document used for debugging can also be represented with the
generic `TextArtifact` model, without creating a second parallel pipeline.
"""

from __future__ import annotations

# Standard-library imports.
import hashlib
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

# Pydantic v2 imports.
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# -----------------------------------------------------------------------------
# Shared type aliases
# -----------------------------------------------------------------------------


# The initial research version uses the standard 10-K and 10-Q forms only.
#
# Amendments such as 10-K/A and 10-Q/A should be handled explicitly later
# rather than silently mixed into the first experiment.
FilingForm = Literal["10-K", "10-Q"]


# Restricted labels requested from the OpenAI structured-output schema.
ExposureLabel = Literal["Direct", "Indirect", "Minimal", "None", "Unclear"]
ExposureDirection = Literal["Increasing", "Stable", "Decreasing", "Unclear"]
EvidenceStrength = Literal["High", "Medium", "Low", "Unclear"]


# -----------------------------------------------------------------------------
# Shared scoring maps
# -----------------------------------------------------------------------------


# Stronger or more direct thematic exposure receives a larger score.
#
# `None` is used for "Unclear" rather than numpy.nan because this is a schema
# module. Downstream pandas code can convert None to NaN when it creates a
# DataFrame.
EXPOSURE_LABEL_SCORE: dict[str, int | None] = {
    "None": 0,
    "Minimal": 1,
    "Indirect": 2,
    "Direct": 3,
    "Unclear": None,
}


# Direction is encoded on a simple -1, 0, +1 scale.
DIRECTION_SCORE: dict[str, int | None] = {
    "Decreasing": -1,
    "Stable": 0,
    "Increasing": 1,
    "Unclear": None,
}


# Stronger filing evidence receives a larger score.
EVIDENCE_STRENGTH_SCORE: dict[str, int | None] = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Unclear": None,
}


# -----------------------------------------------------------------------------
# Small reusable helpers
# -----------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace into single spaces.

    The learning notebooks used the same normalization before hashing and
    sending text to OpenAI. Keeping the rule here makes validation and cache
    keys deterministic.
    """

    # `(text or "")` protects the helper from an accidental None-like value.
    # `split()` removes repeated spaces, tabs, and line breaks.
    # `" ".join(...)` rebuilds a stable single-space representation.
    return " ".join((text or "").split())


def sha256_text(text: str) -> str:
    """Return the SHA-256 hash of an exact string."""

    # Encode text as UTF-8 bytes because hash functions operate on bytes.
    encoded = text.encode("utf-8")

    # Return the standard 64-character hexadecimal digest.
    return hashlib.sha256(encoded).hexdigest()


def _normalize_ticker(value: str) -> str:
    """Normalize a ticker symbol for consistent joins across project files."""

    # Strip accidental whitespace and use uppercase ticker labels.
    return value.strip().upper()


def _normalize_utc_datetime(value: datetime) -> datetime:
    """Return a timezone-aware datetime normalized to UTC.

    The learning notebooks used `datetime.utcnow()`, which creates a naive
    datetime. This validator remains compatible with that input by treating a
    naive timestamp as UTC, while storing an explicit timezone for auditability.
    """

    # Treat a naive datetime as already representing UTC.
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    # Convert any timezone-aware timestamp to UTC.
    return value.astimezone(timezone.utc)


def _validate_sha256(value: str) -> str:
    """Validate and normalize a SHA-256 hexadecimal digest."""

    # Remove accidental surrounding whitespace and use lowercase hex.
    value = value.strip().lower()

    # A SHA-256 hexadecimal digest must contain exactly 64 characters.
    if len(value) != 64:
        raise ValueError("SHA-256 digest must contain exactly 64 characters")

    # Reject any character that is not valid hexadecimal.
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError("SHA-256 digest must contain hexadecimal characters only")

    return value


def _read_text_artifact(path_value: str) -> tuple[Path, str]:
    """Read a text artifact and return both its Path and exact contents."""

    # Convert the stored path string to a Path object.
    path = Path(path_value)

    # The artifact must already exist when the model is validated.
    if not path.exists():
        raise ValueError(f"text artifact does not exist: {path_value}")

    # Directories are not valid text artifacts.
    if not path.is_file():
        raise ValueError(f"text artifact path is not a file: {path_value}")

    # Read the exact stored text used by the downstream pipeline.
    text = path.read_text(encoding="utf-8", errors="ignore")

    return path, text


# -----------------------------------------------------------------------------
# Core filing metadata
# -----------------------------------------------------------------------------


class FilingRecord(BaseModel):
    """Validated metadata for one SEC 10-K or 10-Q filing.

    This model intentionally does not require a locally saved full filing.
    EdgarTools remains free to retrieve and parse its filing object directly.
    """

    # Reject unexpected fields so typos and schema drift fail loudly.
    model_config = ConfigDict(extra="forbid")

    # Stock ticker, normalized to uppercase by validator below. 
    ticker: str = Field(..., min_length=1, max_length=12)
    # SEC CIK. 
    cik: int = Field(..., ge=0)
    # Company name from EdgarTools.
    company_name: str = Field(..., min_length=1)

    # Filing identity.
    # Only allow 10-K or 10-Q. 
    form: FilingForm
    # Filing date from SEC metadata. 
    filing_date: date
    # Period of report can be missing for some filings. 
    period_of_report: date | None = None
    # SEC accession number. 
    accession_number: str = Field(..., min_length=5)

    # Optional SEC/EdgarTools metadata.
    primary_document: str | None = None
    acceptance_datetime: datetime | None = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        """Store ticker symbols in a consistent uppercase form."""

        return _normalize_ticker(value)

    @field_validator("company_name", "accession_number")
    @classmethod
    def clean_required_strings(cls, value: str) -> str:
        """Remove accidental leading and trailing whitespace."""

        cleaned = value.strip()

        if not cleaned:
            raise ValueError("value must not be blank")

        return cleaned

    @field_validator("primary_document")
    @classmethod
    def clean_optional_primary_document(cls, value: str | None) -> str | None:
        """Clean the optional primary-document filename."""

        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @field_validator("acceptance_datetime")
    @classmethod
    def normalize_acceptance_datetime(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        """Normalize an optional SEC acceptance timestamp to UTC."""

        if value is None:
            return None

        return _normalize_utc_datetime(value)

    @model_validator(mode="after")
    def report_period_should_not_follow_filing_date(self):
        """Reject an impossible period-of-report ordering."""

        if (
            self.period_of_report is not None
            and self.period_of_report > self.filing_date
        ):
            raise ValueError(
                "period_of_report cannot be later than filing_date"
            )

        return self


# -----------------------------------------------------------------------------
# Saved text artifacts
# -----------------------------------------------------------------------------


class TextArtifact(BaseModel):
    """Validated local text file used by the research pipeline.

    This generic model can represent:

        - focused Business / Risk Factors / MD&A text,
        - a raw filing saved only for debugging,
        - another exact text artifact added later.

    The model verifies that the saved file still matches its recorded hash and
    size metadata.
    """

    model_config = ConfigDict(extra="forbid")

    # Local file path.
    path: str = Field(..., min_length=1)

    # Method used to produce the text.
    extraction_method: str = Field(..., min_length=1)

    # Exact-file audit metadata.
    sha256: str = Field(..., min_length=64, max_length=64)
    char_count: int = Field(..., ge=0)
    word_count: int = Field(..., ge=0)

    # Timestamp at which this artifact record was created.
    created_at_utc: datetime

    @field_validator("path", "extraction_method")
    @classmethod
    def clean_required_strings(cls, value: str) -> str:
        """Remove accidental whitespace from required string fields."""

        cleaned = value.strip()

        if not cleaned:
            raise ValueError("value must not be blank")

        return cleaned

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        """Require a valid 64-character SHA-256 digest."""

        return _validate_sha256(value)

    @field_validator("created_at_utc")
    @classmethod
    def normalize_created_at_utc(cls, value: datetime) -> datetime:
        """Store artifact timestamps as timezone-aware UTC values."""

        return _normalize_utc_datetime(value)

    @model_validator(mode="after")
    def metadata_must_match_saved_file(self):
        """Verify path, character count, word count, and hash."""

        # Read the exact file referenced by this artifact.
        _, text = _read_text_artifact(self.path)

        # Compare the actual file length with the recorded character count.
        if len(text) != self.char_count:
            raise ValueError(
                "char_count does not match the saved text artifact"
            )

        # Compare the actual whitespace-delimited word count.
        if len(text.split()) != self.word_count:
            raise ValueError(
                "word_count does not match the saved text artifact"
            )

        # Compare the actual SHA-256 digest with the recorded digest.
        if sha256_text(text) != self.sha256:
            raise ValueError(
                "sha256 does not match the saved text artifact"
            )

        return self


# -----------------------------------------------------------------------------
# Focused filing input
# -----------------------------------------------------------------------------


class SectionExtractionDetail(BaseModel):
    """Audit information for one requested filing section.

    This record explains what happened to one section during extraction:

        - whether the researcher requested the section,
        - whether EdgarTools found usable text,
        - how large the original section was,
        - how much text was retained,
        - which character limit was actually applied,
        - whether head-and-tail truncation was required.

    Keeping these details with SelectedFilingInput makes later ablation studies
    and data-quality checks easier to reproduce.
    """

    model_config = ConfigDict(extra="forbid")

    # Stable internal name used throughout the project, for example:
    # management_discussion, business, or risk_factors.
    section_name: str = Field(..., min_length=1)

    # True when this section was enabled for the current extraction run.
    requested: bool

    # True when EdgarTools returned a usable section.
    found: bool

    # Number of characters in the complete extracted section before sampling.
    original_char_count: int = Field(..., ge=0)

    # Number of section-text characters retained after sampling.
    retained_char_count: int = Field(..., ge=0)

    # Actual section-text limit applied during this extraction run.
    # It can be slightly lower than the configured default when the overall
    # max_chars safety cap must also reserve room for section headings.
    char_limit: int = Field(..., ge=1)

    # True when retained_char_count is smaller than original_char_count.
    truncated: bool

    @field_validator("section_name")
    @classmethod
    def clean_section_name(cls, value: str) -> str:
        """Normalize whitespace and reject a blank section name."""

        cleaned = normalize_text(value)

        if not cleaned:
            raise ValueError("section_name must not be blank")

        return cleaned

    @model_validator(mode="after")
    def validate_section_counts(self):
        """Keep status flags and character counts internally consistent."""

        # A section cannot be found when it was not requested.
        if not self.requested and self.found:
            raise ValueError(
                "a section cannot be found when requested is False"
            )

        # Missing or disabled sections should not claim retained text.
        if not self.found:
            if self.original_char_count != 0:
                raise ValueError(
                    "original_char_count must be zero when found is False"
                )

            if self.retained_char_count != 0:
                raise ValueError(
                    "retained_char_count must be zero when found is False"
                )

            if self.truncated:
                raise ValueError(
                    "truncated must be False when found is False"
                )

            return self

        # A found section must contain some usable text.
        if self.original_char_count == 0:
            raise ValueError(
                "original_char_count must be positive when found is True"
            )

        if self.retained_char_count == 0:
            raise ValueError(
                "retained_char_count must be positive when found is True"
            )

        # Sampling cannot create more text than existed originally.
        if self.retained_char_count > self.original_char_count:
            raise ValueError(
                "retained_char_count cannot exceed original_char_count"
            )

        # The retained section must obey the actual limit applied.
        if self.retained_char_count > self.char_limit:
            raise ValueError(
                "retained_char_count cannot exceed char_limit"
            )

        # The boolean should agree with the actual counts.
        expected_truncated = (
            self.retained_char_count < self.original_char_count
        )

        if self.truncated != expected_truncated:
            raise ValueError(
                "truncated does not agree with the section character counts"
            )

        return self


class SelectedFilingInput(BaseModel):
    """Focused filing sections selected for downstream OpenAI analysis.

    The selected_text field stores the exact focused text sent to the model.

    The raw_filing_text field is optional. When raw caching is enabled in the
    EdgarTools ingestion layer, it stores a validated artifact for the complete
    filing markdown or plain text. Normal runs may leave this field as None.
    """

    model_config = ConfigDict(extra="forbid")

    # Validated SEC filing metadata.
    filing: FilingRecord

    # Human-readable section identifiers such as:
    # business, risk_factors, management_discussion.
    selected_sections: list[str] = Field(..., min_length=1, max_length=8)

    # Section-level extraction and truncation audit records. The empty default
    # keeps older manually constructed SelectedFilingInput objects valid.
    section_extraction_details: list[SectionExtractionDetail] = Field(
        default_factory=list,
        max_length=8,
    )

    # Exact focused text artifact sent to the language model.
    selected_text: TextArtifact

    # Optional complete filing retained for debugging, offline inspection,
    # parser comparison, or final archival research runs.
    raw_filing_text: TextArtifact | None = None

    @field_validator("selected_sections")
    @classmethod
    def clean_selected_sections(cls, values: list[str]) -> list[str]:
        """Remove blank and duplicate section names while preserving order."""

        cleaned_sections: list[str] = []

        for value in values:
            # Normalize whitespace but otherwise preserve the researcher's label.
            cleaned = normalize_text(value)

            # Ignore blank values.
            if not cleaned:
                continue

            # Preserve the first occurrence and remove later duplicates.
            if cleaned not in cleaned_sections:
                cleaned_sections.append(cleaned)

        # At least one usable section must remain.
        if not cleaned_sections:
            raise ValueError(
                "selected_sections must contain at least one non-empty section"
            )

        return cleaned_sections

    @field_validator("section_extraction_details")
    @classmethod
    def reject_duplicate_section_details(
        cls,
        values: list[SectionExtractionDetail],
    ) -> list[SectionExtractionDetail]:
        """Require at most one audit record per section name."""

        section_names = [value.section_name for value in values]

        if len(section_names) != len(set(section_names)):
            raise ValueError(
                "section_extraction_details contains duplicate section names"
            )

        return values


# -----------------------------------------------------------------------------
# Theme and OpenAI structured output
# -----------------------------------------------------------------------------


class ThemeConfig(BaseModel):
    """Versioned natural-language definition of the research theme."""

    model_config = ConfigDict(extra="forbid")

    # Short display name.
    theme_name: str = Field(..., min_length=1)

    # Detailed definition used by the prompt and embedding comparison.
    theme_description: str = Field(..., min_length=20)

    # Version identifier for audit and cache invalidation.
    prompt_version: str = Field(..., min_length=1)

    @field_validator("theme_name", "theme_description", "prompt_version")
    @classmethod
    def clean_theme_strings(cls, value: str) -> str:
        """Normalize repeated whitespace in theme fields."""

        cleaned = normalize_text(value)

        if not cleaned:
            raise ValueError("theme field must not be blank")

        return cleaned


class FilingThemeSummary(BaseModel):
    """Validated structured summary generated from selected filing text."""

    model_config = ConfigDict(extra="forbid")

    # Main theme-specific prose summary.
    theme_exposure_summary: str = Field(
        ...,
        min_length=100,
        max_length=1800,
        description=(
            "Theme-specific summary supported only by the selected filing text."
        ),
    )

    # Restricted categorical assessments.
    exposure_label: ExposureLabel
    exposure_direction: ExposureDirection
    evidence_strength: EvidenceStrength

    # Supporting evidence and caveats from the filing.
    # Must contain at least one item and at most six items. 
    key_evidence: list[str] = Field(..., min_length=1, max_length=6)

    # Risks/uncertainties can be empty but no more than six. 
    risks_or_uncertainties: list[str] = Field(
        default_factory=list,
        max_length=6,
    )

    @field_validator("theme_exposure_summary")
    @classmethod
    def clean_summary(cls, value: str) -> str:
        """Normalize summary text and reject very short generic output."""

        cleaned = normalize_text(value)

        # Character limits alone do not guarantee a useful summary.
        if len(cleaned.split()) < 20:
            raise ValueError(
                "theme_exposure_summary must contain at least 20 words"
            )

        return cleaned

    @field_validator("key_evidence")
    @classmethod
    def clean_key_evidence(cls, values: list[str]) -> list[str]:
        """Normalize evidence items and remove blank entries."""

        cleaned = [
            normalize_text(value)
            for value in values
            if normalize_text(value)
        ]

        if not cleaned:
            raise ValueError(
                "key_evidence must contain at least one non-empty item"
            )

        return cleaned

    @field_validator("risks_or_uncertainties")
    @classmethod
    def clean_risks(cls, values: list[str]) -> list[str]:
        """Normalize risk items and remove blank entries."""

        return [
            normalize_text(value)
            for value in values
            if normalize_text(value)
        ]


class FilingSummaryRecord(BaseModel):
    """One complete validated OpenAI summary result."""

    model_config = ConfigDict(extra="forbid")

    # Exact selected filing text and metadata used for the request.
    selected_input: SelectedFilingInput

    # Versioned research theme.
    theme: ThemeConfig

    # Model names are stored for reproducibility.
    summary_model: str = Field(..., min_length=1)
    embedding_model: str = Field(..., min_length=1)

    # Audit timestamp.
    created_at_utc: datetime

    # Structured OpenAI output.
    summary: FilingThemeSummary

    @field_validator("summary_model", "embedding_model")
    @classmethod
    def clean_model_names(cls, value: str) -> str:
        """Remove accidental whitespace from model identifiers."""

        cleaned = value.strip()

        if not cleaned:
            raise ValueError("model name must not be blank")

        return cleaned

    @field_validator("created_at_utc")
    @classmethod
    def normalize_created_at_utc(cls, value: datetime) -> datetime:
        """Store summary timestamps as timezone-aware UTC values."""

        return _normalize_utc_datetime(value)


# -----------------------------------------------------------------------------
# Point-in-time filing selection
# -----------------------------------------------------------------------------


class PointInTimeFilingSelection(BaseModel):
    """One filing selected for one ticker at one historical rebalance date.

    A row exists only when an eligible filing was found. Missing selections can
    remain absent from the selection table rather than being represented by a
    partially empty model.
    """

    model_config = ConfigDict(extra="forbid")

    # Ticker and historical evaluation date.
    ticker: str = Field(..., min_length=1, max_length=12)
    rebalance_date: date

    # The filing selected as of that date.
    filing: FilingRecord

    # First trading date on which the research pipeline permits the filing to
    # influence a signal.
    effective_signal_date: date

    # Record how the point-in-time choice was made.
    selection_method: str = Field(
        default="latest_eligible_filing",
        min_length=1,
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        """Normalize the outer ticker label."""

        return _normalize_ticker(value)

    @field_validator("selection_method")
    @classmethod
    def clean_selection_method(cls, value: str) -> str:
        """Clean the selection-method label."""

        cleaned = normalize_text(value)

        if not cleaned:
            raise ValueError("selection_method must not be blank")

        return cleaned

    @model_validator(mode="after")
    def validate_point_in_time_ordering(self):
        """Enforce the core no-lookahead ordering constraints."""

        # The outer ticker must identify the same company as the filing.
        if self.ticker != self.filing.ticker:
            raise ValueError(
                "selection ticker must match filing ticker"
            )

        # A filing cannot become usable before it was filed.
        if self.effective_signal_date < self.filing.filing_date:
            raise ValueError(
                "effective_signal_date cannot be earlier than filing_date"
            )

        # The selected signal must have been available by the rebalance date.
        if self.effective_signal_date > self.rebalance_date:
            raise ValueError(
                "effective_signal_date cannot be later than rebalance_date"
            )

        # If an acceptance timestamp is known, its calendar date cannot follow
        # the effective date. More detailed after-market-close logic belongs in
        # the future EdgarTools selection module, which has the trading calendar.
        if (
            self.filing.acceptance_datetime is not None
            and self.filing.acceptance_datetime.date()
            > self.effective_signal_date
        ):
            raise ValueError(
                "effective_signal_date cannot precede acceptance_datetime"
            )

        return self


# -----------------------------------------------------------------------------
# Filing-level numeric signal
# -----------------------------------------------------------------------------


class FilingSignalRecord(BaseModel):
    """Numeric thematic signal derived from one validated filing summary."""

    model_config = ConfigDict(extra="forbid")

    # Full source record used to construct the signal.
    summary_record: FilingSummaryRecord

    # Cosine similarity between the filing summary embedding and theme embedding.
    summary_theme_similarity: float = Field(..., ge=-1.0, le=1.0)

    # Numeric versions of the structured categorical labels.
    exposure_label_score: int | None
    direction_score: int | None
    evidence_strength_score: int | None

    # Version and timestamp for reproducibility.
    signal_version: str = Field(..., min_length=1)
    created_at_utc: datetime

    @field_validator("signal_version")
    @classmethod
    def clean_signal_version(cls, value: str) -> str:
        """Clean the signal-version identifier."""

        cleaned = normalize_text(value)

        if not cleaned:
            raise ValueError("signal_version must not be blank")

        return cleaned

    @field_validator("created_at_utc")
    @classmethod
    def normalize_created_at_utc(cls, value: datetime) -> datetime:
        """Store signal timestamps as timezone-aware UTC values."""

        return _normalize_utc_datetime(value)

    @model_validator(mode="after")
    def numeric_scores_must_match_summary_labels(self):
        """Prevent categorical labels and numeric columns from disagreeing."""

        # Read the structured labels from the nested OpenAI summary.
        summary = self.summary_record.summary

        # Calculate the expected numeric values using the shared maps.
        expected_exposure = EXPOSURE_LABEL_SCORE[summary.exposure_label]
        expected_direction = DIRECTION_SCORE[summary.exposure_direction]
        expected_evidence = EVIDENCE_STRENGTH_SCORE[summary.evidence_strength]

        # Validate each stored numeric field against its source label.
        if self.exposure_label_score != expected_exposure:
            raise ValueError(
                "exposure_label_score does not match exposure_label"
            )

        if self.direction_score != expected_direction:
            raise ValueError(
                "direction_score does not match exposure_direction"
            )

        if self.evidence_strength_score != expected_evidence:
            raise ValueError(
                "evidence_strength_score does not match evidence_strength"
            )

        return self

    # Take the already-validated filing summary, read its categorical labels
    # and then translate those into numeric scores using the shared lookup
    # tables. Then build and return FilingSignalRecord.
    @classmethod
    def from_summary_record(
        cls,
        *,
        summary_record: FilingSummaryRecord,
        summary_theme_similarity: float,
        signal_version: str,
        created_at_utc: datetime,
    ) -> "FilingSignalRecord":
        """Build a signal record while deriving label scores consistently."""

        # Read the validated structured summary once.
        summary = summary_record.summary

        # Construct the record using the canonical score mappings.
        return cls(
            summary_record=summary_record,
            summary_theme_similarity=summary_theme_similarity,
            exposure_label_score=EXPOSURE_LABEL_SCORE[
                summary.exposure_label
            ],
            direction_score=DIRECTION_SCORE[
                summary.exposure_direction
            ],
            evidence_strength_score=EVIDENCE_STRENGTH_SCORE[
                summary.evidence_strength
            ],
            signal_version=signal_version,
            created_at_utc=created_at_utc,
        )


# Explicit public API for researchers importing from this module.
__all__ = [
    "FilingForm",
    "ExposureLabel",
    "ExposureDirection",
    "EvidenceStrength",
    "EXPOSURE_LABEL_SCORE",
    "DIRECTION_SCORE",
    "EVIDENCE_STRENGTH_SCORE",
    "normalize_text",
    "sha256_text",
    "FilingRecord",
    "TextArtifact",
    "SectionExtractionDetail",
    "SelectedFilingInput",
    "ThemeConfig",
    "FilingThemeSummary",
    "FilingSummaryRecord",
    "PointInTimeFilingSelection",
    "FilingSignalRecord",
]
