"""
EdgarTools ingestion and filing-text extraction for the SEC signal pipeline.

This module is the primary SEC access layer for the research project.

It deliberately uses EdgarTools instead of recreating SEC archive URLs,
HTML download logic, or a custom 10-K / 10-Q parser. EdgarTools already
provides:

    Company(ticker)
    company.get_filings(...)
    filing.obj()
    filing.markdown()
    filing.text()

The module adds the project-specific behavior that EdgarTools should not own:

    - normalize filing metadata into validated Pydantic records,
    - recover a filing by accession number,
    - select the latest filing available by a historical date,
    - extract Business / Risk Factors / MD&A sections,
    - fall back to full markdown or plain text when structured parsing fails,
    - save the exact selected text used downstream,
    - optionally cache the complete filing as markdown or plain text,
    - create validated SelectedFilingInput records,
    - save and load filing metadata using the project's chosen .pkl format.

This module does not call OpenAI and does not build alpha scores. Those tasks
belong in later modules.

Testing approach
----------------
Every network-facing function accepts an optional dependency such as
`company_factory`. Unit tests can therefore use small fake EdgarTools objects
without requiring internet access or an installed EdgarTools package.

The production path remains simple:

    from edgar import Company
    company = Company("NVDA")
    filings = company.get_filings(form=["10-K", "10-Q"])

The dependency-injection parameters are only there to make the research code
auditable and testable.
"""

from __future__ import annotations

# Standard-library imports.
import re
from collections.abc import Callable, Iterable, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# pandas is already part of the existing project and is useful for:
#   - tabular filing manifests,
#   - point-in-time date comparisons,
#   - saving/loading .pkl outputs.
import pandas as pd

# Reuse the shared Pydantic contracts created in sec_models.py.
from .sec_models import (
    FilingRecord,
    PointInTimeFilingSelection,
    SectionExtractionDetail,
    SelectedFilingInput,
    TextArtifact,
    normalize_text,
    sha256_text,
)


# -----------------------------------------------------------------------------
# Section definitions
# -----------------------------------------------------------------------------


# Each internal section name has:
#
#   attributes:
#       Named properties that recent EdgarTools report objects may expose.
#
#   keys:
#       Dictionary-like report item keys used in the learning notebooks.
#
# Trying both keeps the implementation simple while remaining tolerant of
# report-object differences across forms and EdgarTools versions.
TENK_SECTION_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    # MD&A is intentionally first because it is the most current narrative
    # discussion of operations, demand, margins, liquidity, and capital needs.
    "management_discussion": {
        "attributes": (
            "management_discussion",
            "managements_discussion",
        ),
        "keys": ("7", "Item 7"),
    },
    "business": {
        "attributes": ("business",),
        "keys": ("1", "Item 1"),
    },
    "risk_factors": {
        "attributes": ("risk_factors",),
        "keys": ("1A", "Item 1A"),
    },
}


TENQ_SECTION_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "management_discussion": {
        "attributes": (
            "management_discussion",
            "managements_discussion",
        ),
        "keys": ("2", "Item 2"),
    },
    "risk_factors": {
        "attributes": ("risk_factors",),
        "keys": ("1A", "Item 1A"),
    },
}


# Default section-text budgets for the first narrative SEC experiment.
# These values apply to section content. The overall max_chars safety cap also
# reserves a small amount of room for visible section headings and separators.
DEFAULT_10K_SECTION_CHAR_LIMITS: dict[str, int] = {
    "management_discussion": 22_000,
    "business": 10_000,
    "risk_factors": 8_000,
}


DEFAULT_10Q_SECTION_CHAR_LIMITS: dict[str, int] = {
    "management_discussion": 28_000,
    "risk_factors": 12_000,
}


# Visible marker inserted when the middle of a long section is omitted.
SECTION_MIDDLE_TRUNCATION_MARKER = (
    "\n\n[SECTION MIDDLE TRUNCATED]\n\n"
)


SUPPORTED_FORMS = ("10-K", "10-Q")


# -----------------------------------------------------------------------------
# EdgarTools dependency helpers
# -----------------------------------------------------------------------------


def _load_edgartools() -> tuple[type, Callable[[str], None]]:
    """Load EdgarTools only when a real network-facing function needs it.

    Keeping the import lazy allows unit tests to run with fake objects even when
    EdgarTools is not installed in the test environment.
    """

    try:
        # EdgarTools exposes both Company and set_identity from `edgar`.
        from edgar import Company, set_identity

    except ImportError as exc:
        raise RuntimeError(
            "EdgarTools is required for live SEC access. "
            "Install it with: pip install edgartools"
        ) from exc

    return Company, set_identity


def configure_edgar_identity(
    identity: str,
    *,
    set_identity_func: Callable[[str], None] | None = None,
) -> None:
    """Configure the SEC identity string used by EdgarTools.

    Args:
        identity:
            A descriptive identity such as:
                "Researcher Name researcher@example.com"

        set_identity_func:
            Optional injected function used by unit tests.

    This function does not store the identity in project outputs.
    """

    # Remove accidental leading and trailing whitespace.
    identity = identity.strip()

    # SEC requests should not be made with a blank identity.
    if not identity:
        raise ValueError("identity must not be blank")

    # Use the real EdgarTools function unless a test double was supplied.
    if set_identity_func is None:
        _, set_identity_func = _load_edgartools()

    # Delegate configuration to EdgarTools.
    set_identity_func(identity)


def _resolve_company_factory(
    company_factory: Callable[[str], Any] | None,
) -> Callable[[str], Any]:
    """Return an injected company factory or EdgarTools Company."""

    if company_factory is not None:
        return company_factory

    Company, _ = _load_edgartools()

    return Company


# -----------------------------------------------------------------------------
# General cleaning helpers
# -----------------------------------------------------------------------------


def normalize_ticker(ticker: str) -> str:
    """Normalize a ticker to the project's uppercase string convention."""

    cleaned = str(ticker).strip().upper()

    if not cleaned:
        raise ValueError("ticker must not be blank")

    return cleaned


def normalize_form(form: str) -> str:
    """Normalize and validate one supported filing form."""

    cleaned = str(form).strip().upper()

    if cleaned not in SUPPORTED_FORMS:
        raise ValueError(
            f"Unsupported form {form!r}. Expected one of {SUPPORTED_FORMS}."
        )

    return cleaned


def normalize_forms(forms: Sequence[str]) -> list[str]:
    """Normalize a sequence of forms while preserving order."""

    if isinstance(forms, (str, bytes)):
        raise TypeError("forms must be a sequence such as ('10-K', '10-Q')")

    cleaned_forms: list[str] = []

    for form in forms:
        cleaned = normalize_form(form)

        if cleaned not in cleaned_forms:
            cleaned_forms.append(cleaned)

    if not cleaned_forms:
        raise ValueError("forms must contain at least one supported form")

    return cleaned_forms


def safe_filename(value: Any) -> str:
    """Convert a value into a portable filename component."""

    # Replace spaces and unusual characters with underscores.
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())

    # Avoid an empty filename component.
    return cleaned or "unknown"


def accession_matches(left: Any, right: Any) -> bool:
    """Compare accession numbers after conservative string cleaning."""

    return str(left).strip() == str(right).strip()


def _first_present_attribute(
    obj: Any,
    names: Sequence[str],
    *,
    default: Any = None,
) -> Any:
    """Return the first non-None attribute found on an object."""

    for name in names:
        if not hasattr(obj, name):
            continue

        value = getattr(obj, name)

        if value is not None:
            return value

    return default


def _coerce_optional_datetime(value: Any) -> datetime | None:
    """Convert a value into a timezone-aware UTC datetime when possible."""

    if value is None:
        return None

    # pandas handles Python dates, datetimes, strings, and Timestamps.
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)

    if pd.isna(timestamp):
        return None

    return timestamp.to_pydatetime().astimezone(timezone.utc)


def _coerce_date(value: Any, *, field_name: str) -> date:
    """Convert a date-like value to a Python date or raise a clear error."""

    timestamp = pd.to_datetime(value, errors="coerce")

    if pd.isna(timestamp):
        raise ValueError(f"Could not parse {field_name}: {value!r}")

    return timestamp.date()


def _coerce_optional_date(value: Any) -> date | None:
    """Convert a date-like value to a Python date when available."""

    if value is None:
        return None

    timestamp = pd.to_datetime(value, errors="coerce")

    if pd.isna(timestamp):
        return None

    return timestamp.date()


# -----------------------------------------------------------------------------
# Filing retrieval and metadata normalization
# -----------------------------------------------------------------------------


def get_recent_company_filings(
    ticker: str,
    forms: Sequence[str] = ("10-K", "10-Q"),
    max_filings: int = 40,
    *,
    company_factory: Callable[[str], Any] | None = None,
) -> list[Any]:
    """Retrieve recent EdgarTools filing objects for one ticker.

    This retains the clean pattern from the learning notebook:

        Company(ticker).get_filings(form=forms)

    Args:
        ticker:
            Stock ticker, for example "NVDA".

        forms:
            Filing forms to retrieve. The first project version supports
            standard 10-K and 10-Q filings.

        max_filings:
            Maximum number of filing objects to keep.

        company_factory:
            Optional test dependency. Production code normally leaves this
            argument as None.

    Returns:
        A normal Python list of EdgarTools filing objects.
    """

    ticker = normalize_ticker(ticker)
    forms = normalize_forms(forms)

    if int(max_filings) <= 0:
        raise ValueError("max_filings must be a positive integer")

    # Resolve the real EdgarTools Company class or a test double.
    Company = _resolve_company_factory(company_factory)

    # Create the EdgarTools company object.
    company = Company(ticker)

    # Let EdgarTools query and filter the company's submissions.
    filings_collection = company.get_filings(form=forms)

    # Convert the collection to an ordinary list. This keeps downstream
    # functions independent of EdgarTools collection internals.
    filings: list[Any] = []

    for index, filing in enumerate(filings_collection):
        if index >= int(max_filings):
            break

        filings.append(filing)

    return filings


def filing_to_record(ticker: str, filing: Any) -> FilingRecord:
    """Convert one EdgarTools filing object into a validated FilingRecord.

    The attribute aliases below reflect names used in the learning notebooks
    and common EdgarTools representations. Missing optional fields remain None.
    Required metadata fails loudly through Pydantic validation.
    """

    ticker = normalize_ticker(ticker)

    # Read required metadata.
    cik = _first_present_attribute(filing, ("cik",))
    company_name = _first_present_attribute(
        filing,
        ("company", "company_name", "name"),
    )
    form = _first_present_attribute(filing, ("form",))
    filing_date = _first_present_attribute(filing, ("filing_date",))
    accession_number = _first_present_attribute(
        filing,
        ("accession_no", "accession_number"),
    )

    # Read optional metadata.
    period_of_report = _first_present_attribute(
        filing,
        ("period_of_report", "report_period"),
    )
    primary_document = _first_present_attribute(
        filing,
        ("primary_document", "primary_doc"),
    )
    acceptance_datetime = _first_present_attribute(
        filing,
        (
            "acceptance_datetime",
            "acceptance_date_time",
            "accepted_at",
        ),
    )

    # Build the shared validated record.
    return FilingRecord(
        ticker=ticker,
        cik=cik,
        company_name=company_name,
        form=normalize_form(form),
        filing_date=_coerce_date(
            filing_date,
            field_name="filing_date",
        ),
        period_of_report=_coerce_optional_date(period_of_report),
        accession_number=str(accession_number).strip(),
        primary_document=(
            str(primary_document).strip()
            if primary_document is not None
            else None
        ),
        acceptance_datetime=_coerce_optional_datetime(
            acceptance_datetime
        ),
    )


def filings_to_records(
    ticker: str,
    filings: Iterable[Any],
    *,
    continue_on_error: bool = False,
) -> list[FilingRecord]:
    """Convert EdgarTools filing objects into validated records.

    Args:
        continue_on_error:
            If False, the first malformed filing raises immediately.
            If True, malformed filing objects are skipped.

    Production research runs should normally use False so unexpected schema
    issues are visible. True is useful for exploratory inspection.
    """

    records: list[FilingRecord] = []

    for filing in filings:
        try:
            records.append(filing_to_record(ticker, filing))

        except Exception:
            if not continue_on_error:
                raise

    # Sort newest filing first for convenient inspection.
    records.sort(
        key=lambda record: record.filing_date,
        reverse=True,
    )

    return records


def records_to_dataframe(
    records: Sequence[FilingRecord],
) -> pd.DataFrame:
    """Convert validated filing records to a stable pandas table."""

    columns = [
        "ticker",
        "cik",
        "company_name",
        "form",
        "filing_date",
        "period_of_report",
        "accession_number",
        "primary_document",
        "acceptance_datetime",
    ]

    rows = [
        record.model_dump(mode="python")
        for record in records
    ]

    dataframe = pd.DataFrame(rows, columns=columns)

    if dataframe.empty:
        return dataframe

    # Preserve useful pandas dtypes for inspection and point-in-time joins.
    dataframe["filing_date"] = pd.to_datetime(
        dataframe["filing_date"],
        errors="coerce",
    )
    dataframe["period_of_report"] = pd.to_datetime(
        dataframe["period_of_report"],
        errors="coerce",
    )
    dataframe["acceptance_datetime"] = pd.to_datetime(
        dataframe["acceptance_datetime"],
        errors="coerce",
        utc=True,
    )

    return dataframe.sort_values(
        ["ticker", "filing_date", "accession_number"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def get_company_filings_for_ticker(
    ticker: str,
    forms: Sequence[str] = ("10-K", "10-Q"),
    max_filings: int = 40,
    *,
    company_factory: Callable[[str], Any] | None = None,
) -> tuple[list[FilingRecord], list[Any]]:
    """Retrieve filing objects and their validated metadata records."""

    filings = get_recent_company_filings(
        ticker=ticker,
        forms=forms,
        max_filings=max_filings,
        company_factory=company_factory,
    )

    records = filings_to_records(
        ticker=ticker,
        filings=filings,
    )

    return records, filings


def save_filing_records_pickle(
    records: Sequence[FilingRecord],
    path: str | Path,
) -> Path:
    """Save validated filing metadata using the project's .pkl convention."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = records_to_dataframe(records)
    dataframe.to_pickle(path)

    return path


def load_filing_records_pickle(
    path: str | Path,
) -> list[FilingRecord]:
    """Load a .pkl filing manifest and revalidate every row."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    dataframe = pd.read_pickle(path)

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("filing-record pickle must contain a pandas DataFrame")

    records: list[FilingRecord] = []

    for row in dataframe.to_dict(orient="records"):
        # Convert pandas missing values back to ordinary None values before
        # passing the row through Pydantic.
        cleaned_row = {
            key: (None if pd.isna(value) else value)
            for key, value in row.items()
        }

        records.append(FilingRecord.model_validate(cleaned_row))

    return records


# -----------------------------------------------------------------------------
# Filing recovery by accession number
# -----------------------------------------------------------------------------


def find_filing_by_accession(
    record: FilingRecord,
    max_filings: int = 80,
    *,
    company_factory: Callable[[str], Any] | None = None,
) -> Any:
    """Recover the EdgarTools filing object represented by a FilingRecord."""

    filings = get_recent_company_filings(
        ticker=record.ticker,
        forms=(record.form,),
        max_filings=max_filings,
        company_factory=company_factory,
    )

    for filing in filings:
        filing_accession = _first_present_attribute(
            filing,
            ("accession_no", "accession_number"),
        )

        if accession_matches(
            filing_accession,
            record.accession_number,
        ):
            return filing

    raise LookupError(
        f"Could not find {record.ticker} {record.form} accession "
        f"{record.accession_number}"
    )


# -----------------------------------------------------------------------------
# Point-in-time selection helpers
# -----------------------------------------------------------------------------


def filing_effective_date(record: FilingRecord) -> date:
    """Return the first calendar date on which a filing could be available.

    If an SEC acceptance timestamp is present, its UTC calendar date is used.
    Otherwise the filing date is used.

    A later module can apply a trading-calendar rule, such as moving filings
    accepted after market close to the next trading date.
    """

    if record.acceptance_datetime is not None:
        return record.acceptance_datetime.date()

    return record.filing_date


def latest_filing_record_before(
    records: Sequence[FilingRecord],
    rebalance_date: str | date | datetime,
) -> FilingRecord | None:
    """Select the latest filing record available by a historical date."""

    rebalance = _coerce_date(
        rebalance_date,
        field_name="rebalance_date",
    )

    eligible: list[tuple[date, date, str, FilingRecord]] = []

    for record in records:
        effective_date = filing_effective_date(record)

        if effective_date <= rebalance:
            # The tuple provides deterministic ordering:
            #   1. effective availability date,
            #   2. filing date,
            #   3. accession number.
            eligible.append(
                (
                    effective_date,
                    record.filing_date,
                    record.accession_number,
                    record,
                )
            )

    if not eligible:
        return None

    # Sort in-place (left to right) so we can rank Rank from newest to oldest. 
    # Key on when comparing two items use only the first three fields for ranking. 
    eligible.sort(
        key=lambda item: (item[0], item[1], item[2]),
        reverse=True,
    )

    # Return the first tuple in the sorted-list
    # And take the fourth element, which is the FilingRecord. 
    return eligible[0][3]


def build_point_in_time_selection(
    ticker: str,
    rebalance_date: str | date | datetime,
    records: Sequence[FilingRecord],
) -> PointInTimeFilingSelection | None:
    """Build one validated point-in-time filing selection.

    Returns None when no filing was available by the requested date.
    """

    ticker = normalize_ticker(ticker)

    selected_record = latest_filing_record_before(
        records,
        rebalance_date,
    )

    if selected_record is None:
        return None

    return PointInTimeFilingSelection(
        ticker=ticker,
        rebalance_date=_coerce_date(
            rebalance_date,
            field_name="rebalance_date",
        ),
        filing=selected_record,
        effective_signal_date=filing_effective_date(selected_record),
        selection_method="latest_eligible_filing",
    )


# -----------------------------------------------------------------------------
# Structured and fallback text extraction
# -----------------------------------------------------------------------------


def _section_value_to_text(value: Any) -> str:
    """Convert an EdgarTools section value into normalized plain text."""

    if value is None:
        return ""

    # Some parser objects expose their useful content through a text-like
    # property. Prefer those before falling back to str(value).
    for attribute_name in ("text", "content", "markdown"):
        if not hasattr(value, attribute_name):
            continue

        candidate = getattr(value, attribute_name)

        try:
            candidate = candidate() if callable(candidate) else candidate
        except Exception:
            continue

        text = normalize_text(str(candidate or ""))

        if text:
            return text

    return normalize_text(str(value))


def get_report_section(
    report: Any,
    possible_keys: Sequence[str] = (),
    *,
    possible_attributes: Sequence[str] = (),
    min_chars: int = 200,
) -> str | None:
    """Extract one usable section from an EdgarTools report object.

    The function tries named properties first, then dictionary-like item keys.
    It returns None when no sufficiently long section can be recovered.
    """

    if int(min_chars) < 0:
        raise ValueError("min_chars must be non-negative")

    # Preferred path: named report properties.
    for attribute_name in possible_attributes:
        try:
            raw_section = getattr(report, attribute_name)
            section = _section_value_to_text(raw_section)

            if len(section) > int(min_chars):
                return section

        except Exception:
            # Missing or failing properties are expected across filings.
            pass

    # Compatibility path from the learning notebooks: report[item_key].
    for key in possible_keys:
        try:
            raw_section = report[key]
            section = _section_value_to_text(raw_section)

            if len(section) > int(min_chars):
                return section

        except Exception:
            pass

    return None


def _read_filing_text_method(
    filing: Any,
    method_name: str,
) -> str:
    """Call or read one EdgarTools filing text accessor."""

    if not hasattr(filing, method_name):
        raise AttributeError(
            f"filing has no {method_name!r} accessor"
        )

    accessor = getattr(filing, method_name)

    # EdgarTools normally exposes markdown() and text() as methods.
    # Supporting a direct property as well makes the fallback more tolerant.
    raw_text = accessor() if callable(accessor) else accessor

    return normalize_text(str(raw_text or ""))


def truncate_section_head_tail(
    text: str,
    max_chars: int,
    *,
    head_fraction: float = 0.65,
) -> str:
    """Keep the beginning and end of an oversized filing section.

    The beginning of MD&A usually contains management's operating overview and
    results discussion. The end often contains liquidity, capital resources,
    commitments, or related disclosures. Keeping both ends is therefore more
    informative than retaining only the first max_chars characters.

    The returned string never exceeds max_chars.
    """

    if int(max_chars) <= 0:
        raise ValueError("max_chars must be a positive integer")

    if not 0.0 < float(head_fraction) < 1.0:
        raise ValueError("head_fraction must be strictly between 0 and 1")

    text = str(text)
    max_chars = int(max_chars)

    if len(text) <= max_chars:
        return text

    marker = SECTION_MIDDLE_TRUNCATION_MARKER

    # Very small limits do not leave enough room for a useful marker plus both
    # a head and a tail. In that unusual case, use simple front truncation.
    if max_chars <= len(marker) + 2:
        return text[:max_chars]

    content_budget = max_chars - len(marker)
    head_chars = int(content_budget * float(head_fraction))
    tail_chars = content_budget - head_chars

    head = text[:head_chars]
    tail = text[-tail_chars:]

    return head + marker + tail


def _resolve_section_char_limits(
    form: str,
    section_char_limits: dict[str, int] | None,
) -> dict[str, int]:
    """Return validated section limits for one filing form.

    A caller can override one or more defaults. Unspecified sections keep their
    form-specific default values.
    """

    form = normalize_form(form)

    if form == "10-K":
        resolved_limits = DEFAULT_10K_SECTION_CHAR_LIMITS.copy()
    else:
        resolved_limits = DEFAULT_10Q_SECTION_CHAR_LIMITS.copy()

    if section_char_limits is not None:
        allowed_names = {
            "management_discussion",
            "business",
            "risk_factors",
        }

        unknown_names = set(section_char_limits) - allowed_names

        if unknown_names:
            raise ValueError(
                "Unknown section names in section_char_limits: "
                f"{sorted(unknown_names)}"
            )

        for section_name, limit in section_char_limits.items():
            if int(limit) <= 0:
                raise ValueError(
                    "Every section character limit must be positive"
                )

            # Ignore a Business override for a 10-Q because that section is not
            # part of the quarterly filing specification.
            if section_name in resolved_limits:
                resolved_limits[section_name] = int(limit)

    return resolved_limits


def _requested_section_names(
    section_specs: dict[str, dict[str, tuple[str, ...]]],
    *,
    include_business: bool,
    include_risk_factors: bool,
    include_mda: bool,
) -> list[str]:
    """Return requested section names in the explicit extraction order."""

    requested_flags = {
        "management_discussion": bool(include_mda),
        "business": bool(include_business),
        "risk_factors": bool(include_risk_factors),
    }

    requested_names = [
        section_name
        for section_name in section_specs
        if requested_flags[section_name]
    ]

    if not requested_names:
        raise ValueError(
            "No applicable filing sections are enabled for extraction"
        )

    return requested_names


def _calculate_effective_section_limits(
    requested_names: Sequence[str],
    configured_limits: dict[str, int],
    max_chars: int,
) -> dict[str, int]:
    """Fit section budgets inside the final selected-text safety cap.

    The default section budgets total roughly 40,000 characters. Visible
    section headings and separators also consume a small amount of space. When
    needed, this helper scales all requested section limits proportionally so
    the final joined text remains within max_chars while preserving the intended
    MD&A-heavy allocation.
    """

    max_chars = int(max_chars)

    if max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")

    heading_chars = sum(
        len(f"SECTION: {section_name}\n")
        for section_name in requested_names
    )
    separator_chars = max(0, len(requested_names) - 1) * len("\n\n")
    available_content_chars = (
        max_chars - heading_chars - separator_chars
    )

    if available_content_chars < len(requested_names):
        raise ValueError(
            "max_chars is too small for the requested section headings "
            "and at least one character from each requested section"
        )

    configured_total = sum(
        configured_limits[section_name]
        for section_name in requested_names
    )

    if configured_total <= available_content_chars:
        return {
            section_name: configured_limits[section_name]
            for section_name in requested_names
        }

    scale = available_content_chars / configured_total

    effective_limits = {
        section_name: max(
            1,
            int(configured_limits[section_name] * scale),
        )
        for section_name in requested_names
    }

    # Integer conversion can leave a few unused characters. Add them back in
    # extraction-priority order without exceeding the configured section caps.
    remaining_chars = (
        available_content_chars - sum(effective_limits.values())
    )

    for section_name in requested_names:
        if remaining_chars <= 0:
            break

        available_room = (
            configured_limits[section_name]
            - effective_limits[section_name]
        )
        added_chars = min(available_room, remaining_chars)
        effective_limits[section_name] += added_chars
        remaining_chars -= added_chars

    return effective_limits


def _extract_focused_sections_with_details(
    filing: Any,
    form: str,
    max_chars: int = 40_000,
    *,
    min_section_chars: int = 200,
    min_fallback_chars: int = 500,
    include_business: bool = True,
    include_risk_factors: bool = True,
    include_mda: bool = True,
    section_char_limits: dict[str, int] | None = None,
    head_fraction: float = 0.65,
) -> tuple[
    str,
    list[str],
    str,
    list[SectionExtractionDetail],
]:
    """Extract selected sections and retain section-level audit details."""

    form = normalize_form(form)

    if int(min_section_chars) < 0:
        raise ValueError("min_section_chars must be non-negative")

    if int(min_fallback_chars) < 0:
        raise ValueError("min_fallback_chars must be non-negative")

    if not 0.0 < float(head_fraction) < 1.0:
        raise ValueError("head_fraction must be strictly between 0 and 1")

    section_specs = (
        TENK_SECTION_SPECS
        if form == "10-K"
        else TENQ_SECTION_SPECS
    )

    configured_limits = _resolve_section_char_limits(
        form,
        section_char_limits,
    )

    requested_names = _requested_section_names(
        section_specs,
        include_business=include_business,
        include_risk_factors=include_risk_factors,
        include_mda=include_mda,
    )

    effective_limits = _calculate_effective_section_limits(
        requested_names,
        configured_limits,
        max_chars,
    )

    requested_name_set = set(requested_names)

    # Start with an explicit not-found record for every applicable section.
    # Found sections replace these defaults below.
    details_by_name: dict[str, SectionExtractionDetail] = {}

    for section_name in section_specs:
        requested = section_name in requested_name_set
        applied_limit = (
            effective_limits[section_name]
            if requested
            else configured_limits[section_name]
        )

        details_by_name[section_name] = SectionExtractionDetail(
            section_name=section_name,
            requested=requested,
            found=False,
            original_char_count=0,
            retained_char_count=0,
            char_limit=applied_limit,
            truncated=False,
        )

    errors: list[str] = []

    # Preferred path: use EdgarTools' specialized report object.
    try:
        report = filing.obj()

        if report is not None:
            blocks: list[str] = []
            selected_sections: list[str] = []

            for section_name, specification in section_specs.items():
                if section_name not in requested_name_set:
                    continue

                section_text = get_report_section(
                    report,
                    possible_keys=specification["keys"],
                    possible_attributes=specification["attributes"],
                    min_chars=min_section_chars,
                )

                if section_text is None:
                    continue

                original_char_count = len(section_text)
                applied_limit = effective_limits[section_name]
                retained_text = truncate_section_head_tail(
                    section_text,
                    applied_limit,
                    head_fraction=head_fraction,
                )
                retained_char_count = len(retained_text)

                details_by_name[section_name] = SectionExtractionDetail(
                    section_name=section_name,
                    requested=True,
                    found=True,
                    original_char_count=original_char_count,
                    retained_char_count=retained_char_count,
                    char_limit=applied_limit,
                    truncated=(
                        retained_char_count < original_char_count
                    ),
                )

                blocks.append(
                    f"SECTION: {section_name}\n{retained_text}"
                )
                selected_sections.append(section_name)

            if blocks:
                selected_text = "\n\n".join(blocks)
                section_details = [
                    details_by_name[section_name]
                    for section_name in section_specs
                ]

                # The effective section limits reserve space for headings, so
                # this assertion protects the final total-cap contract.
                if len(selected_text) > int(max_chars):
                    raise RuntimeError(
                        "Internal section budgeting exceeded max_chars"
                    )

                return (
                    selected_text,
                    selected_sections,
                    "filing.obj() selected sections",
                    section_details,
                )

    except Exception as exc:
        errors.append(f"filing.obj(): {exc!r}")

    section_details = [
        details_by_name[section_name]
        for section_name in section_specs
    ]

    # Fallback 1: use the complete markdown representation. The structured
    # details remain marked as not found so the audit trail shows that the
    # requested sections were not recovered separately.
    try:
        markdown = _read_filing_text_method(
            filing,
            "markdown",
        )

        if len(markdown) > int(min_fallback_chars):
            retained_markdown = truncate_section_head_tail(
                markdown,
                int(max_chars),
                head_fraction=head_fraction,
            )

            return (
                retained_markdown,
                ["full_markdown_fallback"],
                "filing.markdown() fallback",
                section_details,
            )

    except Exception as exc:
        errors.append(f"filing.markdown(): {exc!r}")

    # Fallback 2: use the complete plain-text representation.
    try:
        plain_text = _read_filing_text_method(
            filing,
            "text",
        )

        if len(plain_text) > int(min_fallback_chars):
            retained_plain_text = truncate_section_head_tail(
                plain_text,
                int(max_chars),
                head_fraction=head_fraction,
            )

            return (
                retained_plain_text,
                ["full_text_fallback"],
                "filing.text() fallback",
                section_details,
            )

    except Exception as exc:
        errors.append(f"filing.text(): {exc!r}")

    error_detail = "; ".join(errors) if errors else "no usable text returned"

    raise RuntimeError(
        "Could not extract usable filing text. "
        f"Details: {error_detail}"
    )


def extract_focused_sections(
    filing: Any,
    form: str,
    max_chars: int = 40_000,
    *,
    min_section_chars: int = 200,
    min_fallback_chars: int = 500,
    include_business: bool = True,
    include_risk_factors: bool = True,
    include_mda: bool = True,
    section_char_limits: dict[str, int] | None = None,
    head_fraction: float = 0.65,
) -> tuple[str, list[str], str]:
    """Extract focused filing text while preserving the original API.

    The default baseline includes all applicable narrative sections:

        10-K: MD&A, Business, Risk Factors
        10-Q: MD&A, Risk Factors

    The include_* flags support later ablation studies. Section-specific limits
    prevent long Business or Risk Factors text from crowding MD&A out of the
    model input.

    This public wrapper intentionally returns the same three values as the
    previous implementation. build_selected_input() uses the detailed helper so
    it can also attach SectionExtractionDetail records.
    """

    selected_text, selected_sections, extraction_method, _ = (
        _extract_focused_sections_with_details(
            filing=filing,
            form=form,
            max_chars=max_chars,
            min_section_chars=min_section_chars,
            min_fallback_chars=min_fallback_chars,
            include_business=include_business,
            include_risk_factors=include_risk_factors,
            include_mda=include_mda,
            section_char_limits=section_char_limits,
            head_fraction=head_fraction,
        )
    )

    return selected_text, selected_sections, extraction_method


# -----------------------------------------------------------------------------
# Optional complete-filing cache
# -----------------------------------------------------------------------------


def extract_raw_filing_text(
    filing: Any,
    *,
    preferred_format: str = "markdown",
) -> tuple[str, str, str]:
    """Extract a complete filing copy for optional local caching.

    Args:
        filing:
            EdgarTools filing object.

        preferred_format:
            Either "markdown" or "text".

    Returns:
        A tuple containing:

            raw_text
            file_extension
            extraction_method

    The preferred accessor is tried first. If it fails or returns blank text,
    the other EdgarTools accessor is used as a fallback.
    """

    preferred_format = str(preferred_format).strip().lower()

    if preferred_format not in {"markdown", "text"}:
        raise ValueError(
            "preferred_format must be 'markdown' or 'text'"
        )

    # Keep the fallback order explicit and easy to audit.
    if preferred_format == "markdown":
        attempts = [
            ("markdown", ".md", "filing.markdown() raw cache"),
            ("text", ".txt", "filing.text() raw-cache fallback"),
        ]
    else:
        attempts = [
            ("text", ".txt", "filing.text() raw cache"),
            ("markdown", ".md", "filing.markdown() raw-cache fallback"),
        ]

    errors: list[str] = []

    for method_name, extension, extraction_method in attempts:
        try:
            raw_text = _read_filing_text_method(
                filing,
                method_name,
            )

            if raw_text:
                return (
                    raw_text,
                    extension,
                    extraction_method,
                )

        except Exception as exc:
            errors.append(f"{method_name}: {exc!r}")

    error_detail = "; ".join(errors) if errors else "no usable text returned"

    raise RuntimeError(
        "Could not extract complete filing text for raw caching. "
        f"Details: {error_detail}"
    )


def save_raw_filing_text(
    record: FilingRecord,
    filing: Any,
    output_dir: str | Path,
    *,
    preferred_format: str = "markdown",
    created_at_utc: datetime | None = None,
) -> TextArtifact:
    """Save and validate an optional complete-filing cache.

    The cache is not used as the normal OpenAI input. It exists for:

        - offline inspection,
        - parser debugging,
        - reproducibility checks,
        - final archived research runs.
    """

    raw_text, extension, extraction_method = extract_raw_filing_text(
        filing,
        preferred_format=preferred_format,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{safe_filename(record.ticker)}_"
        f"{safe_filename(record.form)}_"
        f"{safe_filename(record.filing_date)}_"
        f"{safe_filename(record.accession_number)}_"
        f"raw{extension}"
    )

    path = output_dir / filename

    # `_read_filing_text_method` already normalizes whitespace. Save exactly
    # that returned text so the stored hash is deterministic.
    path.write_text(raw_text, encoding="utf-8")

    if created_at_utc is None:
        created_at_utc = datetime.now(timezone.utc)

    return TextArtifact(
        path=str(path),
        extraction_method=extraction_method,
        sha256=sha256_text(raw_text),
        char_count=len(raw_text),
        word_count=len(raw_text.split()),
        created_at_utc=created_at_utc,
    )


# -----------------------------------------------------------------------------
# Saving selected text and building Pydantic inputs
# -----------------------------------------------------------------------------


def save_selected_text(
    record: FilingRecord,
    selected_text: str,
    extraction_method: str,
    output_dir: str | Path,
    *,
    created_at_utc: datetime | None = None,
) -> TextArtifact:
    """Save the exact focused text used downstream and validate its metadata."""

    # Keep the exact stored text deterministic.
    selected_text = normalize_text(selected_text)

    if not selected_text:
        raise ValueError("selected_text must not be blank")

    extraction_method = normalize_text(extraction_method)

    if not extraction_method:
        raise ValueError("extraction_method must not be blank")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{safe_filename(record.ticker)}_"
        f"{safe_filename(record.form)}_"
        f"{safe_filename(record.filing_date)}_"
        f"{safe_filename(record.accession_number)}_"
        "selected.txt"
    )

    path = output_dir / filename

    # Save the exact text before constructing TextArtifact. The Pydantic model
    # then re-reads the file and verifies its hash and size metadata.
    path.write_text(selected_text, encoding="utf-8")

    if created_at_utc is None:
        created_at_utc = datetime.now(timezone.utc)

    return TextArtifact(
        path=str(path),
        extraction_method=extraction_method,
        sha256=sha256_text(selected_text),
        char_count=len(selected_text),
        word_count=len(selected_text.split()),
        created_at_utc=created_at_utc,
    )


def build_selected_input(
    record: FilingRecord,
    output_dir: str | Path,
    max_chars: int = 40_000,
    *,
    filing: Any | None = None,
    company_factory: Callable[[str], Any] | None = None,
    max_filings: int = 80,
    created_at_utc: datetime | None = None,
    cache_raw: bool = False,
    raw_output_dir: str | Path | None = None,
    raw_preferred_format: str = "markdown",
    include_business: bool = True,
    include_risk_factors: bool = True,
    include_mda: bool = True,
    section_char_limits: dict[str, int] | None = None,
    head_fraction: float = 0.65,
) -> SelectedFilingInput:
    """Build one validated SelectedFilingInput from an SEC filing.

    Required path:

        1. recover the EdgarTools filing object if it was not supplied,
        2. extract enabled narrative sections,
        3. apply explicit per-section character budgets,
        4. save and hash the exact selected text,
        5. attach section-level extraction details.

    Optional raw cache:

        Set cache_raw=True to also save the complete filing as markdown or
        plain text. The raw cache is retained for debugging and reproducibility;
        it is not the default OpenAI input.

    Ablation controls:

        include_business, include_risk_factors, and include_mda can be changed
        later to test which narrative sections contribute predictive signal.
        All applicable sections remain enabled by default.
    """

    if filing is None:
        filing = find_filing_by_accession(
            record,
            max_filings=max_filings,
            company_factory=company_factory,
        )

    (
        selected_text,
        selected_sections,
        extraction_method,
        section_extraction_details,
    ) = _extract_focused_sections_with_details(
        filing=filing,
        form=record.form,
        max_chars=max_chars,
        include_business=include_business,
        include_risk_factors=include_risk_factors,
        include_mda=include_mda,
        section_char_limits=section_char_limits,
        head_fraction=head_fraction,
    )

    selected_text_artifact = save_selected_text(
        record=record,
        selected_text=selected_text,
        extraction_method=extraction_method,
        output_dir=output_dir,
        created_at_utc=created_at_utc,
    )

    raw_filing_artifact: TextArtifact | None = None

    if cache_raw:
        selected_output_dir = Path(output_dir)

        if raw_output_dir is None:
            raw_output_dir = selected_output_dir.parent / "raw_text"

        raw_filing_artifact = save_raw_filing_text(
            record=record,
            filing=filing,
            output_dir=raw_output_dir,
            preferred_format=raw_preferred_format,
            created_at_utc=created_at_utc,
        )

    return SelectedFilingInput(
        filing=record,
        selected_sections=selected_sections,
        section_extraction_details=section_extraction_details,
        selected_text=selected_text_artifact,
        raw_filing_text=raw_filing_artifact,
    )


# Explicit public API.
__all__ = [
    "TENK_SECTION_SPECS",
    "TENQ_SECTION_SPECS",
    "DEFAULT_10K_SECTION_CHAR_LIMITS",
    "DEFAULT_10Q_SECTION_CHAR_LIMITS",
    "SECTION_MIDDLE_TRUNCATION_MARKER",
    "SUPPORTED_FORMS",
    "configure_edgar_identity",
    "normalize_ticker",
    "normalize_form",
    "normalize_forms",
    "safe_filename",
    "accession_matches",
    "get_recent_company_filings",
    "filing_to_record",
    "filings_to_records",
    "records_to_dataframe",
    "get_company_filings_for_ticker",
    "save_filing_records_pickle",
    "load_filing_records_pickle",
    "find_filing_by_accession",
    "filing_effective_date",
    "latest_filing_record_before",
    "build_point_in_time_selection",
    "get_report_section",
    "truncate_section_head_tail",
    "extract_focused_sections",
    "extract_raw_filing_text",
    "save_raw_filing_text",
    "save_selected_text",
    "build_selected_input",
]
