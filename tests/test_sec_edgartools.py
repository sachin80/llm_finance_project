"""
Unit tests for src.sec_edgartools.

These tests use small fake EdgarTools objects. They do not contact the SEC,
do not call OpenAI, and do not require the `edgartools` package to be installed.

Run from the repository root with:

    pytest -q tests/test_sec_edgartools.py
"""

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.sec_edgartools import (
    accession_matches,
    build_point_in_time_selection,
    DEFAULT_10K_SECTION_CHAR_LIMITS,
    DEFAULT_10Q_SECTION_CHAR_LIMITS,
    SECTION_MIDDLE_TRUNCATION_MARKER,
    build_selected_input,
    configure_edgar_identity,
    extract_focused_sections,
    extract_raw_filing_text,
    filing_to_record,
    filings_to_records,
    find_filing_by_accession,
    get_company_filings_for_ticker,
    get_recent_company_filings,
    get_report_section,
    latest_filing_record_before,
    load_filing_records_pickle,
    normalize_form,
    normalize_forms,
    records_to_dataframe,
    safe_filename,
    save_filing_records_pickle,
    save_raw_filing_text,
    save_selected_text,
    truncate_section_head_tail,
)
from src.sec_models import FilingRecord


# -----------------------------------------------------------------------------
# Fake EdgarTools objects
# -----------------------------------------------------------------------------


LONG_BUSINESS = (
    "The company designs accelerated computing systems, processors, networking "
    "products, and software platforms for data centers. Customers use these "
    "products to build large-scale computing infrastructure and deploy advanced "
    "machine-learning workloads. " * 5
)

LONG_RISKS = (
    "Risk factors include competition, supply constraints, manufacturing "
    "capacity, customer concentration, regulation, cybersecurity, and changes "
    "in demand for data-center infrastructure. " * 5
)

LONG_MDA = (
    "Management discusses revenue growth, data-center demand, capital spending, "
    "gross margin, inventory, supply commitments, and expectations for future "
    "accelerated-computing adoption. " * 5
)

LONG_MARKDOWN = (
    "Full filing markdown fallback containing business discussion, financial "
    "statements, risk factors, management commentary, and additional material. "
    * 20
)

LONG_TEXT = (
    "Full filing plain-text fallback containing enough disclosure to be used "
    "when structured report parsing and markdown extraction are unavailable. "
    * 20
)


class FakeReport:
    """Small report object supporting properties and item access."""

    def __init__(
        self,
        *,
        attributes=None,
        items=None,
    ):
        self._items = items or {}

        for name, value in (attributes or {}).items():
            setattr(self, name, value)

    def __getitem__(self, key):
        if key not in self._items:
            raise KeyError(key)

        return self._items[key]


class FakeFiling:
    """Minimal filing object that resembles the EdgarTools API."""

    def __init__(
        self,
        *,
        ticker="NVDA",
        cik=1045810,
        company="NVIDIA Corporation",
        form="10-Q",
        filing_date_value=date(2024, 5, 29),
        period_of_report=date(2024, 4, 28),
        accession_no="0001045810-24-000124",
        primary_document="nvda-20240428.htm",
        acceptance_datetime=datetime(
            2024,
            5,
            29,
            16,
            22,
            tzinfo=timezone.utc,
        ),
        report=None,
        obj_error=None,
        markdown_value=LONG_MARKDOWN,
        markdown_error=None,
        text_value=LONG_TEXT,
        text_error=None,
    ):
        self.ticker = ticker
        self.cik = cik
        self.company = company
        self.form = form
        self.filing_date = filing_date_value
        self.period_of_report = period_of_report
        self.accession_no = accession_no
        self.primary_document = primary_document
        self.acceptance_datetime = acceptance_datetime

        self._report = report
        self._obj_error = obj_error
        self._markdown_value = markdown_value
        self._markdown_error = markdown_error
        self._text_value = text_value
        self._text_error = text_error

    def obj(self):
        if self._obj_error is not None:
            raise self._obj_error

        return self._report

    def markdown(self):
        if self._markdown_error is not None:
            raise self._markdown_error

        return self._markdown_value

    def text(self):
        if self._text_error is not None:
            raise self._text_error

        return self._text_value


class FakeCompany:
    """Company object that records the form argument passed to get_filings."""

    def __init__(self, ticker, filings):
        self.ticker = ticker
        self.cik = 1045810
        self.name = "NVIDIA Corporation"
        self._filings = list(filings)
        self.requested_forms = None

    def get_filings(self, form):
        self.requested_forms = form
        return list(self._filings)


class FakeCompanyFactory:
    """Callable factory used in place of edgar.Company during tests."""

    def __init__(self, filings_by_ticker):
        self.filings_by_ticker = filings_by_ticker
        self.created_companies = []

    def __call__(self, ticker):
        company = FakeCompany(
            ticker,
            self.filings_by_ticker.get(ticker, []),
        )
        self.created_companies.append(company)
        return company


# -----------------------------------------------------------------------------
# Shared test helpers
# -----------------------------------------------------------------------------


def make_10k_filing(
    *,
    filing_date_value=date(2024, 3, 1),
    period_of_report_value=date(2024, 1, 28),
    accession_no="0001045810-24-000050",
    acceptance_datetime=datetime(
        2024,
        3,
        1,
        13,
        0,
        tzinfo=timezone.utc,
    ),
):
    """Create a fake 10-K with structured report properties."""

    report = FakeReport(
        attributes={
            "business": LONG_BUSINESS,
            "risk_factors": LONG_RISKS,
            "management_discussion": LONG_MDA,
        }
    )

    return FakeFiling(
        form="10-K",
        filing_date_value=filing_date_value,
        period_of_report=period_of_report_value,
        accession_no=accession_no,
        acceptance_datetime=acceptance_datetime,
        report=report,
    )


def make_10q_filing(
    *,
    filing_date_value=date(2024, 5, 29),
    accession_no="0001045810-24-000124",
    acceptance_datetime=datetime(
        2024,
        5,
        29,
        16,
        22,
        tzinfo=timezone.utc,
    ),
):
    """Create a fake 10-Q using dictionary-like report item access."""

    report = FakeReport(
        items={
            "Item 2": LONG_MDA,
            "Item 1A": LONG_RISKS,
        }
    )

    return FakeFiling(
        form="10-Q",
        filing_date_value=filing_date_value,
        period_of_report=date(2024, 4, 28),
        accession_no=accession_no,
        acceptance_datetime=acceptance_datetime,
        report=report,
    )


# -----------------------------------------------------------------------------
# Small cleaning and configuration tests
# -----------------------------------------------------------------------------


def test_configure_edgar_identity_uses_injected_function():
    """Identity configuration should be testable without importing EdgarTools."""

    observed = []

    configure_edgar_identity(
        "Researcher researcher@example.com",
        set_identity_func=observed.append,
    )

    assert observed == ["Researcher researcher@example.com"]


def test_configure_edgar_identity_rejects_blank_value():
    """A blank SEC identity should fail before any external call."""

    with pytest.raises(ValueError, match="must not be blank"):
        configure_edgar_identity(
            "   ",
            set_identity_func=lambda value: None,
        )


def test_normalize_form_and_forms():
    """Only the first-version 10-K and 10-Q forms are accepted."""

    assert normalize_form(" 10-q ") == "10-Q"
    assert normalize_forms(["10-k", "10-Q", "10-k"]) == [
        "10-K",
        "10-Q",
    ]

    with pytest.raises(ValueError, match="Unsupported form"):
        normalize_form("8-K")


def test_safe_filename_and_accession_matches():
    """Filename cleaning and accession comparison should be conservative."""

    assert safe_filename("NVDA 10-Q / test") == "NVDA_10-Q_test"
    assert accession_matches(
        " 0001045810-24-000124 ",
        "0001045810-24-000124",
    )


# -----------------------------------------------------------------------------
# Filing retrieval and metadata tests
# -----------------------------------------------------------------------------


def test_get_recent_company_filings_uses_edgartools_pattern_and_limit():
    """The wrapper should call Company(...).get_filings(form=...)."""

    filings = [
        make_10q_filing(
            accession_no=f"accession-{index}",
        )
        for index in range(5)
    ]

    factory = FakeCompanyFactory({"NVDA": filings})

    result = get_recent_company_filings(
        " nvda ",
        forms=("10-K", "10-Q"),
        max_filings=3,
        company_factory=factory,
    )

    assert result == filings[:3]
    assert factory.created_companies[0].ticker == "NVDA"
    assert factory.created_companies[0].requested_forms == [
        "10-K",
        "10-Q",
    ]


def test_get_recent_company_filings_rejects_nonpositive_limit():
    """A nonpositive filing limit is a caller error."""

    with pytest.raises(ValueError, match="positive integer"):
        get_recent_company_filings(
            "NVDA",
            max_filings=0,
            company_factory=FakeCompanyFactory({}),
        )


def test_filing_to_record_maps_common_edgartools_attributes():
    """EdgarTools metadata should become a validated FilingRecord."""

    filing = make_10q_filing()

    record = filing_to_record(" nvda ", filing)

    assert record.ticker == "NVDA"
    assert record.cik == 1045810
    assert record.company_name == "NVIDIA Corporation"
    assert record.form == "10-Q"
    assert record.filing_date == date(2024, 5, 29)
    assert record.period_of_report == date(2024, 4, 28)
    assert record.accession_number == "0001045810-24-000124"
    assert record.primary_document == "nvda-20240428.htm"
    assert record.acceptance_datetime.tzinfo == timezone.utc


def test_filings_to_records_sorts_newest_first():
    """The manifest should have deterministic newest-first ordering."""

    older = make_10k_filing(
        filing_date_value=date(2023, 3, 1),
        period_of_report_value=date(2023, 1, 29),
        accession_no="older",
        acceptance_datetime=datetime(
            2023,
            3,
            1,
            13,
            0,
            tzinfo=timezone.utc,
        ),
    )
    newer = make_10q_filing(
        filing_date_value=date(2024, 5, 29),
        accession_no="newer",
    )

    records = filings_to_records(
        "NVDA",
        [older, newer],
    )

    assert [record.accession_number for record in records] == [
        "newer",
        "older",
    ]


def test_records_to_dataframe_has_stable_columns_and_dates():
    """Validated records should become a clean tabular manifest."""

    records = [
        filing_to_record("NVDA", make_10q_filing()),
        filing_to_record("NVDA", make_10k_filing()),
    ]

    dataframe = records_to_dataframe(records)

    assert list(dataframe.columns) == [
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
    assert pd.api.types.is_datetime64_any_dtype(
        dataframe["filing_date"]
    )


def test_get_company_filings_for_ticker_returns_records_and_objects():
    """The convenience wrapper should not hide the original filing objects."""

    filings = [make_10q_filing(), make_10k_filing()]
    factory = FakeCompanyFactory({"NVDA": filings})

    records, returned_filings = get_company_filings_for_ticker(
        "NVDA",
        company_factory=factory,
    )

    assert len(records) == 2
    assert returned_filings == filings


def test_filing_manifest_pickle_round_trip(tmp_path):
    """The chosen .pkl format should round-trip through Pydantic validation."""

    records = [
        filing_to_record("NVDA", make_10q_filing()),
        filing_to_record("NVDA", make_10k_filing()),
    ]

    path = save_filing_records_pickle(
        records,
        tmp_path / "filing_manifest.pkl",
    )

    loaded = load_filing_records_pickle(path)

    assert loaded == records


# -----------------------------------------------------------------------------
# Accession recovery tests
# -----------------------------------------------------------------------------


def test_find_filing_by_accession_returns_matching_object():
    """A validated record should recover the correct EdgarTools filing."""

    first = make_10q_filing(accession_no="first")
    target = make_10q_filing(accession_no="target")
    factory = FakeCompanyFactory({"NVDA": [first, target]})

    record = filing_to_record("NVDA", target)

    recovered = find_filing_by_accession(
        record,
        company_factory=factory,
    )

    assert recovered is target
    assert factory.created_companies[0].requested_forms == ["10-Q"]


def test_find_filing_by_accession_raises_for_missing_record():
    """Missing accessions should fail loudly rather than choose a nearby filing."""

    filing = make_10q_filing(accession_no="available")
    factory = FakeCompanyFactory({"NVDA": [filing]})

    missing_record = FilingRecord(
        ticker="NVDA",
        cik=1045810,
        company_name="NVIDIA Corporation",
        form="10-Q",
        filing_date=date(2024, 5, 29),
        period_of_report=date(2024, 4, 28),
        accession_number="missing",
    )

    with pytest.raises(LookupError, match="Could not find"):
        find_filing_by_accession(
            missing_record,
            company_factory=factory,
        )


# -----------------------------------------------------------------------------
# Point-in-time selection tests
# -----------------------------------------------------------------------------


def test_latest_filing_record_before_selects_latest_eligible_record():
    """Selection should use only records available by the requested date."""

    annual = filing_to_record("NVDA", make_10k_filing())
    quarterly = filing_to_record("NVDA", make_10q_filing())

    selected = latest_filing_record_before(
        [annual, quarterly],
        date(2024, 5, 30),
    )

    assert selected == quarterly


def test_latest_filing_record_before_returns_none_when_all_are_future():
    """No filing should be selected before the first eligible filing."""

    annual = filing_to_record("NVDA", make_10k_filing())

    selected = latest_filing_record_before(
        [annual],
        date(2024, 2, 1),
    )

    assert selected is None


def test_build_point_in_time_selection_returns_validated_model():
    """The selection helper should enforce the shared no-lookahead schema."""

    annual = filing_to_record("NVDA", make_10k_filing())
    quarterly = filing_to_record("NVDA", make_10q_filing())

    selection = build_point_in_time_selection(
        "NVDA",
        date(2024, 5, 31),
        [annual, quarterly],
    )

    assert selection is not None
    assert selection.filing == quarterly
    assert selection.effective_signal_date <= selection.rebalance_date


# -----------------------------------------------------------------------------
# Structured section extraction tests
# -----------------------------------------------------------------------------


def test_get_report_section_prefers_named_attribute():
    """Recent report-object properties should be used when available."""

    report = FakeReport(
        attributes={"risk_factors": LONG_RISKS},
        items={"Item 1A": "short"},
    )

    section = get_report_section(
        report,
        possible_keys=("Item 1A",),
        possible_attributes=("risk_factors",),
    )

    assert section == " ".join(LONG_RISKS.split())


def test_get_report_section_falls_back_to_item_key():
    """Dictionary-like report access should remain supported."""

    report = FakeReport(
        items={"Item 2": LONG_MDA},
    )

    section = get_report_section(
        report,
        possible_keys=("2", "Item 2"),
        possible_attributes=("management_discussion",),
    )

    assert section == " ".join(LONG_MDA.split())


def test_extract_focused_sections_10k_uses_expected_sections():
    """10-K extraction should include Business, Risk Factors, and MD&A."""

    filing = make_10k_filing()

    text, sections, method = extract_focused_sections(
        filing,
        form="10-K",
    )

    assert sections == [
        "management_discussion",
        "business",
        "risk_factors",
    ]
    assert method == "filing.obj() selected sections"
    assert "SECTION: business" in text
    assert "SECTION: risk_factors" in text
    assert "SECTION: management_discussion" in text


def test_extract_focused_sections_10q_uses_expected_sections():
    """10-Q extraction should include MD&A and Risk Factors."""

    filing = make_10q_filing()

    text, sections, method = extract_focused_sections(
        filing,
        form="10-Q",
    )

    assert sections == [
        "management_discussion",
        "risk_factors",
    ]
    assert method == "filing.obj() selected sections"
    assert "SECTION: management_discussion" in text


def test_default_section_budgets_match_research_baseline():
    """The default narrative allocation should remain explicit and stable."""

    assert DEFAULT_10K_SECTION_CHAR_LIMITS == {
        "management_discussion": 22_000,
        "business": 10_000,
        "risk_factors": 8_000,
    }
    assert DEFAULT_10Q_SECTION_CHAR_LIMITS == {
        "management_discussion": 28_000,
        "risk_factors": 12_000,
    }


def test_truncate_section_head_tail_keeps_both_ends():
    """Long sections should retain both early and late disclosure text."""

    text = "HEAD-" + ("middle-" * 200) + "-TAIL"

    retained = truncate_section_head_tail(
        text,
        max_chars=200,
        head_fraction=0.65,
    )

    assert len(retained) == 200
    assert retained.startswith("HEAD-")
    assert retained.endswith("-TAIL")
    assert SECTION_MIDDLE_TRUNCATION_MARKER in retained


def test_extract_focused_sections_supports_mda_only_ablation():
    """Ablation controls should permit an MD&A-only experiment."""

    filing = make_10k_filing()

    text, sections, method = extract_focused_sections(
        filing,
        form="10-K",
        include_business=False,
        include_risk_factors=False,
        include_mda=True,
    )

    assert sections == ["management_discussion"]
    assert method == "filing.obj() selected sections"
    assert "SECTION: management_discussion" in text
    assert "SECTION: business" not in text
    assert "SECTION: risk_factors" not in text


def test_extract_focused_sections_supports_business_and_mda_ablation():
    """The researcher should be able to disable Risk Factors explicitly."""

    filing = make_10k_filing()

    text, sections, _ = extract_focused_sections(
        filing,
        form="10-K",
        include_business=True,
        include_risk_factors=False,
        include_mda=True,
    )

    assert sections == [
        "management_discussion",
        "business",
    ]
    assert "SECTION: risk_factors" not in text


def test_extract_focused_sections_rejects_all_disabled_sections():
    """A caller error should be visible instead of silently using a fallback."""

    filing = make_10k_filing()

    with pytest.raises(
        ValueError,
        match="No applicable filing sections are enabled",
    ):
        extract_focused_sections(
            filing,
            form="10-K",
            include_business=False,
            include_risk_factors=False,
            include_mda=False,
        )


def test_custom_section_limits_are_applied_without_truncating_short_sections():
    """Custom limits should be easy to inspect in a controlled experiment."""

    filing = make_10k_filing()

    text, sections, _ = extract_focused_sections(
        filing,
        form="10-K",
        max_chars=5_000,
        section_char_limits={
            "management_discussion": 2_000,
            "business": 1_500,
            "risk_factors": 1_000,
        },
    )

    assert sections == [
        "management_discussion",
        "business",
        "risk_factors",
    ]
    assert len(text) <= 5_000


def test_extract_focused_sections_falls_back_to_markdown():
    """Markdown should be used when report parsing fails."""

    filing = FakeFiling(
        obj_error=RuntimeError("parser failed"),
        markdown_value=LONG_MARKDOWN,
    )

    text, sections, method = extract_focused_sections(
        filing,
        form="10-Q",
    )

    assert sections == ["full_markdown_fallback"]
    assert method == "filing.markdown() fallback"
    assert text == " ".join(LONG_MARKDOWN.split())


def test_extract_focused_sections_falls_back_to_plain_text():
    """Plain text should be used when both structured and markdown paths fail."""

    filing = FakeFiling(
        obj_error=RuntimeError("parser failed"),
        markdown_error=RuntimeError("markdown failed"),
        text_value=LONG_TEXT,
    )

    text, sections, method = extract_focused_sections(
        filing,
        form="10-Q",
    )

    assert sections == ["full_text_fallback"]
    assert method == "filing.text() fallback"
    assert text == " ".join(LONG_TEXT.split())


def test_extract_focused_sections_rejects_unusable_text():
    """All extraction paths failing should produce a clear RuntimeError."""

    filing = FakeFiling(
        report=FakeReport(),
        markdown_value="too short",
        text_value="also too short",
    )

    with pytest.raises(
        RuntimeError,
        match="Could not extract usable filing text",
    ):
        extract_focused_sections(
            filing,
            form="10-Q",
        )


def test_extract_focused_sections_obeys_max_chars():
    """The selected text should never exceed the caller's explicit limit."""

    filing = make_10k_filing()

    text, _, _ = extract_focused_sections(
        filing,
        form="10-K",
        max_chars=500,
    )

    assert len(text) == 500


# -----------------------------------------------------------------------------
# Selected text and SelectedFilingInput tests
# -----------------------------------------------------------------------------


def test_save_selected_text_creates_validated_artifact(tmp_path):
    """Saving should produce a file whose hash and counts are revalidated."""

    record = filing_to_record("NVDA", make_10q_filing())

    artifact = save_selected_text(
        record=record,
        selected_text=LONG_MDA,
        extraction_method="test extraction",
        output_dir=tmp_path,
        created_at_utc=datetime(2026, 6, 17, 12, 0),
    )

    path = Path(artifact.path)

    assert path.exists()
    assert artifact.extraction_method == "test extraction"
    assert artifact.char_count == len(
        " ".join(LONG_MDA.split())
    )
    assert artifact.created_at_utc.tzinfo == timezone.utc


def test_build_selected_input_uses_supplied_filing_without_network(tmp_path):
    """The complete extraction path should work with an existing filing object."""

    filing = make_10q_filing()
    record = filing_to_record("NVDA", filing)

    selected_input = build_selected_input(
        record=record,
        filing=filing,
        output_dir=tmp_path,
        created_at_utc=datetime(2026, 6, 17, 12, 0),
    )

    assert selected_input.filing == record
    assert selected_input.selected_sections == [
        "management_discussion",
        "risk_factors",
    ]
    assert Path(selected_input.selected_text.path).exists()
    assert (
        selected_input.selected_text.extraction_method
        == "filing.obj() selected sections"
    )


def test_build_selected_input_attaches_section_extraction_details(tmp_path):
    """The saved selected input should explain what happened to each section."""

    filing = make_10q_filing()
    record = filing_to_record("NVDA", filing)

    selected_input = build_selected_input(
        record=record,
        filing=filing,
        output_dir=tmp_path,
    )

    details = {
        detail.section_name: detail
        for detail in selected_input.section_extraction_details
    }

    assert set(details) == {
        "management_discussion",
        "risk_factors",
    }
    assert details["management_discussion"].requested is True
    assert details["management_discussion"].found is True
    assert details["management_discussion"].original_char_count > 0
    assert details["management_discussion"].retained_char_count > 0


def test_build_selected_input_records_missing_requested_section(tmp_path):
    """A missing structured section should remain visible in the audit record."""

    report = FakeReport(
        attributes={
            "management_discussion": LONG_MDA,
        }
    )
    filing = FakeFiling(
        form="10-Q",
        report=report,
    )
    record = filing_to_record("NVDA", filing)

    selected_input = build_selected_input(
        record=record,
        filing=filing,
        output_dir=tmp_path,
    )

    details = {
        detail.section_name: detail
        for detail in selected_input.section_extraction_details
    }

    assert details["management_discussion"].found is True
    assert details["risk_factors"].requested is True
    assert details["risk_factors"].found is False
    assert details["risk_factors"].original_char_count == 0


def test_build_selected_input_can_cache_raw_filing(tmp_path):
    """The optional complete filing cache should remain compatible."""

    filing = make_10q_filing()
    record = filing_to_record("NVDA", filing)

    selected_input = build_selected_input(
        record=record,
        filing=filing,
        output_dir=tmp_path / "selected_text",
        cache_raw=True,
        raw_output_dir=tmp_path / "raw_text",
    )

    assert selected_input.raw_filing_text is not None
    assert Path(selected_input.raw_filing_text.path).exists()
    assert Path(selected_input.raw_filing_text.path).suffix == ".md"


def test_extract_raw_filing_text_falls_back_to_plain_text():
    """Raw caching should use plain text when markdown extraction fails."""

    filing = FakeFiling(
        markdown_error=RuntimeError("markdown unavailable"),
        text_value=LONG_TEXT,
    )

    raw_text, extension, method = extract_raw_filing_text(
        filing,
        preferred_format="markdown",
    )

    assert raw_text == " ".join(LONG_TEXT.split())
    assert extension == ".txt"
    assert method == "filing.text() raw-cache fallback"


def test_save_raw_filing_text_creates_validated_artifact(tmp_path):
    """A raw filing cache should be saved with validated metadata."""

    filing = make_10q_filing()
    record = filing_to_record("NVDA", filing)

    artifact = save_raw_filing_text(
        record=record,
        filing=filing,
        output_dir=tmp_path,
        preferred_format="markdown",
        created_at_utc=datetime(2026, 6, 17, 12, 0),
    )

    assert Path(artifact.path).exists()
    assert Path(artifact.path).suffix == ".md"
    assert artifact.extraction_method == "filing.markdown() raw cache"


def test_build_selected_input_can_recover_filing_by_accession(tmp_path):
    """When no object is supplied, the function should recover it by accession."""

    filing = make_10q_filing()
    record = filing_to_record("NVDA", filing)
    factory = FakeCompanyFactory({"NVDA": [filing]})

    selected_input = build_selected_input(
        record=record,
        output_dir=tmp_path,
        company_factory=factory,
    )

    assert selected_input.filing.accession_number == filing.accession_no
    assert factory.created_companies[0].requested_forms == ["10-Q"]
