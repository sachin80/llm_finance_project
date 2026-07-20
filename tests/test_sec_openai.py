"""
Unit tests for src.sec_openai.

These tests:

    - do not call OpenAI,
    - do not require an API key,
    - do not contact the SEC,
    - use small fake response clients,
    - use temporary local text files.

Run from the project root with:

    python -m pytest -q tests/test_sec_openai.py
"""

from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.config import EMBED_MODEL
from src.sec_models import (
    FilingRecord,
    FilingThemeSummary,
    SectionExtractionDetail,
    SelectedFilingInput,
    TextArtifact,
    ThemeConfig,
    sha256_text,
)
from src.sec_openai import (
    DEFAULT_EMBEDDING_MODEL,
    build_section_audit_text,
    build_summary_record,
    build_user_prompt,
    create_openai_client,
    flatten_summary_record,
    get_or_create_summary_record,
    load_summary_cache,
    make_dry_run_summary,
    make_summary_cache_key,
    read_selected_text,
    save_summary_cache,
    save_summary_table_pickle,
    summarize_selected_inputs,
    summarize_with_openai,
    summary_records_to_dataframe,
)


def make_filing_record() -> FilingRecord:
    """Build one valid filing record."""

    return FilingRecord(
        ticker="NVDA",
        cik=1045810,
        company_name="NVIDIA Corporation",
        form="10-Q",
        filing_date=date(2024, 5, 29),
        period_of_report=date(2024, 4, 28),
        accession_number="0001045810-24-000124",
        primary_document="nvda-20240428.htm",
        acceptance_datetime=datetime(
            2024,
            5,
            29,
            20,
            35,
            52,
            tzinfo=timezone.utc,
        ),
    )


def make_text_artifact(
    path: Path,
    text: str,
    extraction_method: str,
) -> TextArtifact:
    """Write and validate one local text artifact."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )

    return TextArtifact(
        path=str(path),
        extraction_method=extraction_method,
        sha256=sha256_text(text),
        char_count=len(text),
        word_count=len(text.split()),
        created_at_utc=datetime(
            2026,
            6,
            18,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )


def make_selected_input(
    tmp_path: Path,
    *,
    suffix: str = "baseline",
    selected_text: str | None = None,
    include_raw: bool = True,
) -> SelectedFilingInput:
    """Build one current-format SelectedFilingInput."""

    if selected_text is None:
        selected_text = (
            "SECTION: management_discussion "
            "Management reports increasing data-center demand, "
            "accelerated-computing revenue, supply commitments, "
            "capital spending, and customer adoption. "
            "[SECTION MIDDLE TRUNCATED] "
            "SECTION: risk_factors "
            "The filing discusses competition, export controls, "
            "manufacturing capacity, and customer concentration."
        )

    selected_artifact = make_text_artifact(
        tmp_path / f"{suffix}_selected.txt",
        selected_text,
        "filing.obj() selected sections",
    )

    raw_artifact = None

    if include_raw:
        raw_text = (
            "Complete filing markdown retained for debugging. "
            "RAW-CACHE-SECRET-SHOULD-NOT-ENTER-PROMPT."
        )

        raw_artifact = make_text_artifact(
            tmp_path / f"{suffix}_raw.md",
            raw_text,
            "filing.markdown() raw cache",
        )

    details = [
        SectionExtractionDetail(
            section_name="management_discussion",
            requested=True,
            found=True,
            original_char_count=35_000,
            retained_char_count=28_000,
            char_limit=28_000,
            truncated=True,
        ),
        SectionExtractionDetail(
            section_name="risk_factors",
            requested=True,
            found=True,
            original_char_count=8_000,
            retained_char_count=8_000,
            char_limit=12_000,
            truncated=False,
        ),
    ]

    return SelectedFilingInput(
        filing=make_filing_record(),
        selected_sections=[
            "management_discussion",
            "risk_factors",
        ],
        section_extraction_details=details,
        selected_text=selected_artifact,
        raw_filing_text=raw_artifact,
    )


def make_theme(
    *,
    prompt_version: str = "sec_theme_summary_v1",
) -> ThemeConfig:
    """Build one stable research theme."""

    return ThemeConfig(
        theme_name="AI infrastructure",
        theme_description=(
            "Companies with direct or indirect exposure to accelerated "
            "computing, AI servers, data centers, networking, semiconductor "
            "equipment, cloud infrastructure, power, and cooling."
        ),
        prompt_version=prompt_version,
    )


def make_live_summary() -> FilingThemeSummary:
    """Build one valid fake live model response."""

    return FilingThemeSummary(
        theme_exposure_summary=(
            "The selected filing text provides direct evidence of exposure to "
            "AI infrastructure through accelerated-computing products, "
            "data-center demand, networking, and related supply commitments. "
            "Management describes increasing customer adoption and operating "
            "activity, while also identifying export controls, competition, "
            "manufacturing capacity, and customer concentration as important "
            "uncertainties that could affect the durability of the exposure."
        ),
        exposure_label="Direct",
        exposure_direction="Increasing",
        evidence_strength="High",
        key_evidence=[
            (
                "Management reports increasing data-center demand and "
                "accelerated-computing revenue."
            )
        ],
        risks_or_uncertainties=[
            (
                "Export controls and manufacturing constraints may limit "
                "future activity."
            )
        ],
    )


class FakeParseResponses:
    """Fake responses API exposing the preferred parse method."""

    def __init__(self, summary: FilingThemeSummary):
        self.summary = summary
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.summary)


class FakeCreateResponses:
    """Fake responses API exposing only the JSON fallback method."""

    def __init__(self, summary: FilingThemeSummary):
        self.summary = summary
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=self.summary.model_dump_json()
        )


class FlakyParseResponses:
    """Fail a fixed number of calls, then return a valid summary."""

    def __init__(
        self,
        summary: FilingThemeSummary,
        failures_before_success: int,
    ):
        self.summary = summary
        self.failures_before_success = failures_before_success
        self.call_count = 0

    def parse(self, **kwargs):
        self.call_count += 1

        if self.call_count <= self.failures_before_success:
            raise RuntimeError("temporary test failure")

        return SimpleNamespace(output_parsed=self.summary)


# -----------------------------------------------------------------------------
# Project configuration and client tests
# -----------------------------------------------------------------------------


def test_default_embedding_model_reuses_project_config():
    """The SEC layer should use the existing project embedding-model setting."""

    assert DEFAULT_EMBEDDING_MODEL == EMBED_MODEL


def test_create_openai_client_reads_environment_key(monkeypatch):
    """The client helper should read OPENAI_API_KEY from the environment."""

    observed_arguments = {}

    def fake_client_factory(**kwargs):
        observed_arguments.update(kwargs)
        return SimpleNamespace(kind="fake-openai-client")

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key-from-environment",
    )

    client = create_openai_client(
        client_factory=fake_client_factory,
    )

    assert client.kind == "fake-openai-client"
    assert observed_arguments == {
        "api_key": "test-key-from-environment"
    }


def test_create_openai_client_prefers_explicit_key(monkeypatch):
    """An explicitly supplied key should override the environment value."""

    observed_arguments = {}

    def fake_client_factory(**kwargs):
        observed_arguments.update(kwargs)
        return SimpleNamespace(kind="fake-openai-client")

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "environment-key",
    )

    create_openai_client(
        api_key="explicit-key",
        client_factory=fake_client_factory,
    )

    assert observed_arguments == {
        "api_key": "explicit-key"
    }


def test_create_openai_client_rejects_missing_key(monkeypatch):
    """A missing .env/environment key should fail with a clear explanation."""

    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="OPENAI_API_KEY not found",
    ):
        create_openai_client(
            client_factory=lambda **kwargs: None,
        )


# -----------------------------------------------------------------------------
# Selected-text and prompt tests
# -----------------------------------------------------------------------------


def test_read_selected_text_returns_exact_saved_text(tmp_path):
    selected_input = make_selected_input(tmp_path)

    text = read_selected_text(selected_input)

    assert text == Path(
        selected_input.selected_text.path
    ).read_text(encoding="utf-8")


def test_read_selected_text_supports_inspection_limit(tmp_path):
    selected_input = make_selected_input(tmp_path)

    shortened = read_selected_text(
        selected_input,
        max_chars=25,
    )

    assert len(shortened) == 25
    assert selected_input.selected_text.char_count > 25


def test_read_selected_text_detects_post_validation_change(tmp_path):
    selected_input = make_selected_input(tmp_path)

    Path(selected_input.selected_text.path).write_text(
        "changed after validation",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="character count no longer matches",
    ):
        read_selected_text(selected_input)


def test_build_section_audit_text_contains_budget_details(tmp_path):
    selected_input = make_selected_input(tmp_path)

    audit = build_section_audit_text(selected_input)

    assert "management_discussion" in audit
    assert "retained_chars=28000" in audit
    assert "truncated=True" in audit


def test_build_user_prompt_uses_selected_not_raw_text(tmp_path):
    selected_input = make_selected_input(tmp_path)

    prompt = build_user_prompt(
        selected_input,
        make_theme(),
    )

    assert "AI infrastructure" in prompt
    assert "accelerated-computing revenue" in prompt
    assert "RAW-CACHE-SECRET-SHOULD-NOT-ENTER-PROMPT" not in prompt
    assert "[SECTION MIDDLE TRUNCATED]" in prompt


def test_make_dry_run_summary_is_schema_valid(tmp_path):
    summary = make_dry_run_summary(
        make_selected_input(tmp_path)
    )

    assert summary.exposure_label == "Unclear"
    assert summary.exposure_direction == "Unclear"
    assert summary.evidence_strength == "Low"


def test_summarize_with_openai_uses_responses_parse(tmp_path):
    summary = make_live_summary()
    responses = FakeParseResponses(summary)
    client = SimpleNamespace(responses=responses)

    result = summarize_with_openai(
        client=client,
        selected_input=make_selected_input(tmp_path),
        theme=make_theme(),
        summary_model="test-summary-model",
        max_attempts=1,
    )

    assert result == summary
    assert len(responses.calls) == 1
    assert responses.calls[0]["text_format"] is FilingThemeSummary


def test_summarize_with_openai_supports_json_fallback(tmp_path):
    summary = make_live_summary()
    responses = FakeCreateResponses(summary)
    client = SimpleNamespace(responses=responses)

    result = summarize_with_openai(
        client=client,
        selected_input=make_selected_input(tmp_path),
        theme=make_theme(),
        summary_model="test-summary-model",
        max_attempts=1,
    )

    assert result == summary
    assert len(responses.calls) == 1
    assert responses.calls[0]["text"]["format"]["type"] == "json_schema"


def test_summarize_with_openai_retries_explicitly(tmp_path):
    summary = make_live_summary()
    responses = FlakyParseResponses(
        summary,
        failures_before_success=2,
    )
    client = SimpleNamespace(responses=responses)
    observed_waits = []

    result = summarize_with_openai(
        client=client,
        selected_input=make_selected_input(tmp_path),
        theme=make_theme(),
        summary_model="test-summary-model",
        max_attempts=3,
        retry_wait_seconds=0.5,
        sleep_func=observed_waits.append,
    )

    assert result == summary
    assert responses.call_count == 3
    assert observed_waits == [0.5, 1.0]


def test_summary_cache_key_is_stable(tmp_path):
    selected_input = make_selected_input(tmp_path)
    theme = make_theme()

    first = make_summary_cache_key(
        selected_input=selected_input,
        theme=theme,
        summary_model="model-a",
    )
    second = make_summary_cache_key(
        selected_input=selected_input,
        theme=theme,
        summary_model="model-a",
    )

    assert first == second
    assert len(first) == 64


def test_summary_cache_key_changes_with_prompt_version(tmp_path):
    selected_input = make_selected_input(tmp_path)

    first = make_summary_cache_key(
        selected_input=selected_input,
        theme=make_theme(prompt_version="v1"),
        summary_model="model-a",
    )
    second = make_summary_cache_key(
        selected_input=selected_input,
        theme=make_theme(prompt_version="v2"),
        summary_model="model-a",
    )

    assert first != second


def test_summary_cache_round_trip_revalidates_records(tmp_path):
    selected_input = make_selected_input(tmp_path)
    theme = make_theme()
    summary = make_live_summary()

    record = build_summary_record(
        selected_input=selected_input,
        theme=theme,
        summary=summary,
        summary_model="summary-model",
        embedding_model="embedding-model",
        created_at_utc=datetime(
            2026,
            6,
            18,
            13,
            0,
            tzinfo=timezone.utc,
        ),
    )

    key = make_summary_cache_key(
        selected_input=selected_input,
        theme=theme,
        summary_model="summary-model",
    )

    path = tmp_path / "summary_cache.pkl"

    save_summary_cache({key: record}, path)

    loaded = load_summary_cache(path)

    assert loaded[key] == record


def test_get_or_create_summary_record_uses_cache(tmp_path):
    selected_input = make_selected_input(tmp_path)
    theme = make_theme()
    responses = FakeParseResponses(make_live_summary())
    client = SimpleNamespace(responses=responses)
    cache_path = tmp_path / "cache.pkl"

    first = get_or_create_summary_record(
        selected_input=selected_input,
        theme=theme,
        summary_model="summary-model",
        embedding_model="embedding-model",
        cache_path=cache_path,
        client=client,
        dry_run=False,
        max_attempts=1,
    )

    second = get_or_create_summary_record(
        selected_input=selected_input,
        theme=theme,
        summary_model="summary-model",
        embedding_model="embedding-model",
        cache_path=cache_path,
        client=client,
        dry_run=False,
        max_attempts=1,
    )

    assert first == second
    assert len(responses.calls) == 1


def test_summarize_selected_inputs_dry_run_and_progress(tmp_path):
    inputs = [
        make_selected_input(tmp_path, suffix="first"),
        make_selected_input(
            tmp_path,
            suffix="second",
            selected_text=(
                "SECTION: management_discussion "
                "Second distinct filing text with a different hash and "
                "different operating discussion for cache testing."
            ),
        ),
    ]

    observed_progress = []

    records = summarize_selected_inputs(
        inputs,
        theme=make_theme(),
        summary_model="summary-model",
        embedding_model="embedding-model",
        cache_path=tmp_path / "batch_cache.pkl",
        dry_run=True,
        progress_callback=(
            lambda index, total, selected_input:
            observed_progress.append(
                (index, total, selected_input.filing.ticker)
            )
        ),
    )

    assert len(records) == 2
    assert observed_progress == [
        (1, 2, "NVDA"),
        (2, 2, "NVDA"),
    ]

    loaded_cache = load_summary_cache(
        tmp_path / "batch_cache.pkl"
    )

    assert len(loaded_cache) == 2


def test_flatten_and_dataframe_preserve_research_fields(tmp_path):
    record = build_summary_record(
        selected_input=make_selected_input(tmp_path),
        theme=make_theme(),
        summary=make_live_summary(),
        summary_model="summary-model",
        embedding_model="embedding-model",
        created_at_utc=datetime(
            2026,
            6,
            18,
            13,
            0,
            tzinfo=timezone.utc,
        ),
    )

    row = flatten_summary_record(record)

    assert row["ticker"] == "NVDA"
    assert row["raw_filing_cached"] is True
    assert row["exposure_label"] == "Direct"

    dataframe = summary_records_to_dataframe([record])

    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.loc[0, "exposure_direction"] == "Increasing"


def test_save_summary_table_pickle(tmp_path):
    record = build_summary_record(
        selected_input=make_selected_input(tmp_path),
        theme=make_theme(),
        summary=make_live_summary(),
        summary_model="summary-model",
        embedding_model="embedding-model",
    )

    path = save_summary_table_pickle(
        [record],
        tmp_path / "summary_table.pkl",
    )

    loaded = pd.read_pickle(path)

    assert loaded.loc[0, "ticker"] == "NVDA"
    assert loaded.loc[0, "exposure_label"] == "Direct"
