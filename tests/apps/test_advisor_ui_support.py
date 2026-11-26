import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pydantic import SecretStr

from apps.advisor_ui_support import (
    DEMO_CASES,
    build_engine,
    case_report,
    discover_snapshots,
    format_cost,
    import_corpus,
    load_case_packs,
    model_status,
    run_case_pack,
    run_consultation,
    selected_scope,
    verdict,
)
from core.advisor_history import RunOutcome, RunSummary
from core.config import ModelSettings, Settings
from core.contracts import (
    AdvisorRequest,
    Observation,
    PortalScope,
    PortalSnapshot,
    SnapshotMetadata,
    snapshot_hash,
)
from modules.advisor.service import local_route

CORPUS = Path(__file__).resolve().parents[2] / "docs/knowledge/bitrix24.json"
CASES = CORPUS.parents[1] / "cases"
CASE_IDS = (
    "01_lead_leak_and_dormant_base",
    "02_contract_approval_bottleneck",
    "03_production_and_claims_spa",
    "04_permissions_and_crm_analytics",
)


def make_snapshot(tenant: str) -> PortalSnapshot:
    observations = [Observation(
        area="crm", status="observed", summary="Воронка 0", source_ref="fixture",
    )]
    metadata = SnapshotMetadata()
    return PortalSnapshot(
        scope=PortalScope(tenant_id=tenant, portal_id="fixture.invalid"),
        request_id=uuid4(), snapshot_id=uuid4(), captured_at=datetime.now(UTC),
        metadata=metadata, observations=observations, sha256=snapshot_hash(metadata, observations),
    )


def test_demo_cases_are_distinct_valid_questions() -> None:
    assert len({label for label, _ in DEMO_CASES}) == 4
    assert len({question for _, question in DEMO_CASES}) == 4
    scope = selected_scope("local", None)
    for (_, question), term in zip(DEMO_CASES, ("времени", "юристу", "рекламац", "аналитик"), strict=True):
        assert term in question
        assert AdvisorRequest(scope=scope, request_id=uuid4(), question=question).question == question


def test_discover_snapshots_filters_invalid_exports_and_other_tenants(tmp_path: Path) -> None:
    snapshot = make_snapshot("local")
    (tmp_path / "snapshot.json").write_text(snapshot.model_dump_json(), encoding="utf-8")
    (tmp_path / "other.json").write_text(make_snapshot("other").model_dump_json(), encoding="utf-8")
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    tampered = snapshot.model_dump(mode="json")
    tampered["sha256"] = "0" * 64
    (tmp_path / "tampered.json").write_text(json.dumps(tampered), encoding="utf-8")
    assert discover_snapshots(tmp_path, "local") == {"snapshot.json": snapshot}
    with pytest.raises(ValueError, match="tenant mismatch"):
        selected_scope("other", snapshot)


def test_dry_runs_bind_only_selected_snapshot_and_record_scoped_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_transport(*args: object, **kwargs: object) -> None:
        pytest.fail("dry run called the model transport")

    settings = Settings(models=ModelSettings.model_validate({
        "base_url": "https://llm.example", "fast_model": "fast",
        "smart_model": "smart", "embedding_model": "embed",
    }))
    engine = build_engine(tmp_path / "advisor_ui.sqlite3", settings)
    monkeypatch.setattr(engine.advisor.gateway, "transport", no_transport)
    snapshot = make_snapshot(settings.tenant_id)
    detached_scope = selected_scope(settings.tenant_id, None)
    for scope in (detached_scope, snapshot.scope):
        assert import_corpus(engine.advisor.knowledge, scope, CORPUS) > 0

    question = DEMO_CASES[0][1]
    detached, detached_record = run_consultation(
        engine, settings.tenant_id, question, None,
        dry_run=True, approved_for_provider=True,
    )
    first, first_record = run_consultation(
        engine, settings.tenant_id, question, snapshot, dry_run=True,
    )
    second, second_record = run_consultation(
        engine, settings.tenant_id, question, snapshot, dry_run=True,
    )
    assert detached.response.snapshot_id is None
    assert "missing_snapshot" in detached.review_codes
    assert first.response.snapshot_id == second.response.snapshot_id == snapshot.snapshot_id
    assert "missing_snapshot" not in first.review_codes
    assert first_record.request_id != second_record.request_id
    for record in (detached_record, first_record, second_record):
        assert record.telemetry.mode == "deterministic"
        assert record.telemetry.cost_rub_e8 == 0
    assert engine.history.list_completed(detached_scope) == [detached_record]
    assert engine.history.list_completed(snapshot.scope) == [second_record, first_record]
    summary = RunSummary.from_records(engine.history.list_completed(snapshot.scope))
    assert (summary.total, summary.native, summary.known_cost_rub_e8, summary.unknown_cost_count) == (
        2, 0, 0, 0,
    )


@pytest.mark.parametrize(
    ("value", "expected"), [(None, "Неизвестно"), (0, "0 ₽"), (1, "0,00000001 ₽"), (125_000_000, "1,25 ₽")],
)
def test_format_cost(value: int | None, expected: str) -> None:
    assert format_cost(value) == expected


def test_verdict_and_model_status() -> None:
    record = RunOutcome(
        scope=selected_scope("local", None), request_id=uuid4(), snapshot_id=None,
        tier="native", review_codes=[],
    )
    assert verdict(record) == "Штатное решение"
    assert verdict(record.model_copy(update={"review_codes": ["missing_snapshot"]})) == "Условный штатный план"
    assert verdict(record.model_copy(update={"tier": "extension"})) == "Требуется расширение/API"
    assert verdict(record.model_copy(update={"tier": "undetermined"})) == "Нужна ручная проверка"
    assert model_status(ModelSettings(fast_model="fast", api_key=SecretStr("example-secret"))) == {
        "FAST_LLM_MODEL": "fast", "SMART_LLM_MODEL": "Не задана", "Ключ API": "Задан",
    }
    assert model_status(ModelSettings())["Ключ API"] == "Не задан"


@pytest.mark.parametrize(
    ("case_index", "step_ids"),
    [
        (0, ["routing", "response", "reactivation"]),
        (1, ["document", "approval", "payment"]),
        (2, ["separation", "handoff", "monitoring"]),
        (3, ["isolation", "role_access_and_export", "management_reporting"]),
    ],
)
def test_case_pack_loads_and_validates(tmp_path: Path, case_index: int, step_ids: list[str]) -> None:
    cases, fragments = load_case_packs(CASES, CORPUS)
    assert tuple(case.case_id for case in cases) == CASE_IDS
    case = cases[case_index]
    assert [step.step_id for step in case.steps] == step_ids
    for step in case.steps:
        assert step.areas == set(local_route(step.question).areas)
        assert all(ref in fragments for ref in step.fragments)
    data = case.model_dump(mode="json")
    path = tmp_path / f"{case.case_id}.json"
    data["steps"][0]["extra"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="extra_forbidden"):
        load_case_packs(tmp_path, CORPUS)
    del data["steps"][0]["extra"]
    data["steps"][0]["fragments"] = ["unknown"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid fragment references"):
        load_case_packs(tmp_path, CORPUS)


@pytest.mark.parametrize("with_snapshot", [True, False])
@pytest.mark.parametrize("case_index", range(len(CASE_IDS)), ids=CASE_IDS)
def test_dry_case_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_snapshot: bool, case_index: int,
) -> None:
    def no_transport(*args: object, **kwargs: object) -> None:
        pytest.fail("dry case called the model transport")

    settings = Settings()
    engine = build_engine(tmp_path / "case.sqlite3", settings)
    monkeypatch.setattr(engine.advisor.gateway, "transport", no_transport)
    snapshot = make_snapshot(settings.tenant_id) if with_snapshot else None
    scope = selected_scope(settings.tenant_id, snapshot)
    cases, fragments = load_case_packs(CASES, CORPUS)
    case = cases[case_index]
    import_corpus(engine.advisor.knowledge, scope, CORPUS)
    with pytest.raises(ValueError, match="spending"):
        run_case_pack(engine, settings.tenant_id, case, snapshot, dry_run=False)
    assert engine.history.list_completed(scope) == []
    results = run_case_pack(engine, settings.tenant_id, case, snapshot, dry_run=True)
    assert len(results) == 3
    assert {result.record.request_id for result in results} == {
        record.request_id for record in engine.history.list_completed(scope)
    }
    for result in results:
        assert result.record.scope == scope
        assert result.run.response.snapshot_id == (snapshot.snapshot_id if snapshot else None)
        assert ("missing_snapshot" in result.record.review_codes) is not with_snapshot
        assert "missing_sources" not in result.record.review_codes
        assert result.record.telemetry.mode == "deterministic"
        assert result.record.telemetry.cost_rub_e8 == 0
    report = case_report(case, results, fragments)
    assert report.startswith(f"# {case.title}\n")
    assert f"Слепок расчёта: {snapshot.snapshot_id if snapshot else 'отсутствует'}" in report
    for step, result in zip(case.steps, results, strict=True):
        assert step.title in report
        assert step.checks[0] in report
        assert str(result.record.request_id) in report
        for ref in step.fragments:
            assert str(fragments[ref].url) in report
    assert "Суммарный расход — 0 ₽" in report
    assert "Штатная архитектура Битрикс24" in report


def test_case_pack_ui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from apps import advisor_ui

    monkeypatch.setattr(advisor_ui, "LOCAL", tmp_path)
    settings = Settings(models=ModelSettings.model_validate({
        "base_url": "https://llm.example", "fast_model": "fast", "smart_model": "smart",
    }))
    monkeypatch.setattr(advisor_ui, "load_settings", lambda _: settings)
    monkeypatch.delenv("B24_UI_PRICES", raising=False)
    app = AppTest.from_string("from apps.advisor_ui import main\nmain()", default_timeout=15).run()
    assert not app.exception
    assert any(tab.label == "Прикладные кейс-паки" for tab in app.tabs)
    case_picker = next(widget for widget in app.selectbox if widget.label == "Кейс-пак")
    assert case_picker.options == [
        "Кейс 1 · Утечка входящих лидов и мёртвая база",
        "Кейс 2 · Договорной паралич и бесконечные согласования",
        "Кейс 3 · Производство, закупки и рекламации внутри одной воронки продаж",
        "Кейс 4 · Бардак в правах, утечка базы и непрозрачная аналитика",
    ]
    assert case_picker.value == case_picker.options[0]
    next(button for button in app.button if button.label == "Загрузить / обновить корпус").click().run()
    next(button for button in app.button if button.label == "Запустить разбор кейса").click().run()
    assert not app.exception
    assert app.get("download_button")[0].proto.label == "Скачать Markdown-отчёт"
    assert any("Суммарный расход — 0 ₽" in element.value for element in app.markdown)
    app.radio[0].set_value(advisor_ui.LIVE).run()
    assert not app.get("download_button")
    assert next(button for button in app.button if button.label == "Запустить разбор кейса").disabled
    app.checkbox(key="case_provider_approved").check().run()
    assert not next(button for button in app.button if button.label == "Запустить разбор кейса").disabled
    app.radio[0].set_value(advisor_ui.DRY).run()
    app.radio[0].set_value(advisor_ui.LIVE).run()
    assert not app.checkbox(key="case_provider_approved").value


@pytest.mark.parametrize("with_snapshot", [True, False])
@pytest.mark.parametrize("case_index", range(1, len(CASE_IDS)), ids=CASE_IDS[1:])
def test_select_case_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_snapshot: bool, case_index: int,
) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from apps import advisor_ui

    settings = Settings()
    monkeypatch.setattr(advisor_ui, "LOCAL", tmp_path)
    monkeypatch.setattr(advisor_ui, "load_settings", lambda _: settings)
    monkeypatch.delenv("B24_UI_PRICES", raising=False)
    snapshot = make_snapshot(settings.tenant_id) if with_snapshot else None
    if snapshot:
        (tmp_path / "snapshot.json").write_text(snapshot.model_dump_json(), encoding="utf-8")

    app = AppTest.from_string("from apps.advisor_ui import main\nmain()", default_timeout=15).run()
    if snapshot:
        next(widget for widget in app.selectbox if widget.label == "Слепок портала").select("snapshot.json").run()
    next(button for button in app.button if button.label == "Загрузить / обновить корпус").click().run()
    next(button for button in app.button if button.label == "Запустить разбор кейса").click().run()
    assert len(app.get("download_button")) == 1

    case_picker = next(widget for widget in app.selectbox if widget.label == "Кейс-пак")
    case_picker.select(case_picker.options[case_index]).run()
    assert not app.get("download_button")
    next(button for button in app.button if button.label == "Запустить разбор кейса").click().run()
    assert not app.exception
    (download,) = app.get("download_button")
    assert download.proto.label == "Скачать Markdown-отчёт"
    report = app.session_state["case_result"]
    cases, _ = load_case_packs(CASES, CORPUS)
    case = cases[case_index]
    assert report.startswith(f"# {case.title}\n")
    assert f"Слепок расчёта: {snapshot.snapshot_id if snapshot else 'отсутствует'}" in report
    assert all(f"### {index}. {step.title}" in report for index, step in enumerate(case.steps, 1))
    assert "Суммарный расход — 0 ₽" in report
    assert any(element.value == report.strip() for element in app.markdown)


def test_approved_case_uses_prepared_questions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(models=ModelSettings.model_validate({
        "base_url": "https://llm.example", "fast_model": "fast", "smart_model": "smart",
    }))
    engine = build_engine(tmp_path / "live_case.sqlite3", settings)
    scope = selected_scope(settings.tenant_id, None)
    import_corpus(engine.advisor.knowledge, scope, CORPUS)
    cases, fragments = load_case_packs(CASES, CORPUS)
    case = cases[0]
    questions = []

    def transport(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert endpoint == "chat/completions"
        request = json.loads(payload["messages"][1]["content"])
        if payload["model"] == "fast":
            question = request["question"]
            questions.append(question)
            content = local_route(question).model_dump_json()
        else:
            assert request["conditional"] and not request["observations"]
            content = json.dumps({
                "tier": "undetermined", "recommendation": "Требуется проверка.",
                "native_gap": None, "fragment_ids": [], "observation_ids": [], "review_reasons": [],
            })
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    monkeypatch.setattr(engine.advisor.gateway, "transport", transport)
    results = run_case_pack(
        engine, settings.tenant_id, case, None, dry_run=False, approved_for_provider=True,
    )
    assert questions == [step.question for step in case.steps]
    assert all(result.record.telemetry.mode == "llm" for result in results)
    report = case_report(case, results, fragments)
    assert "Шагов без оценки — 3" in report
    assert "токены вход/выход 10/5" in report
