import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from core.config import ModelSettings
from core.contracts import (
    AdvisorRequest,
    Area,
    Observation,
    PortalScope,
    PortalSnapshot,
)
from core.knowledge import Fragment, KnowledgeBase, load_fragments
from core.llm import LLMGateway
from core.prompts import PromptRegistry
from core.storage import SnapshotStore
from modules.advisor.engine import AdvisorEngine
from modules.advisor.service import Advisor

NOW = datetime(2026, 11, 25, 12, tzinfo=UTC)
SCOPE = PortalScope(tenant_id="tenant", portal_id="portal")
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "knowledge.json"


def fixture() -> Fragment:
    return load_fragments(FIXTURE.read_text(encoding="utf-8-sig"))[0]


def search(kb: KnowledgeBase, scope: PortalScope = SCOPE) -> list[Fragment]:
    return kb.search(scope, "поля CRM", ["crm"], now=NOW)[0]


def test_knowledge_scoping_and_freshness(tmp_path: Path) -> None:
    kb = KnowledgeBase(SnapshotStore(tmp_path / "test.db"), LLMGateway(ModelSettings()))
    source = fixture()
    at_limit = source.model_copy(update={"checked_at": NOW - timedelta(days=30)})
    expired = source.model_copy(update={
        "fragment_id": "expired", "checked_at": NOW - timedelta(days=30, seconds=1),
    })
    future = source.model_copy(update={"fragment_id": "future", "checked_at": NOW + timedelta(1)})
    kb.replace(SCOPE, [at_limit, expired, future], dry_run=True)
    assert search(kb) == [at_limit]
    for other in (PortalScope(tenant_id="other", portal_id=SCOPE.portal_id),
                  PortalScope(tenant_id=SCOPE.tenant_id, portal_id="other")):
        assert search(kb, other) == []
        kb.replace(other, [], dry_run=True)
        assert search(kb) == [at_limit]


@pytest.mark.parametrize("url", [
    "http://helpdesk.bitrix24.ru/open/24633726/",
    "https://helpdesk.bitrix24.ru.evil.example/open/24633726/",
    "https://helpdesk.bitrix24.ru/open/24633726/?token=secret",
])
def test_corpus_rejects_noncanonical_sources(url: str) -> None:
    with pytest.raises(ValidationError):
        Fragment.model_validate({**fixture().model_dump(), "url": url})


def test_semantic_search_and_safe_model_change(tmp_path: Path) -> None:
    inputs = []

    def transport(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert endpoint == "embeddings"
        inputs.append(payload["input"])
        assert payload["model"] == "vector-v1"
        return {"data": [{"embedding": [0.2, 0.8]}]}

    settings = ModelSettings.model_validate({
        "base_url": "http://127.0.0.1/v1", "embedding_model": "vector-v1",
    })
    store = SnapshotStore(tmp_path / "test.db")
    gateway = LLMGateway(settings, transport=transport)
    kb = KnowledgeBase(store, gateway)
    source = fixture()
    kb.replace(SCOPE, [source])
    assert kb.search(SCOPE, "запрос", ["crm"], now=NOW)[0] == [source]
    changed_gateway = LLMGateway(
        settings.model_copy(update={"embedding_model": "vector-v2"}), transport=transport,
    )
    changed = KnowledgeBase(store, changed_gateway)
    assert changed.search(SCOPE, "запрос", ["crm"], now=NOW)[0] == []
    assert search(changed) == [source]
    assert len(inputs) == 2


@pytest.mark.parametrize("case", [
    "native", "unknown_doc", "unknown_observation", "no_snapshot", "tenant", "portal",
    "wrong_snapshot_id", "incomplete", "expired", "conflict", "fake_gap", "inline_url", "dry_run",
    "known_gap", "known_tariff", "unknown_tariff", "unsafe_question",
    "unrelated_contact_error", "empty_observations", "extension_inline_url",
    "extension_empty_observations", "conditional_extension", "conditional_inline_url",
    "conditional_unknown_doc", "conditional_unknown_observation", "conditional_fake_gap",
    "conditional_missing_sources", "unknown_area", "mixed_visibility", "mixed_empty_observations",
])
def test_advisor_binds_only_valid_evidence(tmp_path: Path, case: str) -> None:
    calls = []
    conditional = case in {"incomplete", "no_snapshot", "tenant", "portal", "wrong_snapshot_id"} or case.startswith(
        ("conditional_", "mixed_"),
    )
    source = fixture()
    question = "Как настроить поля CRM?"
    if case == "unrelated_contact_error":
        question = "Как настроить поля CRM для сделок?"
    if case == "unknown_area":
        question = "Что сделать?"
    if case.startswith("mixed_"):
        question += " Как настроить роботы?"
        source = source.model_copy(update={"areas": ["crm", "robots"]})
    request = AdvisorRequest(scope=SCOPE, request_id=uuid4(), question=question)
    observation = Observation(
        area="crm", status="unsupported" if conditional and not case.startswith("mixed_") else "observed",
        summary="PRIVATE_CARD_VALUE", source_ref="PRIVATE_SOURCE",
    )
    original_snapshot = PortalSnapshot(
        scope=SCOPE, request_id=uuid4(), snapshot_id=uuid4(),
        captured_at=NOW, observations=[observation],
    )
    if case == "unrelated_contact_error":
        observation.source_ref = "crm.item.fields/2"
        original_snapshot.observations.extend([
            Observation(area="crm", status="error", summary="PRIVATE_FAILURE", source_ref="crm.category.list/3"),
            Observation(area="crm", status="observed", summary="PRIVATE_FIELD", source_ref="crm.item.fields/3"),
        ])
    if case.startswith("mixed_"):
        original_snapshot.observations.append(Observation(
            area="robots", status="no_access", summary="PRIVATE_ROBOTS", source_ref="synthetic",
        ))
    snapshot: PortalSnapshot | None = original_snapshot
    if case == "tenant":
        original_snapshot.scope = PortalScope(tenant_id="other", portal_id=SCOPE.portal_id)
    elif case == "portal":
        original_snapshot.scope = PortalScope(tenant_id=SCOPE.tenant_id, portal_id="other")
    elif case == "wrong_snapshot_id":
        request.snapshot_id = uuid4()
    elif case == "no_snapshot":
        snapshot = None

    def transport(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert endpoint == "chat/completions"
        model = payload["model"]
        calls.append(model)
        result: dict[str, object]
        assert "PRIVATE_" not in json.dumps(payload)
        if model == "fast":
            result = {
                "task": "Настройка полей", "success_criterion": None,
                "areas": [] if case == "unknown_area" else ["crm"], "clarification_questions": [],
            }
        else:
            assert model == "smart"
            assert payload["messages"][0]["content"].startswith(
                PromptRegistry().render("advisor.recommend", "1.2.0") + "\nJSON Schema:\n",
            )
            context = json.loads(payload["messages"][-1]["content"])
            assert context["conditional"] == conditional
            if conditional:
                if case.startswith("mixed_"):
                    assert "Проверьте роботов и триггеры" in context["manual_checks"][0]
                    assert len(context["observations"]) == 1
                else:
                    assert "Проверьте в CRM" in context["manual_checks"][0]
                    assert context["observations"] == []
            if case == "unrelated_contact_error":
                assert len(context["observations"]) == 1
            fake_gap = case in {"fake_gap", "conditional_fake_gap"}
            observation_ids = [row["observation_id"] for row in context["observations"]]
            if case in {"unknown_observation", "conditional_unknown_observation"}:
                observation_ids = [1]
            elif case in {"empty_observations", "extension_empty_observations", "mixed_empty_observations"}:
                observation_ids = []
            result = {
                "tier": "extension" if source.native_gap or fake_gap else "native",
                "native_gap": "Требование вне штатного варианта" if fake_gap else source.native_gap,
                "recommendation": (
                    f"Шаги CRM: {source.url}" if case in {"inline_url", "extension_inline_url", "conditional_inline_url"}
                    else "Шаги CRM"
                ),
                "fragment_ids": [
                    "invented" if case in {"unknown_doc", "conditional_unknown_doc"} else source.fragment_id,
                ],
                "observation_ids": observation_ids,
                "review_reasons": ["Сверьте обязательность нужного поля."] if conditional else [],
            }
        return {"choices": [{"message": {"content": json.dumps(result)}}]}

    gateway = LLMGateway(ModelSettings.model_validate({
        "base_url": "http://127.0.0.1/v1", "fast_model": "fast", "smart_model": "smart",
    }), transport=transport)
    kb = KnowledgeBase(SnapshotStore(tmp_path / "test.db"), gateway)
    fragments = [source]
    if case in {"expired", "conditional_missing_sources"}:
        fragments = [source.model_copy(update={"checked_at": NOW - timedelta(days=31)})]
    elif case == "conflict":
        fragments = [
            source.model_copy(update={"claim_key": "crm.fields", "claim_value": "yes"}),
            source.model_copy(update={
                "fragment_id": "opposite", "claim_key": "crm.fields", "claim_value": "no",
            }),
        ]
    elif case in {"known_gap", "extension_inline_url", "extension_empty_observations", "conditional_extension"}:
        source = source.model_copy(update={"native_gap": "Требование вне штатного варианта"})
        fragments = [source]
    elif case in {"known_tariff", "unknown_tariff"}:
        source = source.model_copy(update={"tariffs": ["test-plan"]})
        fragments = [source]
    kb.replace(SCOPE, fragments, dry_run=True)
    result = Advisor(gateway, kb, PromptRegistry()).run(
        request, snapshot, now=NOW, dry_run=case == "dry_run",
        public_question="token=PRIVATE_SECRET" if case == "unsafe_question" else request.question,
        tariff="test-plan" if case == "known_tariff" else None,
    )
    assert result.response.scope == SCOPE
    assert result.response.request_id == request.request_id
    assert result.prompt_versions == {"advisor.route": "1.0.0", "advisor.recommend": "1.2.0"}
    if case in {"native", "known_gap", "known_tariff", "unrelated_contact_error"}:
        assert result.response.tier == ("extension" if case == "known_gap" else "native")
        assert result.response.documentation == [source.evidence()]
        assert result.response.observations == [observation]
        assert result.review_codes == []
        assert calls == ["fast", "smart"]
    elif case in {"incomplete", "no_snapshot", "tenant", "portal", "wrong_snapshot_id",
                  "conditional_extension", "mixed_visibility"}:
        assert result.response.tier == ("extension" if case == "conditional_extension" else "native")
        assert result.response.documentation == [source.evidence()]
        assert result.response.observations == ([observation] if case == "mixed_visibility" else [])
        assert result.response.need_human
        assert "условны" in result.response.recommendation
        assert "Сверьте обязательность нужного поля." in result.response.review_reasons
        if case in {"incomplete", "conditional_extension", "mixed_visibility"}:
            assert result.response.snapshot_id == original_snapshot.snapshot_id
            assert result.review_codes == ["visibility_gap"]
        else:
            assert result.response.snapshot_id is None
            assert result.review_codes == ["missing_snapshot"]
        assert calls == ["fast", "smart"]
        assert result.recommend.mode == "llm"
    else:
        assert result.response.tier == "undetermined"
        assert result.response.need_human and result.response.review_reasons
        assert not result.response.documentation and not result.response.observations
    if case in {"inline_url", "empty_observations", "extension_inline_url", "extension_empty_observations"}:
        assert result.review_codes == ["unsupported_conclusion"]
        assert result.recommend.mode == "llm"
        assert calls == ["fast", "smart"]
    if case in {"expired", "conflict", "unknown_tariff", "conditional_missing_sources", "unknown_area"}:
        assert calls == ["fast"]
        assert result.recommend.mode == "deterministic"
    if case in {"conditional_inline_url", "conditional_fake_gap", "mixed_empty_observations"}:
        assert result.review_codes == ["visibility_gap", "unsupported_conclusion"]
        assert calls == ["fast", "smart"]
    if case in {"conditional_unknown_doc", "conditional_unknown_observation"}:
        assert result.review_codes == ["visibility_gap", "invalid_evidence"]
        assert calls == ["fast", "smart"]
    if case in {"dry_run", "unsafe_question"}:
        assert calls == []
        assert result.route.cost_rub_e8 == result.recommend.cost_rub_e8 == 0





def test_recommend_prompt_versions() -> None:
    registry = PromptRegistry()
    original = registry.render("advisor.recommend", "1.0.0")
    active = registry.render("advisor.recommend")
    assert "Для выбранного решения укажи источник" in original
    assert "observation_ids" not in original
    assert active == registry.render("advisor.recommend", "1.2.0")
    previous = registry.render("advisor.recommend", "1.1.0")
    assert "заполни оба списка fragment_ids и observation_ids" in previous
    assert "Если observations пуст и conditional=true" in active
    assert "Не вставляй URL или ссылки" in active
    assert "Не додумывай отсутствующие настройки, права или наблюдения" in active
    assert "верни undetermined" in active


@pytest.mark.parametrize(("question", "entity", "area", "observed", "failure", "has_gap"), [
    ("Поля CRM сделок", None, "crm", "crm.item.fields/2", "crm.category.list/2", True),
    ("Поля CRM сделок", None, "crm", "crm.item.fields/2", "crm.status.list/DEAL_STAGE_4", True),
    ("Поля CRM сделок", None, "crm", "crm.item.fields/2", "crm.status.list/STATUS", False),
    ("Поля CRM сделок и контактов", None, "crm", "crm.item.fields/2", "crm.category.list/3", True),
    ("Поля CRM сделок и контактов", None, "crm", "crm.item.fields/2", None, True),
    ("Поля CRM сделок", None, "crm", "crm.item.fields/3", None, True),
    ("Поля CRM сделок", None, "crm", "synthetic", "crm.category.list/3", True),
    ("Поля CRM сделок", None, "crm", "crm.item.fields/2", "unknown", True),
    ("Поля CRM", None, "crm", "crm.item.fields/2", "crm.category.list/3", True),
    ("Поля CRM", "deal", "crm", "crm.item.fields/2", "crm.category.list/3", False),
    ("Поля CRM сделок", "contact", "crm", "crm.item.fields/2", "crm.category.list/3", True),
    ("Поля CRM сделок", "unknown", "crm", "crm.item.fields/2", "crm.category.list/3", True),
    ("Поля CRM сделок", None, "crm", None, None, True),
    ("Поля CRM и роботы сделок", None, "robots", None, "bizproc.robot.list", True),
    ("Поля CRM и бизнес-процессы сделок", None, "workflows", None, None, True),
    ("Поля CRM и права сделок", None, "permissions", "method.get/crm.item.fields", "method.get", True),
    ("Поля CRM и аналитика сделок", None, "analytics", None, "portal_audit/analytics", True),
    ("Поля CRM и смарт-процессы сделок", None, "smart_processes", "crm.type.list", "crm.item.fields/1048", True),
])
def test_visibility_gap_and_history(
    tmp_path: Path, question: str, entity: str | None, area: Area,
    observed: str | None, failure: str | None, has_gap: bool,
) -> None:
    store = SnapshotStore(tmp_path / "test.db")
    gateway = LLMGateway(ModelSettings())
    kb = KnowledgeBase(store, gateway)
    areas = ["crm"] if area == "crm" else ["crm", area]
    kb.replace(SCOPE, [fixture().model_copy(update={"areas": areas})], dry_run=True)
    observations = []
    if area != "crm":
        observations.append(Observation(
            area="crm", status="observed", summary="visible", source_ref="crm.item.fields/2",
        ))
    if observed is not None:
        observations.append(Observation(area=area, status="observed", summary="visible", source_ref=observed))
    if failure is not None:
        observations.append(Observation(
            area=area, status="error" if area == "crm" else "unsupported", summary="gap", source_ref=failure,
        ))
    if not observations:
        observations.append(Observation(
            area="robots", status="observed", summary="visible", source_ref="synthetic",
        ))
    snapshot = PortalSnapshot(
        scope=SCOPE, request_id=uuid4(), snapshot_id=uuid4(), captured_at=NOW, observations=observations,
    )
    store.save(snapshot)
    engine = AdvisorEngine(Advisor(gateway, kb, PromptRegistry()))
    request = AdvisorRequest(scope=SCOPE, request_id=uuid4(), question=question, snapshot_id=snapshot.snapshot_id)
    run = engine.run(request, dry_run=True, now=NOW, entity=entity)
    assert run.response.tier == "undetermined" and run.response.need_human
    assert run.review_codes == (["visibility_gap"] if has_gap else ["model_review"])
    record = engine.history.get(SCOPE, request.request_id)
    assert record is not None
    assert record.review_codes == run.review_codes
    assert record.prompt_versions == {"advisor.route": "1.0.0", "advisor.recommend": "1.2.0"}
    assert record.telemetry.recommend == run.recommend
    assert record.telemetry.recommend.mode == "deterministic"


def test_official_seed_has_valid_hashes_and_covers_all_areas() -> None:
    path = FIXTURE.parents[2] / "docs" / "knowledge" / "bitrix24.json"
    fragments = load_fragments(path.read_text(encoding="utf-8"))
    assert {item.url.host for item in fragments} == {"helpdesk.bitrix24.ru", "apidocs.bitrix24.ru"}
    assert {area for item in fragments for area in item.areas} == {
        "crm", "robots", "workflows", "smart_processes", "permissions", "analytics",
    }
    for fragment in fragments:
        digest = hashlib.sha256(fragment.snippet.encode("utf-8")).hexdigest()
        assert fragment.revision_or_hash == f"sha256:{digest}"
