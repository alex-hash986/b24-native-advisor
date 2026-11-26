from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from core.config import ModelSettings
from core.contracts import Area, PortalScope
from core.knowledge import Fragment, KnowledgeBase, load_fragments
from core.llm import LLMGateway
from core.storage import SnapshotStore

CORPUS = Path(__file__).resolve().parents[2] / "docs/knowledge/bitrix24.json"
REVIEW_DATE = datetime(2026, 11, 26, 23, 59, 59, tzinfo=UTC)
LEGACY_IDS = {
    "robots", "workflows", "smart-processes", "crm-roles", "crm-analytics",
    "rest-fields", "rest-limits", "crm-form-routing", "crm-stage-task",
    "crm-stage-task-conditions", "crm-stage-task-contract-example",
    "crm-contract-workflow-task", "smart-claims-stages-and-access",
    "smart-claims-deal-link", "smart-claims-workflow-options",
}


@pytest.fixture(scope="module")
def corpus() -> list[Fragment]:
    return load_fragments(CORPUS.read_text(encoding="utf-8"))


def test_editorial_integrity(corpus: list[Fragment]) -> None:
    assert len(corpus) >= 50
    ids = {fragment.fragment_id for fragment in corpus}
    assert len(ids) == len(corpus)
    assert LEGACY_IDS <= ids
    assert {area for fragment in corpus for area in fragment.areas} == {
        "crm", "robots", "workflows", "smart_processes", "permissions", "analytics",
    }

    claims: dict[str, str | None] = {}
    for fragment in corpus:
        assert REVIEW_DATE - timedelta(days=30) <= fragment.checked_at <= REVIEW_DATE
        assert fragment.url.scheme == "https"
        assert fragment.url.host in {"helpdesk.bitrix24.ru", "apidocs.bitrix24.ru"}
        assert fragment.revision_or_hash == "sha256:" + sha256(fragment.snippet.encode("utf-8")).hexdigest()
        if fragment.claim_key is not None:
            assert claims.setdefault(fragment.claim_key, fragment.claim_value) == fragment.claim_value

    assert {fragment.fragment_id for fragment in corpus if fragment.native_gap} == {
        "workflow-external-scoring-gap", "rest-cross-system-rollback-gap",
    }


@pytest.fixture(scope="module")
def indexed_corpus(
    corpus: list[Fragment], tmp_path_factory: pytest.TempPathFactory,
) -> tuple[KnowledgeBase, PortalScope]:
    store = SnapshotStore(tmp_path_factory.mktemp("knowledge") / "index.sqlite3")
    knowledge = KnowledgeBase(store, LLMGateway(ModelSettings()))
    scope = PortalScope(tenant_id="test", portal_id="knowledge.invalid")
    executions = knowledge.replace(scope, corpus, dry_run=True)
    assert len(executions) == len(corpus)
    assert all(item.mode == "deterministic" and item.cost_rub_e8 == 0 for item in executions)
    return knowledge, scope


@pytest.mark.parametrize(
    ("area", "question", "expected_id"),
    [
        ("crm", "Как туннели продаж переносят сделки между воронками?", "deal-sales-tunnels"),
        ("crm", "Как задать модификаторы и символьные коды в шаблоне договора?", "crm-document-template-modifiers"),
        ("crm", "Как подставить реквизиты компании и контакта в договор?", "crm-client-requisites"),
        ("crm", "Как отобрать сделки по наличию связей в фильтре CRM?", "crm-related-elements-filter"),
        ("smart_processes", "Как копировать записи таймлайна, звонки и письма между сделкой и смарт-процессом?",
         "spa-timeline-copy-marketplace"),
        ("crm", "Как равномерно распределить чаты в очереди операторов Открытых линий?", "open-lines-queue"),
        ("robots", "Как настроить перевод сделки после оплаты связанного счёта?", "trigger-invoice-paid"),
        ("workflows", "Как организовать параллельное согласование договора?", "workflow-parallel-approval"),
        ("smart_processes", "Подходит ли смарт-процесс для закупок, производства и аренды?", "spa-independent-process"),
        ("permissions", "Как ограничить видимость пользовательских полей для сотрудников?", "crm-custom-field-visibility"),
        ("permissions", "Как убрать разрешение «Экспорт» для выгрузки контактов, компаний или сделок?",
         "crm-role-export-permission"),
        ("analytics", "Почему штатная CRM-аналитика не подходит для новых счетов?", "crm-reports-permissions-and-scope"),
        ("workflows", "Как получить асинхронный внешний скоринг в бизнес-процессе?", "workflow-external-scoring-gap"),
    ],
)
def test_practical_queries(
    indexed_corpus: tuple[KnowledgeBase, PortalScope], area: Area, question: str, expected_id: str,
) -> None:
    knowledge, scope = indexed_corpus
    hits, execution = knowledge.search(scope, question, [area], now=REVIEW_DATE, dry_run=True)
    assert expected_id in [hit.fragment_id for hit in hits[:5]]
    assert execution is None
