import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import pytest

from core.advisor_history import AdvisorHistory
from core.config import ModelSettings
from core.contracts import AdvisorRequest, PortalScope
from core.knowledge import KnowledgeBase
from core.llm import LLMGateway
from core.prompts import PromptRegistry
from core.storage import SnapshotStore
from modules.advisor.engine import AdvisorEngine
from modules.advisor.service import Advisor, local_route
from ops.runners.advisor_benchmark import BenchmarkCase, evaluate, load_cases, run_case

CASES = load_cases()


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_labelled_advisor_cases(tmp_path: Path, case: BenchmarkCase) -> None:
    store = SnapshotStore(tmp_path / "advisor.sqlite3")
    run, record, cited = run_case(case, store)
    assert all(evaluate(case, run, cited).values())
    assert record.request_id == run.response.request_id
    assert record.scope == run.response.scope
    assert record.snapshot_id == run.response.snapshot_id
    assert record.tier == run.response.tier
    assert record.review_codes == run.review_codes
    assert record.prompt_versions == run.prompt_versions
    assert record.telemetry.route == run.route
    assert record.telemetry.recommend == run.recommend
    assert record.telemetry.embedding == run.embedding
    assert record.telemetry.duration_ms >= 0
    assert record.telemetry.cost_rub_e8 == 0
    assert record.telemetry.route.tokens_in == (0 if case.dry_run else 12)
    if case.embedding_fallback:
        assert record.telemetry.mode == "fallback"
        assert run.embedding is not None and run.embedding.fallback_reason == "invalid_response"
        assert run.embedding.tokens_in == 3
    if case.dry_run:
        assert record.telemetry.mode == "deterministic"
    if case.required_claim:
        run.response.recommendation = "Включите неподтверждённую возможность."
        assert not evaluate(case, run, cited)["claims"]


def test_history_scope_privacy_and_atomicity(tmp_path: Path) -> None:
    path = tmp_path / "advisor.sqlite3"
    store = SnapshotStore(path)
    gateway = LLMGateway(ModelSettings())
    engine = AdvisorEngine(Advisor(gateway, KnowledgeBase(store, gateway), PromptRegistry()))
    scope = PortalScope(tenant_id="a", portal_id="portal")
    request = AdvisorRequest(scope=scope, request_id=uuid4(), question="CRM PRIVATE_CARD_CONTENT")
    engine.run(request, public_question="CRM")

    history = AdvisorHistory(SnapshotStore(path))
    record = history.get(scope, request.request_id)
    assert record is not None
    assert record.telemetry.cost_rub_e8 is None
    assert record.telemetry.mode == "fallback"
    assert record.telemetry.route.fallback_reason == "unconfigured"
    assert record.review_codes == ["missing_snapshot", "missing_sources"]
    assert history.get(PortalScope(tenant_id="b", portal_id="portal"), request.request_id) is None
    assert history.get(PortalScope(tenant_id="a", portal_id="other"), request.request_id) is None
    with pytest.raises(sqlite3.IntegrityError):
        history.save(record)
    assert history.get(scope, request.request_id) == record

    with closing(store.connect()) as connection, connection:
        row = connection.execute("SELECT response_json, cost_rub_e8 FROM advisor_runs").fetchone()
        assert row is not None and row[1] is None
        assert all(key not in row[0] for key in ("PRIVATE", "recommendation", "observations"))
        connection.execute(
            """CREATE TRIGGER reject_telemetry BEFORE INSERT ON advisor_run_telemetry
               BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END"""
        )
    request.request_id = uuid4()
    with pytest.raises(sqlite3.IntegrityError):
        engine.run(request, dry_run=True)
    with closing(store.connect()) as connection:
        assert connection.execute("SELECT COUNT(*) FROM advisor_runs").fetchone() == (1,)


def test_route_distinguishes_smart_process_and_workflow() -> None:
    assert local_route("Смарт-процесс CRM").areas == ["crm", "smart_processes"]
    assert local_route("Бизнес-процесс согласования").areas == ["workflows"]
