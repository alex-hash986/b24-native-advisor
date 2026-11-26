import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from typing import Any, Literal
from uuid import uuid4

from pydantic import TypeAdapter

from core.advisor_history import ReviewCode, RunRecord
from core.config import ModelSettings
from core.contracts import (
    AdvisorRequest,
    Contract,
    Observation,
    PortalScope,
    PortalSnapshot,
    Tier,
)
from core.knowledge import KnowledgeBase, load_fragments
from core.llm import LLMGateway, Price
from core.prompts import PromptRegistry
from core.storage import SnapshotStore
from modules.advisor.engine import AdvisorEngine
from modules.advisor.service import Advisor, AdvisorRun, Draft, local_route

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "ops" / "fixtures" / "advisor_cases.json"
NOW = datetime(2026, 11, 25, 12, tzinfo=UTC)
GAP = "Передать метаданные полей во внешний реестр с его собственным протоколом."


class BenchmarkCase(Contract):
    name: str
    question: str
    sources: list[str]
    snapshot: Literal["complete", "partial", "absent", "foreign"] = "complete"
    age_days: int = 0
    embedding_fallback: bool = False
    dry_run: bool = False
    draft: Draft | None = None
    expected_tier: Tier
    expected_review: ReviewCode | None = None
    expected_sources: list[str] = []
    required_claim: str = ""


class CaseResult(Contract):
    case: str
    tier: Tier
    conditional: bool
    escalated: bool
    review_codes: list[ReviewCode]
    mode: str
    duration_ms: int
    cost_rub_e8: int | None
    checks: dict[str, bool]


def load_cases() -> list[BenchmarkCase]:
    return TypeAdapter(list[BenchmarkCase]).validate_json(CASES.read_text(encoding="utf-8"))


def run_case(case: BenchmarkCase, store: SnapshotStore) -> tuple[AdvisorRun, RunRecord, set[str]]:
    def transport(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        if case.dry_run:
            raise AssertionError("transport used by dry-run case")
        if endpoint == "embeddings" and case.embedding_fallback:
            data = [] if payload["input"] == case.question else [{"embedding": [1.0]}]
            return {"data": data, "usage": {"prompt_tokens": 3}}
        if endpoint != "chat/completions":
            raise AssertionError("unexpected model endpoint")
        if payload["model"] == "route":
            content = local_route(case.question).model_dump_json()
        elif payload["model"] == "recommend" and case.draft is not None:
            content = case.draft.model_dump_json()
        else:
            raise AssertionError("unexpected recommendation call")
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        }

    gateway = LLMGateway(
        ModelSettings.model_validate({
            "base_url": "http://127.0.0.1/v1",
            "fast_model": "route", "smart_model": "recommend", "embedding_model": "vector",
        }),
        transport=transport,
        prices={
            model: Price(input_per_million_rub=Decimal(0), output_per_million_rub=Decimal(0))
            for model in ("route", "recommend", "vector")
        },
    )
    kb = KnowledgeBase(store, gateway)
    engine = AdvisorEngine(Advisor(gateway, kb, PromptRegistry()))
    scope = PortalScope(tenant_id="benchmark", portal_id=f"{case.name}.invalid")
    source_by_id = {
        item.fragment_id: item
        for item in load_fragments((ROOT / "docs/knowledge/bitrix24.json").read_text(encoding="utf-8"))
    }
    fragments = [
        source_by_id[key].model_copy(update={
            "checked_at": NOW - timedelta(days=case.age_days),
            "native_gap": GAP if key == "rest-fields" else None,
        })
        for key in case.sources
    ]
    kb.replace(scope, fragments, dry_run=not case.embedding_fallback)
    snapshot_id = None
    if case.snapshot != "absent":
        snapshot_id = uuid4()
        snapshot_scope = scope
        if case.snapshot == "foreign":
            snapshot_scope = PortalScope(tenant_id="foreign", portal_id=scope.portal_id)
        store.save(PortalSnapshot(
            scope=snapshot_scope, request_id=uuid4(), snapshot_id=snapshot_id, captured_at=NOW,
            observations=[
                Observation(
                    area=area, source_ref="synthetic", summary="synthetic",
                    status="no_access" if case.snapshot == "partial" else "observed",
                )
                for area in local_route(case.question).areas
            ],
        ))
    request = AdvisorRequest(
        scope=scope, request_id=uuid4(), snapshot_id=snapshot_id, question=case.question,
    )
    result = engine.run(request, dry_run=case.dry_run, now=NOW, public_question=case.question)
    record = engine.history.get(scope, request.request_id)
    if record is None:
        raise AssertionError("missing run history")
    cited = {
        fragment.fragment_id for fragment in fragments
        if fragment.evidence() in result.response.documentation
    }
    return result, record, cited


def evaluate(case: BenchmarkCase, result: AdvisorRun, cited: set[str]) -> dict[str, bool]:
    response = result.response
    conditional = case.snapshot in {"absent", "foreign"}
    expected_claim = case.required_claim or "Уточните данные и подтвердите применимость штатного варианта."
    if conditional:
        expected_claim = "Состояние портала не проверено. Выводы о его настройках условны. " + expected_claim
    return {
        "classification": response.tier == case.expected_tier
        and result.review_codes == ([case.expected_review] if case.expected_review else []),
        "citations": cited == set(case.expected_sources)
        and len(cited) == len(response.documentation),
        "claims": response.recommendation == expected_claim
        and response.native_gap == (GAP if case.expected_tier == "extension" else None),
        "conditional": (response.snapshot_id is None) == conditional
        and ("условны" in response.recommendation) == (response.snapshot_id is None),
    }


def benchmark(store: SnapshotStore) -> dict[str, object]:
    started = perf_counter_ns()
    results = []
    for case in load_cases():
        run, record, cited = run_case(case, store)
        results.append(CaseResult(
            case=case.name, tier=run.response.tier, conditional=run.response.snapshot_id is None,
            escalated=run.response.need_human, review_codes=record.review_codes,
            mode=record.telemetry.mode, duration_ms=record.telemetry.duration_ms,
            cost_rub_e8=record.telemetry.cost_rub_e8, checks=evaluate(case, run, cited),
        ))
    timings = sorted(item.duration_ms for item in results)
    costs = [item.cost_rub_e8 for item in results]
    checks = Counter(key for item in results for key, success in item.checks.items() if not success)
    count = len(results)
    return {
        "cases": count,
        "tier_counts": dict(Counter(item.tier for item in results)),
        "native_share": sum(item.tier == "native" for item in results) / count,
        "conditional_share": sum(item.conditional for item in results) / count,
        "escalated_share": sum(item.escalated for item in results) / count,
        "p50_ms": timings[math.ceil(count * 0.50) - 1],
        "p95_ms": timings[math.ceil(count * 0.95) - 1],
        "wall_ms": (perf_counter_ns() - started) // 1_000_000,
        "total_cost_rub_e8": None if None in costs else sum(cost for cost in costs if cost is not None),
        "citation_failures": checks["citations"],
        "claim_failures": checks["claims"],
        "failed_checks": dict(checks),
        "results": [item.model_dump(mode="json") for item in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальный сквозной smoke и бенчмарк советника")
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.parse_args()
    local = ROOT / ".local"
    local.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="advisor-benchmark-", dir=local) as directory:
        report = benchmark(SnapshotStore(Path(directory) / "benchmark.sqlite3"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failed_checks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
