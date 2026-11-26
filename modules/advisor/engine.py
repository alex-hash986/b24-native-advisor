from datetime import UTC, datetime
from time import perf_counter_ns

from core.advisor_history import AdvisorHistory, RunRecord, RunTelemetry
from core.contracts import AdvisorRequest
from modules.advisor.service import Advisor, AdvisorRun


class AdvisorEngine:
    def __init__(self, advisor: Advisor) -> None:
        self.advisor = advisor
        self.store = advisor.knowledge.store
        self.history = AdvisorHistory(self.store)

    def run(
        self, request: AdvisorRequest, *, dry_run: bool = False, now: datetime | None = None,
        public_question: str | None = None, entity: str | None = None, tariff: str | None = None,
    ) -> AdvisorRun:
        started = perf_counter_ns()
        request = AdvisorRequest.model_validate_json(request.model_dump_json())
        snapshot = (
            self.store.get(request.scope, request.snapshot_id) if request.snapshot_id else None
        )
        result = self.advisor.run(
            request, snapshot, dry_run=dry_run, now=now, public_question=public_question,
            entity=entity, tariff=tariff,
        )
        stages = [result.route, result.recommend]
        if result.embedding is not None:
            stages.append(result.embedding)
        modes = {stage.mode for stage in stages}
        self.history.save(RunRecord(
            scope=request.scope, request_id=request.request_id,
            snapshot_id=result.response.snapshot_id, tier=result.response.tier,
            prompt_versions=result.prompt_versions, review_codes=result.review_codes,
            telemetry=RunTelemetry(
                scope=request.scope, request_id=request.request_id, completed_at=datetime.now(UTC),
                duration_ms=(perf_counter_ns() - started) // 1_000_000,
                mode="fallback" if "fallback" in modes else "llm" if "llm" in modes else "deterministic",
                route=result.route, recommend=result.recommend, embedding=result.embedding,
            ),
        ))
        return result
