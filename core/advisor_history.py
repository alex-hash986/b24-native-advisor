from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime
from typing import Literal, cast
from uuid import UUID

from pydantic import AwareDatetime

from core.contracts import (
    Contract,
    NonnegativeId,
    PortalScope,
    ScopedRequest,
    Text,
    Tier,
)
from core.llm import Execution
from core.storage import SnapshotStore

ReviewCode = Literal[
    "unknown_area", "unsafe_input", "missing_snapshot", "visibility_gap",
    "conflicting_sources", "missing_sources", "invalid_evidence", "unsupported_conclusion",
    "model_review",
]


class RunOutcome(ScopedRequest):
    snapshot_id: UUID | None
    tier: Tier
    review_codes: list[ReviewCode]

    @property
    def conditional_native(self) -> bool:
        return self.tier == "native" and (
            "missing_snapshot" in self.review_codes or "visibility_gap" in self.review_codes
        )


class RunTelemetry(ScopedRequest):
    completed_at: AwareDatetime
    duration_ms: NonnegativeId
    mode: Literal["llm", "deterministic", "fallback"]
    route: Execution
    recommend: Execution
    embedding: Execution | None

    @property
    def cost_rub_e8(self) -> int | None:
        costs = [self.route.cost_rub_e8, self.recommend.cost_rub_e8]
        if self.embedding is not None:
            costs.append(self.embedding.cost_rub_e8)
        if any(cost is None for cost in costs):
            return None
        total = sum(cost for cost in costs if cost is not None)
        return total if total < 2**63 else None


class PromptVersions(Contract):
    prompt_versions: dict[str, Text]


class RunRecord(RunOutcome, PromptVersions):
    telemetry: RunTelemetry


class RunSummary(Contract):
    total: NonnegativeId
    native: NonnegativeId
    conditional_native: NonnegativeId
    known_cost_rub_e8: NonnegativeId
    unknown_cost_count: NonnegativeId
    mean_duration_ms: float | None

    @classmethod
    def from_records(cls, records: Sequence[RunRecord]) -> "RunSummary":
        costs = [record.telemetry.cost_rub_e8 for record in records]
        return cls(
            total=len(records),
            native=sum(record.tier == "native" for record in records),
            conditional_native=sum(record.conditional_native for record in records),
            known_cost_rub_e8=sum(cost for cost in costs if cost is not None),
            unknown_cost_count=costs.count(None),
            mean_duration_ms=(
                sum(record.telemetry.duration_ms for record in records) / len(records)
                if records else None
            ),
        )


HistoryRow = tuple[str, str, str, int | None, str | None, str, str]

HISTORY_QUERY = """SELECT r.response_json, r.prompt_versions_json, t.payload_json,
                          r.cost_rub_e8, r.snapshot_id, r.created_at, r.request_id
                   FROM advisor_runs r JOIN advisor_run_telemetry t
                      USING (tenant_id, portal_id, request_id)
                   WHERE tenant_id = ? AND portal_id = ?"""


class AdvisorHistory:
    def __init__(self, store: SnapshotStore) -> None:
        self.store = store

    def save(self, record: RunRecord) -> None:
        record = RunRecord.model_validate_json(record.model_dump_json())
        telemetry = record.telemetry
        if (record.scope, record.request_id) != (telemetry.scope, telemetry.request_id):
            raise ValueError("advisor telemetry correlation mismatch")
        outcome = RunOutcome.model_validate(record.model_dump(include=set(RunOutcome.model_fields)))
        key = (record.scope.tenant_id, record.scope.portal_id, str(record.request_id))
        with closing(self.store.connect()) as connection, connection:
            connection.execute(
                """INSERT INTO advisor_runs
                   (tenant_id, portal_id, request_id, snapshot_id, created_at,
                    response_json, prompt_versions_json, cost_rub_e8)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (*key, str(record.snapshot_id) if record.snapshot_id else None,
                 telemetry.completed_at.astimezone(UTC).isoformat(), outcome.model_dump_json(),
                 record.model_dump_json(include={"prompt_versions"}), telemetry.cost_rub_e8),
            )
            connection.execute(
                """INSERT INTO advisor_run_telemetry (tenant_id, portal_id, request_id, payload_json)
                   VALUES (?, ?, ?, ?)""",
                (*key, telemetry.model_dump_json()),
            )

    def get(self, scope: PortalScope, request_id: UUID) -> RunRecord | None:
        with closing(self.store.connect()) as connection:
            row = cast(HistoryRow | None, connection.execute(
                HISTORY_QUERY + " AND request_id = ?",
                (scope.tenant_id, scope.portal_id, str(request_id)),
            ).fetchone())
        return self._decode(row, scope) if row is not None else None

    def list_completed(self, scope: PortalScope) -> list[RunRecord]:
        with closing(self.store.connect()) as connection:
            rows = cast(list[HistoryRow], connection.execute(
                HISTORY_QUERY + " ORDER BY r.created_at DESC, r.request_id DESC",
                (scope.tenant_id, scope.portal_id),
            ).fetchall())
        return [self._decode(row, scope) for row in rows]

    @staticmethod
    def _decode(row: HistoryRow, scope: PortalScope) -> RunRecord:
        request_id = UUID(row[6])
        outcome = RunOutcome.model_validate_json(row[0])
        telemetry = RunTelemetry.model_validate_json(row[2])
        if (
            (outcome.scope, outcome.request_id) != (scope, request_id)
            or (telemetry.scope, telemetry.request_id) != (scope, request_id)
            or telemetry.cost_rub_e8 != row[3]
            or (str(outcome.snapshot_id) if outcome.snapshot_id else None) != row[4]
            or telemetry.completed_at != datetime.fromisoformat(row[5])
        ):
            raise ValueError("advisor row metadata mismatch")
        return RunRecord(
            scope=outcome.scope, request_id=outcome.request_id, snapshot_id=outcome.snapshot_id,
            tier=outcome.tier, review_codes=outcome.review_codes, telemetry=telemetry,
            prompt_versions=PromptVersions.model_validate_json(row[1]).prompt_versions,
        )
