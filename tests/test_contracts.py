import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from core.contracts import AdvisorResponse, SnapshotRequest


def test_snapshot_rejects_extra_nested_field() -> None:
    with pytest.raises(ValidationError, match="scope.write_access"):
        SnapshotRequest.model_validate({
            "request_id": uuid4(),
            "scope": {"tenant_id": "tenant", "portal_id": "portal", "write_access": True},
            "areas": ["crm"],
        })


def test_extension_requires_native_gap() -> None:
    with pytest.raises(ValidationError, match="documented native gap"):
        AdvisorResponse.model_validate({
            "request_id": uuid4(),
            "scope": {"tenant_id": "tenant", "portal_id": "portal"},
            "tier": "extension",
            "recommendation": "Extension",
            "documentation": [{
                "url": "https://helpdesk.bitrix24.ru/open/18913884/",
                "checked_at": "2026-11-25T00:00:00Z",
                "revision": "fixture",
                "excerpt": "Fixture",
            }],
            "observations": [],
            "need_human": False,
            "review_reasons": [],
        })


@pytest.mark.parametrize("need_human", [True, False])
def test_recommendation_without_observations_requires_review(need_human: bool) -> None:
    payload = {
        "request_id": uuid4(),
        "scope": {"tenant_id": "tenant", "portal_id": "portal"},
        "tier": "native",
        "recommendation": "План настройки при выполнении предпосылок.",
        "documentation": [{
            "url": "https://helpdesk.bitrix24.ru/open/18913884/",
            "checked_at": "2026-11-25T00:00:00Z",
            "revision": "fixture",
            "excerpt": "Fixture",
        }],
        "observations": [],
        "need_human": need_human,
        "review_reasons": ["Проверьте настройки в интерфейсе."] if need_human else [],
    }
    if need_human:
        assert AdvisorResponse.model_validate(payload).need_human
    else:
        with pytest.raises(ValidationError, match="without observations requires review"):
            AdvisorResponse.model_validate(payload)


def test_schema_enables_wal_and_checks_portal_scope(tmp_path: Path) -> None:
    schema = Path(__file__).resolve().parents[1] / "infra" / "schema.sql"
    connection = sqlite3.connect(tmp_path / "test.sqlite3")
    try:
        connection.executescript(schema.read_text(encoding="utf-8"))
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        connection.execute(
            "INSERT INTO portal_snapshots VALUES (?, ?, ?, ?, ?, ?)",
            ("tenant", "portal", "snapshot", "audit", "2026-11-25T00:00:00Z", "{}"),
        )
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(
                "INSERT INTO advisor_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("tenant", "other", "advice", "snapshot", "2026-11-25T00:00:00Z", "{}", "{}", 0),
            )
    finally:
        connection.close()
