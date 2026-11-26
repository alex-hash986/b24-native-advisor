import json
import sys
from itertools import count
from pathlib import Path
from uuid import uuid4

import pytest

from core.contracts import PortalSnapshot, SnapshotRequest, snapshot_hash
from core.storage import SnapshotStore
from modules.portal_audit.client import BitrixReadOnlyClient, PortalLimiter
from modules.portal_audit.collector import PortalSnapshotCollector
from ops.runners.collect_snapshot import dry_transport, live_dispatcher, main


def test_partial_paginated_snapshot_round_trip(tmp_path: Path) -> None:
    calls = []

    def transport(method: str, params: dict[str, object]) -> tuple[int, str | None, bytes]:
        calls.append((method, params))
        if method == "crm.type.list":
            if "start" in params:
                return 403, None, b'{"error_description":"secret"}'
            return 200, None, json.dumps({
                "result": {"types": [{
                    "entityTypeId": 1040, "title": "Mock", "isCategoriesEnabled": "N",
                    "isStagesEnabled": "Y", "createdBy": 123456,
                }]},
                "next": 50,
            }).encode()
        if method == "method.get":
            return 200, None, b'{"result":{}}'
        return dry_transport(method, params)

    client = BitrixReadOnlyClient(transport, PortalLimiter(clock=count().__next__))
    request = SnapshotRequest.model_validate({
        "request_id": uuid4(),
        "scope": {"tenant_id": "fixture", "portal_id": "fixture.invalid"},
        "areas": ["smart_processes", "permissions", "robots"],
    })
    snapshot = PortalSnapshotCollector(client).collect(request)
    assert snapshot.metadata is not None
    assert snapshot.metadata.smart_processes[0].entity_type_id == 1040
    assert snapshot.metadata.stages[0].directory == "DYNAMIC_1040_STAGE_0"
    assert ("crm.type.list", {"start": 50}) in calls
    assert {item.status for item in snapshot.observations} == {
        "unsupported", "no_access", "observed", "error",
    }
    assert {item.area for item in snapshot.observations} == request.areas
    assert not any(method == "bizproc.robot.list" for method, _ in calls)
    assert "123456" not in snapshot.model_dump_json()
    assert "secret" not in snapshot.model_dump_json()
    assert snapshot.sha256 == snapshot_hash(snapshot.metadata, snapshot.observations[::-1])
    store = SnapshotStore(tmp_path / "snapshot.sqlite3")
    store.save(snapshot)
    assert store.get(request.scope, snapshot.snapshot_id) == snapshot


def test_dry_run_writes_fixture_and_sqlite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    fixture = Path(".local/example.json")
    monkeypatch.setattr(sys, "argv", ["collect_snapshot", "--dry-run", "--output", str(fixture)])
    assert main() == 0
    snapshot = PortalSnapshot.model_validate_json(fixture.read_text(encoding="utf-8"))
    store = SnapshotStore(Path(".local/dry-run.sqlite3"))
    assert store.get(snapshot.scope, snapshot.snapshot_id) == snapshot
    assert snapshot.metadata is not None
    assert len(snapshot.metadata.fields) == 7
    assert {item.area for item in snapshot.observations} == {
        "crm", "robots", "workflows", "smart_processes", "permissions", "analytics",
    }
    with pytest.raises(SystemExit, match="2"):
        main()


def test_live_cli_uses_configured_scope_and_store_on_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ops.runners.collect_snapshot.time.sleep", lambda _: None)
    monkeypatch.setattr("modules.portal_audit.client.WebhookTransport", lambda _: dry_transport)
    monkeypatch.setattr(
        "modules.portal_audit.client.portal_limiter",
        lambda _: PortalLimiter(clock=count().__next__),
    )
    Path(".env").write_text(
        "BITRIX24_WEBHOOK_URL=https://mock.invalid/rest/1/synthetic-secret/\n"
        "B24_TENANT_ID=test-tenant\nSQLITE_PATH=.local/live-test.sqlite3\n",
        encoding="utf-8",
    )
    fixture = Path(".local/live-test.json")
    monkeypatch.setattr(sys, "argv", ["collect_snapshot", "--live", "--output", str(fixture)])
    assert main() == 0
    raw = fixture.read_text(encoding="utf-8")
    assert "synthetic-secret" not in raw
    snapshot = PortalSnapshot.model_validate_json(raw)
    assert snapshot.scope.tenant_id == "test-tenant" and snapshot.scope.portal_id == "mock.invalid"
    assert SnapshotStore(Path(".local/live-test.sqlite3")).get(
        snapshot.scope, snapshot.snapshot_id,
    ) == snapshot


def test_live_dispatcher_excludes_concurrent_runs_per_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    monkeypatch.setattr("ops.runners.collect_snapshot.time.sleep", lambda _: None)
    with live_dispatcher(tmp_path, "mock.invalid"):
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            with live_dispatcher(tmp_path, "mock.invalid"):
                pytest.fail("concurrent dispatcher entered")
        with live_dispatcher(tmp_path, "another.invalid"):
            pass
    with live_dispatcher(tmp_path, "mock.invalid"):
        pass
