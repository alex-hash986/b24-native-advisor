from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from core.config import load_settings
from core.contracts import (
    Observation,
    PortalScope,
    PortalSnapshot,
    SnapshotMetadata,
    snapshot_hash,
)
from core.storage import SnapshotStore


def test_settings_load_pin_and_hide_webhook(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "BITRIX24_WEBHOOK_URL=https://example.test/rest/1/synthetic-secret/\n"
        "BITRIX24_PINNED_IP=2001:0db8::1\n"
        "B24_TENANT_ID=demo\nSQLITE_PATH=local.sqlite3\n",
        encoding="utf-8",
    )
    settings = load_settings(env)
    assert settings.tenant_id == "demo"
    assert settings.sqlite_path == Path("local.sqlite3")
    assert settings.webhook and settings.webhook.hostname == "example.test"
    assert settings.webhook.pinned_ip == "2001:db8::1"
    assert "synthetic-secret" not in repr(settings)


@pytest.mark.parametrize("url", [
    "http://example.test/rest/1/synthetic-secret/",
    "https://user@example.test/rest/1/synthetic-secret/",
    "https://example.test/rest/1/synthetic-secret/?auth=other",
])
def test_invalid_webhook_is_redacted(tmp_path: Path, url: str) -> None:
    env = tmp_path / ".env"
    env.write_text(f"BITRIX24_WEBHOOK_URL={url}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid snapshot configuration") as exc:
        load_settings(env)
    assert "synthetic-secret" not in str(exc.value)
    assert exc.value.__context__ is None


def test_store_round_trip_and_tenant_isolation(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "test.sqlite3")
    metadata = SnapshotMetadata()
    observations = [Observation(area="robots", status="unsupported", summary="limit", source_ref="test")]
    snapshot = PortalSnapshot(
        request_id=uuid4(), snapshot_id=uuid4(), scope=PortalScope(tenant_id="a", portal_id="p"),
        captured_at=datetime.fromisoformat("2026-11-25T07:00:00+07:00"),
        metadata=metadata, observations=observations, sha256=snapshot_hash(metadata, observations),
    )
    assert snapshot.captured_at.hour == 0
    store.save(snapshot)
    connection = store.connect()
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
        assert connection.execute("PRAGMA busy_timeout").fetchone() == (5000,)
    finally:
        connection.close()
    assert store.get(snapshot.scope, snapshot.snapshot_id) == snapshot
    assert store.get(PortalScope(tenant_id="b", portal_id="p"), snapshot.snapshot_id) is None
    assert store.get(PortalScope(tenant_id="a", portal_id="q"), snapshot.snapshot_id) is None
    snapshot.observations[0].summary = "changed"
    with pytest.raises(ValidationError, match="hash mismatch"):
        store.save(snapshot)


def test_nested_metadata_rejects_extra_keys_and_coercion() -> None:
    with pytest.raises(ValidationError) as exc:
        SnapshotMetadata.model_validate({"categories": [{
            "entity_type_id": "2", "category_id": 0, "name": "Mock", "unexpected": True,
        }]})
    assert {error["type"] for error in exc.value.errors()} == {"int_type", "extra_forbidden"}
