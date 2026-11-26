import argparse
import hashlib
import json
import sqlite3
import tempfile
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, closing, contextmanager, nullcontext
from itertools import count
from pathlib import Path
from uuid import uuid4

from core.config import load_settings
from core.contracts import PortalScope, SnapshotRequest
from core.storage import SnapshotStore
from modules.portal_audit.client import BitrixReadOnlyClient, PortalLimiter
from modules.portal_audit.collector import PortalSnapshotCollector

AREAS = {"crm", "robots", "workflows", "smart_processes", "permissions", "analytics"}


@contextmanager
def live_dispatcher(root: Path, hostname: str) -> Iterator[None]:
    directory = root / "dispatchers"
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(hostname.encode("ascii")).hexdigest()
    with closing(sqlite3.connect(directory / f"{key}.sqlite3", timeout=0)) as lock:
        lock.execute("BEGIN IMMEDIATE")
        # Preserve request spacing across sequential invocations and after a crash.
        time.sleep(0.5)
        yield


def dry_transport(method: str, params: dict[str, object]) -> tuple[int, str | None, bytes]:
    if method == "crm.status.list":
        filters = params["filter"]
        assert isinstance(filters, dict)
        result: object = [{
            "ENTITY_ID": filters["ENTITY_ID"], "STATUS_ID": "MOCK_NEW", "NAME": "Mock", "SORT": 10,
        }]
    else:
        result = {
            "method.get": {"isExisting": True, "isAvailable": True, "scope": "crm"},
            "crm.type.list": {"types": [{
                "entityTypeId": 1040, "title": "Mock", "isCategoriesEnabled": "Y",
                "isStagesEnabled": "Y",
            }]},
            "crm.category.list": {"categories": [{"id": 0, "name": "Mock"}]},
            "crm.item.fields": {"fields": {"id": {
                "type": "integer", "isRequired": False, "isMultiple": False, "isReadOnly": True,
            }}},
        }[method]
    return 200, None, json.dumps({"result": result}).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Bitrix24 metadata snapshot")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--live", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, help="New JSON file (defaults to .local)")
    args = parser.parse_args()
    try:
        request_id = uuid4()
        root = Path(".local").resolve()
        output = (args.output or root / f"snapshot-{request_id}.json").resolve()
        if output.suffix.lower() != ".json":
            parser.error("--output must be a JSON file")
        if output.exists():
            parser.error("--output must name a new file")
        dispatch: AbstractContextManager[None]
        if args.live:
            settings = load_settings(args.env_file)
            if settings.webhook is None:
                parser.error("BITRIX24_WEBHOOK_URL is required for --live")
            scope = PortalScope(tenant_id=settings.tenant_id, portal_id=settings.webhook.hostname)
            client = BitrixReadOnlyClient.for_webhook(settings.webhook)
            database = settings.sqlite_path.resolve()
            dispatch = live_dispatcher(root, settings.webhook.hostname)
        else:
            scope = PortalScope(tenant_id="fixture", portal_id="fixture.invalid")
            database = root / "dry-run.sqlite3"
            client = BitrixReadOnlyClient(dry_transport, PortalLimiter(clock=count(0.0, 0.5).__next__))
            dispatch = nullcontext()

        db = SnapshotStore(database)
        request = SnapshotRequest.model_validate({
            "request_id": request_id, "scope": scope, "areas": AREAS,
        })
        with dispatch:
            snapshot = PortalSnapshotCollector(client).collect(request)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Stage JSON before committing SQLite. Publish with an exclusive hard link.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False,
        ) as fixture:
            pending = Path(fixture.name)
        try:
            pending.write_text(snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8")
            output.hardlink_to(pending)
            try:
                db.save(snapshot)
            except (OSError, ValueError, sqlite3.Error):
                output.unlink()
                raise
        finally:
            pending.unlink(missing_ok=True)
    except (OSError, ValueError, sqlite3.Error, RuntimeError):
        parser.exit(1, "Snapshot was not saved to both destinations. Check local configuration and paths.\n")
    print(f"snapshot_id={snapshot.snapshot_id} sha256={snapshot.sha256} fixture={output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
