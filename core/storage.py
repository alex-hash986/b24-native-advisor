import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import UUID

from core.contracts import PortalScope, PortalSnapshot

SCHEMA = Path(__file__).resolve().parents[1] / "infra" / "schema.sql"


class SnapshotStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection:
            connection.executescript(SCHEMA.read_text(encoding="utf-8"))

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
            if connection.execute("PRAGMA journal_mode = WAL").fetchone() != ("wal",):
                raise RuntimeError("SQLite WAL unavailable")
            if connection.execute("PRAGMA foreign_keys").fetchone() != (1,):
                raise RuntimeError("SQLite foreign keys unavailable")
        except BaseException:
            connection.close()
            raise
        return connection

    def save(self, snapshot: PortalSnapshot) -> None:
        snapshot = PortalSnapshot.model_validate_json(snapshot.model_dump_json())
        with closing(self.connect()) as connection, connection:
            connection.execute(
                """INSERT INTO portal_snapshots
                   (tenant_id, portal_id, snapshot_id, request_id, captured_at, payload_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    snapshot.scope.tenant_id,
                    snapshot.scope.portal_id,
                    str(snapshot.snapshot_id),
                    str(snapshot.request_id),
                    snapshot.captured_at.isoformat(),
                    snapshot.model_dump_json(),
                ),
            )

    def ensure_saved(self, snapshot: PortalSnapshot) -> None:
        existing = self.get(snapshot.scope, snapshot.snapshot_id)
        if existing is None:
            try:
                self.save(snapshot)
                return
            except sqlite3.IntegrityError:
                existing = self.get(snapshot.scope, snapshot.snapshot_id)
                if existing is None:
                    raise
        if existing != snapshot:
            raise ValueError("conflicting snapshot for the same scope and ID")

    def get(self, scope: PortalScope, snapshot_id: UUID) -> PortalSnapshot | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                """SELECT payload_json, request_id FROM portal_snapshots
                   WHERE tenant_id = ? AND portal_id = ? AND snapshot_id = ?""",
                (scope.tenant_id, scope.portal_id, str(snapshot_id)),
            ).fetchone()
        if row is None:
            return None
        snapshot = PortalSnapshot.model_validate_json(row[0])
        if (
            snapshot.scope != scope
            or snapshot.snapshot_id != snapshot_id
            or str(snapshot.request_id) != row[1]
        ):
            raise ValueError("snapshot row metadata mismatch")
        return snapshot
