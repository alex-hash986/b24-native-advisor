PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

BEGIN;

CREATE TABLE IF NOT EXISTS portal_snapshots (
    tenant_id TEXT NOT NULL,
    portal_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    PRIMARY KEY (tenant_id, portal_id, snapshot_id),
    UNIQUE (tenant_id, portal_id, request_id)
) STRICT;

CREATE TABLE IF NOT EXISTS advisor_runs (
    tenant_id TEXT NOT NULL,
    portal_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    snapshot_id TEXT,
    created_at TEXT NOT NULL,
    response_json TEXT NOT NULL CHECK (json_valid(response_json)),
    prompt_versions_json TEXT NOT NULL CHECK (json_valid(prompt_versions_json)),
    cost_rub_e8 INTEGER CHECK (cost_rub_e8 >= 0),
    PRIMARY KEY (tenant_id, portal_id, request_id),
    FOREIGN KEY (tenant_id, portal_id, snapshot_id)
        REFERENCES portal_snapshots (tenant_id, portal_id, snapshot_id)
) STRICT;

CREATE TABLE IF NOT EXISTS knowledge_fragments (
    tenant_id TEXT NOT NULL,
    portal_id TEXT NOT NULL,
    fragment_id TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    index_revision TEXT NOT NULL,
    vector_json TEXT CHECK (json_valid(vector_json)),
    PRIMARY KEY (tenant_id, portal_id, fragment_id)
) STRICT;

CREATE TABLE IF NOT EXISTS advisor_run_telemetry (
    tenant_id TEXT NOT NULL,
    portal_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    PRIMARY KEY (tenant_id, portal_id, request_id),
    FOREIGN KEY (tenant_id, portal_id, request_id)
        REFERENCES advisor_runs (tenant_id, portal_id, request_id) ON DELETE CASCADE
) STRICT;

COMMIT;
