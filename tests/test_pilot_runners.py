import json
from pathlib import Path

import pytest

from ops.runners.advisor_pilot import fixture_snapshot, main
from ops.runners.verify_snapshot import checklist, load_export


def test_snapshot_checklist_rejects_tampered_export(tmp_path: Path) -> None:
    snapshot = fixture_snapshot()
    export = tmp_path / "snapshot.json"
    export.write_text(snapshot.model_dump_json(), encoding="utf-8")
    report = checklist(load_export(export))
    assert report["ui_verification"] == "pending"
    assert report["sha256"] == snapshot.sha256
    payload = snapshot.model_dump(mode="json")
    payload["metadata"]["categories"][0]["name"] = "Modified"
    export.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_export(export)


def test_pilot_dry_run_binds_export_and_preserves_evidence_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot = fixture_snapshot()
    export = tmp_path / "snapshot.json"
    export.write_text(snapshot.model_dump_json(), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["advisor_pilot", "--dry-run", "--snapshot", str(export)])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["snapshot_id"] == str(snapshot.snapshot_id)
    assert report["snapshot_input"] == "export"
    results = {row["case"]: row for row in report["results"]}
    assert set(results) == {"routing", "stage-task", "access", "smart-process"}
    for row in results.values():
        assert row["snapshot_bound"]
        assert row["mode"] == "deterministic"
        assert row["tier"] == "undetermined"
        assert row["cost_rub_e8"] == 0
    assert results["stage-task"]["review_codes"] == ["visibility_gap"]
    assert results["access"]["review_codes"] == ["visibility_gap"]
