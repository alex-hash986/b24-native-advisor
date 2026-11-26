import argparse
import json
from collections import Counter
from pathlib import Path

from core.contracts import PortalSnapshot


def load_export(path: Path) -> PortalSnapshot:
    snapshot = PortalSnapshot.model_validate_json(path.read_bytes())
    if snapshot.metadata is None:
        raise ValueError("export must include hashed metadata")
    return snapshot


def checklist(snapshot: PortalSnapshot) -> dict[str, object]:
    metadata = snapshot.metadata
    assert metadata is not None
    return {
        "snapshot_id": str(snapshot.snapshot_id),
        "captured_at": snapshot.captured_at.isoformat(),
        "sha256": snapshot.sha256,
        "observation_counts": dict(Counter(
            f"{row.area}/{row.status}" for row in snapshot.observations
        )),
        "gaps": [
            {"area": row.area, "status": row.status, "source": row.source_ref, "code": row.summary}
            for row in snapshot.observations if row.status != "observed"
        ],
        "structure_for_ui_review": {
            "smart_processes": [row.model_dump(mode="json") for row in metadata.smart_processes],
            "categories": [row.model_dump(mode="json") for row in metadata.categories],
            "stages": [row.model_dump(mode="json") for row in metadata.stages],
            "fields": [row.model_dump(mode="json") for row in metadata.fields],
        },
        "ui_verification": "pending",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a snapshot and prepare a local UI checklist")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New report under .local/")
    args = parser.parse_args()
    local = Path(".local").resolve()
    output = args.output.resolve()
    if not output.is_relative_to(local) or output.suffix.lower() != ".json":
        parser.error("--output must be a new JSON path under .local/")
    try:
        report = checklist(load_export(args.snapshot))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.write("\n")
    except (OSError, ValueError):
        parser.exit(1, "Snapshot validation or local checklist export failed.\n")
    print(f"snapshot_id={report['snapshot_id']} ui_verification=pending")


if __name__ == "__main__":
    main()
