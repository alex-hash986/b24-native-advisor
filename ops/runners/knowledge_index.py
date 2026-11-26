import argparse
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from core.config import ModelSettings, load_settings
from core.contracts import AdvisorRequest, Observation, PortalScope, PortalSnapshot
from core.knowledge import KnowledgeBase, load_fragments
from core.llm import LLMGateway
from core.prompts import PromptRegistry
from core.storage import SnapshotStore
from modules.advisor.service import Advisor

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "knowledge.json"
FIXTURE_NOW = datetime(2026, 11, 25, 12, tzinfo=UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description="Индекс документации Битрикс24")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--index", action="store_true")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--portal-id")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if args.dry_run:
        if args.portal_id:
            parser.error("--portal-id используется с --index")
        gateway = LLMGateway(ModelSettings())
        scope = PortalScope(tenant_id="synthetic", portal_id="dry-run.invalid")
        knowledge = KnowledgeBase(SnapshotStore(Path(".local/knowledge-dry-run.sqlite3")), gateway)
        fragments = load_fragments((args.corpus or FIXTURE).read_text(encoding="utf-8-sig"))
        knowledge.replace(scope, fragments, dry_run=True)
        snapshot = PortalSnapshot(
            scope=scope, request_id=uuid4(), snapshot_id=uuid4(), captured_at=FIXTURE_NOW,
            observations=[Observation(
                area="crm", status="observed", summary="visible_metadata",
                source_ref="crm.item.fields",
            )],
        )
        request = AdvisorRequest(
            scope=scope, request_id=uuid4(), snapshot_id=snapshot.snapshot_id,
            question="Как настроить поля CRM?",
        )
        hits, _ = knowledge.search(scope, request.question, ["crm"], now=FIXTURE_NOW, dry_run=True)
        result = Advisor(gateway, knowledge, PromptRegistry()).run(
            request, snapshot, dry_run=True, now=FIXTURE_NOW,
        )
        print(f"fragments={len(fragments)} matches={len(hits)}")
        print(result.model_dump_json(indent=2))
        return

    if args.corpus is None or args.portal_id is None:
        parser.error("--index требует --corpus и --portal-id")
    settings = load_settings(args.env_file)
    gateway = LLMGateway(settings.models)
    scope = PortalScope(tenant_id=settings.tenant_id, portal_id=args.portal_id)
    fragments = load_fragments(args.corpus.read_text(encoding="utf-8-sig"))
    knowledge = KnowledgeBase(
        SnapshotStore(settings.sqlite_path), gateway, max_age_days=settings.docs_max_age_days,
    )
    executions = knowledge.replace(scope, fragments)
    print(f"fragments={len(fragments)} embedded={sum(item.mode == 'llm' for item in executions)}")


if __name__ == "__main__":
    main()
