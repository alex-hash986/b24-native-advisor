import argparse
import json
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from pydantic import TypeAdapter

from core.config import ModelSettings, load_settings
from core.contracts import AdvisorRequest, PortalScope, PortalSnapshot, SnapshotRequest
from core.knowledge import KnowledgeBase, load_fragments
from core.llm import LLMGateway, Price
from core.prompts import PromptRegistry
from core.storage import SnapshotStore
from modules.advisor.engine import AdvisorEngine
from modules.advisor.service import Advisor
from modules.portal_audit.client import BitrixReadOnlyClient, PortalLimiter
from modules.portal_audit.collector import PortalSnapshotCollector
from ops.runners.collect_snapshot import AREAS, dry_transport
from ops.runners.verify_snapshot import load_export

CORPUS = Path(__file__).resolve().parents[2] / "docs/knowledge/bitrix24.json"
CASES = (
    ("routing", "Как распределять входящие лиды из CRM-формы по очереди между менеджерами?"),
    ("stage-task", "Как робот может создать задачу юристу на стадии «Договор» по сделке?"),
    ("access", "Как дать отделу доступ только к своим сделкам в отдельной воронке CRM?"),
    ("smart-process", "Что выбрать для рекламаций со своими стадиями и полями: сделку или смарт-процесс?"),
)


def fixture_snapshot() -> PortalSnapshot:
    client = BitrixReadOnlyClient(dry_transport, PortalLimiter(clock=count(0.0, 0.5).__next__))
    return PortalSnapshotCollector(client).collect(SnapshotRequest.model_validate({
        "request_id": uuid4(), "scope": PortalScope(tenant_id="pilot", portal_id="fixture.invalid"),
        "areas": AREAS,
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four Bitrix24 consultation cases through AdvisorEngine")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--provider", action="store_true")
    parser.add_argument("--snapshot", type=Path, help="Hashed snapshot export; defaults to a fixture")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--prices", type=Path, help="JSON mapping model names to Price records")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    args = parser.parse_args()
    if args.dry_run and args.prices:
        parser.error("--prices requires --provider")

    try:
        settings = load_settings(args.env_file) if args.provider else None
        models = settings.models if settings else ModelSettings()
        if args.provider and not (models.base_url and models.fast_model and models.smart_model):
            parser.error("--provider requires LLM_BASE_URL, FAST_LLM_MODEL and SMART_LLM_MODEL")
        prices = (
            TypeAdapter(dict[str, Price]).validate_json(args.prices.read_bytes()) if args.prices else {}
        )
        snapshot = load_export(args.snapshot) if args.snapshot else fixture_snapshot()
        fragments = load_fragments(args.corpus.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        parser.exit(1, "Invalid pilot configuration, corpus or snapshot.\n")

    now = datetime.now(UTC)
    if args.dry_run and fragments:
        # Deterministic replay at the date the supplied corpus was checked.
        now = max(fragment.checked_at for fragment in fragments)
    local = Path(".local")
    local.mkdir(exist_ok=True)
    results: list[dict[str, object]] = []
    with TemporaryDirectory(prefix="advisor-pilot-", dir=local) as directory:
        store = SnapshotStore(Path(directory) / "pilot.sqlite3")
        store.save(snapshot)
        gateway = LLMGateway(models, prices=prices)
        knowledge = KnowledgeBase(store, gateway, max_age_days=settings.docs_max_age_days if settings else 30)
        knowledge.replace(snapshot.scope, fragments, dry_run=True)
        engine = AdvisorEngine(Advisor(gateway, knowledge, PromptRegistry()))
        for name, question in CASES:
            request = AdvisorRequest(
                scope=snapshot.scope, request_id=uuid4(), snapshot_id=snapshot.snapshot_id,
                question=question,
            )
            run = engine.run(request, dry_run=args.dry_run, public_question=question, now=now)
            record = engine.history.get(request.scope, request.request_id)
            if record is None:
                raise RuntimeError("missing pilot history")
            results.append({
                "case": name, "request_id": str(request.request_id), "tier": run.response.tier,
                "snapshot_bound": run.response.snapshot_id == snapshot.snapshot_id,
                "review_codes": run.review_codes, "mode": record.telemetry.mode,
                "duration_ms": record.telemetry.duration_ms, "cost_rub_e8": record.telemetry.cost_rub_e8,
                "route": run.route.model_dump(mode="json"),
                "recommend": run.recommend.model_dump(mode="json"),
                "recommendation": run.response.recommendation,
                "sources": sorted({str(row.url) for row in run.response.documentation}),
            })
    print(json.dumps({
        "snapshot_id": str(snapshot.snapshot_id),
        "snapshot_input": "export" if args.snapshot else "fixture",
        "evidence_at": now.isoformat(),
        "results": results,
    }, ensure_ascii=False, indent=2))
    if args.provider and any(result["mode"] == "fallback" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
