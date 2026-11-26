from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from pydantic import Field, TypeAdapter, model_validator

from core.advisor_history import RunOutcome, RunRecord, RunSummary
from core.config import ModelSettings, Settings
from core.contracts import (
    AdvisorRequest,
    Area,
    Contract,
    PortalScope,
    PortalSnapshot,
    Text,
)
from core.knowledge import Fragment, KnowledgeBase, load_fragments
from core.llm import LLMGateway, Price
from core.prompts import PromptRegistry
from core.storage import SnapshotStore
from modules.advisor.engine import AdvisorEngine
from modules.advisor.service import Advisor, AdvisorRun

DEMO_CASES = (
    (
        "Очередь лидов из CRM-формы",
        "Как распределять лиды из CRM-формы по очереди между менеджерами "
        "с учётом их рабочего времени?",
    ),
    (
        "Согласование договора",
        "Как настроить бизнес-процесс согласования договора и создать роботом "
        "задачу юристу при переходе сделки на стадию «Договор»?",
    ),
    (
        "Смарт-процесс для рекламаций",
        "Как настроить смарт-процесс для рекламаций со своими полями и стадиями "
        "и связать рекламацию со сделкой CRM?",
    ),
    (
        "Права по воронкам и аналитика",
        "Как разграничить права на сделки по воронкам CRM и дать руководителям "
        "доступ к аналитике продаж своего отдела?",
    ),
)

NO_SNAPSHOT = "Без слепка портала — условный план по справке"


class CaseStep(Contract):
    step_id: Text
    title: Text
    question: Text = Field(max_length=4000)
    areas: set[Area] = Field(min_length=1)
    fragments: list[Text] = Field(min_length=1)
    native_plan: Text
    checks: list[Text] = Field(min_length=1)


class CasePack(Contract):
    case_id: Text = Field(pattern=r"^[0-9]{2}_[a-z0-9]+(?:_[a-z0-9]+)*$")
    title: Text
    business_area: Text
    pain: Text
    costly_mistakes: list[Text] = Field(min_length=1)
    steps: list[CaseStep] = Field(min_length=1)
    architecture_assessment: list[Text] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_steps(self) -> "CasePack":
        if len({step.step_id for step in self.steps}) != len(self.steps):
            raise ValueError("duplicate case step ID")
        return self


def load_case_packs(directory: Path, corpus: Path) -> tuple[list[CasePack], dict[str, Fragment]]:
    fragments = {item.fragment_id: item for item in load_fragments(corpus.read_text(encoding="utf-8"))}
    cases = []
    for path in sorted(directory.glob("*.json")):
        case = CasePack.model_validate_json(path.read_bytes())
        if case.case_id != path.stem:
            raise ValueError("case ID must match its filename")
        for step in case.steps:
            if len(set(step.fragments)) != len(step.fragments) or set(step.fragments) - fragments.keys():
                raise ValueError(f"invalid fragment references in {case.case_id}/{step.step_id}")
            areas = {area for ref in step.fragments for area in fragments[ref].areas}
            if not step.areas <= areas:
                raise ValueError(f"incomplete area coverage in {case.case_id}/{step.step_id}")
        cases.append(case)
    return cases, fragments


class CaseStepResult(Contract):
    run: AdvisorRun
    record: RunRecord


def run_case_pack(
    engine: AdvisorEngine, tenant_id: str, case: CasePack, snapshot: PortalSnapshot | None,
    *, dry_run: bool, approved_for_provider: bool = False,
) -> list[CaseStepResult]:
    if not dry_run and not approved_for_provider:
        raise ValueError("provider spending must be approved for the case")
    results = []
    for step in case.steps:
        run, record = run_consultation(
            engine, tenant_id, step.question, snapshot,
            dry_run=dry_run, approved_for_provider=approved_for_provider,
        )
        results.append(CaseStepResult(run=run, record=record))
    return results


def case_report(
    case: CasePack, results: Sequence[CaseStepResult], fragments: dict[str, Fragment],
) -> str:
    pairs = list(zip(case.steps, results, strict=True))
    summary = RunSummary.from_records([result.record for _, result in pairs])
    snapshot_id = pairs[0][1].record.snapshot_id
    lines = [
        f"# {case.title}", "", case.business_area, "",
        f"Слепок расчёта: {snapshot_id or 'отсутствует'}", "",
        "## Боль клиента", "", case.pain, "", "**Дорогие ошибки**", "",
        *(f"- {mistake}" for mistake in case.costly_mistakes), "",
        "## Сводка по шагам", "",
    ]
    for index, (step, result) in enumerate(pairs, 1):
        lines.extend([
            f"### {index}. {step.title}", "", step.question, "",
            f"**{verdict(result.record)}**", "", result.run.response.recommendation, "",
        ])
    lines.extend([
        "## Штатная архитектура Битрикс24", "",
        "Проект по справке для ручной сверки на портале.", "",
        *(f"- **{step.title}.** {step.native_plan}" for step, _ in pairs), "",
        "### Оценка штатных средств и кастома", "",
        *(f"- {point}" for point in case.architecture_assessment), "",
        "## Официальная справка", "", "### Источники кейса", "",
    ])
    sources = {
        str(fragments[ref].url): fragments[ref].sections[0]
        for step, _ in pairs for ref in step.fragments
    }
    lines.extend(f"- [{title}]({url})" for url, title in sources.items())
    cited = {
        str(source.url): source.title or str(source.url)
        for _, result in pairs for source in result.run.response.documentation
    }
    lines.extend(["", "### Цитаты в выводах советника", ""])
    lines.extend(f"- [{title}]({url})" for url, title in cited.items())
    if not cited:
        lines.append("В итогах шагов нет подтверждённых цитат.")
    lines.extend(["", "## Чек-лист ручной проверки портала", ""])
    for step, result in pairs:
        lines.extend([f"### {step.title}", ""])
        lines.extend(f"- [ ] {check}" for check in dict.fromkeys(
            [*step.checks, *result.run.response.review_reasons],
        ))
        lines.append("")
    lines.extend(["## Телеметрия и стоимость расчёта", ""])
    for step, result in pairs:
        telemetry = result.record.telemetry
        codes = ", ".join(result.record.review_codes)
        lines.append(
            f"- **{step.title}** · {telemetry.mode} · {telemetry.duration_ms} мс · "
            f"{format_cost(telemetry.cost_rub_e8)} · ID `{result.record.request_id}`"
            + (f" · {codes}" if codes else "")
        )
        for name, stage in (
            ("Маршрутизация", telemetry.route), ("Рекомендация", telemetry.recommend),
            ("Поисковый эмбеддинг", telemetry.embedding),
        ):
            if stage is not None:
                incoming = stage.tokens_in if stage.tokens_in is not None else "—"
                outgoing = stage.tokens_out if stage.tokens_out is not None else "—"
                lines.append(
                    f"  - {name} · {stage.mode} · токены вход/выход {incoming}/{outgoing} · "
                    f"{format_cost(stage.cost_rub_e8)}"
                )
    if summary.unknown_cost_count:
        known = format_cost(summary.known_cost_rub_e8) if summary.unknown_cost_count < summary.total else "нет"
        lines.extend(["", f"Известный расход — {known}. Шагов без оценки — {summary.unknown_cost_count}."])
    else:
        lines.extend(["", f"Суммарный расход — {format_cost(summary.known_cost_rub_e8)}."])
    return "\n".join(lines) + "\n"


def discover_snapshots(local_dir: Path, tenant_id: str) -> dict[str, PortalSnapshot]:
    snapshots = {}
    for path in sorted(local_dir.glob("*.json")):
        try:
            if not path.resolve().is_relative_to(local_dir.resolve()):
                continue
            snapshot = PortalSnapshot.model_validate_json(path.read_bytes())
        except (OSError, ValueError):
            continue
        if snapshot.metadata is not None and snapshot.scope.tenant_id == tenant_id:
            snapshots[path.name] = snapshot
    return snapshots


def selected_scope(tenant_id: str, snapshot: PortalSnapshot | None) -> PortalScope:
    if snapshot is None:
        return PortalScope(tenant_id=tenant_id, portal_id="documentation.invalid")
    if snapshot.scope.tenant_id != tenant_id:
        raise ValueError("snapshot tenant mismatch")
    return snapshot.scope


def build_engine(
    db_path: Path, settings: Settings, prices_path: Path | None = None,
) -> AdvisorEngine:
    prices = (
        TypeAdapter(dict[str, Price]).validate_json(prices_path.read_bytes())
        if prices_path is not None else {}
    )
    gateway = LLMGateway(settings.models, prices=prices)
    knowledge = KnowledgeBase(
        SnapshotStore(db_path), gateway, max_age_days=settings.docs_max_age_days,
    )
    return AdvisorEngine(Advisor(gateway, knowledge, PromptRegistry()))


def import_corpus(knowledge: KnowledgeBase, scope: PortalScope, path: Path) -> int:
    fragments = load_fragments(path.read_text(encoding="utf-8"))
    knowledge.replace(scope, fragments, dry_run=True)
    return len(fragments)


def run_consultation(
    engine: AdvisorEngine, tenant_id: str, question: str, snapshot: PortalSnapshot | None,
    *, dry_run: bool, approved_for_provider: bool = False,
) -> tuple[AdvisorRun, RunRecord]:
    request = AdvisorRequest(
        scope=selected_scope(tenant_id, snapshot), request_id=uuid4(), question=question,
        snapshot_id=snapshot.snapshot_id if snapshot is not None else None,
    )
    if snapshot is not None:
        engine.store.ensure_saved(snapshot)
    result = engine.run(
        request, dry_run=dry_run,
        public_question=request.question if approved_for_provider and not dry_run else None,
    )
    record = engine.history.get(request.scope, request.request_id)
    if record is None:
        raise RuntimeError("missing advisor run history")
    return result, record


def verdict(record: RunOutcome) -> str:
    if record.conditional_native:
        return "Условный штатный план"
    return {
        "native": "Штатное решение",
        "extension": "Требуется расширение/API",
        "undetermined": "Нужна ручная проверка",
    }[record.tier]


def format_cost(cost_rub_e8: int | None) -> str:
    if cost_rub_e8 is None:
        return "Неизвестно"
    rubles = f"{Decimal(cost_rub_e8).scaleb(-8):.8f}".rstrip("0").rstrip(".")
    return f"{rubles.replace('.', ',')} ₽"


def model_status(models: ModelSettings) -> dict[str, str]:
    return {
        "FAST_LLM_MODEL": models.fast_model or "Не задана",
        "SMART_LLM_MODEL": models.smart_model or "Не задана",
        "Ключ API": "Задан" if models.api_key and models.api_key.get_secret_value() else "Не задан",
    }
