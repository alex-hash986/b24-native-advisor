import re
from datetime import UTC, datetime
from typing import Annotated

from pydantic import Field

from core.advisor_history import ReviewCode
from core.contracts import (
    AdvisorRequest,
    AdvisorResponse,
    Area,
    Contract,
    NonnegativeId,
    Observation,
    PortalSnapshot,
    Text,
    Tier,
)
from core.knowledge import Fragment, KnowledgeBase, terms
from core.llm import Execution, LLMGateway
from core.prompts import ACTIVE_VERSIONS, PromptRegistry
from modules.portal_audit.client import READ_METHODS
from modules.portal_audit.collector import CRM_ENTITIES

AREA_TERMS: dict[Area, str] = {
    "crm": "crm срм сделк лид ворон пол стад контакт компани",
    "robots": "робот триггер",
    "workflows": "бизнес согласован",
    "smart_processes": "смарт",
    "permissions": "прав доступ рол",
    "analytics": "аналитик отчет bi",
}
MAX_FRAGMENTS = 12
MAX_OBSERVATIONS = 50
CRM_ENTITY_NAMES = {
    1: ("lead", r"лид(?:а|у|ом|е|ы|ов|ам|ами|ах)?"),
    2: ("deal", r"сдел(?:ка|ки|ке|ку|кой|ок|кам|ками|ках)"),
    3: ("contact", r"контакт(?:а|у|ом|е|ы|ов|ам|ами|ах)?"),
    4: ("company", r"компани[яиюей]|компаний|компаниям[и]?|компаниях"),
    7: ("quote", r"предложени[еяюим]|предложений|предложениями|предложениях"),
    31: ("smart_invoice", r"счет(?:а|у|ом|е|ов|ам|ами|ах)?"),
}
SENSITIVE_INPUT = re.compile(r"/rest/|(?:token|api[_ -]?key|password|пароль)\s*[:=]", re.I)
INLINE_LINK = re.compile(r"https?://|www\.|\]\(", re.I)
STOP_CODES = {"unknown_area", "unsafe_input", "missing_sources", "conflicting_sources"}
UI_CHECKS: dict[Area, str] = {
    "crm": "Проверьте в CRM целевую сущность, воронку, стадии и параметры полей или формы по задаче.",
    "robots": "Проверьте роботов и триггеры на целевой стадии, их условия и порядок выполнения.",
    "workflows": "Проверьте шаблон бизнес-процесса, условия запуска и участников.",
    "smart_processes": "Проверьте параметры типа смарт-процесса, его поля, стадии и связи.",
    "permissions": "Проверьте роли и права на нужную сущность и воронку под учётной записью сотрудника.",
    "analytics": "Проверьте источник данных отчёта, фильтры и доступ пользователей.",
}
REVIEW_MESSAGES: dict[ReviewCode, str] = {
    "unknown_area": "Уточните область аудита.",
    "unsafe_input": "Для вызова модели нужна обезличенная формулировка без ссылок и секретов.",
    "missing_snapshot": "Нужен слепок указанного портала и тенанта.",
    "visibility_gap": "Слепок содержит пробелы в необходимых областях аудита.",
    "conflicting_sources": "Свежие источники противоречат друг другу.",
    "missing_sources": "Нужны свежие источники по всем областям с подтверждённой применимостью.",
    "invalid_evidence": "Ответ ссылается на доказательства вне переданного набора.",
    "unsupported_conclusion": "Вывод не подтверждён переданными источниками и наблюдениями.",
    "model_review": "Требуется проверить вывод по доказательствам.",
}


class Route(Contract):
    task: Text
    success_criterion: Text | None
    areas: list[Area] = Field(max_length=6)
    clarification_questions: list[Text]


class Draft(Contract):
    tier: Tier
    recommendation: Annotated[Text, Field(max_length=8000)]
    native_gap: Text | None
    fragment_ids: list[Text]
    observation_ids: list[NonnegativeId]
    review_reasons: list[Text]


class AdvisorRun(Contract):
    response: AdvisorResponse
    prompt_versions: dict[str, Text]
    route: Execution
    recommend: Execution
    embedding: Execution | None = None
    review_codes: list[ReviewCode] = Field(default_factory=list)


def local_route(question: str) -> Route:
    words = terms(question)
    return Route(
        task=question,
        success_criterion=None,
        areas=[area for area, stems in AREA_TERMS.items()
               if any(word.startswith(stem) for word in words for stem in stems.split())],
        clarification_questions=[],
    )


def unresolved(reasons: list[str]) -> Draft:
    return Draft(
        tier="undetermined",
        recommendation="Уточните данные и подтвердите применимость штатного варианта.",
        native_gap=None, fragment_ids=[], observation_ids=[], review_reasons=reasons,
    )


def conflicting_claims(fragments: list[Fragment]) -> bool:
    claims = set()
    keys = set()
    for fragment in fragments:
        if fragment.claim_key is not None and fragment.claim_value is not None:
            keys.add(fragment.claim_key)
            claims.add((fragment.claim_key, fragment.claim_value))
    return len(claims) > len(keys)


def crm_targets(question: str, entity: str | None) -> set[int]:
    words = terms(question)
    targets = {
        entity_id for entity_id, (alias, pattern) in CRM_ENTITY_NAMES.items()
        if any(word in (alias, alias + "s") or re.fullmatch(pattern, word) for word in words)
    }
    if entity is not None:
        name = entity.strip().casefold()
        confirmed = {
            entity_id for entity_id, (alias, _) in CRM_ENTITY_NAMES.items()
            if name in (alias, str(entity_id))
        }
        if not confirmed:
            return set()
        targets.update(confirmed)
    return targets


def crm_observation_entity(item: Observation) -> int | None:
    if item.area != "crm":
        return None
    method, _, parameter = item.source_ref.partition("/")
    if method in ("crm.item.fields", "crm.category.list") and re.fullmatch(r"[0-9]{1,18}", parameter):
        entity_id = int(parameter)
        return entity_id if entity_id in CRM_ENTITIES else None
    if method == "crm.status.list":
        if parameter == "STATUS":
            return 1
        if parameter == "QUOTE_STATUS":
            return 7
        if re.fullmatch(r"DEAL_STAGE(?:_[0-9]+)?", parameter):
            return 2
        if re.fullmatch(r"SMART_INVOICE_STAGE_[0-9]+", parameter):
            return 31
    return None


def public_observation(
    index: int, item: Observation, snapshot: PortalSnapshot | None,
) -> dict[str, object]:
    fact: dict[str, object] = {"observation_id": index, "area": item.area, "status": item.status}
    method, _, parameter = item.source_ref.partition("/")
    if method not in READ_METHODS:
        return fact
    fact["method"] = method
    metadata = snapshot.metadata if snapshot is not None else None
    if metadata is None:
        return fact
    if method == "crm.type.list":
        fact["visible_types"] = [
            {"entity_type_id": row.entity_type_id, "categories_enabled": row.categories_enabled,
             "stages_enabled": row.stages_enabled}
            for row in metadata.smart_processes[:50]
        ]
    elif method in ("crm.item.fields", "crm.category.list") and re.fullmatch(r"[0-9]{1,18}", parameter):
        entity_id = int(parameter)
        rows = metadata.fields if method == "crm.item.fields" else metadata.categories
        fact["entity_type_id"] = entity_id
        fact["visible_count"] = sum(row.entity_type_id == entity_id for row in rows)
    elif method == "crm.status.list":
        fact["visible_count"] = sum(row.directory == parameter for row in metadata.stages)
    elif method == "method.get" and parameter in READ_METHODS:
        fact["checked_method"] = parameter
        fact["visibility"] = [
            {"exists": row.exists, "available": row.available}
            for row in metadata.methods if row.method == parameter
        ]
    return fact


class Advisor:
    def __init__(self, gateway: LLMGateway, knowledge: KnowledgeBase, prompts: PromptRegistry):
        self.gateway = gateway
        self.knowledge = knowledge
        self.prompts = prompts

    def run(
        self, request: AdvisorRequest, snapshot: PortalSnapshot | None = None, *,
        dry_run: bool = False, now: datetime | None = None,
        public_question: str | None = None, entity: str | None = None, tariff: str | None = None,
    ) -> AdvisorRun:
        # Remote input is prepared by the application. Only aggregate observation facts travel.
        offline = (
            dry_run or public_question is None
            or bool(INLINE_LINK.search(public_question) or SENSITIVE_INPUT.search(public_question))
        )
        question = request.question
        if public_question is not None and not offline:
            question = AdvisorRequest.model_validate({
                **request.model_dump(), "question": public_question,
            }).question

        route, route_execution = self.gateway.structured(
            "fast", self.prompts.render("advisor.route"), {"question": question},
            Route, lambda: local_route(question), dry_run=offline,
        )
        areas = set(route.areas) | set(local_route(question).areas)
        review_codes: list[ReviewCode] = []
        if not areas:
            review_codes.append("unknown_area")
        if offline and not dry_run:
            review_codes.append("unsafe_input")
        if snapshot is not None:
            snapshot = PortalSnapshot.model_validate_json(snapshot.model_dump_json())
        if snapshot is None or snapshot.scope != request.scope or (
            request.snapshot_id is not None and snapshot.snapshot_id != request.snapshot_id
        ):
            snapshot = None
            review_codes.append("missing_snapshot")
        observations: list[Observation] = []
        unchecked_areas = set(areas) if snapshot is None else set()
        if snapshot is not None:
            relevant = [
                (item, crm_observation_entity(item)) for item in snapshot.observations
                if item.area in areas
            ]
            targets = crm_targets(question, entity) if "crm" in areas else set()
            # Legacy snapshots can have area-level observations only.
            if not any(entity_id is not None for _, entity_id in relevant):
                targets.clear()
            if targets:
                relevant = [(item, entity_id) for item, entity_id in relevant
                            if entity_id is None or entity_id in targets]
            observed = [(item, entity_id) for item, entity_id in relevant
                        if item.status == "observed"][:MAX_OBSERVATIONS]
            observations = [item for item, _ in observed]
            unchecked_areas = areas - {item.area for item in observations}
            unchecked_areas.update(item.area for item, _ in relevant if item.status != "observed")
            if targets - {entity_id for _, entity_id in observed}:
                unchecked_areas.add("crm")
            if unchecked_areas:
                review_codes.append("visibility_gap")

        fragments, embedding = self.knowledge.search(
            request.scope, question, sorted(areas), now=now or datetime.now(UTC),
            dry_run=offline, entity=entity, tariff=tariff,
        )
        fragments = [
            item for item in fragments
            if (entity is not None or not item.entities)
            and (tariff is not None or not item.tariffs)
        ]
        if conflicting_claims(fragments):
            review_codes.append("conflicting_sources")
        fragments = fragments[:MAX_FRAGMENTS]
        if not fragments or areas - {area for item in fragments for area in item.areas}:
            review_codes.append("missing_sources")

        conditional = "missing_snapshot" in review_codes or "visibility_gap" in review_codes
        manual_checks = [UI_CHECKS[area] for area in sorted(unchecked_areas)]
        draft, recommend_execution = self.gateway.structured(
            "smart", self.prompts.render("advisor.recommend"),
            {
                "question": question,
                "route": route.model_dump(mode="json"),
                "areas": sorted(areas),
                "conditional": conditional,
                "manual_checks": manual_checks,
                "documentation": [item.model_dump(mode="json") for item in fragments],
                "observations": [
                    public_observation(index, item, snapshot)
                    for index, item in enumerate(observations)
                ],
            },
            Draft, lambda: unresolved([REVIEW_MESSAGES[code] for code in review_codes or ["model_review"]]),
            dry_run=offline or bool(STOP_CODES.intersection(review_codes)),
        )
        response, binding_code = self._bind(
            request, snapshot, observations, fragments, areas, draft, manual_checks,
        )
        if binding_code is not None:
            review_codes.append(binding_code)
        elif response.need_human and not review_codes:
            review_codes.append("model_review")
        return AdvisorRun(
            response=response, route=route_execution, recommend=recommend_execution,
            embedding=embedding,
            review_codes=review_codes,
            prompt_versions={prompt_id: version for prompt_id, version in ACTIVE_VERSIONS.items()},
        )

    @staticmethod
    def _bind(
        request: AdvisorRequest, snapshot: PortalSnapshot | None, observations: list[Observation],
        fragments: list[Fragment], areas: set[Area], draft: Draft, manual_checks: list[str],
    ) -> tuple[AdvisorResponse, ReviewCode | None]:
        by_id = {item.fragment_id: item for item in fragments}
        cited_docs: list[Fragment] = []
        cited_observations: list[Observation] = []
        binding_code: ReviewCode | None = None
        if any(key not in by_id for key in draft.fragment_ids) or any(
            not 0 <= key < len(observations) for key in draft.observation_ids
        ):
            binding_code = "invalid_evidence"
            draft = unresolved([REVIEW_MESSAGES[binding_code]])
        elif draft.tier != "undetermined":
            cited_docs = [by_id[key] for key in dict.fromkeys(draft.fragment_ids)]
            cited_observations = [observations[key] for key in dict.fromkeys(draft.observation_ids)]
            if (
                not cited_docs
                or (not manual_checks and (not cited_observations or draft.review_reasons))
                or any(INLINE_LINK.search(reason) for reason in draft.review_reasons)
                or INLINE_LINK.search(draft.recommendation)
                or areas - {area for item in cited_docs for area in item.areas}
                or {item.area for item in observations} - {item.area for item in cited_observations}
                or (draft.tier == "native" and (
                    draft.native_gap is not None
                    or areas - {area for item in cited_docs
                                if item.url.host == "helpdesk.bitrix24.ru" for area in item.areas}
                ))
                or (draft.tier == "extension" and (
                    draft.native_gap is None
                    or not any(item.native_gap == draft.native_gap for item in cited_docs)
                ))
            ):
                binding_code = "unsupported_conclusion"
                draft = unresolved([REVIEW_MESSAGES[binding_code]])
        if draft.tier == "undetermined":
            cited_docs, cited_observations = [], []
            draft = unresolved([
                reason for reason in draft.review_reasons if not INLINE_LINK.search(reason)
            ] or ["Недостаточно подтверждений для вывода."])
        review_reasons = list(dict.fromkeys([*draft.review_reasons, *manual_checks]))
        prefix = ""
        if snapshot is None:
            prefix = "Состояние портала не проверено. Выводы о его настройках условны. "
        elif manual_checks and draft.tier != "undetermined":
            prefix = "Слепок неполон. Выводы о непроверенных настройках портала условны. "
        return AdvisorResponse(
            request_id=request.request_id, scope=request.scope,
            snapshot_id=snapshot.snapshot_id if snapshot is not None else None,
            tier=draft.tier, recommendation=prefix + draft.recommendation, native_gap=draft.native_gap,
            documentation=[item.evidence() for item in cited_docs], observations=cited_observations,
            need_human=bool(review_reasons), review_reasons=review_reasons,
        ), binding_code
