import os
from hashlib import sha256
from pathlib import Path
from sqlite3 import Error as SQLiteError
from typing import Literal

import streamlit as st

from apps.advisor_ui_support import (
    DEMO_CASES,
    NO_SNAPSHOT,
    build_engine,
    case_report,
    discover_snapshots,
    format_cost,
    import_corpus,
    load_case_packs,
    model_status,
    run_case_pack,
    run_consultation,
    selected_scope,
    verdict,
)
from core.advisor_history import AdvisorHistory, RunRecord, RunSummary
from core.config import load_settings
from core.contracts import PortalScope, PortalSnapshot, Tier
from modules.advisor.engine import AdvisorEngine
from modules.advisor.service import AdvisorRun

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
CORPUS = ROOT / "docs/knowledge/bitrix24.json"
CASES = ROOT / "docs/cases"
DRY = "Сухой прогон без траты токенов"
LIVE = "Живой провайдер LLM"
MODE_LABELS = {"llm": "llm", "fallback": "fallback", "deterministic": "Локально"}
BADGE_COLORS: dict[Tier, Literal["green", "orange", "gray"]] = {
    "native": "green", "extension": "orange", "undetermined": "gray",
}


def unapprove_question() -> None:
    st.session_state.provider_approved = False


def select_demo(question: str) -> None:
    st.session_state.question = question
    unapprove_question()


def render_result(run: AdvisorRun, record: RunRecord) -> None:
    response = run.response
    with st.container(border=True):
        st.badge(verdict(record), color=BADGE_COLORS[response.tier])
        st.text(response.recommendation)
        if response.review_reasons or run.review_codes:
            st.subheader("Что проверить вручную")
            for reason in response.review_reasons:
                st.text(f"• {reason}")
            if run.review_codes:
                st.caption("Коды проверки · " + ", ".join(run.review_codes))
        st.subheader("Использованная справка")
        for source in response.documentation:
            st.link_button(str(source.title or source.url), str(source.url))
            st.caption(f"{source.url} · Проверено {source.checked_at:%d.%m.%Y} UTC")
        if not response.documentation:
            st.caption("В итоговой рекомендации нет ссылок на справку.")
        mode, latency, cost = st.columns(3)
        mode.metric("Выполнение", MODE_LABELS[record.telemetry.mode])
        latency.metric("Задержка", f"{record.telemetry.duration_ms} мс")
        cost.metric("Расход", format_cost(record.telemetry.cost_rub_e8))


def render_history(history: AdvisorHistory, scope: PortalScope) -> None:
    st.subheader("Завершённые запросы")
    st.caption(f"Область истории · {scope.tenant_id} / {scope.portal_id}")
    try:
        records = history.list_completed(scope)
        summary = RunSummary.from_records(records)
    except (SQLiteError, ValueError):
        st.error("Не удалось прочитать историю запусков.")
        return
    total, share, cost, duration = st.columns(4)
    total.metric("Всего запросов", summary.total)
    share.metric(
        "Штатные и условные",
        f"{summary.native / summary.total:.1%}".replace(".", ",") if summary.total else "—",
    )
    if summary.total:
        share.caption(f"Условных штатных планов · {summary.conditional_native}")
    cost.metric(
        "Суммарный расход" if not summary.unknown_cost_count else "Известный расход",
        format_cost(summary.known_cost_rub_e8)
        if summary.total > summary.unknown_cost_count or not summary.total else "Неизвестно",
    )
    if summary.unknown_cost_count:
        cost.caption(f"Без оценки стоимости · {summary.unknown_cost_count}")
    duration.metric(
        "Средняя задержка",
        f"{summary.mean_duration_ms:.1f} мс".replace(".", ",")
        if summary.mean_duration_ms is not None else "—",
    )
    if records:
        st.dataframe([
            {
                "Завершён (UTC)": record.telemetry.completed_at.strftime("%d.%m.%Y %H:%M:%S"),
                "Вердикт": verdict(record),
                "Режим": MODE_LABELS[record.telemetry.mode],
                "Задержка, мс": record.telemetry.duration_ms,
                "Расход": format_cost(record.telemetry.cost_rub_e8),
                "ID запроса": str(record.request_id),
            }
            for record in records
        ], hide_index=True)
    else:
        st.info("В выбранной области пока нет завершённых запросов.")


def render_case_packs(
    engine: AdvisorEngine, scope: PortalScope, snapshot: PortalSnapshot | None,
    *, dry_run: bool, provider_ready: bool,
) -> None:
    try:
        cases, fragments = load_case_packs(CASES, CORPUS)
    except (OSError, ValueError):
        st.error("Проверьте схему кейс-паков и ссылки на фрагменты корпуса.")
        return
    if not cases:
        st.info("В каталоге docs/cases пока нет кейс-паков.")
        return
    by_label = {f"Кейс {int(case.case_id[:2])} · {case.title}": case for case in cases}
    selection = st.selectbox("Кейс-пак", tuple(by_label), index=0)
    if selection is None:
        return
    case = by_label[selection]
    st.subheader(case.title)
    st.caption(case.business_area)
    st.write(case.pain)
    with st.expander("Паспорт кейса"):
        st.write("Дорогие ошибки")
        for mistake in case.costly_mistakes:
            st.write(f"- {mistake}")
        for step in case.steps:
            st.markdown(f"**{step.title}** · {', '.join(sorted(step.areas))}")
            st.write(step.question)

    fingerprint = sha256(case.model_dump_json().encode()).hexdigest()
    context = (scope, snapshot.snapshot_id if snapshot else None, dry_run, fingerprint)
    if st.session_state.get("case_context") != context:
        st.session_state.case_context = context
        st.session_state.pop("case_provider_approved", None)
        st.session_state.pop("case_result", None)
    approved = False
    if not dry_run:
        if not provider_ready:
            st.warning("Задайте LLM_BASE_URL, FAST_LLM_MODEL и SMART_LLM_MODEL.")
        approved = st.checkbox(
            f"Разрешаю сетевой расчёт и расход по всем шагам ({len(case.steps)}). "
            "Итоговый расход зависит от тарифа и числа токенов.",
            key="case_provider_approved",
        )
    if st.button(
        "Запустить разбор кейса", type="primary",
        disabled=not dry_run and not (provider_ready and approved),
    ):
        st.session_state.pop("case_result", None)
        try:
            with st.spinner("Выполняем сквозной разбор…"):
                results = run_case_pack(
                    engine, scope.tenant_id, case, snapshot,
                    dry_run=dry_run, approved_for_provider=approved,
                )
                st.session_state.case_result = case_report(case, results, fragments)
        except (OSError, SQLiteError, ValueError, RuntimeError):
            st.error("Разбор прерван. Перед повторным запуском проверьте завершённые шаги и расход в истории.")
    report = st.session_state.get("case_result")
    if report:
        st.subheader("Итоговая карта решений")
        st.download_button(
            "Скачать Markdown-отчёт", report, file_name=f"{case.case_id}.md", mime="text/markdown",
        )
        st.markdown(report)


def main() -> None:
    st.set_page_config(page_title="Консультант Битрикс24", layout="wide")
    st.title("Консультант Битрикс24")
    try:
        settings = load_settings(ROOT / ".env")
        prices = os.environ.get("B24_UI_PRICES")
        engine = build_engine(
            LOCAL / "advisor_ui.sqlite3", settings, ROOT / prices if prices else None,
        )
    except (OSError, SQLiteError, ValueError, RuntimeError):
        st.error("Проверьте локальную БД, настройки в .env и файл тарифов B24_UI_PRICES.")
        return

    models = settings.models
    snapshots = discover_snapshots(LOCAL, settings.tenant_id)
    with st.sidebar:
        st.header("Настройки")
        dry_run = st.radio("Режим", (DRY, LIVE), on_change=unapprove_question) == DRY
        selection = st.selectbox("Слепок портала", (NO_SNAPSHOT, *snapshots))
        snapshot = snapshots.get(selection) if selection else None
        scope = selected_scope(settings.tenant_id, snapshot)
        if snapshot:
            st.caption(f"{scope.portal_id} · Слепок от {snapshot.captured_at:%d.%m.%Y} UTC")
        for name, status in model_status(models).items():
            st.text(f"{name} · {status}")
        st.subheader("Корпус справки")
        st.caption("Импортируйте корпус для выбранной области перед первым запросом.")
        if st.button("Загрузить / обновить корпус"):
            try:
                count = import_corpus(engine.advisor.knowledge, scope, CORPUS)
            except (OSError, SQLiteError, ValueError, RuntimeError):
                st.error("Не удалось загрузить локальный корпус.")
            else:
                st.success(f"Загружено фрагментов · {count}")

    advisor_tab, cases_tab, history_tab = st.tabs((
        "Консультант Битрикс24", "Прикладные кейс-паки", "История и экономика",
    ))
    provider_ready = bool(models.base_url and models.fast_model and models.smart_model)
    with advisor_tab:
        st.subheader("Быстрый старт")
        for column, (label, question) in zip(st.columns(4), DEMO_CASES, strict=True):
            column.button(label, on_click=select_demo, args=(question,), width="stretch")
        question = st.text_area(
            "Задача клиента", key="question", max_chars=4000, height=145,
            on_change=unapprove_question,
            placeholder="Опишите, что нужно настроить в Битрикс24",
        )
        approved = False
        if not dry_run:
            if not provider_ready:
                st.warning("Задайте LLM_BASE_URL, FAST_LLM_MODEL и SMART_LLM_MODEL.")
            approved = st.checkbox(
                "В задаче нет секретов, персональных данных и содержимого карточек. "
                "Разрешаю отправить этот текст провайдеру LLM.",
                key="provider_approved",
            )
        if st.button(
            "Подобрать штатное решение", type="primary",
            disabled=not question.strip() or (not dry_run and not (provider_ready and approved)),
        ):
            st.session_state.pop("consultation", None)
            try:
                with st.spinner("Подбираем решение…"):
                    run, record = run_consultation(
                        engine, settings.tenant_id, question, snapshot,
                        dry_run=dry_run, approved_for_provider=approved,
                    )
            except (OSError, SQLiteError, ValueError, RuntimeError):
                st.error("Не удалось завершить запрос. Проверьте настройки, слепок и локальную БД.")
            else:
                st.session_state.consultation = (selection, dry_run, question, run, record)
        previous = st.session_state.get("consultation")
        if previous is not None and previous[:3] == (selection, dry_run, question):
            run, record = previous[3:]
            if record.scope == scope:
                render_result(run, record)
    with cases_tab:
        render_case_packs(engine, scope, snapshot, dry_run=dry_run, provider_ready=provider_ready)
    with history_tab:
        render_history(engine.history, scope)


if __name__ == "__main__":
    main()
