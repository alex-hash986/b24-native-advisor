import json
import math
import re
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime, timedelta
from typing import Annotated, Self

from pydantic import (
    AwareDatetime,
    Field,
    HttpUrl,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from core.contracts import Area, Contract, DocumentationEvidence, PortalScope, Text
from core.llm import Embedding, Execution, LLMGateway, Vector
from core.storage import SnapshotStore

INDEX_FORMAT = "1"
OFFICIAL_HOSTS = {"helpdesk.bitrix24.ru", "apidocs.bitrix24.ru"}


class Fragment(Contract):
    fragment_id: Text
    url: HttpUrl
    checked_at: AwareDatetime
    revision_or_hash: Text
    snippet: Annotated[Text, Field(max_length=2500)]
    areas: list[Area] = Field(min_length=1)
    sections: list[Text] = Field(min_length=1)
    entities: list[Text]
    tariffs: list[Text]
    claim_key: Text | None = None
    claim_value: Text | None = None
    native_gap: Text | None = None

    @model_validator(mode="after")
    def normalize_source(self) -> Self:
        self.checked_at = self.checked_at.astimezone(UTC)
        if (
            self.url.scheme != "https" or self.url.host not in OFFICIAL_HOSTS
            or self.url.port != 443 or self.url.username or self.url.password
            or self.url.query is not None or self.url.fragment is not None
            or not self.url.path or self.url.path == "/"
        ):
            raise ValueError("expected a public official Bitrix24 source")
        if (self.claim_key is None) != (self.claim_value is None):
            raise ValueError("claim key and value must be supplied together")
        return self

    def evidence(self) -> DocumentationEvidence:
        return DocumentationEvidence(
            title=self.sections[0], url=self.url, checked_at=self.checked_at,
            revision=self.revision_or_hash, excerpt=self.snippet,
        )


def load_fragments(payload: str) -> list[Fragment]:
    return TypeAdapter(list[Fragment]).validate_json(payload)


def terms(text: str) -> set[str]:
    return set(re.findall(r"\w{3,}", text.casefold().replace("ё", "е")))


def searchable(fragment: Fragment) -> str:
    return " ".join([fragment.snippet, *fragment.sections, *fragment.entities, *fragment.tariffs])


def cosine(left: Vector, right: Vector) -> float | None:
    if len(left) != len(right):
        return None
    left_norm, right_norm = math.hypot(*left), math.hypot(*right)
    if not (0 < left_norm < math.inf and 0 < right_norm < math.inf):
        return None
    return math.fsum(a / left_norm * (b / right_norm) for a, b in zip(left, right))


class KnowledgeBase:
    def __init__(
        self, store: SnapshotStore, gateway: LLMGateway, *, max_age_days: int = 30,
    ) -> None:
        if max_age_days < 1:
            raise ValueError("freshness threshold must be positive")
        self.store = store
        self.gateway = gateway
        self.max_age = timedelta(days=max_age_days)

    def _revision(self) -> str:
        settings = self.gateway.settings
        return json.dumps([INDEX_FORMAT, str(settings.base_url), settings.embedding_model])

    def replace(
        self, scope: PortalScope, fragments: Sequence[Fragment], *, dry_run: bool = False,
    ) -> list[Execution]:
        checked = load_fragments(json.dumps([item.model_dump(mode="json") for item in fragments]))
        if len({fragment.fragment_id for fragment in checked}) != len(checked):
            raise ValueError("duplicate documentation fragment ID")
        scope_key = (scope.tenant_id, scope.portal_id)
        revision = self._revision()
        rows = []
        executions = []
        for fragment in checked:
            vector, execution = self.gateway.embed(searchable(fragment), dry_run=dry_run)
            executions.append(execution)
            rows.append((*scope_key, fragment.fragment_id, fragment.model_dump_json(),
                         revision, json.dumps(vector) if vector is not None else None))
        with closing(self.store.connect()) as connection, connection:
            connection.execute(
                "DELETE FROM knowledge_fragments WHERE tenant_id = ? AND portal_id = ?", scope_key
            )
            connection.executemany(
                """INSERT INTO knowledge_fragments
                   (tenant_id, portal_id, fragment_id, payload_json, index_revision, vector_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return executions

    def search(
        self, scope: PortalScope, query: str, areas: Sequence[Area], *, now: datetime,
        dry_run: bool = False, entity: str | None = None, tariff: str | None = None,
    ) -> tuple[list[Fragment], Execution | None]:
        if now.utcoffset() is None:
            raise ValueError("search requires timezone-aware time")
        with closing(self.store.connect()) as connection:
            rows = connection.execute(
                """SELECT fragment_id, payload_json, index_revision, vector_json
                   FROM knowledge_fragments WHERE tenant_id = ? AND portal_id = ?
                   ORDER BY fragment_id""",
                (scope.tenant_id, scope.portal_id),
            ).fetchall()
        eligible = []
        revision = self._revision()
        for fragment_id, payload, indexed_as, vector_json in rows:
            fragment = Fragment.model_validate_json(payload)
            if fragment.fragment_id != fragment_id:
                raise ValueError("documentation row metadata mismatch")
            if not (timedelta(0) <= now - fragment.checked_at <= self.max_age):
                continue
            if not set(areas).intersection(fragment.areas):
                continue
            if entity and fragment.entities and entity not in fragment.entities:
                continue
            if tariff and fragment.tariffs and tariff not in fragment.tariffs:
                continue
            eligible.append((fragment, vector_json if indexed_as == revision else None))
        vector, execution = None, None
        if any(stored for _, stored in eligible):
            vector, execution = self.gateway.embed(query, dry_run=dry_run)
        query_terms = terms(query)
        ranked = []
        for fragment, stored in eligible:
            lexical = len(query_terms & terms(searchable(fragment)))
            similarity = None
            if vector is not None and stored is not None:
                try:
                    embedded = Embedding.model_validate_json('{"vector":' + stored + '}', strict=True)
                    similarity = cosine(vector, embedded.vector)
                except ValidationError:
                    pass
            if lexical or (similarity is not None and similarity > 0):
                ranked.append((lexical, similarity or 0.0, fragment))
        ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        matches = [fragment for _, _, fragment in ranked]
        matched_claims = {item.claim_key for item in matches if item.claim_key is not None}
        matched_ids = {item.fragment_id for item in matches}
        matches.extend(
            item for item, _ in eligible
            if item.claim_key in matched_claims and item.fragment_id not in matched_ids
        )
        return matches, execution
