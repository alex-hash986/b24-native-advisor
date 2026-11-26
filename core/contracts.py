import hashlib
import json
from datetime import UTC
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)

Text = Annotated[str, Field(strict=True, min_length=1)]
Area = Literal["crm", "robots", "workflows", "smart_processes", "permissions", "analytics"]
Tier = Literal["native", "extension", "undetermined"]
ObservationStatus = Literal["observed", "no_access", "unsupported", "error"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PortalScope(Contract):
    tenant_id: Text
    portal_id: Text


class ScopedRequest(Contract):
    schema_version: Literal["1"] = "1"
    request_id: UUID
    scope: PortalScope


class SnapshotRequest(ScopedRequest):
    areas: set[Area] = Field(min_length=1)


class Observation(Contract):
    area: Area
    status: ObservationStatus
    summary: Text
    source_ref: Text


PositiveId = Annotated[int, Field(strict=True, gt=0)]
NonnegativeId = Annotated[int, Field(strict=True, ge=0)]
StrictFlag = Annotated[bool, Field(strict=True)]


class MethodVisibility(Contract):
    method: Text
    exists: StrictFlag
    available: StrictFlag
    scope: Text | None = None


class SmartProcess(Contract):
    entity_type_id: PositiveId
    title: Text
    categories_enabled: StrictFlag
    stages_enabled: StrictFlag


class Category(Contract):
    entity_type_id: PositiveId
    category_id: NonnegativeId
    name: Text


class Stage(Contract):
    directory: Text
    status_id: Text
    name: Text
    sort: NonnegativeId


class FieldDefinition(Contract):
    entity_type_id: PositiveId
    code: Text
    type: Text
    required: StrictFlag
    multiple: StrictFlag
    read_only: StrictFlag


class SnapshotMetadata(Contract):
    methods: list[MethodVisibility] = Field(default_factory=list)
    smart_processes: list[SmartProcess] = Field(default_factory=list)
    categories: list[Category] = Field(default_factory=list)
    stages: list[Stage] = Field(default_factory=list)
    fields: list[FieldDefinition] = Field(default_factory=list)


class PortalSnapshot(ScopedRequest):
    snapshot_id: UUID
    captured_at: AwareDatetime
    observations: list[Observation] = Field(min_length=1)
    metadata: SnapshotMetadata | None = None
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None

    @model_validator(mode="after")
    def check_content_hash(self) -> Self:
        self.captured_at = self.captured_at.astimezone(UTC)
        if self.metadata is None and self.sha256 is None:
            return self
        if self.metadata is None or self.sha256 != snapshot_hash(self.metadata, self.observations):
            raise ValueError("snapshot content hash mismatch")
        return self


def snapshot_hash(metadata: SnapshotMetadata, observations: list[Observation]) -> str:
    def canonical(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    records = metadata.model_dump(mode="json")
    records["observations"] = [item.model_dump(mode="json") for item in observations]
    for rows in records.values():
        rows.sort(key=canonical)
    return hashlib.sha256(canonical({"version": "1", **records}).encode("utf-8")).hexdigest()


class AdvisorRequest(ScopedRequest):
    question: Text = Field(max_length=4000)
    snapshot_id: UUID | None = None


class DocumentationEvidence(Contract):
    title: Text | None = None
    url: HttpUrl
    checked_at: AwareDatetime
    revision: Text
    excerpt: Text


class AdvisorResponse(ScopedRequest):
    snapshot_id: UUID | None = None
    tier: Tier
    recommendation: Text
    native_gap: Text | None = None
    documentation: list[DocumentationEvidence]
    observations: list[Observation]
    need_human: bool = Field(strict=True)
    review_reasons: list[Text]

    @model_validator(mode="after")
    def check_evidence(self) -> Self:
        if self.need_human != bool(self.review_reasons):
            raise ValueError("need_human must match the presence of review reasons")
        if self.observations and self.snapshot_id is None:
            raise ValueError("observations require a snapshot_id")
        if self.tier != "undetermined" and not self.documentation:
            raise ValueError("a recommendation requires documentation evidence")
        if self.tier == "extension" and not self.native_gap:
            raise ValueError("extension requires a documented native gap")
        if self.tier != "undetermined" and not self.observations and not self.need_human:
            raise ValueError("a recommendation without observations requires review")
        if self.tier == "undetermined" and not self.need_human:
            raise ValueError("an undetermined result requires review")
        return self
