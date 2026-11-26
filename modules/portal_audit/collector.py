from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from typing import TypeVar
from uuid import uuid4

from core.contracts import (
    Area,
    Category,
    FieldDefinition,
    MethodVisibility,
    Observation,
    ObservationStatus,
    PortalSnapshot,
    SmartProcess,
    SnapshotMetadata,
    SnapshotRequest,
    Stage,
    snapshot_hash,
)
from modules.portal_audit.client import READ_METHODS, ApiError, BitrixReadOnlyClient

T = TypeVar("T")
CRM_ENTITIES = (1, 2, 3, 4, 7, 31)
CRM_CATEGORY_ENTITIES = (2, 3, 4, 31)
MAX_PAGES = 1000


def object_data(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("expected object")
    return dict(value)


def list_data(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected list")
    return list(value)


def integer(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    raise ValueError("expected integer")


def flag(value: object) -> bool:
    if value is True or value == "Y":
        return True
    if value is False or value == "N":
        return False
    raise ValueError("expected flag")


class PortalSnapshotCollector:
    def __init__(self, client: BitrixReadOnlyClient) -> None:
        self.client = client

    def collect(self, request: SnapshotRequest) -> PortalSnapshot:
        metadata = SnapshotMetadata()
        observations: list[Observation] = []

        def fetch(
            area: Area,
            method: str,
            params: dict[str, object],
            source: str,
            normalize: Callable[[object], list[T]],
        ) -> list[T]:
            rows: list[T] = []
            previous_offset = 0
            try:
                for _ in range(MAX_PAGES):
                    result, cursor = self.client.call(method, params)
                    rows.extend(normalize(result))
                    if cursor is None:
                        observations.append(Observation(
                            area=area, status="observed", summary="visible_metadata", source_ref=source,
                        ))
                        return rows
                    offset = integer(cursor)
                    if offset <= previous_offset:
                        raise ValueError("invalid cursor")
                    previous_offset = offset
                    params = {**params, "start": offset}
                raise ApiError("page_limit")
            except ApiError as exc:
                status: ObservationStatus = "no_access" if exc.code == "no_access" else (
                    "unsupported" if exc.code == "unsupported" else "error"
                )
                observations.append(Observation(
                    area=area, status=status, summary=exc.code, source_ref=source,
                ))
            except ValueError:
                observations.append(Observation(
                    area=area, status="error", summary="invalid_response", source_ref=source,
                ))
            return rows

        if "permissions" in request.areas:
            for method in sorted(READ_METHODS):
                metadata.methods.extend(fetch(
                    "permissions", "method.get", {"name": method}, f"method.get/{method}",
                    partial(normalize_method, method),
                ))
            observations.append(Observation(
                area="permissions", status="unsupported", summary="user_permission_matrix_unavailable",
                source_ref="method.get",
            ))

        def fields(area: Area, entity: int) -> None:
            metadata.fields.extend(fetch(
                area, "crm.item.fields", {"entityTypeId": entity}, f"crm.item.fields/{entity}",
                lambda value: normalize_fields(entity, value),
            ))

        def categories(area: Area, entity: int) -> list[Category]:
            result = fetch(
                area, "crm.category.list", {"entityTypeId": entity}, f"crm.category.list/{entity}",
                lambda value: normalize_categories(entity, value),
            )
            metadata.categories.extend(result)
            return result

        def stages(area: Area, directory: str) -> None:
            metadata.stages.extend(fetch(
                area, "crm.status.list", {"filter": {"ENTITY_ID": directory}},
                f"crm.status.list/{directory}", lambda value: normalize_stages(directory, value),
            ))

        if "crm" in request.areas:
            for entity in CRM_ENTITIES:
                fields("crm", entity)
            stages("crm", "STATUS")
            stages("crm", "QUOTE_STATUS")
            for entity in CRM_CATEGORY_ENTITIES:
                for category in categories("crm", entity):
                    category_id = category.category_id
                    if entity == 2:
                        stages("crm", f"DEAL_STAGE_{category_id}" if category_id else "DEAL_STAGE")
                    elif entity == 31:
                        stages("crm", f"SMART_INVOICE_STAGE_{category_id}")

        if "smart_processes" in request.areas:
            metadata.smart_processes = fetch(
                "smart_processes", "crm.type.list", {}, "crm.type.list", normalize_types,
            )
            for process in metadata.smart_processes:
                entity = process.entity_type_id
                fields("smart_processes", entity)
                if process.categories_enabled or process.stages_enabled:
                    for category in categories("smart_processes", entity):
                        if process.stages_enabled:
                            stages("smart_processes", f"DYNAMIC_{entity}_STAGE_{category.category_id}")

        limitations: tuple[tuple[Area, str, str], ...] = (
            ("robots", "app_robot_registry_does_not_show_stage_robots", "bizproc.robot.list"),
            ("workflows", "workflow_inventory_out_of_scope", "portal_audit/workflows"),
            ("analytics", "analytics_inventory_out_of_scope", "portal_audit/analytics"),
        )
        for area, code, source in limitations:
            if area in request.areas:
                observations.append(Observation(
                    area=area, status="unsupported", summary=code, source_ref=source,
                ))

        return PortalSnapshot(
            request_id=request.request_id,
            scope=request.scope,
            snapshot_id=uuid4(),
            captured_at=datetime.now(UTC),
            observations=observations,
            metadata=metadata,
            sha256=snapshot_hash(metadata, observations),
        )


def normalize_method(method: str, value: object) -> list[MethodVisibility]:
    row = object_data(value)
    return [MethodVisibility.model_validate({
        "method": method, "exists": row.get("isExisting"), "available": row.get("isAvailable"),
        "scope": row.get("scope") or None,
    })]


def normalize_types(value: object) -> list[SmartProcess]:
    rows = map(object_data, list_data(object_data(value).get("types")))
    return [SmartProcess.model_validate({
        "entity_type_id": integer(row.get("entityTypeId")), "title": row.get("title"),
        "categories_enabled": flag(row.get("isCategoriesEnabled")),
        "stages_enabled": flag(row.get("isStagesEnabled")),
    }) for row in rows]


def normalize_categories(entity: int, value: object) -> list[Category]:
    rows = map(object_data, list_data(object_data(value).get("categories")))
    return [Category.model_validate({
        "entity_type_id": entity, "category_id": integer(row.get("id")), "name": row.get("name"),
    }) for row in rows]


def normalize_stages(directory: str, value: object) -> list[Stage]:
    stages = []
    for value_row in list_data(value):
        row = object_data(value_row)
        if row.get("ENTITY_ID") != directory:
            raise ValueError("unexpected status directory")
        stages.append(Stage.model_validate({
            "directory": directory, "status_id": row.get("STATUS_ID"),
            "name": row.get("NAME"), "sort": integer(row.get("SORT")),
        }))
    return stages


def normalize_fields(entity: int, value: object) -> list[FieldDefinition]:
    rows = object_data(object_data(value).get("fields"))
    fields = []
    for code, definition in rows.items():
        row = object_data(definition)
        fields.append(FieldDefinition.model_validate({
            "entity_type_id": entity, "code": code, "type": row.get("type"),
            "required": row.get("isRequired"), "multiple": row.get("isMultiple"),
            "read_only": row.get("isReadOnly"),
        }))
    return fields
