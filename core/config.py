import re
from ipaddress import ip_address
from os import environ
from pathlib import Path
from typing import Annotated, Self
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import Field, HttpUrl, SecretStr, ValidationError, model_validator

from core.contracts import Contract, Text

WEBHOOK_PATH = re.compile(r"/rest/[1-9][0-9]*/[A-Za-z0-9_-]+/?")


class WebhookEndpoint(Contract):
    url: SecretStr
    pinned_ip: str | None = None

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
        value = self.url.get_secret_value()
        parts = urlsplit(value)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", parts.hostname)
            or parts.netloc.lower() not in {parts.hostname, f"{parts.hostname}:443"}
            or not WEBHOOK_PATH.fullmatch(parts.path)
            or "?" in value
            or "#" in value
            or any(c.isspace() for c in value)
        ):
            raise ValueError("invalid HTTPS Bitrix24 webhook")
        if self.pinned_ip is not None:
            self.pinned_ip = str(ip_address(self.pinned_ip))
        return self

    @property
    def hostname(self) -> str:
        hostname = urlsplit(self.url.get_secret_value()).hostname
        assert hostname is not None
        return hostname


class ModelSettings(Contract):
    base_url: HttpUrl | None = None
    api_key: SecretStr | None = None
    fast_model: Text | None = None
    smart_model: Text | None = None
    embedding_model: Text | None = None
    timeout_seconds: Annotated[float, Field(gt=0, le=120)] = 30

    @model_validator(mode="after")
    def validate_url(self) -> Self:
        if self.base_url is not None:
            url = self.base_url
            if url.username or url.password or url.query is not None or url.fragment is not None:
                raise ValueError("model URL must not contain credentials, query or fragment")
            if url.scheme != "https" and url.host != "127.0.0.1":
                raise ValueError("model URL requires HTTPS or loopback")
        return self


class Settings(Contract):
    models: ModelSettings = Field(default_factory=ModelSettings)
    docs_max_age_days: Annotated[int, Field(gt=0, le=36500)] = 30
    sqlite_path: Path = Path(".local/advisor.sqlite3")
    tenant_id: Text = "local"
    webhook: WebhookEndpoint | None = None


def load_settings(path: Path = Path(".env")) -> Settings:
    values = dotenv_values(path, interpolate=False) if path.is_file() else {}
    try:
        url = values.get("BITRIX24_WEBHOOK_URL")
        pin = values.get("BITRIX24_PINNED_IP")
        webhook = WebhookEndpoint(url=SecretStr(url), pinned_ip=pin or None) if url else None
        if pin and webhook is None:
            raise ValueError("IP pin requires webhook")
        return Settings(
            sqlite_path=Path(values.get("SQLITE_PATH") or ".local/advisor.sqlite3"),
            tenant_id=values.get("B24_TENANT_ID") or "local",
            webhook=webhook,
            models=ModelSettings.model_validate({
                "base_url": environ.get("LLM_BASE_URL", values.get("LLM_BASE_URL")) or None,
                "api_key": environ.get("LLM_API_KEY", values.get("LLM_API_KEY")) or None,
                "fast_model": environ.get("FAST_LLM_MODEL", values.get("FAST_LLM_MODEL")) or None,
                "smart_model": environ.get("SMART_LLM_MODEL", values.get("SMART_LLM_MODEL")) or None,
                "embedding_model": environ.get("EMBEDDING_MODEL", values.get("EMBEDDING_MODEL")) or None,
            }),
            docs_max_age_days=int(
                environ.get("B24_DOCS_MAX_AGE_DAYS", values.get("B24_DOCS_MAX_AGE_DAYS")) or 30
            ),
        )
    except (ValidationError, ValueError):
        pass
    raise ValueError("invalid snapshot configuration or model settings")
