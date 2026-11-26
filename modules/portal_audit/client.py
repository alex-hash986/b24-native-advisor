import json
import socket
import ssl
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from http.client import HTTPException, HTTPSConnection
from threading import Lock
from typing import Protocol
from urllib.parse import urlsplit

from core.config import WebhookEndpoint

READ_METHODS = frozenset({
    "method.get", "crm.type.list", "crm.category.list", "crm.status.list", "crm.item.fields",
})
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class Transport(Protocol):
    def __call__(self, method: str, params: dict[str, object]) -> tuple[int, str | None, bytes]: ...


class PortalLimiter:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.clock = clock
        self.sleep = sleep
        self.lock = Lock()
        self.next_start = 0.0

    def send(
        self, transport: Transport, method: str, params: dict[str, object]
    ) -> tuple[int, str | None, bytes]:
        with self.lock:
            while (delay := self.next_start - self.clock()) > 0:
                self.sleep(delay)
            self.next_start = self.clock() + 0.5
            return transport(method, params)


_limiters: dict[str, PortalLimiter] = {}
_registry_lock = Lock()


def portal_limiter(hostname: str) -> PortalLimiter:
    with _registry_lock:
        if hostname not in _limiters:
            _limiters[hostname] = PortalLimiter()
        return _limiters[hostname]


class PinnedHTTPSConnection(HTTPSConnection):
    def __init__(self, hostname: str, ip: str, context: ssl.SSLContext) -> None:
        super().__init__(hostname, timeout=15, context=context)
        self.ip = ip
        self.tls_context = context

    def connect(self) -> None:
        raw = socket.create_connection((self.ip, self.port), timeout=15)
        try:
            self.sock = self.tls_context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


class WebhookTransport:
    def __init__(self, endpoint: WebhookEndpoint) -> None:
        self.endpoint = endpoint
        self.context = ssl.create_default_context()

    def __call__(self, method: str, params: dict[str, object]) -> tuple[int, str | None, bytes]:
        if method not in READ_METHODS:
            raise ApiError("method_blocked")
        host = self.endpoint.hostname
        ip = self.endpoint.pinned_ip
        connection = (
            PinnedHTTPSConnection(host, ip, self.context)
            if ip
            else HTTPSConnection(host, timeout=15, context=self.context)
        )
        try:
            base = urlsplit(self.endpoint.url.get_secret_value()).path.rstrip("/")
            connection.request(
                "POST", f"{base}/{method}.json", json.dumps(params),
                {"Content-Type": "application/json", "Accept": "application/json"},
            )
            response = connection.getresponse()
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ApiError("response_too_large")
            return response.status, response.getheader("Retry-After"), body
        finally:
            connection.close()


def retry_delay(header: str | None, fallback: float) -> float:
    if header:
        if header.isascii() and header.isdecimal():
            seconds = header.lstrip("0")
            return min(int(seconds or "0"), 61) if len(seconds) <= 2 else 61
        try:
            retry_at = parsedate_to_datetime(header)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            pass
    return fallback


class BitrixReadOnlyClient:
    def __init__(self, transport: Transport, limiter: PortalLimiter) -> None:
        self.transport = transport
        self.limiter = limiter

    @classmethod
    def for_webhook(cls, endpoint: WebhookEndpoint) -> "BitrixReadOnlyClient":
        return cls(WebhookTransport(endpoint), portal_limiter(endpoint.hostname))

    def call(
        self, method: str, params: dict[str, object] | None = None
    ) -> tuple[object, object]:
        if method not in READ_METHODS:
            raise ApiError("method_blocked")

        for attempt in range(3):
            delay = float(2 ** attempt)
            try:
                status, header, body = self.limiter.send(self.transport, method, params or {})
            except TimeoutError:
                code = "timeout"
            except (OSError, HTTPException):
                break
            else:
                code = "query_limit"
                if status == 429:
                    delay = retry_delay(header, delay)
                elif 300 <= status < 400:
                    raise ApiError("redirect_blocked")
                else:
                    result = self._decode(status, body)
                    if result is not None:
                        return result
                    delay = retry_delay(header, delay)
            if attempt == 2 or delay > 60:
                raise ApiError(code)
            self.limiter.sleep(delay)
        raise ApiError("transport_error")

    @staticmethod
    def _decode(status: int, body: bytes) -> tuple[object, object] | None:
        try:
            payload = json.loads(body)
        except (ValueError, RecursionError):
            payload = None
        code = payload.get("error") if isinstance(payload, dict) else None
        if code == "QUERY_LIMIT_EXCEEDED":
            return None
        if status in (401, 403) or code in (
            "ACCESS_DENIED", "access_denied", "insufficient_scope", "INVALID_CREDENTIALS",
            "NO_AUTH_FOUND", "expired_token", "ERROR_ACCESS_DENIED",
        ):
            raise ApiError("no_access")
        if code in ("ERROR_METHOD_NOT_FOUND", "METHOD_NOT_FOUND", "ENTITY_TYPE_NOT_SUPPORTED"):
            raise ApiError("unsupported")
        if not 200 <= status < 300 or code is not None:
            raise ApiError("api_error")
        if not isinstance(payload, dict) or "result" not in payload:
            raise ApiError("invalid_response")
        return payload["result"], payload.get("next")
