import ssl
from unittest.mock import Mock, patch

import pytest
from pydantic import SecretStr

from core.config import WebhookEndpoint
from modules.portal_audit.client import (
    ApiError,
    BitrixReadOnlyClient,
    PinnedHTTPSConnection,
    PortalLimiter,
)


def test_shared_clients_space_attempts_and_honor_retry_after() -> None:
    now = 0.0
    starts = []

    def clock() -> float:
        return now

    def sleep(delay: float) -> None:
        nonlocal now
        now += delay

    def transport(method: str, params: dict[str, object]) -> tuple[int, str | None, bytes]:
        starts.append(now)
        if len(starts) == 2:
            return 503, "3", b'{"error":"QUERY_LIMIT_EXCEEDED"}'
        if len(starts) == 3:
            raise TimeoutError("secret")
        return 200, None, b'{"result":[],"next":50}'

    limiter = PortalLimiter(clock, sleep)
    first = BitrixReadOnlyClient(transport, limiter)
    second = BitrixReadOnlyClient(transport, limiter)
    assert first.call("crm.status.list") == ([], 50)
    assert second.call("crm.status.list") == ([], 50)
    assert starts == [0, 0.5, 3.5, 5.5]


def test_webhooks_on_same_hostname_share_limiter() -> None:
    first = WebhookEndpoint(url=SecretStr("https://example.test/rest/1/mock/"))
    second = WebhookEndpoint(url=SecretStr("https://EXAMPLE.test/rest/2/mock/"))
    assert BitrixReadOnlyClient.for_webhook(first).limiter is BitrixReadOnlyClient.for_webhook(second).limiter


@pytest.mark.parametrize("method", ["crm.item.add", "batch", "crm.item.delete", "CRM.TYPE.LIST"])
def test_blocked_method_does_not_reach_transport(method: str) -> None:
    transport = Mock()
    client = BitrixReadOnlyClient(transport, PortalLimiter())
    with pytest.raises(ApiError, match="method_blocked"):
        client.call(method)
    transport.assert_not_called()


@pytest.mark.parametrize(("status", "body", "code"), [
    (401, b'{"error_description":"synthetic-token"}', "no_access"),
    (200, b'{"error":"insufficient_scope"}', "no_access"),
    (404, b'{"error":"ERROR_METHOD_NOT_FOUND"}', "unsupported"),
    (200, b'not JSON', "invalid_response"),
    (302, b'{"result":[]}', "redirect_blocked"),
    (500, b'{"error":"synthetic-token"}', "api_error"),
])
def test_errors_are_classified_and_redacted(status: int, body: bytes, code: str) -> None:
    transport = Mock(return_value=(status, None, body))
    client = BitrixReadOnlyClient(transport, PortalLimiter())
    with pytest.raises(ApiError) as exc:
        client.call("crm.type.list")
    assert str(exc.value) == code
    transport.assert_called_once()


def test_timeout_retries_are_bounded_and_redacted() -> None:
    transport = Mock(side_effect=TimeoutError("secret"))
    now = 0.0

    def clock() -> float:
        return now

    def sleep(delay: float) -> None:
        nonlocal now
        now += delay

    client = BitrixReadOnlyClient(transport, PortalLimiter(clock, sleep))
    with pytest.raises(ApiError, match="^timeout$") as exc:
        client.call("crm.type.list")
    assert transport.call_count == 3
    assert now == 3
    assert exc.value.__context__ is None


def test_tcp_pin_keeps_tls_hostname_and_certificate_verification() -> None:
    context = ssl.create_default_context()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    connection = PinnedHTTPSConnection("portal.test", "192.0.2.1", context)
    with (
        patch("socket.create_connection") as dial,
        patch.object(context, "wrap_socket") as wrap,
    ):
        connection.connect()
        dial.assert_called_once_with(("192.0.2.1", 443), timeout=15)
        wrap.assert_called_once_with(dial.return_value, server_hostname="portal.test")
