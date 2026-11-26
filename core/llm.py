import json
import re
from collections.abc import Callable, Mapping
from decimal import ROUND_HALF_EVEN, Decimal, DecimalException, localcontext
from http.client import HTTPException
from typing import Annotated, Any, Literal, TypeVar
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import Field, FiniteFloat, ValidationError

from core.config import ModelSettings
from core.contracts import Contract, NonnegativeId

Output = TypeVar("Output", bound=Contract)
Role = Literal["fast", "smart"]
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]
Vector = list[FiniteFloat]
FENCE = re.compile(r"\A```(?:json)?\s*\n(.*?)\n\s*```\Z", re.I | re.S)


def unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


class Usage(Contract):
    tokens_in: NonnegativeId | None = None
    tokens_out: NonnegativeId | None = None
    cost_rub_e8: NonnegativeId | None = None


class Execution(Usage):
    mode: Literal["llm", "deterministic", "fallback"]
    fallback_reason: Literal["unconfigured", "transport_error", "invalid_response"] | None = None


class Price(Contract):
    input_per_million_rub: Annotated[Decimal, Field(ge=0, allow_inf_nan=False, max_digits=24, decimal_places=12)]
    output_per_million_rub: Annotated[Decimal, Field(ge=0, allow_inf_nan=False, max_digits=24, decimal_places=12)]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: Any, msg: Any,
                         headers: Any, newurl: Any) -> None:
        return None


class LLMGateway:
    def __init__(
        self,
        settings: ModelSettings,
        *,
        transport: Transport | None = None,
        prices: Mapping[str, Price] | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport or self._post
        self.prices = dict(prices or {})

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        base = self.settings.base_url
        if base is None:
            raise ValueError("model endpoint unavailable")
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key is not None:
            headers["Authorization"] = "Bearer " + self.settings.api_key.get_secret_value()
        request = Request(f"{str(base).rstrip('/')}/{endpoint}",
                          data=json.dumps(payload).encode(), headers=headers, method="POST")
        with build_opener(NoRedirect()).open(request, timeout=self.settings.timeout_seconds) as reply:
            body = reply.read(2_000_001)
        if len(body) > 2_000_000:
            raise ValueError("model response too large")
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError("invalid model envelope")
        return dict(data)

    def _usage(self, reply: dict[str, Any], model: str, *, embedding: bool = False) -> Usage:
        raw = reply.get("usage")
        if not isinstance(raw, dict):
            return Usage()
        tokens_in = raw.get("prompt_tokens")
        tokens_out = raw.get("completion_tokens", 0 if embedding else None)

        def known(value: object) -> int | None:
            return value if type(value) is int and 0 <= value < 2**63 else None

        incoming, outgoing = known(tokens_in), known(tokens_out)
        price = self.prices.get(model)
        cost = None
        if price is not None and incoming is not None and outgoing is not None:
            try:
                with localcontext() as context:
                    context.prec = 50
                    scaled = (price.input_per_million_rub * incoming
                              + price.output_per_million_rub * outgoing) * 100
                    rounded = scaled.to_integral_value(rounding=ROUND_HALF_EVEN)
                    if 0 <= rounded < 2**63:
                        cost = int(rounded)
            except DecimalException:
                pass
        return Usage(tokens_in=incoming, tokens_out=outgoing, cost_rub_e8=cost)

    def structured(
        self, role: Role, instructions: str, data: Mapping[str, object], output: type[Output],
        local: Callable[[], Output], *, dry_run: bool = False,
    ) -> tuple[Output, Execution]:
        if dry_run:
            return output.model_validate_json(local().model_dump_json()), Execution(
                mode="deterministic", tokens_in=0, tokens_out=0, cost_rub_e8=0
            )
        model = self.settings.fast_model if role == "fast" else self.settings.smart_model
        usage = Usage()
        reason: Literal["unconfigured", "transport_error", "invalid_response"] = "unconfigured"
        if self.settings.base_url and model:
            try:
                reply = self.transport("chat/completions", {
                    "model": model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": instructions + "\nJSON Schema:\n"
                         + json.dumps(output.model_json_schema(), ensure_ascii=False)},
                        {"role": "user", "content": json.dumps(dict(data), ensure_ascii=False)},
                    ],
                })
                usage = self._usage(reply, model)
                choice = reply["choices"][0]
                content = choice["message"]["content"].strip()
                if choice.get("finish_reason") not in (None, "stop"):
                    raise ValueError("incomplete output")
                fence = FENCE.fullmatch(content)
                document = fence[1] if fence else content
                json.loads(document, object_pairs_hook=unique_keys)
                result = output.model_validate_json(document, strict=True)
                return result, Execution(mode="llm", **usage.model_dump())
            except (URLError, OSError, HTTPException):
                reason = "transport_error"
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError):
                reason = "invalid_response"
        return output.model_validate_json(local().model_dump_json()), Execution(
            mode="fallback", fallback_reason=reason, **usage.model_dump()
        )

    def embed(self, value: str, *, dry_run: bool = False) -> tuple[Vector | None, Execution]:
        model = self.settings.embedding_model
        if dry_run:
            return None, Execution(mode="deterministic", tokens_in=0, tokens_out=0, cost_rub_e8=0)
        if not self.settings.base_url or not model:
            return None, Execution(mode="fallback", fallback_reason="unconfigured")
        usage = Usage()
        try:
            reply = self.transport("embeddings", {"model": model, "input": value})
            usage = self._usage(reply, model, embedding=True)
            vector = Embedding.model_validate({"vector": reply["data"][0]["embedding"]}, strict=True)
            return vector.vector, Execution(mode="llm", **usage.model_dump())
        except (URLError, OSError, HTTPException):
            return None, Execution(mode="fallback", fallback_reason="transport_error")
        except (ValidationError, ValueError, TypeError, IndexError, KeyError):
            return None, Execution(mode="fallback", fallback_reason="invalid_response",
                                   **usage.model_dump())


class Embedding(Contract):
    vector: Annotated[Vector, Field(min_length=1, max_length=16384)]
