import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from core.config import ModelSettings, load_settings
from core.contracts import Contract
from core.llm import LLMGateway, Price


class Output(Contract):
    value: int


def local() -> Output:
    return Output(value=0)


def test_fenced_json_roles_and_prices() -> None:
    calls = []

    def transport(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert endpoint == "chat/completions"
        assert payload["response_format"] == {"type": "json_object"}
        assert json.loads(payload["messages"][1]["content"]) == {"question": "hello"}
        calls.append(payload["model"])
        return {
            "choices": [{"message": {"content": "```json\n{\"value\": 7}\n```"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5},
        }

    gateway = LLMGateway(
        ModelSettings.model_validate({
            "base_url": "http://127.0.0.1/v1", "fast_model": "fast", "smart_model": "smart",
        }),
        transport=transport,
        prices={"fast": Price(input_per_million_rub=Decimal("1.25"),
                              output_per_million_rub=Decimal("2"))},
    )
    result, run = gateway.structured("fast", "prompt", {"question": "hello"}, Output, local)
    assert (result.value, run.mode, run.tokens_in, run.tokens_out, run.cost_rub_e8) == (
        7, "llm", 100, 5, 13500,
    )
    _, run = gateway.structured("smart", "prompt", {"question": "hello"}, Output, local)
    assert run.cost_rub_e8 is None
    assert calls == ["fast", "smart"]


@pytest.mark.parametrize("content", [
    '{"value": "7"}', '{"value": 7, "extra": true}', 'text {"value": 7}',
    '{"value": 7, "value": 8}', 'NaN',
])
def test_invalid_output_falls_back_with_known_usage(content: str) -> None:
    def transport(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 18}}

    gateway = LLMGateway(
        ModelSettings.model_validate({"base_url": "http://127.0.0.1/v1", "fast_model": "fast"}),
        transport=transport,
    )
    result, run = gateway.structured("fast", "prompt", {}, Output, local)
    assert (result.value, run.mode, run.fallback_reason, run.tokens_in, run.tokens_out) == (
        0, "fallback", "invalid_response", 18, None,
    )


def test_dry_run_and_unconfigured_gateway_never_use_transport() -> None:
    def transport(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        pytest.fail("unexpected network")

    gateway = LLMGateway(ModelSettings(), transport=transport)
    _, run = gateway.structured("fast", "prompt", {}, Output, local, dry_run=True)
    assert (run.mode, run.cost_rub_e8) == ("deterministic", 0)
    _, run = gateway.structured("fast", "prompt", {}, Output, local)
    assert (run.mode, run.fallback_reason, run.cost_rub_e8) == ("fallback", "unconfigured", None)
    assert gateway.embed("test", dry_run=True)[1].mode == "deterministic"


@pytest.mark.parametrize(("rate", "expected"), [
    ("0.005", 0), ("0.015", 2), ("900000000000", None),
])
def test_cost_rounding_and_overflow(rate: str, expected: int | None) -> None:
    gateway = LLMGateway(
        ModelSettings.model_validate({
            "base_url": "http://127.0.0.1/v1", "embedding_model": "vector",
        }),
        transport=lambda *_: {"data": [{"embedding": [1, 0]}], "usage": {"prompt_tokens": 1_000_000}
                              if expected is None else {"prompt_tokens": 1}},
        prices={"vector": Price(input_per_million_rub=Decimal(rate),
                                output_per_million_rub=Decimal(0))},
    )
    vector, run = gateway.embed("test")
    assert vector == [1, 0]
    assert run.tokens_out == 0
    assert run.cost_rub_e8 == expected



def test_model_settings_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "FAST_LLM_MODEL", "SMART_LLM_MODEL",
                 "EMBEDDING_MODEL", "B24_DOCS_MAX_AGE_DAYS"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "LLM_BASE_URL=http://127.0.0.1:8080/v1\nLLM_API_KEY=fake-key\n"
        "FAST_LLM_MODEL=file-fast\nSMART_LLM_MODEL=smart\nEMBEDDING_MODEL=vector\n"
        "B24_DOCS_MAX_AGE_DAYS=45\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAST_LLM_MODEL", "env-fast")
    settings = load_settings(path)
    assert settings.docs_max_age_days == 45
    assert (settings.models.fast_model, settings.models.smart_model,
            settings.models.embedding_model) == ("env-fast", "smart", "vector")
    assert "fake-key" not in repr(settings)
