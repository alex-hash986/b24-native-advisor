import re
from pathlib import Path
from typing import Literal

PromptId = Literal["advisor.route", "advisor.recommend"]
PROMPT_REGISTRY = Path(__file__).resolve().parents[1] / "docs" / "prompts_registry.md"
ACTIVE_VERSIONS: dict[PromptId, str] = {
    "advisor.route": "1.0.0",
    "advisor.recommend": "1.2.0",
}


class PromptRegistry:
    def __init__(self, path: Path = PROMPT_REGISTRY) -> None:
        self.markdown = path.read_text(encoding="utf-8-sig")

    def render(self, prompt_id: PromptId, version: str | None = None) -> str:
        if version is None:
            version = ACTIVE_VERSIONS[prompt_id]
        blocks = re.findall(
            rf"^## {re.escape(prompt_id)} {re.escape(version)}[ \t]*\r?\n(.*?)(?=^## |\Z)",
            self.markdown, flags=re.M | re.S,
        )
        rules = re.findall(r"^## Правила исполнения\s*\n(.*?)(?=^## |\Z)",
                           self.markdown, flags=re.M | re.S)
        if len(blocks) != 1 or len(rules) != 1:
            raise ValueError("missing or duplicate prompt registry section")
        template = "\n".join(re.findall(r"^> ?(.*)$", blocks[0], flags=re.M))
        guidance = "\n".join(re.findall(r"^- .*$", rules[0], flags=re.M))
        if not template or not guidance:
            raise ValueError("empty prompt registry section")
        return f"{template}\n\n{guidance}"
