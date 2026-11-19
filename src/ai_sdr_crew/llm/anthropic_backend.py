"""Live backend: Anthropic Messages API with tool use and prompt caching."""

from __future__ import annotations

from typing import Any

from ai_sdr_crew.llm.types import LLMRequest, LLMResponse, TextBlock, ToolUseBlock
from ai_sdr_crew.models import Usage


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY; SDK retries 429/5xx
        self._client = client

    @staticmethod
    def build_params(req: LLMRequest) -> dict[str, Any]:
        system: list[dict[str, Any]] = []
        for block in req.system:
            item: dict[str, Any] = {"type": "text", "text": block.text}
            if block.cache:
                item["cache_control"] = {"type": "ephemeral"}
            system.append(item)
        params: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "system": system,
            "messages": [m.model_dump() for m in req.messages],
        }
        if req.tools:
            params["tools"] = [t.model_dump() for t in req.tools]
            params["tool_choice"] = req.tool_choice
        return params

    def create(self, req: LLMRequest) -> LLMResponse:
        resp = self._client.messages.create(**self.build_params(req))
        blocks: list[TextBlock | ToolUseBlock] = []
        for b in resp.content:
            if b.type == "text":
                blocks.append(TextBlock(text=b.text))
            elif b.type == "tool_use":
                blocks.append(ToolUseBlock(id=b.id, name=b.name, input=dict(b.input)))
        u = resp.usage
        usage = Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        )
        return LLMResponse(content=blocks, usage=usage, stop_reason=resp.stop_reason or "end_turn")
