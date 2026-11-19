"""Provider-neutral message types that mirror the Anthropic Messages API shapes."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field

from ai_sdr_crew.models import Usage


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[TextBlock | ToolUseBlock | ToolResultBlock, Field(discriminator="type")]
OutputBlock = Annotated[TextBlock | ToolUseBlock, Field(discriminator="type")]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: list[ContentBlock]

    def text(self) -> str:
        parts: list[str] = []
        for b in self.content:
            if isinstance(b, TextBlock):
                parts.append(b.text)
            elif isinstance(b, ToolResultBlock):
                parts.append(b.content)
        return "\n".join(parts)


class SystemBlock(BaseModel):
    text: str
    cache: bool = False


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class LLMRequest(BaseModel):
    role: str
    model: str
    system: list[SystemBlock]
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    tool_choice: dict[str, Any] = Field(default_factory=lambda: {"type": "auto"})
    max_tokens: int = 2048


class LLMResponse(BaseModel):
    content: list[OutputBlock]
    usage: Usage
    stop_reason: str = "end_turn"


class LLMBackend(Protocol):
    name: str

    def create(self, req: LLMRequest) -> LLMResponse: ...
