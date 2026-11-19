from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ai_sdr_crew.llm.anthropic_backend import AnthropicBackend
from ai_sdr_crew.llm.types import LLMRequest, Message, SystemBlock, TextBlock, ToolSpec


class FakeMessages:
    def __init__(self) -> None:
        self.params: dict[str, Any] = {}

    def create(self, **params: Any) -> SimpleNamespace:
        self.params = params
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="ok"),
                SimpleNamespace(type="tool_use", id="toolu_1", name="submit_angle", input={"a": 1}),
            ],
            usage=SimpleNamespace(
                input_tokens=50,
                output_tokens=10,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=1200,
            ),
            stop_reason="tool_use",
        )


def make_req() -> LLMRequest:
    return LLMRequest(
        role="strategist",
        model="claude-sonnet-5-5",
        system=[SystemBlock(text="shared", cache=True), SystemBlock(text="role")],
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
        tools=[ToolSpec(name="submit_angle", description="d", input_schema={"type": "object"})],
        tool_choice={"type": "tool", "name": "submit_angle"},
    )


def test_request_shape_has_cache_control_on_shared_block_only() -> None:
    fake = FakeMessages()
    AnthropicBackend(client=SimpleNamespace(messages=fake)).create(make_req())
    p = fake.params
    assert p["model"] == "claude-sonnet-5-5"
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in p["system"][1]
    assert p["tools"][0]["name"] == "submit_angle"
    assert p["tool_choice"] == {"type": "tool", "name": "submit_angle"}
    assert p["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_response_and_usage_mapping() -> None:
    resp = AnthropicBackend(client=SimpleNamespace(messages=FakeMessages())).create(make_req())
    assert [b.type for b in resp.content] == ["text", "tool_use"]
    assert resp.usage.cache_read_input_tokens == 1200
    assert resp.usage.cache_creation_input_tokens == 0
    assert resp.stop_reason == "tool_use"


def test_default_model_from_env(monkeypatch) -> None:
    from ai_sdr_crew.config import Settings

    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test-model")
    assert Settings.from_env().model == "claude-test-model"
    monkeypatch.delenv("ANTHROPIC_MODEL")
    assert Settings.from_env().model == "claude-sonnet-5-5"


def test_live_mode_requires_key(monkeypatch) -> None:
    import pytest

    from ai_sdr_crew.config import Settings
    from ai_sdr_crew.factory import make_backend

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        make_backend(Settings(live=True))
