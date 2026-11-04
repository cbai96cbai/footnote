from decimal import Decimal
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel

from footnote.budget.limits import LIMITS, PRICES, CallLimits, call_max_cost
from footnote.config import Settings
from footnote.llm.client import CallFailed, InputTooLarge, LLMClient

REQ = httpx.Request("POST", "https://x.openai.azure.com/chat")
SETTINGS = Settings(azure_endpoint="https://x.openai.azure.com", azure_api_key="k")
MSGS = [{"role": "user", "content": "hi"}]


def ok(
    content: str = "hello",
    finish: str = "stop",
    pt: int = 100,
    ct: int = 50,
    parsed: Any = None,
    tool_calls: Any = None,
) -> Any:
    from openai.types.chat.parsed_chat_completion import ParsedChatCompletion

    return ParsedChatCompletion.model_validate(
        {
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "parsed": parsed,
                        "tool_calls": tool_calls,
                    },
                }
            ],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        }
    )


def status_error(code: int) -> openai.APIStatusError:
    resp = httpx.Response(code, request=REQ)
    cls = openai.RateLimitError if code == 429 else openai.InternalServerError
    return cls("boom", response=resp, body=None)


class FakeSDK:
    """Plays a script of responses or exceptions, one per call."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.chat = self
        self.completions = self

    def _next(self, kwargs: dict[str, Any]) -> Any:
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def create(self, **kwargs: Any) -> Any:
        return self._next(kwargs)

    parse = create


def client(script: list[Any], sleeps: list[float] | None = None) -> tuple[LLMClient, FakeSDK]:
    sdk = FakeSDK(script)
    rec = sleeps if sleeps is not None else []
    return LLMClient(SETTINGS, sdk=sdk, sleep=rec.append), sdk


def test_charges_returned_usage() -> None:
    c, sdk = client([ok(pt=1000, ct=100)])
    r = c.chat(MSGS, deployment="footnote-agent", limits=LIMITS["agent"])
    assert r.usage.cost == Decimal("0.00045")
    assert not r.usage.charged_max
    assert (r.usage.input_tokens, r.usage.output_tokens) == (1000, 100)
    assert r.message["content"] == "hello"
    kw = sdk.calls[0]
    assert kw["model"] == "footnote-agent"
    assert kw["max_completion_tokens"] == 3000
    assert kw["reasoning_effort"] == "low"
    assert "temperature" not in kw and "store" not in kw


def test_grader_deployment_uses_grader_price_and_no_reasoning_effort() -> None:
    c, sdk = client([ok(pt=1000, ct=1000)])
    r = c.chat(MSGS, deployment="footnote-grader", limits=LIMITS["grader"])
    assert r.usage.cost == Decimal("0.002")
    assert "reasoning_effort" not in sdk.calls[0]


def test_deployment_names_follow_settings() -> None:
    s = Settings(agent_deployment="my-agent", grader_deployment="my-grader")
    sdk = FakeSDK([ok(pt=1_000_000, ct=0)])
    r = LLMClient(s, sdk=sdk, sleep=lambda _: None).chat(
        MSGS, deployment="my-agent", limits=LIMITS["agent"]
    )
    assert r.usage.cost == Decimal("0.25")
    with pytest.raises(ValueError):
        LLMClient(s, sdk=sdk).chat(MSGS, deployment="other", limits=LIMITS["agent"])


def test_charging_rules() -> None:
    maxc = call_max_cost(LIMITS["agent"], PRICES["gpt-5-mini"])
    # 429, 429, 200 -> three attempts, only the usage is charged, backoff 2s then 8s
    sleeps: list[float] = []
    c, sdk = client([status_error(429), status_error(429), ok(pt=10, ct=10)], sleeps)
    r = c.chat(MSGS, deployment="footnote-agent", limits=LIMITS["agent"])
    assert len(sdk.calls) == 3 and sleeps == [2, 8]
    assert r.usage.cost == Decimal("0.0000225") and not r.usage.charged_max
    # three 429s -> failed, charged 0
    c, sdk = client([status_error(429)] * 3)
    with pytest.raises(CallFailed) as e:
        c.chat(MSGS, deployment="footnote-agent", limits=LIMITS["agent"])
    assert len(sdk.calls) == 3 and e.value.usage.cost == 0 and not e.value.usage.charged_max
    # timeout, dropped connection, 500 -> charged the maximum, no retry
    for exc in (
        openai.APITimeoutError(request=REQ),
        openai.APIConnectionError(request=REQ),
        status_error(500),
    ):
        c, sdk = client([exc, ok()])
        with pytest.raises(CallFailed) as e:
            c.chat(MSGS, deployment="footnote-agent", limits=LIMITS["agent"])
        assert len(sdk.calls) == 1
        assert e.value.usage.charged_max and e.value.usage.cost == maxc


def test_missing_usage_on_success_is_charged_max() -> None:
    resp = ok()
    resp.usage = None
    c, _ = client([resp])
    r = c.chat(MSGS, deployment="footnote-agent", limits=LIMITS["agent"])
    assert r.usage.charged_max and r.usage.cost == Decimal("0.010")


def test_other_client_error_is_failed_and_free() -> None:
    err = openai.BadRequestError("bad", response=httpx.Response(400, request=REQ), body=None)
    c, sdk = client([err])
    with pytest.raises(CallFailed) as e:
        c.chat(MSGS, deployment="footnote-agent", limits=LIMITS["agent"])
    assert len(sdk.calls) == 1 and e.value.usage.cost == 0


def test_input_too_large_raises_before_send() -> None:
    c, sdk = client([ok()])
    with pytest.raises(InputTooLarge):
        c.chat(MSGS, deployment="footnote-agent", limits=CallLimits(100, 3000, None))
    assert sdk.calls == []


def test_finish_reason_exposed() -> None:
    c, _ = client([ok(finish="length")])
    r = c.chat(MSGS, deployment="footnote-agent", limits=CallLimits(16000, 16, "low"))
    assert r.finish_reason == "length"


class Verdict(BaseModel):
    answer: str


def test_structured_output_returns_parsed() -> None:
    c, sdk = client([ok(content='{"answer": "42"}', parsed={"answer": "42"})])
    r = c.chat(MSGS, deployment="footnote-grader", limits=LIMITS["grader"], response_schema=Verdict)
    assert sdk.calls[0]["response_format"] is Verdict
    assert r.parsed == Verdict(answer="42") or r.parsed == {"answer": "42"}


def test_tools_passed_through_and_tool_calls_in_message() -> None:
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    resp = ok(
        content=None,
        tool_calls=[  # type: ignore[arg-type]
            {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ],
    )
    c, sdk = client([resp])
    r = c.chat(MSGS, deployment="footnote-agent", limits=LIMITS["agent"], tools=tools)
    assert sdk.calls[0]["tools"] == tools
    assert r.message["tool_calls"][0]["function"]["name"] == "f"
    assert "parsed" not in r.message
