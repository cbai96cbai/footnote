import json
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
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(
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
                        "tool_calls": tool_calls,
                    },
                }
            ],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        }
    )


def status_error(code: int, headers: dict[str, str] | None = None) -> openai.APIStatusError:
    resp = httpx.Response(code, request=REQ, headers=headers)
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
    # 429, 429, 200 -> three attempts, only the usage is charged, backoff 2s then 5s
    sleeps: list[float] = []
    c, sdk = client([status_error(429), status_error(429), ok(pt=10, ct=10)], sleeps)
    r = c.chat(MSGS, deployment="footnote-agent", limits=LIMITS["agent"])
    assert len(sdk.calls) == 3 and sleeps == [2, 5]
    assert r.usage.cost == Decimal("0.0000225") and not r.usage.charged_max
    # six 429s -> failed and marked rate limited, charged 0, five waits none over 60 s
    sleeps = []
    c, sdk = client([status_error(429)] * 6, sleeps)
    with pytest.raises(CallFailed) as e:
        c.chat(MSGS, deployment="footnote-agent", limits=LIMITS["agent"])
    assert len(sdk.calls) == 6 and e.value.usage.cost == 0 and not e.value.usage.charged_max
    assert e.value.rate_limited and len(sleeps) == 5 and max(sleeps) <= 60
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
        assert not e.value.rate_limited


def test_retry_after_headers_are_honored_and_capped() -> None:
    sleeps: list[float] = []
    script = [
        status_error(429, {"retry-after-ms": "1500"}),
        status_error(429, {"retry-after": "12"}),
        status_error(429, {"retry-after": "600"}),
        status_error(429, {"retry-after": "soon"}),  # unreadable: the default backoff
        ok(),
    ]
    c, sdk = client(script, sleeps)
    c.chat(MSGS, deployment="footnote-agent", limits=LIMITS["agent"])
    assert sleeps == [1.5, 12, 60, 30]


def test_other_rejections_are_not_rate_limited() -> None:
    c, _ = client([status_error(404)])
    with pytest.raises(CallFailed) as e:
        c.chat(MSGS, deployment="footnote-agent", limits=LIMITS["agent"])
    assert not e.value.rate_limited


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


def structured(content: str, finish: str = "stop") -> Any:
    return ok(content=content, finish=finish, pt=700, ct=40)


def test_structured_output_returns_parsed() -> None:
    c, sdk = client([structured('{"answer": "42"}')])
    r = c.chat(MSGS, deployment="footnote-grader", limits=LIMITS["grader"], response_schema=Verdict)
    rf = sdk.calls[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert r.parsed == Verdict(answer="42") and r.parse_error is None


def test_input_count_uses_the_payload_sent() -> None:
    from footnote.llm.tokens import count_input_tokens, count_text_tokens, response_format_payload

    base = count_input_tokens(MSGS, None, None)
    expected = base + count_text_tokens(json.dumps(response_format_payload(Verdict)))
    assert count_input_tokens(MSGS, None, Verdict) == expected


def test_length_on_structured_call_is_charged_and_signalled() -> None:
    c, _ = client([structured('{"answer": "4', finish="length")])
    r = c.chat(MSGS, deployment="footnote-grader", limits=LIMITS["grader"], response_schema=Verdict)
    assert r.finish_reason == "length" and r.parsed is None
    assert r.usage.cost == Decimal("0.00028") + Decimal("0.000064") and not r.usage.charged_max


def test_content_filter_is_charged_and_signalled() -> None:
    c, _ = client([structured("", finish="content_filter")])
    r = c.chat(MSGS, deployment="footnote-grader", limits=LIMITS["grader"], response_schema=Verdict)
    assert r.finish_reason == "content_filter" and r.parsed is None and r.usage.cost > 0


def test_invalid_json_is_charged_and_distinguishable_from_length() -> None:
    for bad in ("not json", '{"wrong": 1}'):
        c, _ = client([structured(bad)])
        r = c.chat(
            MSGS, deployment="footnote-grader", limits=LIMITS["grader"], response_schema=Verdict
        )
        assert r.finish_reason == "stop" and r.parsed is None and r.parse_error
        assert r.usage.input_tokens == 700 and r.usage.cost > 0


def test_empty_choices_is_charged_and_signalled() -> None:
    resp = ok(pt=500, ct=0)
    resp.choices = []
    c, _ = client([resp])
    r = c.chat(MSGS, deployment="footnote-agent", limits=LIMITS["agent"])
    assert r.finish_reason == "no_choices" and r.message == {}
    assert r.usage.input_tokens == 500 and r.usage.cost == Decimal("0.000125")


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
