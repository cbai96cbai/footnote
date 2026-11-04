"""Azure OpenAI Chat Completions client that enforces limits and the charging rule.

A call that returns usage is charged that usage. A call that was sent but returned no usage
(timeout, dropped connection, any 5xx) is charged its maximum and not retried. HTTP 429 is
rejected before processing: retried (6 attempts in all, waiting what the service's Retry-After
header says, else 2s, 5s, 15s, 30s, 60s; no wait over 60s) and charged nothing. A call that was
rate limited on every attempt raises CallFailed with rate_limited set, so callers can leave the
question to be retried instead of scoring it.

Contract: once a response with usage arrives, chat() always returns a ChatResult carrying that
charge; it never raises afterwards. Callers tell outcomes apart by finish_reason ("stop",
"tool_calls", "length", "content_filter", or "no_choices" for an empty choices list) and, for
structured output, by parsed (the validated model, or None) and parse_error (why parsing failed,
None when parsing was not needed or succeeded). A cut-off call is therefore finish_reason ==
"length", while unparseable content under another finish_reason has parse_error set.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import openai
from pydantic import BaseModel, ValidationError

from footnote.budget.limits import PRICES, CallLimits, Price, call_cost, call_max_cost
from footnote.config import Settings
from footnote.llm.tokens import count_input_tokens, response_format_payload

API_VERSION = "2025-04-01-preview"
TIMEOUT_S = 300
MAX_ATTEMPTS = 6
BACKOFF_S = (2.0, 5.0, 15.0, 30.0, 60.0)
MAX_WAIT_S = 60.0
# fields the SDK adds to its message objects that the API does not accept back
_DROP_MESSAGE = ("annotations", "audio")


class InputTooLarge(Exception):
    """The conservative input count exceeds the call's limit; nothing was sent."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cost: Decimal
    charged_max: bool = False


class CallFailed(Exception):
    def __init__(self, message: str, usage: Usage, *, rate_limited: bool = False) -> None:
        super().__init__(message)
        self.usage = usage
        self.rate_limited = rate_limited  # refused before processing on every attempt


def _retry_wait(exc: openai.RateLimitError, attempt: int) -> float:
    """Seconds to wait before the next try: the service's own figure when readable, capped."""
    headers = exc.response.headers
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            return min(max(float(headers[name]) * scale, 0.0), MAX_WAIT_S)
        except (KeyError, ValueError):
            continue
    return min(BACKOFF_S[attempt], MAX_WAIT_S)


@dataclass(frozen=True)
class ChatResult:
    message: dict[str, Any]
    usage: Usage
    finish_reason: str
    latency_s: float
    parsed: Any = None
    parse_error: str | None = None


def _plain_message(message: Any) -> dict[str, Any]:
    out: dict[str, Any] = message.model_dump(exclude_none=True)
    for key in _DROP_MESSAGE:
        out.pop(key, None)
    for call in out.get("tool_calls") or []:
        call.get("function", {}).pop("parsed_arguments", None)
    return out


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        *,
        sdk: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._sleep = sleep
        self._sdk = sdk or openai.AzureOpenAI(
            azure_endpoint=settings.azure_endpoint,
            api_key=settings.azure_api_key,
            api_version=API_VERSION,
            max_retries=0,
            timeout=TIMEOUT_S,
        )

    def price_for(self, deployment: str) -> Price:
        if deployment == self.settings.agent_deployment:
            return PRICES["gpt-5-mini"]
        if deployment == self.settings.grader_deployment:
            return PRICES["gpt-4.1-mini"]
        raise ValueError(f"unknown deployment {deployment!r}")

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        deployment: str,
        limits: CallLimits,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_schema: type[BaseModel] | None = None,
        tool_choice: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        price = self.price_for(deployment)
        counted = count_input_tokens(messages, tools, response_schema)
        if counted > limits.max_input_tokens:
            raise InputTooLarge(
                f"{counted} input tokens exceed the {limits.max_input_tokens} limit"
            )
        max_cost = call_max_cost(limits, price)
        kwargs: dict[str, Any] = {
            "model": deployment,
            "messages": list(messages),
            "max_completion_tokens": limits.max_completion_tokens,
        }
        if limits.reasoning_effort:
            kwargs["reasoning_effort"] = limits.reasoning_effort
        if tools:
            kwargs["tools"] = list(tools)
        if tool_choice is not None:
            kwargs["tool_choice"] = dict(tool_choice)
        if response_schema is not None:
            kwargs["response_format"] = response_format_payload(response_schema)
        started = time.monotonic()
        completion = self._send(kwargs, max_cost)
        latency = time.monotonic() - started
        if completion.usage is None:
            usage = Usage(0, 0, max_cost, charged_max=True)
        else:
            u = completion.usage
            usage = Usage(
                u.prompt_tokens,
                u.completion_tokens,
                call_cost(u.prompt_tokens, u.completion_tokens, price),
            )
        # from here on the charge is fixed: nothing below may raise
        if not completion.choices:
            return ChatResult({}, usage, "no_choices", latency)
        choice = completion.choices[0]
        message = _plain_message(choice.message)
        finish = str(choice.finish_reason)
        parsed: Any = None
        parse_error: str | None = None
        if response_schema is not None and finish != "length":
            try:
                parsed = response_schema.model_validate_json(message.get("content") or "")
            except ValidationError as exc:
                parse_error = f"{type(exc).__name__}: {exc.error_count()} errors"
        return ChatResult(message, usage, finish, latency, parsed, parse_error)

    def _send(self, kwargs: dict[str, Any], max_cost: Decimal) -> Any:
        endpoint = self._sdk.chat.completions.create
        for attempt in range(MAX_ATTEMPTS):
            try:
                return endpoint(**kwargs)
            except openai.RateLimitError as exc:
                if attempt + 1 == MAX_ATTEMPTS:
                    raise CallFailed(
                        "rate limited on every attempt",
                        Usage(0, 0, Decimal(0)),
                        rate_limited=True,
                    ) from exc
                self._sleep(_retry_wait(exc, attempt))
            except openai.APIConnectionError as exc:  # includes timeouts
                raise CallFailed(
                    f"no response ({type(exc).__name__}); charged the maximum",
                    Usage(0, 0, max_cost, charged_max=True),
                ) from exc
            except openai.APIStatusError as exc:
                if exc.status_code >= 500:
                    raise CallFailed(
                        f"server error {exc.status_code}; charged the maximum",
                        Usage(0, 0, max_cost, charged_max=True),
                    ) from exc
                raise CallFailed(
                    f"request rejected with {exc.status_code}", Usage(0, 0, Decimal(0))
                ) from exc
        raise AssertionError("unreachable")
