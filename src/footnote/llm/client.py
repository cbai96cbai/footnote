"""Azure OpenAI Chat Completions client that enforces limits and the charging rule.

A call that returns usage is charged that usage. A call that was sent but returned no usage
(timeout, dropped connection, any 5xx) is charged its maximum and not retried. HTTP 429 is
rejected before processing: retried (3 attempts in all, backoff 2s then 8s) and charged nothing.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import openai
from pydantic import BaseModel

from footnote.budget.limits import PRICES, CallLimits, Price, call_cost, call_max_cost
from footnote.config import Settings
from footnote.llm.tokens import count_input_tokens

API_VERSION = "2025-04-01-preview"
TIMEOUT_S = 300
MAX_ATTEMPTS = 3
BACKOFF_S = (2.0, 8.0)
# fields the SDK adds to its message objects that the API does not accept back
_DROP_MESSAGE = ("parsed", "annotations", "audio")


class InputTooLarge(Exception):
    """The conservative input count exceeds the call's limit; nothing was sent."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cost: Decimal
    charged_max: bool = False


class CallFailed(Exception):
    def __init__(self, message: str, usage: Usage) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class ChatResult:
    message: dict[str, Any]
    usage: Usage
    finish_reason: str
    latency_s: float
    parsed: Any = None


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
        if response_schema is not None:
            kwargs["response_format"] = response_schema
        started = time.monotonic()
        completion = self._send(kwargs, response_schema is not None, max_cost)
        latency = time.monotonic() - started
        choice = completion.choices[0]
        if completion.usage is None:
            usage = Usage(0, 0, max_cost, charged_max=True)
        else:
            u = completion.usage
            usage = Usage(
                u.prompt_tokens,
                u.completion_tokens,
                call_cost(u.prompt_tokens, u.completion_tokens, price),
            )
        return ChatResult(
            message=_plain_message(choice.message),
            usage=usage,
            finish_reason=str(choice.finish_reason),
            latency_s=latency,
            parsed=getattr(choice.message, "parsed", None) if response_schema else None,
        )

    def _send(self, kwargs: dict[str, Any], structured: bool, max_cost: Decimal) -> Any:
        endpoint = (
            self._sdk.chat.completions.parse if structured else (self._sdk.chat.completions.create)
        )
        for attempt in range(MAX_ATTEMPTS):
            try:
                return endpoint(**kwargs)
            except openai.RateLimitError as exc:
                if attempt + 1 == MAX_ATTEMPTS:
                    raise CallFailed(
                        "rate limited on every attempt", Usage(0, 0, Decimal(0))
                    ) from exc
                self._sleep(BACKOFF_S[attempt])
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
