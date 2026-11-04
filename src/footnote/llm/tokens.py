"""Conservative input-token counting, done before a call is sent."""

import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import tiktoken

# private SDK converter, used so the counted payload is exactly the one sent; uv.lock pins the SDK
from openai.lib._parsing._completions import type_to_response_format_param
from pydantic import BaseModel

PER_MESSAGE = 4
FRAMING = 300


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


def count_text_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def _content_tokens(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return count_text_tokens(content)
    return sum(
        count_text_tokens(part["text"])
        if isinstance(part, Mapping) and "text" in part
        else count_text_tokens(json.dumps(part))
        for part in content
    )


def _text_tokens(value: Any) -> int:
    return count_text_tokens(value) if isinstance(value, str) else 0


def _message_tokens(message: Mapping[str, Any]) -> int:
    total = PER_MESSAGE + _content_tokens(message.get("content"))
    for key in ("name", "tool_call_id"):
        total += _text_tokens(message.get(key))
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        total += _text_tokens(call.get("id")) + _text_tokens(fn.get("name"))
        total += _text_tokens(fn.get("arguments"))
    return total


def response_format_payload(model: type[BaseModel]) -> Any:
    """The strict JSON-schema response_format sent for a pydantic model."""
    return type_to_response_format_param(model)


def count_input_tokens(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    response_schema: type[BaseModel] | None,
) -> int:
    total = FRAMING + sum(_message_tokens(m) for m in messages)
    if tools:
        total += count_text_tokens(json.dumps(tools))
    if response_schema is not None:
        total += count_text_tokens(json.dumps(response_format_payload(response_schema)))
    return total
