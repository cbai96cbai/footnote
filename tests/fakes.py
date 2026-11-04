"""Test doubles for the agent: a scripted LLM client and a fixed-page search index."""

import json
from decimal import Decimal
from typing import Any

from footnote.config import Settings
from footnote.llm.client import CallFailed, ChatResult, Usage
from footnote.retrieval.chunk import Passage
from footnote.retrieval.index import ScoredPassage

U = Usage(100, 50, Decimal("0.001"))


class ScriptedClient:
    """Returns the scripted outcomes in order; an Exception outcome is raised."""

    settings = Settings()

    def __init__(self, outcomes: list[ChatResult | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[list[dict[str, Any]]] = []
        self.tool_sets: list[Any] = []

    def chat(self, messages: Any, **kw: Any) -> ChatResult:
        self.calls.append(json.loads(json.dumps(list(messages))))  # a snapshot of this call
        self.tool_sets.append(kw.get("tools"))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def tool_call(name: str, args: Any, call_id: str = "c1") -> dict[str, Any]:
    raw = args if isinstance(args, str) else json.dumps(args)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}


def reply(*calls: dict[str, Any], finish: str = "tool_calls", usage: Usage = U) -> ChatResult:
    msg: dict[str, Any] = {"role": "assistant", "tool_calls": list(calls)}
    return ChatResult(msg, usage, finish, 0.1)


def text_reply(text: str, finish: str = "stop") -> ChatResult:
    return ChatResult({"role": "assistant", "content": text}, U, finish, 0.1)


def failed(cost: str = "0.5") -> CallFailed:
    return CallFailed("timeout", Usage(0, 0, Decimal(cost), True))


class FakeIndex:
    def __init__(self, hits: list[tuple[int, str]]) -> None:
        self.hits = hits

    def search(self, query: str, k: int = 8) -> list[ScoredPassage]:
        return [
            ScoredPassage(Passage(f"D:p{p}:0", "D", p, text), 1.0 / (i + 1))
            for i, (p, text) in enumerate(self.hits[:k])
        ]
