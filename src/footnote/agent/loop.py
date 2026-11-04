"""The agent: an explicit tool loop of at most 8 model calls, ending in a verified answer."""

import json
import time
from typing import Any, Literal

from pydantic import ValidationError

from footnote.agent.prompts import (
    LAST_REPLY,
    NUDGE_LENGTH,
    NUDGE_NO_TOOL,
    REPLIES_LEFT,
    RETRY_TEMPLATE,
    SYSTEM_PROMPT,
    TOOLS,
    USER_TEMPLATE,
)
from footnote.agent.tools import SearchIndex, Toolbox, ToolError, cap, trace_args
from footnote.answers import AgentAnswer
from footnote.benchmark import AgentQuestion
from footnote.budget.limits import LIMITS, MAX_CALLS
from footnote.llm.client import CallFailed, ChatResult, LLMClient, Usage
from footnote.llm.tokens import count_input_tokens
from footnote.prediction import NO_USAGE, RATE_LIMITED, Prediction, Step, add_usage
from footnote.verify import verify_answer

FAMILY = "agent"
VERSION = "1"
SUBMIT = "submit_answer"
_Status = Literal["answered", "declined", "error"]


def _declined(reason: str) -> AgentAnswer:
    return AgentAnswer(
        answer="", final_value=None, declined=True, decline_reason=reason, citations=[]
    )


class _Agent:
    def __init__(
        self, q: AgentQuestion, client: LLMClient, pages: dict[int, str], index: SearchIndex
    ) -> None:
        self.q = q
        self.client = client
        self.pages = pages
        self.tools = Toolbox(pages, index)
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    doc_name=q.doc_name, company=q.company, question=q.question
                ),
            },
        ]
        self.summaries: dict[str, str] = {}  # tool_call_id -> the summary that replaces a result
        self.trace: list[Step] = []
        self.usage = NO_USAGE
        self.calls = 0
        self.pages_read: list[int] = []
        self.retried = False
        self.last_problems: list[str] = []
        self.started = time.monotonic()

    # -- the result -------------------------------------------------------------------------
    def _result(
        self,
        status: _Status,
        answer: AgentAnswer | None,
        verify_ok: bool | None = None,
        problems: list[str] | None = None,
    ) -> Prediction:
        if status == "declined" and answer is not None:
            answer = answer.model_copy(update={"citations": []})  # never checked, never shown
        return Prediction(
            question_id=self.q.id,
            system=FAMILY,
            version=VERSION,
            answer=answer,
            status=status,
            verify_ok=verify_ok,
            verify_problems=problems or [],
            trace=self.trace,
            usage=self.usage,
            latency_s=time.monotonic() - self.started,
            pages_read=self.pages_read,
        )

    def _decline(self, reason: str) -> Prediction:
        self.trace.append(Step("decline", {}, cap(reason), NO_USAGE))
        failed = bool(self.last_problems)
        return self._result(
            "declined", _declined(reason), False if failed else None, self.last_problems
        )

    def _error(self, reason: str, *, rate_limited: bool = False) -> Prediction:
        args = {"reason": RATE_LIMITED} if rate_limited else {}
        self.trace.append(Step("error", args, cap(reason), NO_USAGE))
        return self._result("error", None)

    # -- context ----------------------------------------------------------------------------
    def _size(self) -> int:
        return count_input_tokens(self.messages, TOOLS, None)

    def _fit(self) -> bool:
        """Replace the oldest tool results by their summaries until the call fits its limit."""
        limit = LIMITS["agent"].max_input_tokens
        for m in self.messages:
            if self._size() <= limit:
                break
            summary = self.summaries.get(m.get("tool_call_id", ""))
            if m.get("role") == "tool" and summary is not None and m["content"] != summary:
                m["content"] = summary
        return self._size() <= limit

    # -- the loop ---------------------------------------------------------------------------
    def run(self) -> Prediction:
        while True:
            if self.calls >= MAX_CALLS["agent"]:
                return self._decline("call limit")
            if not self._fit():
                return self._decline("context limit")
            try:
                res = self.client.chat(
                    self.messages,
                    deployment=self.client.settings.agent_deployment,
                    limits=LIMITS["agent"],
                    tools=TOOLS,
                )
            except CallFailed as exc:
                self.usage = add_usage(self.usage, exc.usage)
                self.trace.append(Step("chat", {"call": self.calls + 1}, cap(str(exc)), exc.usage))
                return self._error(f"call failed: {exc}", rate_limited=exc.rate_limited)
            except Exception as exc:  # nothing was sent or charged
                return self._error(f"call not made: {type(exc).__name__}: {exc}")
            self.calls += 1
            self.usage = add_usage(self.usage, res.usage)
            try:
                done = self._handle(res)
            except Exception as exc:  # the charge is already in self.usage
                return self._error(f"failed handling a reply: {type(exc).__name__}: {exc}")
            if done is not None:
                return done

    def _handle(self, res: ChatResult) -> Prediction | None:
        if res.finish_reason in ("content_filter", "no_choices"):
            self.trace.append(Step("chat", {}, res.finish_reason, res.usage))
            return self._error(f"finish_reason {res.finish_reason}")
        calls = res.message.get("tool_calls") or []
        if res.finish_reason == "length" or not calls:
            nudge = NUDGE_LENGTH if res.finish_reason == "length" else NUDGE_NO_TOOL
            self.trace.append(Step("chat", {}, f"finish_reason={res.finish_reason}", res.usage))
            if res.finish_reason != "length":
                self.messages.append(
                    {"role": "assistant", "content": res.message.get("content") or ""}
                )
            self.messages.append({"role": "user", "content": f"{nudge} {self._left()}"})
            return None
        self.messages.append(res.message)
        final: Prediction | None = None
        for i, call in enumerate(calls):
            usage = res.usage if i == 0 else NO_USAGE  # one charge, shown once
            if final is not None:
                self._reply(call, "ignored: the answer was already submitted", usage)
            else:
                final = self._tool_call(call, usage)
        if final is None and self.messages[-1].get("role") == "tool":
            self.messages[-1]["content"] += "\n" + self._left()
        return final

    def _left(self) -> str:
        n = MAX_CALLS["agent"] - self.calls
        return LAST_REPLY if n == 1 else REPLIES_LEFT.format(n=n)

    def _reply(self, call: dict[str, Any], content: str, usage: Usage) -> None:
        fn = call.get("function", {})
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        self.summaries[call["id"]] = cap(content)
        self.trace.append(Step(str(fn.get("name")), {}, cap(content), usage))

    def _tool_call(self, call: dict[str, Any], usage: Usage) -> Prediction | None:
        fn = call.get("function", {})
        name = str(fn.get("name"))
        raw = fn.get("arguments") or ""
        try:
            args = json.loads(raw)
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as exc:
            self._tool_error(call, name, {"raw": cap(str(raw))}, f"bad arguments: {exc}", usage)
            return None
        if name == SUBMIT:
            return self._submit(call, args, usage)
        try:
            out = self.tools.run(name, args)
        except ToolError as exc:
            self._tool_error(call, name, args, str(exc), usage)
            return None
        for p in out.pages_seen:
            if p not in self.pages_read:
                self.pages_read.append(p)
        self._record(call, name, args, out.content, out.summary, usage)
        return None

    def _record(
        self,
        call: dict[str, Any],
        name: str,
        args: dict[str, Any],
        content: str,
        summary: str,
        usage: Usage,
    ) -> None:
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        self.summaries[call["id"]] = summary
        self.trace.append(Step(name, trace_args(args), cap(summary), usage))

    def _tool_error(
        self, call: dict[str, Any], name: str, args: dict[str, Any], message: str, usage: Usage
    ) -> None:
        self._record(call, name, args, f"error: {message}", cap(f"error: {message}"), usage)

    def _submit(
        self, call: dict[str, Any], args: dict[str, Any], usage: Usage
    ) -> Prediction | None:
        try:
            answer = AgentAnswer.model_validate(args)
        except ValidationError as exc:
            self._tool_error(
                call,
                SUBMIT,
                args,
                f"answer does not match the schema ({exc.error_count()} errors)",
                usage,
            )
            return None
        if answer.declined:
            self._record(call, SUBMIT, args, "declined", "declined", usage)
            return self._result("declined", answer)
        ok, problems = verify_answer(answer, self.pages, self.q.question, self.tools.calcs)
        if ok:
            self._record(call, SUBMIT, args, "accepted", "verified", usage)
            return self._result("answered", answer, True)
        self.last_problems = problems
        self._record(
            call,
            SUBMIT,
            args,
            "problems: " + "; ".join(problems),
            cap("verification failed: " + "; ".join(problems)),
            usage,
        )
        if self.retried:
            return self._result("declined", _declined("; ".join(problems)), False, problems)
        self.retried = True
        self.messages[-1]["content"] = RETRY_TEMPLATE.format(problems="; ".join(problems))
        return None


def run_agent(
    q: AgentQuestion, client: LLMClient, pages: dict[int, str], index: SearchIndex
) -> Prediction:
    return _Agent(q, client, pages, index).run()
