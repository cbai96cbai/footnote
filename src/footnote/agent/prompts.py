"""The agent's instructions, user message and tool definitions (all of it is registered config)."""

from typing import Any

from footnote.answers import AgentAnswer

SYSTEM_PROMPT = """\
You answer one question about one SEC filing. The filing is named in the user message. Answer only
from the filing, using the tools; do not use your own knowledge of the company.

Tools: search_filing finds passages, read_page shows one whole page, calculate does arithmetic,
submit_answer ends the work with your answer. Pages are numbered by page_index, exactly as the tools
show it; read_page and citations use the same numbers.

Rules for the answer you submit:
- Quote exactly, character for character, from text the tools returned. Each quote is at most 400
  characters and carries the page_index of the page it came from. Cite every figure you rely on.
- final_value is the single number the question asks for, in the unit the question asks for (if it
  asks "in USD millions", the value is the number of millions). Use unit "usd" for money, "percent"
  for percentages (65.4% is 65.4), "ratio", "count", or "other". Use null when the question has no
  single numeric answer.
- Use calculate for every piece of arithmetic; never compute in your head. Write the figures from
  the filing into the expression.
- Every number in your answer must come from a cited quote or a calculation on cited quotes.
- Arithmetic constants: for an average of two figures use (a+b)/2; for a percentage use a/b*100;
  never use 0.5 or other decimals as constants. Only whole numbers 1 to 12, 360, 365, 100, 1000,
  1000000 and 1000000000 may appear in an expression without being quoted, and at least one number
  in each expression must come from a cited quote.
- Page numbers are page_index (0-based), not the page number printed on the page. Read table rows
  across their columns carefully: match the row label to the right column heading and period.
- Set declined to true, with a decline_reason, only when the filing cannot support an answer.
- Your answer is checked against the filing. If the check finds problems you will be told once and
  may submit again; searching again first is allowed.
- Budget: you get at most 8 replies in total, and the submission and any retry count among them.
  Several tool calls in one reply cost one reply. Each tool result tells you how many replies are
  left. Submit before they run out; an unsubmitted question counts as declined."""

USER_TEMPLATE = "The question is about this filing: {doc_name} ({company}).\n\nQuestion: {question}"

NUDGE_LENGTH = "Your last reply was cut off. Reply again, briefly, with one tool call."
NUDGE_NO_TOOL = "Reply with a tool call. When you are ready, call submit_answer."
RETRY_TEMPLATE = (
    "Verification found problems: {problems}. Fix them and call submit_answer again, once. "
    "You may search or read again first if you need to."
)
REPLIES_LEFT = "[replies left: {n}]"
LAST_REPLY = "[replies left: 1. This is your last reply: call submit_answer now.]"


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> Any:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS: list[Any] = [
    _tool(
        "search_filing",
        "Search the filing with plain words (no Boolean, wildcard or exact-phrase syntax); "
        "returns up to 8 passages, each with its page_index.",
        {"query": {"type": "string", "description": "words likely to appear in the passage"}},
        ["query"],
    ),
    _tool(
        "read_page",
        "Show the text of one page of the filing (long pages are cut at 3,000 tokens).",
        {"page_index": {"type": "integer", "description": "the 0-based page_index"}},
        ["page_index"],
    ),
    _tool(
        "calculate",
        "Evaluate arithmetic: numbers, + - * / **, parentheses, unary minus.",
        {"expression": {"type": "string"}},
        ["expression"],
    ),
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer; its quotes and numbers are checked.",
            "parameters": AgentAnswer.model_json_schema(),
        },
    },
]
