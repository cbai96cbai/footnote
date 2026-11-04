from pydantic import BaseModel

from footnote.llm.tokens import count_input_tokens, count_text_tokens

MSGS = [{"role": "user", "content": "What was revenue in 2022?"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read one page",
            "parameters": {"type": "object", "properties": {"page": {"type": "integer"}}},
        },
    }
]


class Verdict(BaseModel):
    answer: str
    page: int


def test_count_text_tokens() -> None:
    assert count_text_tokens("hello world") == 2
    assert count_text_tokens("") == 0


def test_input_count_base_is_content_plus_4_plus_300() -> None:
    assert count_input_tokens(MSGS, None, None) == count_text_tokens(MSGS[0]["content"]) + 4 + 300


def test_input_count_includes_tools_and_schema() -> None:
    base = count_input_tokens(MSGS, None, None)
    with_tools = count_input_tokens(MSGS, TOOLS, None)
    with_schema = count_input_tokens(MSGS, None, Verdict)
    assert with_tools > base + 20
    assert with_schema > base + 10
    assert count_input_tokens(MSGS, TOOLS, Verdict) > max(with_tools, with_schema)


def test_input_count_includes_tool_calls_and_results() -> None:
    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "read_page", "arguments": '{"page": 12345}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "Total revenue was 9,999 million"},
    ]
    n = count_input_tokens(msgs, None, None)
    expected = (
        count_text_tokens("q")
        + count_text_tokens('{"page": 12345}')
        + count_text_tokens("Total revenue was 9,999 million")
        + 2 * count_text_tokens("1")  # call id and tool_call_id
        + count_text_tokens("read_page")
        + 3 * 4
        + 300
    )
    assert n == expected


def test_input_count_handles_content_parts() -> None:
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hello world"}]}]
    assert count_input_tokens(msgs, None, None) == 2 + 4 + 300


def test_tool_names_and_ids_are_counted() -> None:
    call_id = "call_" + "abcdef0123456789" * 8
    long_name = "read_page_" + "x" * 200
    history = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": long_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "name": long_name, "content": "r"},
    ]
    bare = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"arguments": "{}"}}]},
        {"role": "tool", "content": "r"},
    ]
    extra = count_input_tokens(history, None, None) - count_input_tokens(bare, None, None)
    assert extra >= 2 * count_text_tokens(call_id) + 2 * count_text_tokens(long_name)
