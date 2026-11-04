from decimal import Decimal

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from footnote.answers import AgentAnswer, Citation, FinalValue


def test_round_trip() -> None:
    a = AgentAnswer(
        answer="Revenue was $1.6 billion.",
        final_value=FinalValue(value=Decimal("1.6e9"), unit="usd"),
        declined=False,
        decline_reason=None,
        citations=[Citation(page_index=3, quote="Net sales 1,577")],
    )
    assert AgentAnswer.model_validate_json(a.model_dump_json()) == a


def test_decline_shape() -> None:
    a = AgentAnswer(
        answer="Not in the filing.",
        final_value=None,
        declined=True,
        decline_reason="not disclosed",
        citations=[],
    )
    assert a.final_value is None


def test_unit_is_closed_set() -> None:
    with pytest.raises(ValidationError):
        FinalValue(value=Decimal(1), unit="eur")  # type: ignore[arg-type]


def test_fields_have_no_defaults() -> None:
    with pytest.raises(ValidationError):
        AgentAnswer.model_validate({"answer": "x"})


def test_strict_schema_converts_and_all_fields_required() -> None:
    schema = to_strict_json_schema(AgentAnswer)
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    for name, sub in schema["$defs"].items():
        assert set(sub["required"]) == set(sub["properties"]), name


def test_final_value_schema_is_a_plain_number() -> None:
    from openai.lib._parsing._completions import type_to_response_format_param

    fmt = type_to_response_format_param(AgentAnswer)
    schema = fmt["json_schema"]["schema"]  # type: ignore[typeddict-item]
    assert schema["$defs"]["FinalValue"]["properties"]["value"]["type"] == "number"
    assert fmt["json_schema"]["strict"] is True  # type: ignore[typeddict-item]


@pytest.mark.parametrize("raw", ["1577", "1577.00", "-5", "1.6e9", "0.1"])
def test_exact_decimals_still_validate(raw: str) -> None:
    f = FinalValue.model_validate_json(f'{{"value": {raw}, "unit": "usd"}}')
    assert f.value == Decimal(raw)
