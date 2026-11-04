"""The structured answer every system returns (also the strict JSON schema sent to the model)."""

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, WithJsonSchema


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Citation(_Strict):
    page_index: int
    quote: str


class FinalValue(_Strict):
    # a plain JSON number for the model; the default Decimal schema also admits strings
    value: Annotated[Decimal, WithJsonSchema({"type": "number"})]
    unit: Literal["usd", "percent", "ratio", "count", "other"]


class AgentAnswer(_Strict):
    answer: str
    final_value: FinalValue | None
    declined: bool
    decline_reason: str | None
    citations: list[Citation]
