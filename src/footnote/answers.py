"""The structured answer every system returns (also the strict JSON schema sent to the model)."""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Citation(_Strict):
    page_index: int
    quote: str


class FinalValue(_Strict):
    value: Decimal
    unit: Literal["usd", "percent", "ratio", "count", "other"]


class AgentAnswer(_Strict):
    answer: str
    final_value: FinalValue | None
    declined: bool
    decline_reason: str | None
    citations: list[Citation]
