"""Least-autonomy controls, request budget and cache."""
from __future__ import annotations
import re
from dataclasses import dataclass

class BudgetExceeded(ValueError): pass
class UnauthorizedToolUse(PermissionError): pass

# Application-layer risk classification (Task 15).
# Low = summarization/transcription; Medium = code generation / customer-support
# tickets; High = medical, hiring, financial data. This advisory support tool is Medium.
RISK_CLASSIFICATION = "Medium"
RISK_JUSTIFICATION = (
    "Customer-support ticket tool: answers policy questions and surfaces synthetic "
    "order status. Advisory only - no order modification, payment, refund, or account "
    "action. Phone numbers and card last-4 are masked before the model and logging; "
    "free-text names/addresses are out of scope for a keyless masker (fabricated "
    "examples only)."
)

def enforce_tool_access(agent_name: str, tool_name: str):
    allowed={"Lookup Agent":{"check_order_status"}}
    if tool_name not in allowed.get(agent_name,set()):
        raise UnauthorizedToolUse(f"{agent_name!r} is not authorized to call {tool_name!r}")

@dataclass
class RequestBudget:
    max_chars: int = 3000
    def check(self, text: str):
        if len(text) > self.max_chars:
            raise BudgetExceeded(f"Request has {len(text)} characters; budget is {self.max_chars}.")

class ResponseCache:
    def __init__(self):
        self._data={}
        self.calls=0
        self.hits=0
    @staticmethod
    def key(query):
        return re.sub(r"\s+"," ",query.strip().lower())
    def get_or_compute(self, query, fn):
        key=self.key(query)
        if key in self._data:
            self.hits += 1
            return {**self._data[key],"cache_hit":True}
        self.calls += 1
        value=fn()
        self._data[key]=dict(value)
        return {**value,"cache_hit":False}
