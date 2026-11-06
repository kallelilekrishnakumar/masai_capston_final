"""Deterministic MOCK_LLM "brain" for CrewAI so ``.kickoff()`` runs with zero API keys.

CrewAI's agent loop calls ``LLM.call(messages, ...)`` and parses the returned
text as ReAct output (``Thought / Action / Action Input`` or
``Thought / Final Answer``). This mock returns deterministic ReAct text so the
real CrewAI orchestration executes and the real tools (``retrieve_policy`` and
``check_order_status``) are genuinely invoked through CrewAI's own loop.

It deliberately does NOT define ``supports_function_calling`` so CrewAI falls
back to its text-parsed ReAct path (no native OpenAI tool-call schema needed).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from crewai.llms.base_llm import BaseLLM

ORDER_RE = re.compile(r"NYK-\d{4}", re.IGNORECASE)
POLICY_KW = (
    "return", "refund", "delivery", "warranty", "cancel", "loyalty", "payment",
    "exchange", "damage", "pickup", "shipping", "sla", "policy", "escalat",
    "international", "redeem", "cod",
)

_DEBUG = os.getenv("MOCK_LLM_DEBUG") == "1"
_DEBUG_PATH = Path(__file__).parent / "artifacts" / "crewai_debug.log"


def _text_of(message) -> str:
    """Flatten a CrewAI message (str/dict/object, possibly multimodal) to text."""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "") or ""
    if isinstance(content, list):  # multimodal content blocks
        content = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content or ""


class MockCrewLLM(BaseLLM):
    """Keyless, deterministic ReAct model that drives a real CrewAI crew."""

    model: str = "mock-llm"

    def supports_stop_words(self) -> bool:  # keep parsing simple
        return False

    # -- the one abstract method -------------------------------------------------
    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ) -> str:
        role = getattr(from_agent, "role", "") if from_agent is not None else ""
        blob = "\n".join(_text_of(m) for m in messages)
        if _DEBUG:
            _DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _DEBUG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(f"\n=== role={role!r} ===\n{blob[:2000]}\n")

        # The prompt template literally contains "Observation: the result of the
        # action", so we cannot detect a prior tool call by searching for that
        # word. Instead: a tool has already run iff an assistant turn exists.
        acted = self._acted(messages)
        order = ORDER_RE.search(blob)
        query = self._query(blob)

        if role == "Retrieval Agent":
            if acted:
                return self._finish("retrieved policy context", self._obs(blob))
            if self._is_policy(query):
                arg = json.dumps({"query": query or "Nykaa policy"})
                return self._action(
                    "This is a policy question, so I will retrieve the relevant policy.",
                    "retrieve_policy",
                    arg,
                )
            return self._finish(
                "not a policy question, so no policy retrieval is needed",
                "No policy retrieval needed for this request.",
            )

        if role == "Lookup Agent":
            if acted:
                return self._finish("order status", self._obs(blob))
            if order:
                return self._action(
                    "An order ID is present, so I will look it up.",
                    "check_order_status",
                    json.dumps({"record_id": order.group(0)}),
                )
            return self._finish(
                "no order ID is present, so no lookup is needed",
                "No order lookup was needed for this query.",
            )

        # Response Composer has no tools -> always produce a final answer.
        return self._finish(
            "compose a grounded final answer from the available context",
            self._compose(blob, query, order),
        )

    # -- helpers ------------------------------------------------------------------
    @staticmethod
    def _action(thought: str, tool: str, action_input: str) -> str:
        return f"Thought: {thought}\nAction: {tool}\nAction Input: {action_input}"

    @staticmethod
    def _finish(thought: str, answer: str) -> str:
        return f"Thought: {thought}\nFinal Answer: {answer}"

    @staticmethod
    def _query(blob: str) -> str:
        for marker in ("policy question:", "grounded answer to:"):
            if marker in blob:
                tail = blob.split(marker, 1)[1].strip()
                return tail.split("\n", 1)[0].strip().strip(".").strip()
        return ""

    @staticmethod
    def _is_policy(query: str) -> bool:
        low = (query or "").lower()
        return any(k in low for k in POLICY_KW)

    @staticmethod
    def _obs(blob: str) -> str:
        obs = blob.rsplit("Observation:", 1)[-1].strip()
        # CrewAI appends format reminders/notes to observations; drop them so
        # the agent's final answer is the clean tool result only.
        for cut in ("Analyze the tool result.", "Do not use a tool you have already used.", "\n(Tool"):
            if cut in obs:
                obs = obs.split(cut)[0].strip()
        return obs

    @staticmethod
    def _role(message) -> str:
        if isinstance(message, dict):
            return str(message.get("role", ""))
        return str(getattr(message, "role", "") or "")

    @staticmethod
    def _acted(messages) -> bool:
        # A tool has already run iff the conversation already has an assistant
        # turn (the Action CrewAI appended) before this LLM call.
        return any(MockCrewLLM._role(m) == "assistant" for m in messages)

    @staticmethod
    def _compose(blob: str, query: str, order: re.Match | None) -> str:
        # The composer's context (t1 + t2 outputs) follows this header; isolate it
        # so we don't echo the surrounding task/format template text.
        ctx = ""
        if "This is the context you're working with:" in blob:
            ctx = blob.split("This is the context you're working with:", 1)[1]
            ctx = ctx.split("Current Task:")[0].strip()
        parts = ctx.split("----------") if "----------" in ctx else [ctx]

        if order:
            seg = parts[-1] if len(parts) > 1 else ctx
            m = re.search(r"Order NYK-\d{4} is \w+[^\n]*", seg or blob, re.IGNORECASE)
            if m:
                return m.group(0).strip()

        seg = (parts[0] if parts else ctx).strip()
        if seg:
            return seg[:700].strip()
        return f"Based on the retrieved context, here is the grounded answer to: {query or 'your question'}."
