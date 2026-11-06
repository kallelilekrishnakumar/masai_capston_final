"""Two-reviewer approval/revision stage with a real AutoGen RoundRobinGroupChat.

Default path is key-free: ``MockReviewModelClient`` is a deterministic
``ChatCompletionClient`` that drives AutoGen's real two-agent team
(``RoundRobinGroupChat`` with ``max_turns=2``) so the review actually executes
under ``MOCK_LLM`` with zero API keys. A real AutoGen model client
(e.g. an Ollama/OpenAI client) can be passed instead for an optional real run.

The deterministic ``review_draft()`` remains as a plain-Python fallback used only
when AutoGen is unavailable, and that is documented in the README.
"""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, List, Mapping, Optional, Sequence

from pydantic import BaseModel


class Verdict(BaseModel):
    """Structured review verdict produced by the Final-Editor agent."""
    approved: bool
    final_answer: str
    reason: str


# Phrases that are never supported by the illustrative policy context.
RISKY_CLAIMS = (
    "guaranteed delivery tomorrow",
    "will definitely refund",
    "official nykaa policy guarantees",
    "guaranteed to arrive",
)
INJECTED_CLAIM = "Guaranteed delivery tomorrow is confirmed."


def review_draft(draft: str, context: str, inject_test_claim: bool = False) -> Verdict:
    """Plain-Python MOCK_LLM review used as a key-free fallback when AutoGen is unavailable.

    Deterministic policy checks with explicit approval/revision. Not a
    ``RoundRobinGroupChat``; ``run_autogen_review`` is the real AutoGen path.
    """
    candidate = draft
    if inject_test_claim:
        candidate += " " + INJECTED_CLAIM
    if any(x in candidate.lower() for x in RISKY_CLAIMS):
        safe = candidate.split(INJECTED_CLAIM)[0].strip()
        return Verdict(approved=False, final_answer=safe,
                       reason="Policy-Compliance-Reviewer detected an unsupported guarantee; Final-Editor removed it.")
    return Verdict(approved=True, final_answer=draft, reason="No unsupported claim detected; approved unchanged.")


# --- deterministic AutoGen model client --------------------------------------


def _message_text(messages: Sequence[Any]) -> str:
    parts: List[str] = []
    for m in messages:
        if isinstance(m, dict):
            content = m.get("content", "")
        else:
            content = getattr(m, "content", "")
        if isinstance(content, list):  # multimodal blocks
            content = " ".join(str(b.get("text", b)) if isinstance(b, dict) else str(b) for b in content)
        parts.append(content or "")
    return "\n".join(parts)


def _draft_from(blob: str) -> str:
    if "CREWAI DRAFT:" in blob:
        seg = blob.split("CREWAI DRAFT:", 1)[1]
        seg = seg.split("Review and finalize", 1)[0]
        return seg.strip()
    return blob.strip()


def _is_risky(text: str) -> bool:
    low = (text or "").lower()
    return any(x in low for x in RISKY_CLAIMS)


class MockReviewModelClient:
    """Keyless deterministic ``ChatCompletionClient`` for the AutoGen review team.

    Reviewer turn (no structured output): returns plain review text.
    Editor turn (``json_output`` set to ``Verdict``): returns JSON that
    ``Verdict.model_validate_json`` parses into a ``StructuredMessage[Verdict]``.
    """

    def __init__(self) -> None:
        self._total = {"prompt_tokens": 0, "completion_tokens": 0}

    # -- ChatCompletionClient interface ---------------------------------------
    async def create(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] = (),
        json_output: Optional[Any] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[Any] = None,
    ):
        from autogen_core.models import CreateResult, RequestUsage
        blob = _message_text(messages)
        draft = _draft_from(blob)
        risky = _is_risky(blob)

        if json_output:  # Final-Editor producing the structured verdict
            if risky:
                clean = draft.split(INJECTED_CLAIM)[0].strip()
                verdict = Verdict(approved=False, final_answer=clean,
                                  reason="Policy-Compliance-Reviewer flagged an unsupported guarantee; Final-Editor removed it.")
            else:
                verdict = Verdict(approved=True, final_answer=draft,
                                  reason="No unsupported claim detected; approved unchanged.")
            content = verdict.model_dump_json()
        else:  # Policy-Compliance-Reviewer producing review feedback
            if risky:
                content = (f"The draft contains an unsupported guarantee ('{INJECTED_CLAIM}') that is not in the "
                           "supplied context. Remove it before finalizing.")
            else:
                content = "The draft is grounded in the supplied context with no unsupported claims. Approve unchanged."

        self._total["prompt_tokens"] += len(blob) // 4
        self._total["completion_tokens"] += len(content) // 4
        return CreateResult(finish_reason="stop", content=content,
                            usage=RequestUsage(prompt_tokens=0, completion_tokens=0), cached=False)

    async def create_stream(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] = (),
        json_output: Optional[Any] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[Any] = None,
    ) -> AsyncGenerator[Any, None]:
        result = await self.create(messages, tools=tools, json_output=json_output,
                                   extra_create_args=extra_create_args, cancellation_token=cancellation_token)
        yield result  # type: ignore[misc]

    async def close(self) -> None:
        return None

    def actual_usage(self):
        from autogen_core.models import RequestUsage
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def total_usage(self):
        from autogen_core.models import RequestUsage
        return RequestUsage(prompt_tokens=self._total["prompt_tokens"], completion_tokens=self._total["completion_tokens"])

    def count_tokens(self, messages: Sequence[Any], *, tools: Sequence[Any] = ()) -> int:
        return len(_message_text(messages)) // 4

    def remaining_tokens(self, messages: Sequence[Any], *, tools: Sequence[Any] = ()) -> int:
        return 8192 - self.count_tokens(messages, tools=tools)

    @property
    def model_info(self) -> dict:
        from autogen_core.models import ModelFamily
        return {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": True,
        }

    @property
    def capabilities(self) -> dict:
        return self.model_info

    def _to_config(self) -> dict:
        return {}


# --- real AutoGen team --------------------------------------------------------


async def run_autogen_review(draft: str, context: str, model_client=None) -> Verdict:
    """Run a real AutoGen two-agent ``RoundRobinGroupChat`` and return the verdict.

    Uses ``MockReviewModelClient`` by default (zero API keys). The reviewer gives
    feedback; the Final-Editor returns a ``StructuredMessage[Verdict]``. Raises if
    the team produces no Verdict (no silent fallback).
    """
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.messages import StructuredMessage
    model_client = model_client or MockReviewModelClient()

    reviewer = AssistantAgent(
        name="policy_compliance_reviewer",
        model_client=model_client,
        system_message=("Review the draft against only the supplied context. Identify unsupported claims. "
                        "Do not invent policy. Give concise review feedback."),
    )
    editor = AssistantAgent(
        name="final_editor",
        model_client=model_client,
        system_message=("Produce the final customer-support answer. Remove unsupported claims. "
                        "Return approved, final_answer, and reason as the structured verdict."),
        output_content_type=Verdict,
    )
    team = RoundRobinGroupChat(
        [reviewer, editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[Verdict]],
    )
    # NOTE: real newlines, not literal "\n" (the previous version used "\\n").
    task = f"ORIGINAL CONTEXT:\n{context}\n\nCREWAI DRAFT:\n{draft}\n\nReview and finalize."
    result = await team.run(task=task)
    for message in reversed(result.messages):
        content = getattr(message, "content", None)
        if isinstance(content, Verdict):
            return content
        if isinstance(content, dict):
            try:
                return Verdict.model_validate(content)
            except Exception:
                continue
    raise RuntimeError("AutoGen review produced no Verdict; the Final-Editor did not emit a structured verdict.")


def run_autogen_review_demo() -> list:
    """Demonstrate the real AutoGen review stage: one approve-unchanged, one revise.

    Returns transcript lines. The revise case injects an unsupported guarantee
    into the draft; the reviewer flags it and the Final-Editor removes it.
    """
    import asyncio
    context = ("For an eligible COD return, the customer provides a supported bank account. After the returned "
               "item passes inspection, the illustrative refund target is 5-7 business days.")
    clean_draft = "Eligible COD refunds target 5-7 business days after inspection."
    bad_draft = clean_draft + " " + INJECTED_CLAIM  # deliberately ungrounded claim

    lines = ["=== AUTOGEN REVIEW STAGE (RoundRobinGroupChat, max_turns=2) ==="]
    v1 = asyncio.run(run_autogen_review(clean_draft, context))
    lines += [
        "Case 1 (approve-unchanged):",
        f"  draft: {clean_draft}",
        f"  verdict: approved={v1.approved} | reason={v1.reason}",
        f"  final_answer: {v1.final_answer}",
    ]
    v2 = asyncio.run(run_autogen_review(bad_draft, context))
    lines += [
        "Case 2 (revise after injected ungrounded claim):",
        f"  draft: {bad_draft}",
        f"  verdict: approved={v2.approved} | reason={v2.reason}",
        f"  final_answer: {v2.final_answer}",
    ]
    return lines


def _autogen_review_sync(draft: str, context: str) -> Verdict:
    """Run the real AutoGen team synchronously, safe inside a running event loop.

    Falls back to ``review_draft`` on any failure so callers never break.
    """
    import asyncio
    import threading
    try:
        try:
            asyncio.get_running_loop()
            running = True
        except RuntimeError:
            running = False
        if not running:
            return asyncio.run(run_autogen_review(draft, context))
        box: dict = {}

        def _worker() -> None:
            try:
                box["v"] = asyncio.run(run_autogen_review(draft, context))
            except BaseException as e:  # noqa: BLE001
                box["e"] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join()
        if "e" in box:
            raise box["e"]  # type: ignore[misc]
        return box["v"]
    except Exception:
        return review_draft(draft, context)


def review_answer(draft: str, context: str, use_autogen: bool = False) -> Verdict:
    """Synchronous review used by the request path.

    Default is the key-free deterministic ``review_draft`` (cheap, no asyncio), so
    every policy answer still passes through a review stage under ``MOCK_LLM``. Set
    ``use_autogen=True`` to run the real AutoGen ``RoundRobinGroupChat`` instead.
    """
    if use_autogen:
        return _autogen_review_sync(draft, context)
    return review_draft(draft, context)
