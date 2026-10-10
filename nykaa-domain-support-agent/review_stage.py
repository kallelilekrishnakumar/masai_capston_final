"""Two-reviewer approval/revision stage; optional AutoGen RoundRobinGroupChat adapter."""
from pydantic import BaseModel
from typing import Optional

class Verdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str

def review_draft(draft: str, context: str, inject_test_claim: bool=False) -> Verdict:
    """MOCK_LLM review: deterministic policy checks with explicit approval/revision."""
    candidate=draft
    if inject_test_claim:
        candidate += " Guaranteed delivery tomorrow is confirmed."
    risky=("guaranteed delivery tomorrow","will definitely refund","official nykaa policy guarantees")
    if any(x in candidate.lower() for x in risky):
        safe=draft.split(" Guaranteed delivery tomorrow is confirmed.")[0]
        return Verdict(approved=False,final_answer=safe,
                       reason="Policy-Compliance-Reviewer detected an unsupported guarantee; Final-Editor removed it.")
    # If draft has text and a supplied context, the mock reviewer leaves it unchanged.
    return Verdict(approved=True,final_answer=draft,reason="No unsupported claim detected; approved unchanged.")

async def run_autogen_review(draft: str, context: str, model_client):
    """Run an AutoGen two-agent RoundRobinGroupChat using a caller-supplied local model client.

    The caller supplies a configured AutoGen-compatible model_client (for example a local
    Ollama client). This keeps the default MOCK_LLM path free of API keys.
    """
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.messages import StructuredMessage
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
                        "Return approved, final_answer, and reason as structured verdict fields."),
        output_content_type=Verdict,
    )
    team = RoundRobinGroupChat(
        [reviewer, editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[Verdict]],
    )
    task = f"ORIGINAL CONTEXT:\\n{context}\\n\\nCREWAI DRAFT:\\n{draft}\\n\\nReview and finalize."
    result = await team.run(task=task)
    for message in reversed(result.messages):
        content=getattr(message,"content",None)
        if isinstance(content,Verdict): return content
        if isinstance(content,dict):
            try: return Verdict.model_validate(content)
            except Exception: pass
    return review_draft(draft,context)
