"""Guardrails, deterministic orchestration, memory and optional CrewAI adapter."""
from __future__ import annotations
import re
from typing import Optional, Literal
from pydantic import BaseModel, Field
from dataset import ORDERS
from rag import LocalRetriever
from governance import enforce_tool_access, ResponseCache

PHONE_RE=re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
CARD_RE=re.compile(r"(?i)(card(?:\s*(?:last[- ]?4|ending))?\s*[:#]?\s*)(\d{4})")
INJECTION_PATTERNS=("ignore all previous instructions","reveal your system prompt","bypass guardrails","act as unrestricted")

def mask_pii(text: str) -> str:
    text=PHONE_RE.sub("[PHONE_MASKED]",text)
    return CARD_RE.sub(lambda m:m.group(1)+"[LAST4_MASKED]",text)

def input_guardrail(text: str) -> dict:
    masked=mask_pii(text)
    lowered=masked.lower()
    injection=next((p for p in INJECTION_PATTERNS if p in lowered),None)
    return {"allowed":injection is None,"masked_text":masked,"injection_detected":injection is not None,
            "reason":f"Prompt injection phrase detected: {injection}" if injection else None}

class SupportResponse(BaseModel):
    answer: str
    intent: Literal["policy","order_status","mixed","unsupported","blocked"]
    grounded: bool
    sources: list[str] = Field(default_factory=list)
    order_status: Optional[str] = None
    escalation_score: Optional[float] = None
    escalation_recommended: Optional[bool] = None
    trace_notes: list[str] = Field(default_factory=list)

def check_order_status(record_id: str) -> dict:
    for row in ORDERS:
        if row["record_id"].lower()==record_id.strip().lower():
            age=row["days_since_created"]/30
            score=round(min(1,max(0,0.65*int(row["delayed_shipment"])+0.35*age)),3)
            return {"record_id":row["record_id"],"status":row["status"],"order_value_inr":row["order_value_inr"],
                    "days_since_created":row["days_since_created"],"delayed_shipment":row["delayed_shipment"],
                    "escalation_score":score,"escalation_recommended":score>=0.65}
    return {"record_id":record_id,"error":"Order not found"}

class SessionMemory:
    def __init__(self): self.history={}
    def add(self, session_id, role, content):
        self.history.setdefault(session_id,[]).append({"role":role,"content":content})
    def get(self, session_id): return list(self.history.get(session_id,[]))
    def reset(self, session_id): self.history.pop(session_id,None)

class SupportOrchestrator:
    def __init__(self, retriever=None):
        self.retriever=retriever or LocalRetriever("fixed")
        self.cache=ResponseCache()
        self.memory=SessionMemory()
        self.tool_calls={"rag":0,"lookup":0}
    def rag_tool(self, query):
        self.tool_calls["rag"]+=1
        return self.cache.get_or_compute(query,lambda:self.retriever.answer(query))
    def lookup_tool(self, record_id, agent_name="Lookup Agent"):
        enforce_tool_access(agent_name,"check_order_status")
        self.tool_calls["lookup"]+=1
        return check_order_status(record_id)
    def ask(self, query, session_id="default"):
        guard=input_guardrail(query)
        safe_query=guard["masked_text"]
        if not guard["allowed"]:
            result=SupportResponse(answer="I can't help with attempts to override the support system's instructions.",
                intent="blocked",grounded=False,trace_notes=["prompt-injection guard fired"])
        else:
            record=re.search(r"\bNYK-\d{4}\b",safe_query,re.I)
            if record:
                info=self.lookup_tool(record.group(0))
                if "error" in info:
                    result=SupportResponse(answer="I couldn't find that order reference. Please verify it and try again.",
                                           intent="order_status",grounded=True)
                else:
                    result=SupportResponse(answer=(f"Order {info['record_id']} is {info['status']}. "
                        f"Order value: ₹{info['order_value_inr']}. Escalation score: {info['escalation_score']:.3f}. "
                        f"{'Recommended for escalation.' if info['escalation_recommended'] else 'No escalation threshold reached.'}"),
                        intent="order_status",grounded=True,order_status=info["status"],
                        escalation_score=info["escalation_score"],escalation_recommended=info["escalation_recommended"],
                        trace_notes=["Lookup Agent invoked check_order_status"])
            else:
                rag=self.rag_tool(safe_query)
                result=SupportResponse(answer=rag["answer"],intent="policy" if rag["grounded"] else "unsupported",
                    grounded=rag["grounded"],sources=rag["sources"],
                    trace_notes=["Retrieval Agent invoked RAG tool", "cache hit" if rag.get("cache_hit") else "cache miss"])
        self.memory.add(session_id,"user",safe_query)
        self.memory.add(session_id,"assistant",result.answer)
        return result

def build_crewai_crew(orchestrator=None):
    """Optional real CrewAI crew; Lookup Agent alone receives the order tool."""
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import tool
    orchestrator=orchestrator or SupportOrchestrator()
    @tool("retrieve_policy")
    def retrieve_policy(query: str) -> str:
        """Retrieve a grounded answer from the local Nykaa policy knowledge base."""
        return str(orchestrator.rag_tool(query))
    @tool("check_order_status")
    def order_lookup(record_id: str) -> str:
        """Look up a synthetic order by record ID. Only Lookup Agent is given this tool."""
        return str(orchestrator.lookup_tool(record_id,agent_name="Lookup Agent"))
    retrieval=Agent(role="Retrieval Agent",goal="Answer policy questions using retrieved policy text only.",
        backstory="A careful retrieval specialist.",tools=[retrieve_policy],verbose=False)
    lookup=Agent(role="Lookup Agent",goal="Look up synthetic order records and report exact status.",
        backstory="A constrained order-record specialist.",tools=[order_lookup],verbose=False)
    composer=Agent(role="Response Composer",goal="Combine agent findings without inventing facts.",
        backstory="A conservative customer support writer.",tools=[],verbose=False)
    t1=Task(description="For this request, retrieve relevant policy context if it is a policy question: {query}",agent=retrieval,expected_output="Retrieved context or explicit unsupported result.")
    t2=Task(description="If an order ID appears in {query}, look it up. Otherwise say no lookup was needed.",agent=lookup,expected_output="Order details or no lookup needed.")
    t3=Task(description="Compose a concise grounded answer to: {query}. Do not add unsupported policy claims.",agent=composer,context=[t1,t2],expected_output="A safe customer-support answer.")
    return Crew(agents=[retrieval,lookup,composer],tasks=[t1,t2,t3],process=Process.sequential,verbose=False)
