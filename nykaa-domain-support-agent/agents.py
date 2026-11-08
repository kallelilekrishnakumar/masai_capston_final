"""Guardrails, deterministic orchestration, memory and optional CrewAI adapter."""
from __future__ import annotations
import re
from typing import Optional, Literal
from pydantic import BaseModel, Field
from dataset import ORDERS, escalation_score, ESCALATION_THRESHOLD
from rag import LocalRetriever
from governance import enforce_tool_access, ResponseCache
from review_stage import review_answer

PHONE_RE=re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
CARD_RE=re.compile(r"(?i)(card(?:\s*(?:last[- ]?4|ending(?:\s+in)?))?\s*[:#]?\s*)(\d{4})(?!\d)")
FULL_CARD_RE=re.compile(r"(?<!\d)(?:\d[ -]?){12,15}(\d{4})(?!\d)")
INJECTION_REGEXES=(
    re.compile(r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+(?:instructions|prompts?)"),
    re.compile(r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)"),
    re.compile(r"reveal\s+(?:your\s+|the\s+)?system\s+prompt"),
    re.compile(r"bypass\s+(?:the\s+)?guardrails?"),
    re.compile(r"act\s+as\s+(?:an?\s+)?unrestricted"),
)

def mask_pii(text: str) -> str:
    text=PHONE_RE.sub("[PHONE_MASKED]",text)
    text=FULL_CARD_RE.sub("[CARD_MASKED]",text)
    return CARD_RE.sub(lambda m:m.group(1)+"[LAST4_MASKED]",text)

def input_guardrail(text: str) -> dict:
    masked=mask_pii(text)
    lowered=masked.lower()
    hit=next((rx for rx in INJECTION_REGEXES if rx.search(lowered)),None)
    injection=hit.search(lowered).group(0) if hit else None
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
            score=escalation_score(row)
            return {"record_id":row["record_id"],"status":row["status"],"order_value_inr":row["order_value_inr"],
                    "days_since_created":row["days_since_created"],"delayed_shipment":row["delayed_shipment"],
                    "escalation_score":score,"escalation_threshold":ESCALATION_THRESHOLD,
                    "escalation_recommended":score>=ESCALATION_THRESHOLD}
    return {"record_id":record_id,"error":"Order not found"}

class SessionMemory:
    def __init__(self): self.history={}
    def add(self, session_id, role, content):
        self.history.setdefault(session_id,[]).append({"role":role,"content":content})
    def get(self, session_id): return list(self.history.get(session_id,[]))
    def reset(self, session_id): self.history.pop(session_id,None)

class SupportOrchestrator:
    def __init__(self, retriever=None):
        self.retriever=retriever or LocalRetriever("sentence")  # recommended chunking strategy (see README)
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
    def make_lookup_tool(self, agent_name):
        """Return a lookup callable bound to ONE agent identity (least autonomy).

        The identity is fixed at wiring time, so handing this tool to any agent other
        than the Lookup Agent fails with UnauthorizedToolUse when it is called.
        """
        def bound(record_id: str):
            return self.lookup_tool(record_id, agent_name=agent_name)
        return bound
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
                answer=rag["answer"]; grounded=rag["grounded"]
                notes=["Retrieval Agent invoked RAG tool", "cache hit" if rag.get("cache_hit") else "cache miss"]
                if grounded:
                    ctx="\n".join(d["text"] for d in self.retriever.documents if d["doc_id"] in rag["sources"])
                    verdict=review_answer(answer, ctx)
                    answer=verdict.final_answer
                    notes.append(f"review stage: approved={verdict.approved} | {verdict.reason}")
                result=SupportResponse(answer=answer,intent="policy" if grounded else "unsupported",
                    grounded=grounded,sources=rag["sources"],trace_notes=notes)
        self.memory.add(session_id,"user",safe_query)
        self.memory.add(session_id,"assistant",result.answer)
        return result

def build_crewai_crew(orchestrator=None, llm=None):
    """Real CrewAI crew run via ``.kickoff()`` under MOCK_LLM with zero API keys.

    The deterministic ``MockCrewLLM`` (see ``mock_llm.py``) is the brain: it emits
    ReAct text so CrewAI's own orchestration loop drives the real tools. Only the
    Lookup Agent is wired to ``check_order_status``; the Composer has no tools.
    """
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import tool
    from mock_llm import MockCrewLLM
    orchestrator = orchestrator or SupportOrchestrator()
    llm = llm or MockCrewLLM(model="mock-llm")

    @tool("retrieve_policy")
    def retrieve_policy(query: str) -> str:
        """Retrieve a grounded answer from the local Nykaa policy knowledge base."""
        result = orchestrator.rag_tool(query)
        return result.get("answer", str(result)) if isinstance(result, dict) else str(result)

    bound_lookup = orchestrator.make_lookup_tool("Lookup Agent")  # identity bound at wiring time

    @tool("check_order_status")
    def order_lookup(record_id: str) -> str:
        """Look up a synthetic order by record ID. Only Lookup Agent is given this tool."""
        info = bound_lookup(record_id)
        if isinstance(info, dict) and "error" in info:
            return info["error"]
        return (f"Order {info['record_id']} is {info['status']}. Order value: Rs.{info['order_value_inr']}. "
                f"Escalation score: {info['escalation_score']:.3f}. "
                f"{'Recommended for escalation.' if info['escalation_recommended'] else 'No escalation threshold reached.'}")

    retrieval = Agent(role="Retrieval Agent", goal="Answer policy questions using retrieved policy text only.",
        backstory="A careful retrieval specialist.", tools=[retrieve_policy], llm=llm, verbose=False)
    lookup = Agent(role="Lookup Agent", goal="Look up synthetic order records and report exact status.",
        backstory="A constrained order-record specialist.", tools=[order_lookup], llm=llm, verbose=False)
    composer = Agent(role="Response Composer", goal="Combine agent findings without inventing facts.",
        backstory="A conservative customer support writer.", tools=[], llm=llm, verbose=False)
    t1 = Task(description="For this request, retrieve relevant policy context if it is a policy question: {query}", agent=retrieval, expected_output="Retrieved context or explicit unsupported result.")
    t2 = Task(description="If an order ID appears in {query}, look it up. Otherwise say no lookup was needed.", agent=lookup, expected_output="Order details or no lookup needed.")
    t3 = Task(description="Compose a concise grounded answer to: {query}. Do not add unsupported policy claims.", agent=composer, context=[t1, t2], expected_output="A safe customer-support answer.")
    return Crew(agents=[retrieval, lookup, composer], tasks=[t1, t2, t3], process=Process.sequential, verbose=False), orchestrator


_CREW_SINGLETON = {}

def crew_answer(query, orchestrator):
    """Run the real CrewAI crew for one query and return a validated SupportResponse.

    This is how the API exposes the crew (Task 11): the crew's .kickoff() string is
    wrapped into the declared Pydantic schema (Task 9), re-using the order metadata
    the Lookup Agent already produced on the orchestrator. The crew is built once per
    orchestrator and reused across requests.
    """
    key=id(orchestrator)
    crew=_CREW_SINGLETON.get(key)
    if crew is None:
        try:
            from crewai.events.listeners.tracing import set_suppress_tracing_messages
            set_suppress_tracing_messages(True)
        except Exception:
            pass
        crew,_=build_crewai_crew(orchestrator=orchestrator)
        _CREW_SINGLETON[key]=crew
    text=str(crew.kickoff(inputs={"query":query}))
    record=re.search(r"\bNYK-\d{4}\b",query,re.I)
    if record:
        info=check_order_status(record.group(0))
        if "error" not in info:
            return SupportResponse(answer=text,intent="order_status",grounded=True,
                order_status=info["status"],escalation_score=info["escalation_score"],
                escalation_recommended=info["escalation_recommended"],
                trace_notes=["CrewAI .kickoff(): Lookup Agent -> check_order_status"])
    grounded="i don't know" not in text.lower()
    return SupportResponse(answer=text,intent="policy" if grounded else "unsupported",
        grounded=grounded,trace_notes=["CrewAI .kickoff(): Retrieval Agent -> retrieve_policy -> Composer"])

def run_crewai_kickoff_demo(queries=None, orchestrator=None):
    """Run the real CrewAI crew via ``.kickoff()`` and return transcript lines.

    Demonstrates that both the RAG tool (``retrieve_policy``) and the order-lookup
    tool (``check_order_status``) are actually invoked through CrewAI's loop on
    different queries. Evidence is the orchestrator's ``tool_calls`` counter,
    which the real tools increment only when CrewAI calls them.
    """
    try:
        from crewai.events.listeners.tracing import set_suppress_tracing_messages
        set_suppress_tracing_messages(True)
    except Exception:
        pass

    queries = queries or [
        "What is the return window for Beauty products?",
        "Please check order NYK-0001",
    ]
    crew, orchestrator = build_crewai_crew(orchestrator=orchestrator)
    before = dict(orchestrator.tool_calls)
    lines = ["=== CREWAI CREW (.kickoff()) ==="]
    for q in queries:
        result = crew.kickoff(inputs={"query": q})
        lines += [f"Query: {q}", f"Kickoff result: {str(result)}"]
    after = dict(orchestrator.tool_calls)
    lines += [f"Tool invocations via .kickoff(): {after} (was {before})",
              f"  retrieve_policy calls: {after['rag'] - before['rag']}",
              f"  check_order_status calls: {after['lookup'] - before['lookup']}"]
    if after["rag"] <= before["rag"] or after["lookup"] <= before["lookup"]:
        lines += ["WARNING: expected both tools to be invoked via .kickoff()."]
    return lines, orchestrator


# --- Task 8: LangChain session memory that actually changes behaviour --------------------

FOLLOWUP_RE=re.compile(r"(?i)^(?:and\s+)?(?:what|how)\s+about\s+(.+?)[\s?.!]*$")
ORDER_REF_RE=re.compile(r"(?i)\b(?:it|that order|this order|my order)\b")
CATEGORY_RE=re.compile(r"(?i)\b(apparel|electronics|home|footwear|beauty)\b")
ORDER_ID_RE=re.compile(r"(?i)\bNYK-\d{4}\b")

def resolve_followup(query: str, previous_user_turns: list[str]) -> tuple[str, bool]:
    """Resolve a short follow-up from earlier turns of the SAME session.

    Returns (resolved_query, needs_context). `needs_context` is True when the query
    is a follow-up but the session has nothing to resolve it against (fresh session).
    """
    q=query.strip()
    m=FOLLOWUP_RE.match(q)
    if m:                                    # "What about Apparel?"
        topic=m.group(1).strip()
        if CATEGORY_RE.fullmatch(topic):
            for prev in reversed(previous_user_turns):
                if CATEGORY_RE.search(prev) and not FOLLOWUP_RE.match(prev):
                    return CATEGORY_RE.sub(topic.capitalize(),prev,count=1), False
        return q, True
    if ORDER_REF_RE.search(q) and not ORDER_ID_RE.search(q):   # "Is it delayed?"
        for prev in reversed(previous_user_turns):
            oid=ORDER_ID_RE.search(prev)
            if oid: return f"{q} ({oid.group(0)})", False
        return q, True
    return q, False

class MemoryChat:
    """Session memory via LangChain: InMemoryChatMessageHistory + RunnableWithMessageHistory.

    (RunnableWithMessageHistory emits a LangChainDeprecationWarning pointing to LangGraph
    persistence; that is expected and the class still works.) Memory is in-process only.
    """
    def __init__(self, orchestrator=None):
        from langchain_core.chat_history import InMemoryChatMessageHistory
        from langchain_core.runnables import RunnableLambda
        from langchain_core.runnables.history import RunnableWithMessageHistory
        self.orchestrator=orchestrator or SupportOrchestrator()
        self.store: dict = {}
        def get_history(session_id: str):
            return self.store.setdefault(session_id,InMemoryChatMessageHistory())
        def step(inputs: dict) -> dict:
            previous=[m.content for m in inputs.get("history",[]) if getattr(m,"type","")=="human"]
            resolved,needs=resolve_followup(inputs["query"],previous)
            if needs:
                resp=SupportResponse(answer="I need a bit more context: which policy topic or order do you mean? "
                                            "Please include the product category or an order ID like NYK-0001.",
                                     intent="unsupported",grounded=False,
                                     trace_notes=["memory: follow-up with no prior context in this session"])
            else:
                resp=self.orchestrator.ask(resolved,session_id="langchain-memory")
                if resolved!=inputs["query"]:
                    resp.trace_notes.append(f"memory: follow-up resolved from earlier turn -> {resolved!r}")
            return {"answer":resp.answer,"response":resp}
        self.chain=RunnableWithMessageHistory(RunnableLambda(step),get_history,
            input_messages_key="query",history_messages_key="history",output_messages_key="answer")
    def ask(self, query: str, session_id: str) -> SupportResponse:
        safe=mask_pii(query)                 # history is stored masked, never raw
        out=self.chain.invoke({"query":safe},config={"configurable":{"session_id":session_id}})
        return out["response"]
    def turns(self, session_id: str) -> int:
        return len(self.store[session_id].messages) if session_id in self.store else 0

def run_memory_demo(orchestrator=None) -> list[str]:
    """Multi-turn transcript (state carried) plus a separate fresh-session transcript (state absent)."""
    import warnings
    warnings.filterwarnings("ignore")
    chat=MemoryChat(orchestrator)
    lines=["=== SESSION MEMORY (LangChain InMemoryChatMessageHistory + RunnableWithMessageHistory) ===",
           "--- Transcript A: session 'conv-a' (state carried across turns) ---"]
    def turn(session,q):
        r=chat.ask(q,session)
        lines.extend([f"User: {q}",f"  intent={r.intent}; grounded={r.grounded}; messages_in_history={chat.turns(session)}",
                      f"  Agent: {r.answer[:200]}",f"  notes: {r.trace_notes}"])
    for q in ["What is the return window for Beauty products?","What about Apparel?",
              "Please check order NYK-0001","Is it delayed?"]:
        turn("conv-a",q)
    lines.append("--- Transcript B: fresh session 'conv-b' (no state; same follow-ups cannot be resolved) ---")
    for q in ["What about Apparel?","Is it delayed?"]:
        turn("conv-b",q)
    return lines
