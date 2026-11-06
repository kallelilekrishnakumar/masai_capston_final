"""Temporary probe: run the real CrewAI crew with MockCrewLLM and prove tools fire."""
import os
os.environ["MOCK_LLM"] = "1"
os.environ["MOCK_LLM_DEBUG"] = "1"

from pathlib import Path
Path("artifacts/crewai_debug.log").write_text("", encoding="utf-8")

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolUsageStartedEvent, ToolUsageFinishedEvent

from agents import SupportOrchestrator
from mock_llm import MockCrewLLM

try:
    from crewai.events.listeners.tracing import set_suppress_tracing_messages
    set_suppress_tracing_messages(True)
except Exception:
    pass

orch = SupportOrchestrator()
llm = MockCrewLLM(model="mock-llm")

calls = []

@tool("retrieve_policy")
def retrieve_policy(query: str) -> str:
    """Retrieve a grounded answer from the local Nykaa policy knowledge base."""
    calls.append(("retrieve_policy", query))
    r = orch.rag_tool(query)
    return r.get("answer", str(r))

@tool("check_order_status")
def order_lookup(record_id: str) -> str:
    """Look up a synthetic order by record ID."""
    calls.append(("check_order_status", record_id))
    info = orch.lookup_tool(record_id, agent_name="Lookup Agent")
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

t1 = Task(description="For this request, retrieve relevant policy context if it is a policy question: {query}",
          agent=retrieval, expected_output="Retrieved context or explicit unsupported result.")
t2 = Task(description="If an order ID appears in {query}, look it up. Otherwise say no lookup was needed.",
          agent=lookup, expected_output="Order details or no lookup needed.")
t3 = Task(description="Compose a concise grounded answer to: {query}. Do not add unsupported policy claims.",
          agent=composer, context=[t1, t2], expected_output="A safe customer-support answer.")
crew = Crew(agents=[retrieval, lookup, composer], tasks=[t1, t2, t3], process=Process.sequential, verbose=False)

for q in ["What is the return window for Beauty products?",
          "Please check order NYK-0001"]:
    calls.clear()
    result = crew.kickoff(inputs={"query": q})
    print("\n=== QUERY:", q)
    print("RESULT:", str(result)[:400])
    print("TOOL CALLS (via crew):", calls)
    print("ORCH tool_calls:", orch.tool_calls)
