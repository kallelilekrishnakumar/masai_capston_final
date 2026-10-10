"""FastAPI deployment with masked JSONL request logging."""
from __future__ import annotations
import json, logging, os, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from agents import SupportOrchestrator, input_guardrail
from rag import LocalRetriever, load_documents
from governance import BudgetExceeded, RequestBudget

app=FastAPI(title="Nykaa Domain Support Agent",version="1.0.0")
orchestrator=SupportOrchestrator()
budget=RequestBudget(max_chars=int(os.getenv("REQUEST_CHAR_BUDGET","3000")))
LOG_PATH=Path(os.getenv("LOG_PATH","artifacts/requests.jsonl"))
LOG_PATH.parent.mkdir(parents=True,exist_ok=True)

class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str = "default"
class AskResponse(BaseModel):
    trace_id: str
    answer: str
    intent: str
    grounded: bool
    sources: list[str]
    order_status: str | None = None
    escalation_score: float | None = None
    escalation_recommended: bool | None = None
    trace_notes: list[str] = []
class AddDocumentRequest(BaseModel):
    doc_id: str
    title: str
    text: str = Field(min_length=10)
class HealthResponse(BaseModel):
    status: str
    rag_mode: str

def log_request(trace_id, query, started, status):
    safe=input_guardrail(query)["masked_text"]
    event={"timestamp":datetime.now(timezone.utc).isoformat(),"trace_id":trace_id,
           "duration_ms":round((time.perf_counter()-started)*1000,2),"status":status,
           "request_text":safe}
    with LOG_PATH.open("a",encoding="utf-8") as f: f.write(json.dumps(event,ensure_ascii=False)+"\n")

@app.get("/health",response_model=HealthResponse)
def health():
    return {"status":"ok","rag_mode":orchestrator.retriever.mode}

@app.post("/ask",response_model=AskResponse)
def ask(body: AskRequest):
    trace=str(uuid.uuid4()); started=time.perf_counter()
    try:
        budget.check(body.query)
        guard=input_guardrail(body.query)
        if not guard["allowed"]:
            raise HTTPException(status_code=400,detail="Prompt-injection guard blocked the request.")
        result=orchestrator.ask(guard["masked_text"],body.session_id)
        log_request(trace,body.query,started,"ok")
        return {"trace_id":trace,**result.model_dump()}
    except BudgetExceeded as exc:
        log_request(trace,body.query,started,"budget_rejected")
        raise HTTPException(status_code=413,detail=str(exc))
    except HTTPException:
        log_request(trace,body.query,started,"blocked")
        raise
    except Exception:
        log_request(trace,body.query,started,"error")
        raise

@app.post("/add-document")
def add_document(body: AddDocumentRequest):
    # Adds to the in-memory corpus for the lifetime of this process.
    existing=[d for d in orchestrator.retriever.documents if d["doc_id"]!=body.doc_id]
    existing.append({"doc_id":body.doc_id,"title":body.title,"text":body.text})
    orchestrator.retriever=LocalRetriever(orchestrator.retriever.strategy,documents=existing)
    return {"status":"added","doc_id":body.doc_id,"note":"Index is in-memory and rebuilds locally; document is not persisted to disk."}

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload=await websocket.receive_json()
            try:
                req=AskRequest(**payload)
                budget.check(req.query)
                guard=input_guardrail(req.query)
                if not guard["allowed"]:
                    await websocket.send_json({"error":"Prompt-injection guard blocked the request."})
                    continue
                result=orchestrator.ask(guard["masked_text"],req.session_id)
                await websocket.send_json(result.model_dump())
            except (BudgetExceeded, ValueError) as exc:
                await websocket.send_json({"error":str(exc)})
    except WebSocketDisconnect:
        # Disconnect is scoped to this socket; the server and other clients remain available.
        return
