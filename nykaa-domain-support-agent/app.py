"""FastAPI deployment with masked JSONL request logging."""
from __future__ import annotations
import json, os, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, ValidationError
from agents import SupportOrchestrator, input_guardrail, mask_pii
from rag import LocalRetriever
from governance import BudgetExceeded, RequestBudget, ResponseCache

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

app=FastAPI(title="Nykaa Domain Support Agent",version="1.0.0")
orchestrator=SupportOrchestrator()
retriever_lock=Lock()
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
class WSReply(AskResponse):
    """WebSocket success frame (same shape as /ask)."""
class WSError(BaseModel):
    trace_id: str
    error: str
class AddDocumentRequest(BaseModel):
    doc_id: str = Field(min_length=1,max_length=100)
    title: str
    text: str = Field(min_length=10)
class AddDocumentResponse(BaseModel):
    trace_id: str
    status: str
    doc_id: str
    note: str
class HealthResponse(BaseModel):
    status: str
    rag_mode: str

def log_request(trace_id, query, started, status, endpoint="/ask"):
    """One JSON-Lines entry per request; text is masked exactly as the model sees it."""
    event={"timestamp":datetime.now(timezone.utc).isoformat(),"trace_id":trace_id,"endpoint":endpoint,
           "duration_ms":round((time.perf_counter()-started)*1000,2),"status":status,
           "request_text":mask_pii(query)}
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

@app.post("/add-document",response_model=AddDocumentResponse)
def add_document(body: AddDocumentRequest):
    trace=str(uuid.uuid4()); started=time.perf_counter()
    text=mask_pii(body.text)
    try:
        budget.check(text)
    except BudgetExceeded as exc:
        log_request(trace,f"add-document {body.doc_id}",started,"budget_rejected","/add-document")
        raise HTTPException(status_code=413,detail=str(exc))
    with retriever_lock:
        existing=[d for d in orchestrator.retriever.documents if d["doc_id"]!=body.doc_id]
        existing.append({"doc_id":body.doc_id,"title":body.title,"text":text})
        orchestrator.retriever=LocalRetriever(orchestrator.retriever.strategy,documents=existing)
        orchestrator.cache=ResponseCache()  # stale answers must not survive a corpus change
    log_request(trace,f"add-document {body.doc_id}",started,"ok","/add-document")
    return {"trace_id":trace,"status":"added","doc_id":body.doc_id,
            "note":"Index is in-memory and rebuilds locally; document is not persisted to disk."}

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            trace=str(uuid.uuid4()); started=time.perf_counter()
            text=""
            try:
                payload=await websocket.receive_json()
                req=AskRequest(**payload)
                text=req.query
                budget.check(text)
                guard=input_guardrail(text)
                if not guard["allowed"]:
                    log_request(trace,text,started,"blocked","/ws/chat")
                    await websocket.send_json(WSError(trace_id=trace,error="Prompt-injection guard blocked the request.").model_dump())
                    continue
                result=await run_in_threadpool(orchestrator.ask,guard["masked_text"],req.session_id)
                log_request(trace,text,started,"ok","/ws/chat")
                await websocket.send_json(WSReply(trace_id=trace,**result.model_dump()).model_dump())
            except WebSocketDisconnect:
                raise
            except (BudgetExceeded, ValidationError, ValueError, TypeError) as exc:
                # bad JSON, bad schema or over budget: tell the client and keep the socket open
                log_request(trace,text or "<invalid payload>",started,"rejected","/ws/chat")
                await websocket.send_json(WSError(trace_id=trace,error=str(exc)[:300]).model_dump())
            except Exception:
                log_request(trace,text or "<error>",started,"error","/ws/chat")
                await websocket.send_json(WSError(trace_id=trace,error="Internal error.").model_dump())
    except WebSocketDisconnect:
        # Disconnect is scoped to this socket; the server and other clients remain available.
        return
