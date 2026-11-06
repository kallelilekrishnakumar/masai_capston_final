# Technical Handover — Nykaa Domain Support Agent

A practical guide for a developer or technical lead taking over this project. Covers architecture, setup, configuration, integrations, operations, and troubleshooting.

> This is an educational capstone prototype. All data/policies are fabricated and illustrative. It runs fully offline with **no API keys** under `MOCK_LLM=1`.

---

## 1. What this project is

A local-first customer-support agent for a fictional Nykaa e-commerce scenario. It:
- answers policy questions from a 13-document markdown knowledge base using retrieval + extractive grounding,
- looks up synthetic order status and computes an escalation score,
- applies input (PII masking, prompt-injection) and output (groundedness) guardrails,
- enforces governance (tool allow-list, request budget, response cache),
- exposes a FastAPI backend (3 HTTP endpoints + 1 WebSocket),
- ships a deterministic evaluation harness and a demo that writes reproducible artifacts.

Optional adapters (not on the default path): a real CrewAI crew and an AutoGen review team.

---

## 2. Architecture

```
Client ──HTTP/WS──▶ app.py (FastAPI)
                      │  RequestBudget → input_guardrail → SupportOrchestrator → masked JSONL log
                      ▼
             SupportOrchestrator (agents.py)
               ├─ ResponseCache (governance.py)
               ├─ SessionMemory (agents.py)
               ├─ order-ID branch ─▶ check_order_status (dataset.py) + enforce_tool_access
               └─ policy branch   ─▶ LocalRetriever (rag.py)
                                        ├─ SentenceTransformers + ChromaDB  (if installed)
                                        └─ lexical cosine fallback          (default when deps absent)
                                        └─ reads knowledge_base/*.md
```

**Key design decisions:**
- **Default orchestration is a deterministic router**, not a live LLM/crew, so the demo runs without keys.
- **Retrieval degrades gracefully**: if embeddings/Chroma are unavailable, it falls back to lexical cosine and sets `mode="lexical-fallback"`. The fallback is **not** semantically equivalent.
- **Everything is in-memory**: orders, vector index, session memory, cache, and added documents. Nothing persists across restarts.

### Module map

| File | Responsibility | Key symbols |
|---|---|---|
| `dataset.py` | Seeded synthetic orders + validation | `generate_orders`, `validate_orders`, `distribution`, `ORDERS` |
| `rag.py` | Chunking, retrieval, grounding, metrics | `LocalRetriever`, `fixed_chunks`, `sentence_chunks`, `calibrated_threshold`, `precision_recall` |
| `agents.py` | Guardrails, tools, memory, orchestration | `input_guardrail`, `mask_pii`, `check_order_status`, `SupportResponse`, `SessionMemory`, `SupportOrchestrator`, `build_crewai_crew` |
| `review_stage.py` | Second-stage review | `review_draft`, `run_autogen_review`, `Verdict` |
| `governance.py` | Least autonomy, budget, cache | `enforce_tool_access`, `RequestBudget`, `ResponseCache` |
| `app.py` | FastAPI/WebSocket + logging | `ask`, `add_document`, `websocket_chat`, `health`, `log_request` |
| `evaluation.py` | 15-query rubric evaluation | `TEST_QUERIES`, `evaluate` |
| `run_demo.py` | End-to-end demo + artifacts | `main` |
| `tests/test_core.py` | Unit tests | 7 test functions |

---

## 3. Setup instructions

**Prerequisites:** Python 3.11 recommended.

```bash
python -m venv .venv
# Windows (cmd):        .venv\Scripts\activate.bat
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS/Linux:          source .venv/bin/activate

pip install -r requirements.txt
```

**Run the acceptance demo (writes artifacts):**
```bash
python run_demo.py
```
Outputs: `artifacts/orders.json`, `artifacts/rag_metrics.json`, `artifacts/evaluation.json`, `artifacts/demo_transcript.txt`.

**Run the API:**
```bash
uvicorn app:app --reload
# docs at http://127.0.0.1:8000/docs
```

**Run the tests:**
```bash
pytest -q
```

> First run of the embedding path downloads `sentence-transformers/all-MiniLM-L6-v2`. If `sentence-transformers`/`chromadb` are unavailable, the system silently uses the lexical fallback — check `GET /health` → `rag_mode` to confirm which path is active.

---

## 4. Configuration

Environment variables (see `.env.example`; copy to `.env` as needed):

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `MOCK_LLM` | `1` | `run_demo.py` (sets default) | Key-free deterministic mode |
| `RAG_TOP_K` | `3` | (declared; retrieval default is `top_k=3` in `rag.py`) | Chunks retrieved |
| `RAG_MIN_SIMILARITY` | `0.0` | (declared in env; runtime uses calibrated threshold) | Minimum similarity floor |
| `REQUEST_CHAR_BUDGET` | `3000` | `app.py` → `RequestBudget` | Max request characters |
| `LOG_PATH` | `artifacts/requests.jsonl` | `app.py` | JSONL request log location |

> Note: `RAG_TOP_K`/`RAG_MIN_SIMILARITY` are present in `.env.example` but the runtime groundedness decision uses the empirically **calibrated** threshold from `rag.calibrated_threshold`, not the env floor. Treat the env values as documentation unless you wire them in.

---

## 5. API reference

Base URL (dev): `http://127.0.0.1:8000`

### `GET /health`
Returns `{ "status": "ok", "rag_mode": "<lexical-fallback | sentence-transformers+chromadb>" }`. Use this to confirm whether embeddings are active.

### `POST /ask`
Request: `{ "query": "<non-empty>", "session_id": "default" }`
Response (`AskResponse`): `trace_id, answer, intent, grounded, sources, order_status?, escalation_score?, escalation_recommended?, trace_notes`.
Errors: `400` prompt-injection blocked; `413` request budget exceeded.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the return window for Beauty products?","session_id":"demo-1"}'
```

### `POST /add-document`
Request: `{ "doc_id": "...", "title": "...", "text": "<min 10 chars>" }`. Rebuilds the in-memory retriever. **Not persisted to disk.**

### `WS /ws/chat`
Connect to `ws://127.0.0.1:8000/ws/chat`, send `{ "query": "...", "session_id": "ws-1" }`. Returns a `SupportResponse` JSON per message, or `{ "error": "..." }` for budget/injection/validation issues. `WebSocketDisconnect` is handled per socket; the server stays up for other clients.

---

## 6. Business logic reference

**Escalation score** (`agents.check_order_status`):
`score = clamp(0.65 * delayed_shipment + 0.35 * (days_since_created / 30), 0, 1)`; **escalate when `score >= 0.65`**.

**Groundedness** (`rag.answer`): if no hits or top-1 similarity `< threshold`, return the "I don't know … contact customer support" fallback with `grounded=False`. Threshold is calibrated per retriever via `calibrated_threshold` (midpoint between in-scope and out-of-scope similarity clusters when they separate).

**Guardrails** (`agents.input_guardrail`):
- PII masking: Indian phone numbers → `[PHONE_MASKED]`; card last-4 → `[LAST4_MASKED]`.
- Prompt injection: blocks known phrases (e.g. "ignore all previous instructions").
- Output groundedness: enforced by the retrieval threshold above.
- Names/addresses are explicitly **not** masked (acknowledged limitation).

**Governance** (`governance.py`):
- Least autonomy: only `"Lookup Agent"` may call `check_order_status` (`enforce_tool_access`).
- Runtime budget: requests over `REQUEST_CHAR_BUDGET` are rejected (`413`).
- Caching: `ResponseCache` keys on normalized (lowercase, whitespace-collapsed) query; a hit avoids recomputing grounded generation.

---

## 7. Integrations

All optional, all off by default:

| Integration | Where | Status | Notes |
|---|---|---|---|
| SentenceTransformers (`all-MiniLM-L6-v2`) | `rag.LocalRetriever` | Optional | Downloaded on first use; falls back to lexical if unavailable |
| ChromaDB (in-memory client) | `rag.LocalRetriever` | Optional | Two collections `nykaa_fixed_chunks`, `nykaa_sentence_chunks`; recreated per init |
| CrewAI | `agents.build_crewai_crew` | Adapter, **not invoked** | Real 3-agent crew; only Lookup Agent gets the order tool |
| AutoGen AgentChat | `review_stage.run_autogen_review` | Adapter, **not invoked** | `RoundRobinGroupChat(max_turns=2)`, needs caller-supplied `model_client` (e.g. local Ollama) |

> There are **no committed artifacts** proving the embedding/Chroma or CrewAI/AutoGen paths were executed; the committed demo ran in `lexical-fallback` with the deterministic router. See `PENDING_TASKS.md` P-1/P-2/P-3.

---

## 8. Operational procedures

- **Check retrieval mode:** `GET /health` → `rag_mode`.
- **Logs:** JSONL at `LOG_PATH` (default `artifacts/requests.jsonl`). Each line: `timestamp, trace_id, duration_ms, status, request_text` (request text re-masked before writing). Status values: `ok`, `blocked`, `budget_rejected`, `error`. The `.jsonl` file is gitignored.
- **Regenerate artifacts:** `python run_demo.py`.
- **CI:** `.github/workflows/tests.yml` installs `requirements.txt` and runs `pytest -q` on push/PR (Python 3.11, ubuntu-latest).
- **Reset state:** restart the process — all orders/index/memory/cache/added-docs are in-memory.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `GET /health` shows `rag_mode: lexical-fallback` when embeddings expected | `sentence-transformers`/`chromadb` not installed or model download failed | Install deps; ensure network for first model download; re-check `LocalRetriever.fallback_reason` |
| All grounded answers look lexically matched / metrics differ from README | Running in fallback mode | Install embedding deps; rerun `run_demo.py` |
| `413` on `/ask` | Query exceeds `REQUEST_CHAR_BUDGET` | Shorten query or raise the budget env var |
| `400` on `/ask` | Prompt-injection phrase detected | Rephrase; see `INJECTION_PATTERNS` in `agents.py` |
| Added document disappears after restart | `/add-document` is in-memory only | Expected; persistence is not implemented |
| CrewAI/AutoGen import errors | Optional heavy deps / version drift | Pin compatible versions; these adapters are optional |
| Evaluation answers show mid-word cut text (ids 8, 14) | Fixed-chunk boundary artifact | Cosmetic; consider sentence chunks or boundary-aware trimming |
| `pytest` import errors | venv not activated / deps missing | Activate venv; `pip install -r requirements.txt` |

---

## 10. Known limitations (carry forward)

- Mock answer generation is **extractive/deterministic**, not generative.
- No persistence, no authentication/authorization on the API, no rate limiting.
- Evaluation scores are a **rubric proxy**, not independent LLM/human judgment.
- Only fixed-format PII (phone, card last-4) is masked; names/addresses are out of scope.
- Threshold and precision/recall numbers are environment-specific; recompute per backend.
- Deviation from brief: session memory uses a custom class, not LangChain (see `PENDING_TASKS.md` P-5).

---

## 11. Where to look next

- Outstanding work and priorities → `project_docs/PENDING_TASKS.md`
- Full status, evidence, and task tracker → `project_docs/PROJECT_ANALYSIS_AND_STATUS.md`
- Original requirements → `project_docs/# Final Capstone -Ecom- Nykaa Domain Req.txt`
- Design rationale (dataset, thresholds, chunking) → `README.md`
