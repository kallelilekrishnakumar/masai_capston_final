# Project Analysis and Status — Nykaa Domain Support Agent

> **Document type:** Technical analysis and status report
> **Prepared by:** Senior Architect / Technical Lead review (static code + artifact analysis)
> **Analysis basis:** Source code, knowledge base, committed artifacts, CI config, and the capstone requirements document in `project_docs/`.
> **Important limitation:** Tests and the demo were **not executed** during this analysis (the target machine does not have the Python toolchain/dependencies installed). All "pass/produced" statements below are inferred from **committed artifacts** and **static code reading**, not from a live run in this environment. Statements are labelled **[Fact]**, **[Inference]**, **[Recommendation]**, or **[Unknown]**.

---

## 1. Executive Summary

**[Fact]** The repository implements a local-first, API-key-free ("MOCK_LLM") customer-support agent for a fictional Nykaa e-commerce scenario. It is a **capstone / educational prototype**, explicitly not a production policy engine (stated in `README.md` and `knowledge_base/*`).

**[Fact]** All 16 capstone tasks across the 4 required parts have corresponding code, and committed artifacts (`artifacts/demo_transcript.txt`, `evaluation.json`, `rag_metrics.json`, `orders.json`) show a prior successful demo run.

**[Inference]** The project appears **functionally complete against the capstone acceptance criteria** in its default deterministic mode. The main gaps are: (1) the optional CrewAI/AutoGen/ChromaDB/embedding integrations are **adapters that were not exercised** in the committed run (it ran in `lexical-fallback` mode), and (2) one requirement deviation — session memory is a custom in-memory class rather than the LangChain components named in the brief.

**Most important caveats for a new owner:**
- The committed artifacts were generated in **`lexical-fallback`** retrieval mode (no SentenceTransformers/ChromaDB). Real embedding behaviour and the two separate Chroma collections are **unverified in this repo's artifacts**.
- Evaluation scores are a **deterministic rubric proxy**, not independent LLM/human judgment.
- There is **no persistence** (orders, memory, added documents, cache are all in-process only).

---

## 2. Project Overview and Business Objectives

**[Fact — from `project_docs/# Final Capstone -Ecom- Nykaa Domain Req.txt`]**

- **Purpose:** A domain support agent that answers Nykaa policy questions from a self-authored knowledge base and looks up order status from a self-designed dataset, with memory, guardrails, a second-agent review stage, governance, evaluation, and a FastAPI deployment.
- **Business framing (from brief):** Give support agents (and eventually customers) instant, grounded answers instead of escalating everything to a human, built to a standard that could "pass a real production governance review."
- **Scope guardrails:** All data is fabricated; policies are illustrative and must not be presented as Nykaa's actual policies.
- **Deliverable:** One public GitHub repo containing dataset, RAG core, CrewAI crew, AutoGen review stage, FastAPI deployment, transcripts, and a README confirming it runs under `MOCK_LLM` with zero API keys.

**Expected outcomes (acceptance criteria from the brief):** structured dataset ≥40 orders, ≥12 KB docs, two chunking strategies in two Chroma collections, calibrated groundedness threshold, precision/recall per strategy, a ≥3-agent CrewAI crew with tool invocation, session memory, Pydantic structured output, input+output guardrails, FastAPI (2 HTTP + 1 WebSocket), JSONL logging with masked PII, 15-query 4-metric evaluation, AutoGen 2-agent review, four-layer governance, and response caching.

---

## 3. Architecture and Technology Stack

### 3.1 Technology stack **[Fact — `requirements.txt`]**

| Layer | Technology | Version constraint |
|---|---|---|
| Web/API | FastAPI | `>=0.115,<1.0` |
| ASGI server | uvicorn[standard] | `>=0.30,<1.0` |
| Validation/schema | Pydantic | `>=2.8,<3.0` |
| Numerics | numpy | `>=1.26` |
| Vector store | chromadb | `>=0.5,<2.0` |
| Embeddings | sentence-transformers | `>=3.0,<6.0` (`all-MiniLM-L6-v2`) |
| Multi-agent | crewai | `>=0.80` |
| Review agents | autogen-agentchat, autogen-core | `>=0.4` |
| Testing | pytest | `>=8.0` |
| Test HTTP client | httpx | `>=0.27` |

Runtime: Python 3.11 recommended (also pinned in CI, `.github/workflows/tests.yml`).

> **[Inference] Note:** `requirements.txt` lists `langchain`-related packages **nowhere**, yet the brief (Task 8) asks for LangChain session memory. The code implements memory without LangChain. See Section 9 / Pending work.

### 3.2 Component responsibilities **[Fact — module docstrings + code]**

- `dataset.py` — Seeded deterministic generator of 50 synthetic orders; validation; distribution reporting.
- `knowledge_base/*.md` — 13 policy documents (12 required topics + `privacy-and-pii.md`).
- `rag.py` — Document loading, two chunking strategies, retrieval (embeddings+Chroma when available, lexical fallback otherwise), grounded extractive answer, empirical threshold calibration, precision/recall.
- `agents.py` — PII masking + injection guard, `check_order_status` tool with escalation score, `SessionMemory`, `SupportResponse` Pydantic schema, deterministic `SupportOrchestrator`, optional `build_crewai_crew()`.
- `review_stage.py` — Deterministic `review_draft` reviewer + optional async AutoGen `run_autogen_review`.
- `governance.py` — Tool access allow-list, request char budget, normalized-query response cache.
- `app.py` — FastAPI `/health`, `/ask`, `/add-document`, `/ws/chat`; masked JSONL request logging.
- `evaluation.py` — 15-query deterministic rubric evaluation harness.
- `run_demo.py` — Orchestrates an end-to-end acceptance demo and writes artifacts.
- `tests/test_core.py` — 7 pytest test functions.

### 3.3 High-level architecture

```
                         ┌─────────────────────────────────────────────┐
   HTTP / WebSocket  ───▶ │                 app.py (FastAPI)             │
   client               │  /health  /ask  /add-document  /ws/chat       │
                         │  RequestBudget.check → input_guardrail →      │
                         │  SupportOrchestrator.ask → masked JSONL log   │
                         └───────────────┬───────────────────────────────┘
                                         │
                              ┌──────────▼───────────┐
                              │   SupportOrchestrator │ (agents.py)
                              │  - cache (governance) │
                              │  - memory (Session)   │
                              └───┬───────────────┬───┘
                 record NYK-####  │               │  policy query
                        ┌─────────▼──┐      ┌──────▼─────────┐
                        │ lookup_tool│      │   rag_tool     │
                        │ check_order│      │ LocalRetriever │ (rag.py)
                        │  _status   │      │  embeddings OR │
                        │ (dataset)  │      │  lexical fallbk│
                        └─────┬──────┘      └──────┬─────────┘
                              │                    │
                        enforce_tool_access   knowledge_base/*.md
                        (governance.py)        + Chroma collection
```

> **[Inference] Important architectural note:** In the **default runtime path** (`SupportOrchestrator.ask`), orchestration is a **deterministic if/else router**, not a live CrewAI crew. `build_crewai_crew()` and `run_autogen_review()` exist as **optional adapters** and are **not called** by `app.py`, `run_demo.py`, or the tests. This matches the README ("core deterministic mock orchestration is the default").

---

## 4. Module-by-Module Analysis

### 4.1 `dataset.py` **[Fact]**
- `generate_orders(seed=20261011, n=50)` produces `record_id` (`NYK-0001`…), `category`, `status`, `order_value_inr` (₹299–₹24,999), `days_since_created` (0–30), `delayed_shipment` (~20% target).
- Guarantees category/status coverage via index-based seeding; if delay rate falls outside 10–30%, it **recursively regenerates with an incremented seed** rather than hand-editing (lines ~41–49).
- `validate_orders` asserts structural contracts; `ORDERS` is built at import time.
- Committed `orders.json` + transcript show: 50 orders, delayed_percentage **14.0%**, p80 days = 24, all categories ≥3, all statuses ≥1. **[Fact — artifact]**
- **Limitation:** The recursive regeneration returns inside the `while` loop (returns on first attempt regardless), so the loop body executes at most once per call; it still works because the base seed already lands in-band. **[Inference]**

### 4.2 `rag.py` **[Fact]**
- `load_documents()` reads `knowledge_base/*.md`, strips headings, derives title from first `# ` line.
- `fixed_chunks(size=320, overlap=60)` and `sentence_chunks(max_sentences=2)`.
- `LocalRetriever.__init__` tries SentenceTransformers + ChromaDB; on **any** exception falls back to a lexical cosine over token counters, setting `mode="lexical-fallback"` and recording `fallback_reason`.
- `search` queries Chroma (cosine) or the lexical index; `answer` applies a calibrated threshold and returns an **extractive** answer (join of top-2 chunk texts) — not generative.
- `calibrated_threshold` measures top-1 similarity for 3 in-scope + 2 out-of-scope queries and picks a midpoint when clusters separate.
- `relevance_for_query` is a hand-written gold map used by `precision_recall`.
- **[Fact — artifact]** The committed `rag_metrics.json` shows `mode: lexical-fallback` for both strategies, confirming the saved metrics are from the fallback path, **not** real embeddings.

### 4.3 `agents.py` **[Fact]**
- `mask_pii` masks Indian phone numbers and card last-4; `input_guardrail` adds prompt-injection phrase detection.
- `check_order_status` computes `escalation_score = clamp(0.65*delayed + 0.35*(days/30), 0, 1)`, recommends escalation at `>= 0.65`.
- `SupportResponse` is the Pydantic structured output schema (intent is a `Literal`).
- `SessionMemory` is a plain dict-of-lists (custom, **not** LangChain).
- `SupportOrchestrator.ask`: injection block → order-ID branch (`\bNYK-\d{4}\b`) → RAG branch; records memory.
- `build_crewai_crew()` builds a real 3-agent crew with tools (only Lookup Agent gets the order tool). **Not invoked by default.**

### 4.4 `review_stage.py` **[Fact]**
- `review_draft` deterministically flags risky guarantee phrases and returns a `Verdict(approved, final_answer, reason)`.
- `run_autogen_review` is an async AutoGen `RoundRobinGroupChat(max_turns=2)` adapter requiring a caller-supplied `model_client`. **Not invoked by default.**

### 4.5 `governance.py` **[Fact]**
- `enforce_tool_access` allow-list: only `"Lookup Agent"` → `check_order_status`.
- `RequestBudget(max_chars=3000)` raises `BudgetExceeded`.
- `ResponseCache` keys on normalized (lowercased, whitespace-collapsed) query; tracks `calls`/`hits`.

### 4.6 `app.py` **[Fact]**
- Endpoints: `GET /health`, `POST /ask`, `POST /add-document`, `WS /ws/chat`.
- `/ask` flow: budget → injection guard → orchestrator → masked JSONL log; maps `BudgetExceeded`→413, injection→400.
- `/add-document` rebuilds the in-memory retriever (not persisted to disk).
- WebSocket catches `WebSocketDisconnect` per connection.
- `log_request` re-masks text via `input_guardrail` before writing JSONL.

### 4.7 `evaluation.py` **[Fact]**
- 15 queries (12 in-scope topic queries + 3 edge/out-of-scope). Deterministic rubric scores for accuracy/grounding/completeness/safety plus averages. Explicitly labelled a proxy, not a real judge.

### 4.8 `tests/test_core.py` **[Fact]**
- 7 tests: dataset contract, PII+injection, order lookup, budget+least-autonomy, KB chunking, review revision, structured response.

---

## 5. End-to-End Business and Technical Workflows

### 5.1 Policy question (default path) **[Fact]**
1. Client → `POST /ask` with `{query, session_id}`.
2. `RequestBudget.check` rejects if > 3000 chars (413).
3. `input_guardrail` masks PII and blocks injection phrases (400 if blocked).
4. `SupportOrchestrator.ask`: no order-ID match → `rag_tool` (cache → `LocalRetriever.answer`).
5. If top-1 similarity ≥ calibrated threshold → extractive grounded answer + sources; else "I don't know" fallback (`grounded=False`, `intent=unsupported`).
6. Response validated as `SupportResponse`; memory updated; masked JSONL log line written.

### 5.2 Order-status lookup **[Fact]**
1–3 as above. 4. Query matches `NYK-####` → `lookup_tool` → `enforce_tool_access("Lookup Agent",...)` → `check_order_status` returns status + escalation score. 5. Formatted answer with escalation recommendation.

### 5.3 Review stage (demonstration only) **[Fact]**
`run_demo.py` calls `review_draft` twice — once approving unchanged, once revising an injected unsupported guarantee. Not wired into the `/ask` request path.

---

## 6. Database and Data Migration Details

**[Fact]** There is **no database and no migrations.**
- "Order data" is generated in-memory at import of `dataset.py` (`ORDERS`).
- Chroma collections (when deps installed) are created **in-memory** (`chromadb.Client()`), deleted/recreated on each `LocalRetriever` init.
- `/add-document` modifies an in-memory list; the note explicitly says it is **not persisted to disk**.
- Session memory and the response cache are in-process dicts.

**[Inference] Implication:** All state is lost on process restart. No migration activity exists or is required for the current scope. For any production path, persistence (orders DB, vector store on disk, durable logs) would be net-new work.

---

## 7. API and External Integration Details

### 7.1 HTTP/WS API **[Fact — `app.py`]**

| Method | Path | Request model | Response | Notes |
|---|---|---|---|---|
| GET | `/health` | — | `HealthResponse{status, rag_mode}` | Reports retriever mode |
| POST | `/ask` | `AskRequest{query, session_id}` | `AskResponse{trace_id, answer, intent, grounded, sources, order_status?, escalation_score?, escalation_recommended?, trace_notes}` | 400 injection, 413 budget |
| POST | `/add-document` | `AddDocumentRequest{doc_id, title, text}` | `{status, doc_id, note}` | In-memory only |
| WS | `/ws/chat` | JSON `{query, session_id}` | `SupportResponse` JSON or `{error}` | Handles disconnect |

### 7.2 External integrations **[Fact]** — all **optional** and **not exercised in committed artifacts**:
- SentenceTransformers `all-MiniLM-L6-v2` (downloaded on first use).
- ChromaDB in-memory client.
- CrewAI (`build_crewai_crew`).
- AutoGen AgentChat (`run_autogen_review`) — needs caller-supplied model client.

**[Fact]** No outbound network calls exist in the default path; the design is explicitly offline/key-free.

---

## 8. Completed Work

Legend: **Impl** = code present · **Demo** = evidence in committed artifacts · **Tested** = covered by `tests/` · **Deployed** = n/a (no deployment evidence) · **Approved** = n/a (no business sign-off evidence).

| # | Task (capstone) | Impl | Demo | Tested | Evidence |
|---|---|:--:|:--:|:--:|---|
| 1 | Seeded dataset ≥40 orders, coverage, 10–30% delay | ✅ | ✅ | ✅ | `dataset.py`; `orders.json`; transcript (50 orders, 14% delay); `test_dataset_contract` |
| 2 | ≥12 KB documents | ✅ | ✅ | ✅ | 13 files in `knowledge_base/`; `test_kb_chunking` asserts ≥12 |
| 3 | Two chunking strategies | ✅ | ✅ (fallback) | ✅ | `fixed_chunks`/`sentence_chunks`; `rag_metrics.json`; `test_kb_chunking` |
| 4 | Grounded gen + calibrated threshold + OOS fallback | ✅ | ✅ (fallback) | — | `rag.answer`/`calibrated_threshold`; transcript OOS fallback |
| 5 | Precision/recall per strategy with arithmetic | ✅ | ✅ (fallback) | — | `precision_recall`; `rag_metrics.json` per-query arithmetic |
| 6 | `check_order_status` + designed escalation score | ✅ | ✅ | ✅ | `agents.check_order_status`; transcript NYK-0001; `test_order_lookup` |
| 7 | CrewAI ≥3-agent crew via `.kickoff()` | ⚠️ | ❌ | — | `build_crewai_crew` present but **not invoked**; default is deterministic router |
| 8 | Session memory (LangChain) | ⚠️ | ✅ | — | Custom `SessionMemory` (not LangChain); transcript shows 2 vs 0 turns |
| 9 | Pydantic structured output | ✅ | ✅ | ✅ | `SupportResponse`; `test_structured_response` |
| 10 | Input PII+injection and output groundedness guards | ✅ | ✅ | ✅ | `mask_pii`/`input_guardrail`/threshold; transcript; `test_pii_and_injection` |
| 11 | FastAPI 2 HTTP + 1 WebSocket w/ disconnect handling | ✅ | — | — | `app.py` endpoints; no committed HTTP transcript |
| 12 | JSONL logging w/ masked PII + trace ID | ✅ | — | — | `log_request`; `.jsonl` gitignored (not committed) |
| 13 | 15-query evaluation, 4 metrics + averages | ✅ | ✅ | — | `evaluation.py`; `evaluation.json` (avg acc 4.73 / grounding 4.93 / completeness 3.73 / safety 5) |
| 14 | AutoGen 2-agent review (approve + revise) | ⚠️ | ✅ (mock only) | ✅ (mock) | `run_autogen_review` present but not invoked; `review_draft` demoed; `test_review_revision` |
| 15 | Four-layer governance (least autonomy, risk, budget) | ✅/⚠️ | ✅ | ✅ | `enforce_tool_access`, `RequestBudget`; risk classification **not found in repo** |
| 16 | Response caching w/ before/after evidence | ✅ | ✅ | — | `ResponseCache`; transcript cache hit counters |

**Status discipline:** No task is "deployed" (no deployment evidence) or "business-approved" (no sign-off). "Demo" rows rely on committed artifacts from a prior run in **lexical-fallback** mode.

---

## 9. In-Progress / Partially Implemented Work

1. **CrewAI crew (Task 7) — adapter only.** `build_crewai_crew()` exists but no code path calls `.kickoff()`; the default orchestrator is a deterministic router. **[Fact]** The brief's acceptance criteria require demonstrating `.kickoff()` tool invocation. **[Inference] Gap.**
2. **AutoGen review (Task 14) — adapter only.** `run_autogen_review` is never called; only the deterministic `review_draft` is demonstrated. **[Fact]**
3. **Session memory (Task 8) — technology deviation.** Implemented with a custom class rather than LangChain `InMemoryChatMessageHistory`/`RunnableWithMessageHistory`. Behaviour (multi-turn vs fresh) is demonstrated, but not with the named library. **[Fact → Inference: likely acceptable functionally, but a documented deviation from the brief.]**
4. **Real embedding/Chroma path — unverified.** All committed metrics are from `lexical-fallback`. The two-separate-Chroma-collections requirement is implemented in code but there is **no committed artifact proving it ran with real embeddings**. **[Fact]**
5. **Risk classification (Task 15) — not found as an artifact.** The brief asks for a written Low/Medium/High risk classification with justification; the allow-list and budget are implemented, but the risk classification paragraph is **not present** in README or code comments. **[Inference — Unknown whether intended elsewhere.]**

---

## 10. Pending Tasks

| ID | Task | Why needed | Status | Affected files | Risk if skipped | Next action | Priority |
|---|---|---|---|---|---|---|---|
| P-1 | Run `run_demo.py` with SentenceTransformers+ChromaDB installed and commit real-embedding artifacts | Current metrics are lexical-fallback; brief requires real embeddings + 2 Chroma collections | Confirmed pending | `rag.py`, `artifacts/*` | Grader may conclude embeddings never ran | Install deps, run demo, commit artifacts | High |
| P-2 | Add an executed CrewAI `.kickoff()` demonstration | Task 7 acceptance requires live crew tool invocation | Confirmed pending | `agents.py`, `run_demo.py` | Task 7 marked incomplete | Wire `build_crewai_crew().kickoff()` into demo under a flag | High |
| P-3 | Add an executed AutoGen review demonstration (or document local model-client setup) | Task 14 acceptance requires the AutoGen team to run | Confirmed pending | `review_stage.py`, `run_demo.py` | Task 14 relies on mock only | Provide local Ollama client wiring + demo | Medium |
| P-4 | Add written risk classification (Low/Med/High + justification) | Task 15 explicitly requires it | Confirmed pending | `README.md` or `governance.py` | Governance task incomplete | Add a paragraph classifying this as Medium risk | Medium |
| P-5 | Reconcile LangChain memory requirement vs custom `SessionMemory` | Task 8 names LangChain | Deviation | `agents.py` | Possible rubric mismatch | Either adopt LangChain memory or document rationale | Medium |
| P-6 | Commit a sample masked `requests.jsonl` or test asserting masking in logs | Task 12 evidence (file is gitignored) | Potential improvement | `app.py`, `.gitignore`, `tests/` | No committed proof of masked logging | Add a logging unit test | Low |
| P-7 | Add API/integration tests for `/ask`, `/add-document`, `/ws/chat` | No endpoint tests exist | Confirmed gap | `tests/` | Regressions in API layer undetected | Add `httpx`+`TestClient` tests | Medium |
| P-8 | Fix misleading recursive-regeneration loop in `generate_orders` | Loop returns on first attempt | Tech debt | `dataset.py` | Confusing; off-band seeds not actually retried | Replace early `return` with reseed+continue | Low |

**Confirmed pending** = P-1..P-5, P-7, P-8. **Improvement/assumption** = P-6.

---

## 11. Blockers and Known Defects

**Blockers (environment):**
- **[Fact]** This analysis machine lacks the Python toolchain/dependencies, so tests and the demo could not be executed here. Verification relies on committed artifacts. This is an **analysis-environment blocker**, not a code defect.
- **[Inference]** To verify P-1/P-2/P-3, an environment with `pip install -r requirements.txt` succeeding (including `crewai`, `autogen-*`, `sentence-transformers`, `chromadb`) is required. These are heavy dependencies and may need network access and compatible Python.

**Known defects / correctness notes:**
- **[Fact]** Several evaluation answers (`evaluation.json` ids 8 and 14) contain **mid-word truncation artifacts** ("re to the specialist queue", "ider's stated window") caused by fixed-size chunk boundaries. Answers are grounded but visibly cut; cosmetic/quality defect.
- **[Fact]** Default orchestration does not invoke CrewAI/AutoGen despite the brief centering on them (adapters exist but unused). Classified as partial, not defect.
- **[Inference]** `load_documents` is imported in `app.py` but unused there (minor dead import).

---

## 12. Risks and Open Questions

**Risks**
- **Integration/compatibility [Fact→Inference]:** `crewai>=0.80` and `autogen-*>=0.4` evolve rapidly; the adapters may not run against currently resolved versions without changes. README already warns about this.
- **Reproducibility [Fact]:** Threshold and precision/recall numbers are environment-specific; a different embedding backend will change them. The checked-in numbers are lexical-fallback values.
- **Data/state [Fact]:** No persistence; restart loses all state. Fine for a demo, unacceptable for production.
- **PII [Fact]:** Only fixed-format phone/card-last-4 are masked; names/addresses are explicitly out of scope. Real PII must never be used.
- **Evaluation validity [Fact]:** Scores are a rubric proxy, not independent judgment; must not be cited as model-quality evidence.

**Open Questions** (require team confirmation — **not** answered here):
- Is the custom `SessionMemory` acceptable, or must LangChain be used per Task 8? **[Unknown]**
- Must CrewAI/AutoGen be demonstrably executed for full marks, or are documented optional adapters sufficient? **[Unknown]**
- Where should the Task 15 risk classification live, and is "Medium" the intended classification? **[Unknown]**
- Is there a required submission GitHub URL / grader environment spec? **[Unknown]**

---

## 13. Testing and Quality Assessment

**[Fact] Existing tests (`tests/test_core.py`, 7 functions):** dataset contract, PII+injection, order lookup (hit+miss), budget+least-autonomy, KB chunking, review revision, structured response. CI runs `pytest -q` on Python 3.11 (`.github/workflows/tests.yml`).

**[Inference] Not executed in this environment** — results below are expected-pass based on static reading, not confirmed.

**Coverage gaps / recommended additional tests [Recommendation]:**
- API tests for `/ask`, `/add-document`, `/ws/chat` (incl. 400 injection, 413 budget, WebSocket disconnect). None exist.
- Assertion that written JSONL log lines contain no raw PII (Task 12 evidence).
- Cache behaviour test (hit avoids recompute) — demoed but not unit-tested.
- Groundedness fallback test for a clearly out-of-scope query.
- Real-embedding path test (skipif deps missing) to distinguish from fallback.
- Negative/edge: empty query (Pydantic `min_length=1` guards it), oversized query, malformed WebSocket payload.

**Production readiness [Inference]:** Not production-ready by design — no persistence, no auth, in-memory everything, deterministic mock generation, rubric-only evaluation. Appropriate as a capstone prototype only.

---

## 14. Deployment and Environment Readiness

**[Fact]**
- Run locally: `pip install -r requirements.txt`; `python run_demo.py`; `uvicorn app:app --reload`.
- Config via env vars (`.env.example`): `MOCK_LLM=1`, `RAG_TOP_K=3`, `RAG_MIN_SIMILARITY=0.0`, `REQUEST_CHAR_BUDGET=3000`, `LOG_PATH=artifacts/requests.jsonl`.
- No Dockerfile, no deploy pipeline, no production/staging config. CI only installs deps and runs tests.
- `.gitignore` excludes `.venv/`, caches, `.env`, and `artifacts/*.jsonl` (so request logs are not committed).

**[Inference]** No evidence of any deployment to any environment. "Deployment" = runnable locally only.

---

## 15. Consolidated Task and Status Tracker

| Capstone Task | Code | Default-path exercised | Committed demo evidence | Unit test | Overall |
|---|:--:|:--:|:--:|:--:|---|
| 1 Dataset | ✅ | ✅ | ✅ | ✅ | Complete |
| 2 KB ≥12 | ✅ | ✅ | ✅ | ✅ | Complete |
| 3 Two chunkings | ✅ | ✅ | ✅ (fallback) | ✅ | Complete (fallback evidence) |
| 4 Grounded + threshold | ✅ | ✅ | ✅ (fallback) | — | Complete (fallback evidence) |
| 5 Precision/recall | ✅ | ✅ | ✅ (fallback) | — | Complete (fallback evidence) |
| 6 Escalation score | ✅ | ✅ | ✅ | ✅ | Complete |
| 7 CrewAI crew | ✅ | ❌ | ❌ | — | Partial (adapter unused) |
| 8 Session memory | ⚠️ | ✅ | ✅ | — | Partial (not LangChain) |
| 9 Structured output | ✅ | ✅ | ✅ | ✅ | Complete |
| 10 Guardrails | ✅ | ✅ | ✅ | ✅ | Complete |
| 11 FastAPI + WS | ✅ | n/a | ❌ | — | Impl complete, demo missing |
| 12 JSONL logging | ✅ | ✅ | ❌ (gitignored) | — | Impl complete, evidence missing |
| 13 Evaluation 15×4 | ✅ | ✅ | ✅ | — | Complete (rubric proxy) |
| 14 AutoGen review | ⚠️ | ❌ (mock only) | ✅ (mock) | ✅ (mock) | Partial (adapter unused) |
| 15 Governance | ✅/⚠️ | ✅ | ✅ | ✅ | Partial (risk doc missing) |
| 16 Caching | ✅ | ✅ | ✅ | — | Complete |

---

## 16. Prioritised Action Plan

**Critical:** none (no broken core path identified from static analysis).

**High:**
1. P-1 Install real deps, run `run_demo.py`, commit real-embedding artifacts (proves Tasks 3–5 beyond fallback).
2. P-2 Execute CrewAI `.kickoff()` and capture a transcript (Task 7).

**Medium:**
3. P-3 Execute/document AutoGen review (Task 14).
4. P-4 Add written risk classification (Task 15).
5. P-5 Resolve LangChain-memory requirement (Task 8).
6. P-7 Add API/WebSocket tests.

**Low:**
7. P-6 Commit masked-log evidence/test (Task 12).
8. P-8 Fix misleading regeneration loop in `dataset.py`.
9. Clean up unused `load_documents` import in `app.py`; fix truncated evaluation answer text.

---

## 17. Recommended Next Steps

1. Stand up a Python 3.11 venv where `requirements.txt` installs cleanly; run `pytest -q` and `python run_demo.py` to replace fallback artifacts with real-embedding artifacts.
2. Add flagged demonstrations for CrewAI and AutoGen so the brief's "demonstrated" criteria are met with executed evidence.
3. Add the missing governance risk-classification write-up and the API-layer tests.
4. Confirm the open questions in Section 12 with the capstone owner before claiming full completion.

---

## 18. References to Source Files

| Concern | File / symbol |
|---|---|
| Dataset generation/validation | `dataset.py::generate_orders`, `validate_orders`, `distribution` |
| Chunking + retrieval | `rag.py::fixed_chunks`, `sentence_chunks`, `LocalRetriever`, `calibrated_threshold`, `precision_recall` |
| Guardrails + orchestration | `agents.py::mask_pii`, `input_guardrail`, `check_order_status`, `SupportResponse`, `SessionMemory`, `SupportOrchestrator`, `build_crewai_crew` |
| Review stage | `review_stage.py::review_draft`, `run_autogen_review`, `Verdict` |
| Governance | `governance.py::enforce_tool_access`, `RequestBudget`, `ResponseCache` |
| API | `app.py::health`, `ask`, `add_document`, `websocket_chat`, `log_request` |
| Evaluation | `evaluation.py::TEST_QUERIES`, `evaluate` |
| Demo orchestration | `run_demo.py::main` |
| Tests | `tests/test_core.py` (7 functions) |
| CI | `.github/workflows/tests.yml` |
| Artifacts | `artifacts/orders.json`, `rag_metrics.json`, `evaluation.json`, `demo_transcript.txt` |

---

## 19. Assumptions, Limitations, and Items Requiring Confirmation

**Assumptions [labelled]:**
- **[Inference]** The committed artifacts reflect the most recent intended run; they were produced in lexical-fallback mode.
- **[Inference]** Tests pass in a correctly provisioned environment (static reading suggests they will; **not executed here**).

**Limitations of this analysis:**
- No code was executed; no live test/demo results were produced in this environment (per user instruction and missing toolchain).
- Deployment and business-approval statuses are unverifiable and reported as such.

**Requires team confirmation (see Section 12 Open Questions):** LangChain-memory requirement, whether CrewAI/AutoGen must be executed, location/value of the risk classification, and submission/grading environment specifics.
