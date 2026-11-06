# Pending Tasks — Nykaa Domain Support Agent (pre-evaluation checklist)

Last reviewed: 2026-11-06.
Basis: static read of `dataset.py`, `rag.py`, `agents.py`, `governance.py`, `review_stage.py`, `evaluation.py`, `app.py`, `run_demo.py`, `tests/test_core.py`, `README.md`, `.env.example`, `requirements.txt`, CI workflow, committed `artifacts/*`, and the requirements brief (`# Final Capstone -Ecom- Nykaa Domain Req.txt`).
Nothing was executed. Python, CrewAI and AutoGen are assumed installed. Items marked **(verify)** are inferred from code and need a real run to confirm.
`PROJECT_ANALYSIS_AND_STATUS.md` and `TECHNICAL_HANDOVER.md` were not re-read. Check them for stale claims after the fixes below.

---

## 0. Corrections to the previous version of this file

- **Old P-1 ("artifacts are lexical-fallback") is stale.** The committed `artifacts/rag_metrics.json` and `demo_transcript.txt` already show `sentence-transformers+chromadb` for both strategies. No re-run is needed just to flip the mode.
- Because of that, **the README is now out of date.** It still quotes the lexical-fallback numbers (fixed P 0.266 / R 0.800, sentence P 0.233 / R 0.600). The real embedding numbers are different, and its recommendation is no longer supported (see B-3).

---

## 1. BLOCKERS: fix before submitting

### B-1. CrewAI crew is never run, and it will not run with zero API keys as written (Task 7)
- `build_crewai_crew()` exists but is never called, and `.kickoff()` appears nowhere. The brief requires showing the RAG tool and the lookup tool both invoked via `.kickoff()` on different queries.
- **API key issue:** the `Agent(...)` objects are created without an `llm=`. CrewAI then defaults to an OpenAI model and expects `OPENAI_API_KEY`. That contradicts the "zero API keys / `MOCK_LLM`" requirement.
- Action:
  - Provide a deterministic mock LLM for CrewAI (a `BaseLLM` subclass or custom `LLM`, depending on the installed version). It should emit tool-call turns for the retrieval and lookup agents and a plain final answer for the composer. Pass it as `llm=` to all three agents.
  - Use a local model (e.g. Ollama) only as an optional alternative, not as the default.
  - Set `memory=False` and disable CrewAI telemetry (`CREWAI_DISABLE_TELEMETRY=true` / `OTEL_SDK_DISABLED=true`) to avoid network calls.
  - Add a `kickoff` demo to `run_demo.py` (or a new `run_crew_demo.py`): `crew.kickoff(inputs={"query": ...})` for one policy query and one order query. Save the transcript to `artifacts/crew_transcript.txt`, showing the `orchestrator.tool_calls` counters changing.
  - Set `output_pydantic=SupportResponse` (or an equivalent response format) on the composer task. Today the crew output is never validated against the schema (Task 9).
- Note: the tool wrappers return `str(dict)`. That is fine for a mock LLM but brittle for a real one. Prefer `json.dumps`.

### B-2. AutoGen review stage is not executed and not wired into the request path (Tasks 14, 15)
- `run_autogen_review()` is never called. The "approve" and "revise" demos use the plain-Python `review_draft()`, which is not a `RoundRobinGroupChat`. The brief requires an actual 2-agent Autogen team with structured verdicts.
- Nothing in `agents.py` or `app.py` calls any review step. The brief says answers are reviewed before they reach the user. `/ask` and `/ws/chat` currently return the orchestrator output unreviewed.
- Action:
  - Run the real team with a key-free model client. The `ReplayChatCompletionClient` in `autogen-ext` is the likely fit, with scripted replies for the reviewer and for the editor's `Verdict` JSON **(verify that the installed version supports structured output / `output_content_type` with it)**. A local Ollama client is the fallback.
  - **`autogen-ext` is not in `requirements.txt`.** Add it.
  - Demo at least two cases with real Autogen output: one approved unchanged, and one revised after an injected ungrounded claim. Save to `artifacts/autogen_transcript.txt`.
  - Wire the review step into `SupportOrchestrator.ask` (or the API layer) so every policy answer passes through it. Keep `review_draft()` as the key-free default if needed, and say so in the README.
- Bug in `run_autogen_review`: the task string is built with `"\\n"` inside an f-string, so the model sees literal `\n` characters instead of newlines. Use `"\n"`.
- The silent fallback to `review_draft()` when no `Verdict` is found hides failures. Log it or raise.

### B-3. README numbers and chunking recommendation are stale and contradicted by the artifacts (Tasks 4, 5)
- Current committed numbers (from `rag_metrics.json`, embedding mode):
  - fixed: mean precision 0.300, mean recall 0.800
  - sentence: mean precision 0.333, mean recall 0.800
- The README says to deploy **fixed**, but sentence is now equal or better on both measures.
- The default retriever in `SupportOrchestrator` and `app.py` is still `"fixed"`.
- Action: after fixing B-4 and re-running, rewrite the "Chunking comparison" section with the new per-query numbers (2–3 sentences, citing both sets) and make the default retriever match the recommendation.
- The README must also **state the measured calibration values**, which the brief requires. They are currently only in the JSON:
  - in-scope top-1: 0.526 – 0.860 (fixed), 0.531 – 0.885 (sentence)
  - out-of-scope top-1: 0.062 – 0.104
  - chosen thresholds: ≈0.315 (fixed), ≈0.314 (sentence)

### B-4. Gold-relevance map gives a wrong result for one of the five Task 5 queries (Task 5)
- `relevance_for_query` matches on substrings of the query. The phrase `"exchange apparel size"` does not appear in "Can I exchange **an** Apparel size?", so the relevant set is empty. The committed artifact shows `precision=0/3; recall=0/0` for it, which is wrong: `size-exchange` is the right document.
- Action: replace the substring heuristic with an explicit `{query: {relevant_doc_ids}}` dict for the five (or more) metric queries. Re-run, then update the README (B-3). With the fix, both strategies should score precision ≈ 1/3 on that query and recall 1.0 **(verify after re-run)**.

### B-5. Escalation score degenerates to a boolean, and the README claim is false (Task 6)
- `score = 0.65*delayed + 0.35*(days/30)` with threshold `0.65`:
  - A non-delayed order can score at most 0.35, so it can never be escalated.
  - Any delayed order scores at least 0.65, so it is always escalated.
  - So the recommendation equals `delayed_shipment`, which is the "bare boolean" the brief rules out.
- The README says a "very old non-delayed order" can surface. That is incorrect.
- The brief also asks that the threshold be justified with the dataset distribution (e.g. a percentile). The README only says to "treat the distribution as the justification" and states no number.
- Action:
  - Redesign so both signals matter. For example, `0.5*delayed + 0.5*(days/30)` or a weighted blend that includes order status, so Delivered/Refunded orders do not escalate.
  - Compute the score distribution over `ORDERS`, pick the threshold as a stated percentile (e.g. the 80th), and print it in `dataset.py`'s output.
  - Write the exact formula, the percentile and the resulting number in the README. Update tests.

### B-6. Task 15 written risk classification is missing
- Add a README section classifying the system as **Medium** (customer-support / ticket tooling per the brief's scheme), with a one-paragraph justification. Mention that order IDs and masked PII are handled and that the system is advisory only.
- The least-autonomy write-up (one paragraph explaining the guard) should be added next to it.

### B-7. Task 15 least-autonomy demo is partly circular
- Inside the CrewAI tool wrapper, `enforce_tool_access` is called with a hardcoded `"Lookup Agent"`, so it can never fail from the crew path. The demo only calls `enforce_tool_access("Response Composer", ...)` directly.
- Action: bind the agent identity when the tool is created (e.g. `make_order_tool(agent_name)`), then show in the transcript that building or calling it for the composer or retrieval agent raises `UnauthorizedToolUse`. Add a test.

### B-8. Task 8 session memory: not LangChain, and no real carry-over is demonstrated
- The brief names `InMemoryChatMessageHistory` + `RunnableWithMessageHistory`. The code uses a custom `SessionMemory`, and `langchain-core` is not in `requirements.txt`.
- The stored history is never used to answer anything. `run_demo.py` only prints turn counts (`memory-demo turns=2`, `fresh-demo turns=0`). That is not "state correctly carried across a multi-turn exchange".
- Action:
  - Adopt LangChain's history classes (add `langchain-core` to requirements), or get written confirmation that the custom class is acceptable.
  - Make memory affect behaviour. Example: turn 1 "What is the return window for Beauty products?", turn 2 "What about Apparel?" resolved from history. Or remember the last order ID for "is it delayed?".
  - Print a multi-turn transcript and a separate fresh-session transcript where the same follow-up fails or asks for clarification. Save both under `artifacts/`.

### B-9. Task 13 evaluation is not an LLM-as-judge
- `evaluation.py` scores with hard-coded rubric proxies (for example `grounding` is 5 or 4 by fixed rules; `completeness` can never exceed 4). There is no judge prompt.
- The README is honest about this, but the brief asks for an "LLM-as-judge prompt (running under MOCK_LLM)".
- Action: add an explicit judge prompt template (query, retrieved context, answer, rubric for the four properties) and a deterministic `MockJudge` that computes scores from that input. For example, grounding as the share of answer tokens found in the context, and safety from PII/injection flags. Save the prompt text and per-query scores to `artifacts/evaluation.json`. Keep the 15-query set (12 topics + 3 edge cases, which already satisfies the coverage rule).
- Also fix truncated answers in `evaluation.json` (see S-2).

### B-10. Task 12 logging does not cover every request, and there is no committed evidence
- `log_request` is only called from `/ask`. `/ws/chat` messages and `/add-document` write no log line. The brief says every request.
- `artifacts/*.jsonl` is gitignored, so no log evidence is submitted.
- Action: log WebSocket messages (same trace ID + timing + masked text) and `/add-document`. Commit a small sanitized sample as `artifacts/sample_requests.jsonl` (not matching the ignore pattern) and add a test asserting `[PHONE_MASKED]` / `[LAST4_MASKED]` appear and the raw values do not.

### B-11. Task 10 output-side groundedness guard is not visibly demonstrated
- In `run_demo.py` the "Output groundedness guard" step calls `answer(..., threshold=1.1)`, which returns a dict and raises nothing. The `except` never fires, so **no line appears in `demo_transcript.txt`**.
- Action: replace it with `agent.ask("<out-of-scope question>")` and print the resulting `SupportResponse` (`intent="unsupported"`, `grounded=False`, fallback text), labelled "OUTPUT GROUNDEDNESS GUARD". Also demo the PII and injection guards through `ask()` / the API, not just the helper function.

### B-12. Tests likely fail with plain `pytest` and CI will not run from the repo root **(verify)**
- `tests/` has no `conftest.py`/`__init__.py`, and there is no `pytest.ini`. With default import mode, `pytest -q` adds `tests/` (not the project root) to `sys.path`, so `from dataset import ...` can raise `ModuleNotFoundError`. Add an empty root `conftest.py` or a `pytest.ini` with `pythonpath = .`.
- `.github/workflows/tests.yml` lives inside `nykaa-domain-support-agent/`. GitHub only runs workflows from the **repository root**. Make sure the repo root is the `nykaa-domain-support-agent` folder, or move `.github/` up.
- CI installs the full requirements (including PyTorch) and downloads the embedding model on every run. Acceptable, but slow.

### B-13. README claims vs. reality on "zero API keys" and `MOCK_LLM`
- The brief requires the README to confirm everything runs under `MOCK_LLM` with zero API keys. After B-1 and B-2, state exactly which paths are key-free (everything, via mock LLM / replay client) and which are optional (Ollama).
- `MOCK_LLM` is only set by `run_demo.py` via `os.environ.setdefault`. **No code reads it.** Either make it a real switch (mock LLM vs. a user-provided real client) or remove the claim that it controls behaviour.

### B-14. FastAPI deployment gaps (Tasks 11, 12)
Status: the minimum for Task 11 is met. `POST /ask` and `POST /add-document` use Pydantic request models, `GET /health` exists, and `/ws/chat` catches `WebSocketDisconnect`. `/ask` enforces the budget (413) and the injection block (400) and writes a masked JSONL log line. Based on a read of `app.py`; not run.

Gaps to close:
- **API does not expose the crew.** The brief says to expose the crew behind the endpoints. `/ask` and `/ws/chat` call the deterministic `SupportOrchestrator`, so CrewAI and AutoGen are never reached. Route both through the crew and the review stage once B-1 and B-2 are done. Keep the deterministic path as an explicit fallback.
- **Logging is incomplete** (see B-10). `/ws/chat` and `/add-document` write no log line.
- **WebSocket handler is fragile.**
  - `receive_json()` is outside the inner `try`. Malformed JSON or any unexpected exception drops that client without an error message. The server survives, but the client gets no reply.
  - Wrap the whole loop. Send `{"error": ...}` for bad input and keep the connection open. Only exit on `WebSocketDisconnect`.
  - `orchestrator.ask` is blocking and called inside `async def`, so one slow request stalls every connection. Use `run_in_threadpool`.
  - Error shapes differ from `/ask` (an `{"error": ...}` dict versus an HTTP 400). Pick one documented schema and add a Pydantic model for the WebSocket reply.
- **`/add-document` weaknesses.**
  - No response model; it returns a raw dict. Add one.
  - It does not clear the response cache, so stale answers can survive after a document is added (see S-3).
  - It swaps `orchestrator.retriever` without locking and re-embeds the whole corpus on every call. Add a lock, and note the cost in the README.
  - Apply the request budget and PII masking to the document text as well.
- **No evidence it works.** Add:
  - `TestClient` tests for `/ask` (happy path, 400 injection, 413 budget), `/add-document` and `/health`.
  - A WebSocket test: send a message, get a reply, disconnect, then reconnect and confirm the server still answers. This is the "survives a client disconnect" demonstration the brief asks for.
  - A committed transcript with example `curl` calls and one WebSocket session, saved under `artifacts/` (for example `artifacts/api_transcript.txt`).
- **Heavy start-up.** `app.py` builds the orchestrator at import time (model load + Chroma indexing), which slows tests and `--reload`. Move it into a FastAPI `lifespan` handler, or build lazily on first use.
- **Run instructions.** Confirm `uvicorn app:app` works from the project folder. `LOG_PATH` is relative to the working directory, so note this in the README.
- **Not required:** a Dockerfile or hosted deployment. The brief only asks for a FastAPI backend, so `uvicorn` is sufficient. Add Docker only as an optional extra.

---

## 2. SHOULD FIX: quality issues a grader may notice

- **S-1. Retriever recomputes calibration on every query.** `answer()` calls `calibrated_threshold()`, which runs 5 searches (plus repeats) each time. `lru_cache` is imported but unused. Compute the threshold once per retriever.
- **S-2. Answer composition mixes unrelated documents and truncates text.** `answer()` joins the top-2 chunks regardless of the second chunk's similarity. Examples in the transcript: the Beauty return query also returns size-exchange text; the delivery SLA answer includes escalation-matrix text cut mid-word ("Any suspected fraud or exposed p", "re to the specialist queue" in the damaged-item answer). Fix: include only chunks at or above the threshold, and make fixed chunks respect word/sentence boundaries.
- **S-3. Response cache is not invalidated by `/add-document`.** The cache lives on the orchestrator; after a new document is added, stale answers can still be served. Clear it in `add_document`. Also consider a size bound.
- **S-4. Card-number masking is incomplete.** `CARD_RE` masks only the first 4 digits after "card", so `card 4111111111111111` leaves 12 digits visible. Add `(?!\d)` after the group, and handle "ending in 1234" style phrasing. Add tests.
- **S-5. Injection detection is a 4-phrase exact match.** Variants like "ignore previous instructions" or "disregard the above" pass. Use a few regexes. Keep it simple and document the limits.
- **S-6. `intent="mixed"` is declared but never produced.** A query like "check NYK-0001 and tell me the return window" returns only the order lookup. Either handle it or remove `mixed` from the schema.
- **S-7. `dataset.py` retry loop bug.** `while not 0.10 <= rate <= 0.30: ... return generate_orders(seed + attempt, n)` returns on the first pass, so the loop never loops. It works only because the base seed lands in band (14.0%). Rewrite as a proper reseed loop. Note: the recursive call also resets `attempt` each time.
- **S-8. Dataset realism and reporting.** Order values are uniform ₹299–24,999 for every category (a ₹24,999 Beauty order is odd). Use per-category ranges and state the one-sentence reasoning in the README. Also report the actual counts and delayed % in the README (currently: Apparel 8, Electronics 9, Home 11, Footwear 8, Beauty 14; Placed 5, Shipped 12, Delivered 15, Returned 7, Refunded 11; 14.0% delayed).
- **S-9. Budget is character-based.** The brief says token/cost budget. Either estimate tokens (e.g. `len/4`) with a stated per-request cap, or explain why characters are the proxy. Add an API-level test for the 413 response.
- **S-10. Unused `.env.example` settings.** `RAG_TOP_K` and `RAG_MIN_SIMILARITY` are not read anywhere; only `REQUEST_CHAR_BUDGET` and `LOG_PATH` are. `RAG_MIN_SIMILARITY=0.0` would disable the fallback if wired in later. Remove or wire up properly. Also no `.env` loader exists (no `python-dotenv`).
- **S-11. Windows console encoding.** `run_demo.py` prints `₹` and en-dashes. On a cp1252 Windows console this can raise `UnicodeEncodeError`. Run with `PYTHONUTF8=1` or call `sys.stdout.reconfigure(encoding="utf-8")`. Add a note to the README.
- **S-12. App start-up and event loop.** `app.py` builds the orchestrator (model load + Chroma indexing) at import time, and the WebSocket handler calls the blocking `orchestrator.ask` directly inside `async def`. Fine for a demo; consider `run_in_threadpool` for the WebSocket path.
- **S-13. Dependencies.**
  - Add `autogen-ext` (B-2) and `langchain-core` (B-8).
  - `crewai>=0.80` and `chromadb>=0.5,<2.0` are open ranges and CrewAI itself constrains `chromadb`. Install everything in one clean venv and confirm it resolves **(verify)**.
  - Pin exact versions (e.g. a `requirements.lock` from `pip freeze`) so graders reproduce your run.

---

## 3. NICE TO HAVE

- Add API tests with FastAPI `TestClient`: `/ask` happy path, 400 on injection, 413 on budget, `/add-document`, WebSocket send + disconnect + reconnect.
- Add tests for `precision_recall`, the new escalation formula, cache invalidation, and the Autogen/CrewAI demos (skipped when optional deps are missing).
- Replace the "initial local lexical-fallback" wording in the README and keep the lexical fallback documented as a convenience only.
- Move the internal analysis docs out of the public repo, or refresh them so they do not contradict the README.

---

## 4. API keys / configuration summary

| Item | Needed? | Notes |
|---|---|---|
| Hosted LLM key (OpenAI, etc.) | **No**, but only if B-1 is done | Without `llm=` on the CrewAI agents, CrewAI defaults to OpenAI and wants `OPENAI_API_KEY`. |
| Hugging Face token | No | `all-MiniLM-L6-v2` is public; first run needs network. |
| Local model server (Ollama) | Optional | Only if you choose it instead of the mock LLM / replay client. |
| `.env` | Optional | Copy `.env.example` to `.env`. Currently only `REQUEST_CHAR_BUDGET` and `LOG_PATH` are read, and only from the real environment (no dotenv loader). |
| Telemetry | Disable | `CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`. |

---

## 5. Status against the brief

| # | Task | Status |
|---|---|---|
| 1 | Dataset generator | Done, with a retry-loop bug (S-7) and README reporting gaps (S-8) |
| 2 | Knowledge base (13 docs) | Done. Only 2 of 13 docs spot-checked for the 2–5 sentence rule |
| 3 | Two chunkers + two Chroma collections | Done; artifacts are in embedding mode |
| 4 | Grounded generation + calibrated threshold | Works; README must state measured values (B-3); answer composition issues (S-2) |
| 5 | Precision/recall + recommendation | Wrong gold set for one query (B-4); README stale (B-3) |
| 6 | `check_order_status` + escalation score | Formula degenerate (B-5) |
| 7 | CrewAI crew with `.kickoff()` | **Not demonstrated; needs mock LLM** (B-1) |
| 8 | Session memory | Custom class, no real carry-over demo (B-8) |
| 9 | Structured output | `SupportResponse` OK; crew does not enforce it (B-1) |
| 10 | Guardrails | Implemented; output guard demo missing from transcript (B-11); gaps S-4, S-5 |
| 11 | FastAPI + WebSocket | Minimum met; crew not exposed, WebSocket fragile, no tests or transcript (B-14) |
| 12 | JSONL logging | Partial (B-10) |
| 13 | 15-query evaluation | Proxy scoring, no judge prompt (B-9) |
| 14 | Autogen review | **Not demonstrated, not wired in** (B-2) |
| 15 | Governance | Risk classification missing (B-6); least-autonomy demo circular (B-7); budget OK |
| 16 | Response cache | Done; invalidation gap (S-3) |

---

## 6. Final pre-submission checklist

- [ ] B-1 to B-14 done (B-12 verified by actually running `pytest`).
- [ ] API transcript committed (curl calls + WebSocket session including a disconnect and reconnect).
- [ ] Re-run `python run_demo.py` (with UTF-8 enabled) and commit refreshed `artifacts/` (`demo_transcript.txt`, `rag_metrics.json`, `evaluation.json`, `orders.json`).
- [ ] New transcripts committed: CrewAI `.kickoff()`, Autogen approve + revise, multi-turn + fresh memory, sample JSONL log, WebSocket session including a disconnect.
- [ ] README updated: measured thresholds, real P/R numbers + recommendation, dataset counts/price reasoning, escalation formula + percentile threshold, risk classification, least-autonomy paragraph, accurate "zero API keys" statement.
- [ ] `pytest -q` passes locally and in CI; repo root is the project folder so `.github/workflows` runs.
- [ ] Requirements pinned and installable in a fresh venv.
- [ ] No `.env`, `.venv/`, or real personal data committed; all names, addresses and phone numbers are fabricated.
- [ ] Repository is **public** and you have the single GitHub link ready to submit.

## Open questions (confirm with the capstone owner)

- Is the custom `SessionMemory` acceptable, or is LangChain mandatory (Task 8)?
- Is a deterministic mock review stage plus a Replay-client Autogen run acceptable, or must Autogen use a real model?
- Is a deterministic judge acceptable for "LLM-as-judge under MOCK_LLM" (Task 13)?
