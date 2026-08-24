# IMPLEMENTATION_PLAN.md — Self-Correcting RAG Framework

**Source:** finalized REQUIREMENTS.md (standalone build, own Qdrant, own docker-compose)
**Format:** dependency-ordered phases, not days. Every phase has a done-checkpoint.
Phases marked **MUST NOT SPLIT** break something specific if separated — the
reason is stated, not just the flag.

**Repo layout note:** this plan is written against the actual finalized
scaffold — flat `src/` layout (no nested `src/self_correcting_rag/`
package folder), `schemas/state.py` holds the LangGraph state `TypedDict`
(there is no separate top-level `state/` folder), and `src/config.py` is
the single settings module (root-level `config/` holds only YAML data —
`rate_limits.yaml`, `thresholds.yaml` — it is not a Python package). See
`scaffold-self-correcting-rag.ps1` for the full tree; there's no separate
`REPO_STRUCTURE.md` doc, the scaffold script itself is the source of truth
for paths.

---

## 1. Key Architecture Decisions

Resolving REQUIREMENTS.md's §0.2/§0.3 into real module boundaries.

**Module layout:** `config.py` (settings, `src/config.py`), `logging_config.py`,
`schemas/` (Pydantic LLM-output validation models, *plus* `schemas/state.py`
for the LangGraph state `TypedDict` — state lives here, not in a separate
folder), `retrieval/`, `grading/`, `rewriting/`, `fallback/`, `generation/`,
`orchestration/` (the LangGraph graph + node functions), `audit/`,
`streaming/`, `llm_clients/` (new — see below), `api/`, `eval/` (root-level,
outside `src/`). Root-level `config/` is YAML data only
(`rate_limits.yaml`, `thresholds.yaml`), loaded by `src/config.py` — not a
Python package, never imported directly.

**Where the counters live.** All loop-bounding state — `iteration_count`,
`max_iterations`, `rewrite_history`, `fallback_used`, `generation_attempts`,
`max_generation_attempts`, `low_confidence` — lives in one place:
`schemas/state.py`'s `TypedDict`. Every node reads and returns a patch to
this dict; no counter is tracked anywhere else (not in a class instance,
not in a module-level global). This is what makes NFR-9 (concurrency
safety) achievable for free — LangGraph threads state per-invocation, so as
long as nothing is cached outside the state dict, concurrent queries can't
cross-contaminate.

**Flagging a real inconsistency in REQUIREMENTS.md, not silently resolving
it:** §0.2 gap #1 describes `max_iterations` as "a single shared counter"
covering both retrieval-correction and generation-correction. But §3's own
pseudocode implements **two separate counters** with **two separate caps**
— `iteration_count`/`max_iterations` (default 3) gates the retrieve→grade→
rewrite loop, and `generation_attempts`/`max_generation_attempts` (default
2) independently gates the generate→grade→regenerate loop. These aren't the
same counter sharing one budget; they're two budgets that happen to total
5, not 3. I'm implementing exactly what §3's executable pseudocode
specifies (two fields, two caps) because that's the literal spec you
finalized — the "shared" language in §0.2 reads as directional intent
rather than the actual design. Two consequences worth knowing before you
build: (1) §4's ~10–12-calls-per-query worst case is consistent with the
two-counter reading, not the one-counter reading — good, no rework needed
there; (2) if you actually want a single combined budget of 3 total
attempts across both loops (stricter, cheaper, but means a query that
needed 2 rewrites gets only 1 shot at generation), that's a different state
schema and a different §3.4 branch condition. Flag it back to me if you
want that instead — cheap to change now, before Phase 1.

**Routing gate implementation (§0.2 gap #4).** Rule-based, zero LLM calls,
zero added latency on the hot path — not an LLM classifier. Signals: query
token count, presence of multi-clause/comparison markers ("vs", "compare",
"and also", multiple question marks), and a cheap keyword-overlap check
against the corpus's known entity/topic list built at ingestion time (if
the query shares almost no vocabulary with anything in the corpus, that's
itself a signal it probably needs correction, not confident fast-path
generation). **Default-to-correction-path on uncertainty** — a false
positive here just costs one unnecessary grading pass; a false negative
(fast-pathing a query that needed correction) ships an ungraded answer with
no safety net. This is a judgment call REQUIREMENTS.md left open (§0.2 gap
#4 flags the proxy as unvalidated) — I'm picking the conservative default
because the downside is asymmetric.

**New module not explicit in REQUIREMENTS.md: `llm_clients/`.** REQUIREMENTS
specifies *which* model handles each call site (FR-19) but not the
provider-failover/backoff mechanics implied by NFR-10. I'm adding a thin
`llm_clients/` layer (`groq_client.py`, `gemini_client.py`,
`rate_limiter.py`) that every other module calls through, rather than
letting `grading/`, `generation/`, etc. each hold their own Groq/Gemini SDK
calls. This is the seam where §2 (concurrency strategy) actually gets
implemented — one place to add backoff, one place to add cross-provider
failover, one place to swap a model string in config without touching six
files.

**Fast-path generation model.** REQUIREMENTS' FR-18 says fast-path
bypasses grading and goes straight to generation, but doesn't say whether
fast-path uses the same strong-tier model as correction-path generation
(FR-9) or something cheaper/faster. Defaulting to **the same model** for
both, for consistent answer quality — a cheaper fast-path model is a real
latency optimization worth measuring later (Phase 11), but starting with
one generation code path instead of two is simpler and defers a premature
optimization. Flagging this as an open call, not silently deciding it's
obviously right.

---

## 2. Concurrency & Rate-Limit Strategy

**I don't have live access to current Groq/Gemini free-tier limits in this
session** — no web search tool is available here, and my training data on
exact RPM/TPM numbers is both stale (limits change) and not something I'd
trust myself to state precisely without checking. If you want this section
filled in with real numbers, enable web search for a follow-up, or pull
current values yourself from `console.groq.com/settings/limits` and the
Gemini API quota page before you build `rate_limiter.py`. What follows is
the architecture, parameterized so it's correct regardless of what the
actual numbers turn out to be.

**Why this matters more here than in Project 1:** a single `correction_path`
query issues up to ~10–12 LLM calls (per REQUIREMENTS §4), almost all of
them cheap-tier grading calls (document grading × up to 3 retrieval
attempts, groundedness + relevance × up to 2 generation attempts, plus
fallback-content grading). The cheap-tier grader is the TPM/RPM pressure
point, not the generator — one query can hit the grading endpoint 5-8 times
before it hits the generation endpoint even twice.

**Design:**

- **Per-provider, per-model token bucket** in `llm_clients/rate_limiter.py`,
  config-driven (`RPM`, `TPM` values live in `src/config.py`, sourced from
  `config/rate_limits.yaml`, loaded from env — you fill in real numbers
  once you have them). One bucket per (provider, model) pair, since Groq's
  llama-3.1-8b-instant and llama-3.3-70b-versatile have independent limits
  from each other.
- **Exponential backoff with jitter on 429**, capped at 3 retries on the
  *same* provider before failing over.
- **Cross-provider failover**: every call site that can tolerate it
  (grading, groundedness, relevance, generation) is configured with a
  primary provider/model and a fallback provider/model (Groq→Gemini).
  Query rewriting and generation get this; the routing gate doesn't need it
  (it's not an LLM call — see §1). This is the actual mitigation for
  rate-limit exhaustion, independent of what the exact numbers are: two
  providers' limits rarely exhaust at the same moment.
- **Sizing formula you'll plug real numbers into:** once you have
  `TPM_cheap` for the grading model, the sustainable correction-path query
  rate is roughly
  `floor(TPM_cheap / (avg_tokens_per_grading_call × grading_calls_per_query))`.
  `grading_calls_per_query` worst-case is ~5 (3 retrieval-grading passes +
  2 answer-grading passes; fallback-content grading only fires on the zero-
  correct branch so treat it as a tail case, not the average). Use this
  formula in Phase 11's load test to compute your actual demo-safe
  concurrency ceiling once real TPM numbers are in.

---

## 3. Error Taxonomy & Retry Policy

**Retryable:**

| Failure | Policy |
|---|---|
| Groq/Gemini 429 | Exponential backoff + jitter, same provider, up to 3 attempts, then fail over (§2) |
| Transient 5xx / network error (any provider, or Tavily) | Backoff, max 3 attempts |
| Malformed grade output (LLM response fails Pydantic schema validation) | Retry once with a stricter "return only valid JSON matching this schema" reprompt. If still malformed: **fail closed, not open** — default that chunk's grade to `AMBIGUOUS`, never to `CORRECT`. A broken grader should never silently pass bad content through as if it were verified. |
| Tavily returns zero results | Not an error — a valid outcome. Feeds zero chunks into grading, which naturally routes down the `p_correct == 0` branch (§3.2 of REQUIREMENTS) toward terminal/low-confidence. No special-casing needed if grading handles an empty chunk list correctly (add this as an explicit unit test — see §5). |
| Injection-gate rejects fallback content | Not retried against Tavily with the same query (that just re-fetches the same flagged content). Drop the flagged chunk(s); if *all* fallback content gets flagged, treat it identically to the Tavily-empty-results case above. |

**Terminal (not errors — defined outcomes, don't route them through
exception handling):**

- Iteration-cap exhaustion → §3.5's terminal/low-confidence path. By
  design, not a failure.
- Document-grading vs. groundedness-check disagreement (§0.2 gap #3) →
  the defined terminal path (regenerate once, then escalate). Not
  retried differently than a normal groundedness failure.

**Terminal (actual infra failures — do not disguise these as
low-confidence answers):**

- Qdrant connection failure (startup or mid-query) → hard failure,
  propagate a 5xx to the caller. This is a distinct failure class from a
  correction-loop soft failure, and conflating the two would hide a real
  outage behind what looks like "the system tried its best." A client
  demoing this system needs to be able to tell "Qdrant is down" apart from
  "the query was genuinely unanswerable."
- Config validation failure at boot (missing threshold, bad model name,
  etc.) → fail fast at startup. Never silently fall back to a default for
  a threshold that should have been explicitly set.

---

## 4. Model Routing Rule

Every LLM call site, mapped to a specific model — no call site left to
"whatever's configured generically":

| Call site | FR | Primary | Failover | Tier rationale |
|---|---|---|---|---|
| Document/chunk grading | FR-4 | Groq `llama-3.1-8b-instant` | Gemini `gemini-2.5-flash` | Ternary classification — doesn't need reasoning depth |
| Fallback-content grading | FR-7/8 | Groq `llama-3.1-8b-instant` | Gemini `gemini-2.5-flash` | Same grader, same input shape |
| Groundedness check | FR-10 | Groq `llama-3.1-8b-instant` | Gemini `gemini-2.5-flash` | Classification-style task |
| Relevance check | FR-11 | Groq `llama-3.1-8b-instant` | Gemini `gemini-2.5-flash` | Classification-style task |
| Query rewriting | FR-6 | Groq `llama-3.3-70b-versatile` | Gemini `gemini-2.5-flash` | Needs real reasoning to produce a genuinely different, better-targeted query — a cheap model tends to paraphrase rather than reformulate |
| Answer generation (fast-path and correction-path) | FR-9/FR-18 | Groq `gpt-oss-120b` | Gemini `gemini-2.5-flash` | Strong tier, matches Project 1's top-tier generation precedent |
| Routing gate | FR-3 | — (rule-based, no LLM call) | — | Zero cost, zero latency on the hot path — see §1 |

---

## 5. Testing Strategy

**Unit tests** — every node/grader tested against a mocked LLM boundary
(a fake client returning canned, schema-valid `Pydantic` responses, plus at
least one canned malformed response per grader to exercise the retry/fail-
closed path from §3):

- `test_document_grader.py` — canned CORRECT/AMBIGUOUS/INCORRECT responses parse correctly
- `test_aggregation.py` — `p_correct` thresholds route to generate / rewrite / fallback exactly per §3.2's pseudocode, including the boundary cases (`p_correct` exactly 0.5, exactly 0)
- `test_query_rewriter.py` — rewrite differs from original; a second rewrite avoids repeating `rewrite_history`
- `test_injection_guard.py` — a planted injection marker is blocked; clean content passes
- `test_answer_grader.py` — groundedness and relevance are scored and gated independently (a high-relevance/low-groundedness case and a low-relevance/high-groundedness case must route differently)
- `test_routing_gate.py` — a simple factoid query routes fast-path; a multi-clause comparison query routes correction-path
- **Empty-results edge case** (from §3's error taxonomy): grading an empty chunk list doesn't crash and routes down the `p_correct == 0` branch

**Integration tests** — real components wired together, fixture corpus
(small, hand-built) instead of the full demo corpus:

- `test_retrieval_substrate.py` — dense + sparse + RRF fusion + rerank against the fixture corpus returns the expected top document for a known query
- `test_correction_loop_orchestration.py` — full graph run with a mocked LLM boundary, asserting state transitions (iteration count increments correctly, `rewrite_history` accumulates, terminal state sets `low_confidence` correctly when the cap is hit)

**Real-model checkpoint (Phase 3 only, not a mock):** the document-grading
prompt itself is validated against 3 hand-picked real chunks from the
ingested demo corpus with real LLM calls before moving on — a mocked test
proves your code parses a response correctly, it doesn't prove your
*prompt* actually grades well. This is the one place I'd insist on a live
call in the middle of otherwise-mocked test development.

**Permanent named regression tests — the three REQUIREMENTS.md §6 DoD trap
cases, one file each, run against the full stack (Docker, real corpus, real
LLM calls) as part of Phase 13:**

- `test_trap_bad_retrieval_triggers_rewrite_and_recovers.py`
- `test_trap_hallucination_caught_by_groundedness_check.py`
- `test_trap_unanswerable_query_escalates_with_low_confidence.py`

**Load test** (`scripts/load_test.py`, not pytest — run manually or in a
nightly job, not on every commit) — exercises the rate-limiter/backoff
under concurrent correction-path load, feeding into Phase 11's threshold
and concurrency-ceiling tuning.

---

## 6. Phase-by-Phase Plan

### Phase 0 — Scaffolding + Config
Repo scaffold (`scaffold-self-correcting-rag.ps1`), `uv init --package`
(run **before** the scaffold script — see the script's own header comment),
`.env.example`, `docker-compose.yml` skeleton with a standalone Qdrant
service (NFR-7), `src/config.py` loading all FR-21 thresholds + model tiers
+ iteration caps, sourced from `config/rate_limits.yaml` and
`config/thresholds.yaml` plus env vars.
**Checkpoint:** `uv run python -c "from config import get_config; print(get_config())"`
prints a fully populated config object — every threshold from §3 (0.5, 0.7,
0.7), both iteration caps (3, 2), and both model tiers present and
correctly typed, none silently defaulted. **Verify the import itself
resolves first** — the flat `src/` layout (no nested package folder) means
this depends on how `pyproject.toml`'s build backend exposes `src/` as
importable; if `from config import get_config` fails with a `ModuleNotFoundError`
even though `src/config.py` exists, that's a packaging-config fix (e.g.
`packages = ["src"]` under your build backend's tool section), not a code
bug — confirm this works before writing anything that depends on it.
**Safe stopping point:** yes.

### Phase 1 — Schemas & State
`schemas/state.py`'s `TypedDict` (all fields from §1 above), `schemas/`
Pydantic models for chunk grades, answer grades, rewrites.
**Checkpoint:** a canned malformed LLM string (not matching any schema)
raises a clear Pydantic validation error — this is the exact boundary
Phase 3-6's "malformed output" retry policy (§3) depends on; prove it fails
loudly before building anything that relies on it failing loudly.
**Safe stopping point:** yes. Small enough to pair with Phase 0 in one
sitting if you have the time, but not required to.

### Phase 2 — Hybrid Retrieval Substrate (FR-1, FR-2)
`retrieval/dense_retriever.py` (Qdrant), `sparse_retriever.py` (bm25s),
`fusion.py` (RRF), `reranker.py` (cross-encoder), `retriever.py` (ties them
into one `retrieve(query)` call). `scripts/ingest_corpus.py` builds the
demo corpus into both indexes — **this is where you seed the three DoD
trap-case documents intentionally**, not as an afterthought in Phase 13.
**Checkpoint:** for a known query against the ingested demo corpus,
`retriever.retrieve(query)` returns the expected document in the top
reranked results — a golden single-query smoke test.
**MUST NOT SPLIT** (dense + sparse + fusion + rerank land together): Phase
3's grading prompts get calibrated against whatever retrieval quality you
hand them. Calibrate a grader against dense-only output, then add rerank
later, and you've tuned the grading thresholds to a noisier signal than
production will actually produce — you'll redo that calibration once rerank
lands. Land the full substrate first.
**Safe stopping point:** yes, after the checkpoint passes.

### Phase 3 — Document Grading + Aggregation (FR-4, FR-5)
`grading/document_grader.py`, aggregation logic as a pure function of
`[ChunkGrade]` + config thresholds.
**Checkpoint:** the real-model 3-chunk check described in §5 above, plus
the mocked unit tests in `test_document_grader.py`/`test_aggregation.py`.
**Safe stopping point:** yes.

### Phase 4 — Query Rewriting (FR-6)
`rewriting/query_rewriter.py`.
**Checkpoint:** given a deliberately under-specified query and empty
history, output is materially different and more specific; given a
non-empty `rewrite_history`, the next rewrite doesn't repeat it.
**Safe stopping point:** yes. Independent of Phase 5 — can be built in
either order relative to it, both just need Phase 3's aggregation output
shape to exist.

### Phase 5 — Tavily Fallback + Injection Guard (FR-7, FR-8)
`fallback/tavily_client.py`, `fallback/injection_guard.py`.
**Checkpoint:** a planted injection marker in fetched content is blocked
before it would reach generation context; clean fallback content passes
through and gets graded normally by Phase 3's grader.
**MUST NOT SPLIT** (Tavily client + injection guard land together, guard
written first if anything): NFR-11 requires *zero* exceptions to fallback
content being gated. Even a short-lived intermediate state where fallback
fetching works but gating doesn't yet is a state where a real network call
could pull ungated content into a real generation prompt during your own
testing. Don't create that window.
**Safe stopping point:** yes, after the checkpoint passes.

### Phase 6 — Generation + Dual Answer-Grading (FR-9, FR-10, FR-11)
`generation/generator.py`, `grading/answer_grader.py` (both groundedness
and relevance).
**Checkpoint:** a deliberately ungrounded canned answer is flagged by the
groundedness grader; a deliberately off-topic canned answer is flagged by
the relevance grader; a good answer passes both.
**MUST NOT SPLIT** (both grades land together, not groundedness-then-
relevance-later): §3.4's branch tree has three real outcomes (both pass /
relevance fails / groundedness fails), each routing differently. Build
against only one score and you'll architect a binary branch, then have to
restructure it into a three-way branch once the second score exists —
same shape as CorpMind's grounding-capture pairing.
**Safe stopping point:** yes.

### Phase 7 — Loop Orchestration + Bounding + Routing Gate (FR-3, FR-12–FR-15)
`orchestration/graph.py`, `nodes.py`, `routing_gate.py`. This is where
Phases 2–6 get wired into the actual LangGraph state machine per §3 of
REQUIREMENTS, plus the entry-point routing gate (§1 above) and the
escalation queue (`audit/escalation_queue.py`'s interface, even if Phase 8
fills in its actual persistence).
**Checkpoint:** run the full graph against three synthetic scenarios with
a mocked LLM boundary — (1) clean retrieval → straight to generate →
accept; (2) forced low `p_correct` → rewrite fires, `iteration_count`
increments, `rewrite_history` grows; (3) forced cap exhaustion → terminal
state hits with `low_confidence: true` and a reason code, not an
unhandled loop.
**MUST NOT SPLIT** (all four FRs are one state machine): a partial graph
with iteration caps wired but no defined terminal/escalation path is worse
than not building the caps yet — an exhausted loop with nowhere to route is
a bug, not a feature you're partway through.
**Depends on:** Phases 2–6 all complete.
**Safe stopping point:** yes, after the checkpoint passes — this is the
single most important stopping point in the whole plan, since everything
downstream builds on a working graph.

### Phase 8 — Audit Trail (FR-16)
`audit/audit_logger.py` — structured per-query log record capturing every
decision point from Phase 7's graph.
**Checkpoint:** running the same three synthetic scenarios from Phase 7
produces a complete, retrievable-by-query-ID trail for each — not just the
final answer, every grade/rewrite/fallback/iteration-count at every step.
**Not a hard must-not-split, but strongly recommended immediately after
Phase 7, in the same session block if possible:** the graph runs fine
without audit logging, which is exactly why it's the thing that quietly
never gets built if you move on to something more visibly "working" first.
**Safe stopping point:** yes.

### Phase 9 — Streaming Strategy (FR-17, FR-18)
`streaming/stream_handler.py` — fast-path token streaming, correction-path
status events (`retrieving`, `grading`, `rewriting`, `checking answer`)
with tokens withheld until an accepted or terminal state.
**Checkpoint:** a fast-path query streams tokens immediately (observable
latency-to-first-token); a correction-path query with a forced rewrite
emits at least two distinct status events before the final answer streams.
**Depends on:** Phase 7.
**Safe stopping point:** yes.

### Phase 10 — Eval Harness + Gold Set (FR-20)
`eval/eval_harness.py`, `eval/metrics.py`, `data/gold_set/` — planted trap
cases across the four categories from REQUIREMENTS §FR-20 (should-grade-
incorrect, should-catch-hallucination, should-trigger-rewrite, should-
escalate). This gold set is a superset of the three DoD cases — DoD is the
minimum bar, the gold set is the tuning instrument.
**Checkpoint:** running the harness against the current (default,
unvalidated) thresholds produces grader precision/recall numbers per NFR-4
and escalation precision/recall per NFR-5 — actual numbers, not a pass/fail
gate yet. That's what Phase 11 uses.
**Depends on:** Phase 7 (needs a working loop to run queries through),
Phase 2's ingested corpus (needs the trap-case documents already seeded).
**Safe stopping point:** yes.

### Phase 11 — Load Test / Tuning
`scripts/load_test.py` exercises the rate-limiter (§2) under concurrent
correction-path load; threshold tuning uses Phase 10's eval numbers to
adjust the 0.5/0.7/0.7 defaults in `config/thresholds.yaml` (FR-21) if the
gold-set results say they're miscalibrated.
**Checkpoint:** a documented before/after — the default thresholds' gold-
set scores vs. the tuned thresholds' gold-set scores, with the reasoning
for any change. Concurrency ceiling computed from real Groq/Gemini numbers
per §2's sizing formula (fill these in once you have live limits).
**Depends on:** Phase 10.
**Safe stopping point:** yes.

### Phase 12 — Dockerization (NFR-7)
`Dockerfile`, `docker-compose.yml` — own dedicated Qdrant service, no
shared network or container with Project 1.
**Checkpoint:** `docker compose up` from a clean checkout (no local Python
env, no locally-running Qdrant) brings up a working system that answers a
known query correctly — proves the standalone-deployment requirement
(§0.3 Q3) actually holds, not just that the code imports cleanly locally.
**Depends on:** can start any time after Phase 2's architecture is stable;
final validation should happen after Phase 11 so you're containerizing
tuned thresholds, not defaults.
**Safe stopping point:** yes.

### Phase 13 — Final Integration Test Against DoD Cases (§6)
The three permanent trap-case tests (§5 above) run against the full
Dockerized stack, real LLM calls, real corpus.
**Checkpoint:** all three DoD cases pass end-to-end with a complete,
correct audit trail for each — this phase's checkpoint *is* the project's
overall done-checkpoint.
**Depends on:** everything.
**Safe stopping point:** this is the end.

---

## 7. Risk Register (condensed, phase-mapped)

| # | Failure mode | Mitigation | Validated by |
|---|---|---|---|
| 1 | Grader hallucinates a grade, silently corrupting the loop | Real-model calibration check before trusting the grader; fail-closed on malformed output (§3) | Phase 3 checkpoint + Phase 10 (NFR-4) |
| 2 | Unbounded cost blowup under adversarial/ambiguous corpora | Hard, independently-capped iteration counters (§1); rate-limit backoff (§2) | Phase 7 checkpoint + Phase 11 load test |
| 3 | Prompt injection via fallback content bypasses the guard | Guard built in the same phase as the fetch, zero-exception policy | Phase 5 checkpoint + Phase 10's planted-injection gold-set cases |
| 4 | Correction-path latency reads as broken with no client feedback | Status events during correction (FR-17) | Phase 9 checkpoint |
| 5 | Miscalibrated thresholds cause false-escalation flooding | Config-driven, explicitly-unvalidated defaults tuned against real data | Phase 11, using Phase 10's numbers |

---

## 8. Explicitly Out of Scope for This Plan

Everything REQUIREMENTS.md §4 already deferred to v2 stays deferred — not
re-litigated here: numeric calibrated confidence (FR-22), rewrite-diversity
enforcement (FR-23), provisional-stream-with-retraction UX (FR-24),
independent cost-budget abort (FR-25), semantic caching of grading
decisions (FR-26), LangSmith tracing (FR-27), reopening document-grading on
a downstream groundedness failure, split iteration caps, retryable
fallback, multi-tenant rate-limit pooling, active-learning feedback into
the gold set, and multi-hop/synthesis-aware grading. None of these appear
in any phase above; none should get built as a "quick add" mid-phase.
