# REQUIREMENTS.md — Self-Correcting RAG Framework

**Project 2 of 3 — RAG/retrieval-systems portfolio (Gulf/EU/USA freelance targeting)**
**Author:** Max Power, senior AI systems architect (tuned by Sameer)
**Owner:** Sameer
**Status:** Finalized

Selling point: retrieval and generation that catch their own mistakes instead
of silently returning bad answers. This project **rebuilds** Project 1's
proven hybrid-retrieval architecture (Qdrant dense + bm25s sparse + RRF
fusion + cross-encoder rerank) fresh, as a standalone module inside its own
repo — same battle-tested design decisions, new implementation, zero
cross-project dependency. That gets Sameer real from-scratch build reps on
the retrieval layer again while keeping the actual novel engineering effort
concentrated on the correction loop, which is the new, demo-able capability.

---

## 0.1 Project Summary

A LangGraph-orchestrated pipeline, fully standalone from Project 1, that
wraps a freshly-built hybrid retriever in a bounded, self-grading correction
loop: retrieved chunks are graded CRAG-style (correct / ambiguous /
incorrect), insufficient retrieval triggers query rewriting or Tavily web
fallback, generated answers are checked for groundedness *and* relevance
separately, and the whole loop is capped, logged, and escalates to a
low-confidence flag rather than ever silently returning a bad answer with
false confidence.

## 0.2 Open Architecture Gaps

These are flagged rather than silently resolved. Each has a v1 default so
the build isn't blocked, but each default is a judgment call that should be
revisited once the eval harness (§FR-20) produces real data.

1. **Shared vs. split iteration budget.** v1 default: a single shared
   `max_iterations` counter covers retrieval-correction attempts
   (retrieve→grade→rewrite) *and* generation-correction attempts
   (generate→grade→regenerate) combined. Risk: a query that burns its whole
   budget on rewrites gets zero generation-retry headroom, or vice versa.
   No data yet on which phase actually needs more budget in practice.
2. **Fallback retryability.** v1 default: Tavily fallback fires at most once
   per query (single-shot), not retried with a rewritten query. This bounds
   cost but means a bad first fallback query can't self-correct the way
   corpus retrieval can.
3. **Reconciling document-grading vs. answer-groundedness disagreement.**
   If document grading passed a chunk as CORRECT but the groundedness
   checker later flags the generated answer as unsupported, v1 has no
   defined path to say "the document grader was wrong" — it just retries
   generation once, then escalates. There's no mechanism in v1 to feed that
   disagreement back into re-grading the retrieval.
4. **Routing-gate signal design.** v1 default: the pre-correction fast-path
   gate follows the same design pattern as Project 1's routing cascade
   (complexity signal as a proxy for "does this query need correction"),
   freshly implemented in this repo rather than imported. That cascade was
   designed to answer a different question originally (which model tier
   should answer this), not "is retrieval for this query likely to be
   insufficient." Proxy accuracy is unvalidated.
5. **Escalation blocking behavior.** v1 default: the system never blocks a
   response waiting on human review — it always returns a best-effort
   answer synchronously (flagged `low_confidence: true`) and pushes to the
   escalation queue asynchronously. No synchronous "hold for review" mode
   exists in v1.

## 0.3 Clarifying Questions / Decisions

1. The default `max_iterations` below is **3** (shared retrieval +
   generation cap, per gap #1 above). Accepted as the starting point;
   revisit via eval-harness data before locking further.
2. Fallback is single-shot per query (gap #2). Accepted for v1 — retryable
   fallback deferred to v2 (adds cost/latency for a currently-unmeasured
   resilience gain).
3. **Decided:** this project stands up its **own dedicated** Qdrant
   instance/collection and its own standalone docker-compose — it does
   **not** share Project 1's running container or network. Each portfolio
   project must deploy and demo independently; a client evaluating Project
   2 should never need Project 1 running alongside it. This also means the
   retrieval layer (FR-1, FR-2) is written fresh in this repo rather than
   imported as a package.
4. **Decided:** demo corpus does not need to stay domain-agnostic like
   Project 1 — the loop/grading logic is the selling point here, not the
   domain, so a specific vertical is fine. Pick whichever corpus makes the
   three Definition-of-Done trap cases (§6) easiest to construct and
   explain in under 2 minutes.

---

## 1. Functional Requirements

**Retrieval substrate (rebuilt fresh, standalone — not imported from Project 1)**

- **FR-1** [DoD-gate] — Build a hybrid retriever in this repo, from
  scratch: Qdrant dense search + bm25s sparse search, fused via RRF, top-N
  candidates. Same architecture as Project 1's proven design; new
  implementation, own Qdrant instance/collection (§0.3 Q3) — no
  cross-project code or runtime dependency on Project 1.
- **FR-2** [DoD-gate] — Build a local cross-encoder reranking step in this
  repo, applying the same reranking design as Project 1, to re-order the
  fused candidate set before any grading occurs.

**Cost/latency-aware triggering**

- **FR-3** [DoD-gate] — Implement a pre-correction routing gate that
  classifies each incoming query as `fast_path` (skip correction entirely,
  retrieve→generate→stream immediately) or `correction_path` (enter the
  loop below). Same design pattern as Project 1's routing cascade,
  implemented fresh per §0.2 gap #4.
- **FR-18** [DoD-gate] — `fast_path` queries bypass grading entirely and go
  straight to generation + immediate token streaming, preserving
  latency-first behavior for the queries that don't need correction.

**Document-level correction (retrieval side)**

- **FR-4** [DoD-gate] — For each reranked chunk, a cheap-tier LLM grader
  classifies it `CORRECT` / `AMBIGUOUS` / `INCORRECT` relative to the query
  (CRAG-style), independent of any single end-to-end quality gate.
- **FR-5** [DoD-gate] — Aggregate per-chunk grades into a retrieval-level
  verdict using the thresholds in §3, determining whether generation
  proceeds, a rewrite is triggered, or fallback is triggered.
- **FR-6** [DoD-gate] — Query rewriting: when the aggregated verdict is
  insufficient, a cheap-to-mid-tier LLM rewrites the query, given the full
  rewrite history (so it doesn't repeat a prior failed reformulation), and
  retriggers retrieval (FR-1→FR-2→FR-4→FR-5).
- **FR-7** [DoD-gate] — Corrective fallback retrieval via Tavily, triggered
  per the branch logic in §3 when the corpus itself can't support the
  query.
- **FR-8** [DoD-gate] — Prompt-injection guard on all fallback web content:
  same deterministic regex-based injection-marker gate design as
  CorpMind's (not an LLM-only judge, per the earlier finding that an
  LLM faithfulness judge alone proved unreliable against a planted
  injection), implemented fresh in this repo — before web content ever
  enters the generation context.

**Answer-level self-correction**

- **FR-9** [DoD-gate] — Strong-tier model generates the answer from
  whichever context was accepted (corpus chunks and/or fallback content).
- **FR-10** [DoD-gate] — Groundedness/faithfulness check: a cheap-tier
  grader checks whether the generated answer is actually supported by the
  accepted context. Scored and gated separately from FR-11.
- **FR-11** [DoD-gate] — Answer-relevance check: a cheap-tier grader checks
  whether the answer addresses the original question — kept as a distinct
  score/gate from FR-10, since a fully-grounded answer can still be
  off-topic and a relevant answer can still be ungrounded.

**Loop orchestration and bounding**

- **FR-12** [DoD-gate] — Implement the full correction loop as a LangGraph
  state machine per the flow in §3, with `iteration_count`, `rewrite_history`,
  `fallback_used`, and `generation_attempts` threaded through shared state.
- **FR-13** [DoD-gate] — Hard iteration cap: a single configurable
  `max_iterations` (default 3, per §0.3 Q1) bounds total loop attempts;
  the loop terminates deterministically the instant the cap is hit,
  regardless of grade outcomes.
- **FR-14** [DoD-gate] — Terminal low-confidence behavior: if the cap is
  reached without an accepted answer, return the best-effort last-generated
  answer explicitly flagged `low_confidence: true` with a machine-readable
  reason code — never presented as if it were a confident, verified answer.
- **FR-15** [DoD-gate] — Escalation queue: every low-confidence terminal
  case is pushed to a human-review queue (same shape as CorpMind's) with
  its full trace attached, asynchronously (per §0.2 gap #5).

**Observability**

- **FR-16** [DoD-gate] — Full audit trail: every node's decision (chunk
  grades, aggregation verdict, rewrite text, fallback trigger, groundedness
  score, relevance score, iteration count at each step) is logged to a
  structured, per-query-ID-retrievable record — not just the final answer.

**Streaming**

- **FR-17** [DoD-gate] — Explicit streaming strategy (see §3.4): `fast_path`
  queries stream tokens immediately (FR-18); `correction_path` queries do
  **not** stream the answer until the loop has converged to an accepted (or
  terminal low-confidence) state, but do stream intermediate status events
  (e.g. `retrieving`, `grading`, `rewriting`, `checking answer`) so the
  client isn't left staring at a frozen UI during multi-second loops.

**Model tiering**

- **FR-19** [DoD-gate] — Explicit cheap/expensive tiering, config-driven,
  not hardcoded per call site:
  - Document grading (FR-4), groundedness check (FR-10), relevance check
    (FR-11), fallback-content grading → cheap/fast Groq tier
  - Query rewriting (FR-6) → cheap-to-mid tier
  - Answer generation (FR-9) → strong tier (Groq strong model or Gemini,
    per Project 1's tiering precedent)

**Config**

- **FR-21** [DoD-gate] — All grading/aggregation thresholds (§3) are
  externalized to config, not hardcoded, so the eval harness (FR-20) can
  tune them without a code change.

**Evaluation**

- **FR-20** [DoD-gate] — Correction-focused eval harness, same design
  pattern as Project 1's RAGAS-style approach (freshly implemented here),
  with a gold set of planted trap cases across four categories: retrieval
  that should grade `INCORRECT`, hallucinations that should be caught by
  groundedness grading, queries that should trigger a rewrite, and queries
  that should trigger escalation. Scores grader precision/recall against
  labels, not just end-to-end answer quality.

**Aspirational / extended (v1.1+, listed but not gating v1 completion)**

- **FR-22** [Aspirational] — Numeric calibrated confidence score exposed to
  the API caller, not just the boolean `low_confidence` flag.
- **FR-23** [Aspirational] — Rewrite-diversity enforcement: use rewrite
  history to actively push subsequent rewrites toward semantically distinct
  reformulations rather than near-duplicates.
- **FR-24** [Aspirational] — Provisional-stream-with-retraction UX: stream a
  tentative answer during correction and emit a retraction/correction event
  if it's later downgraded.
- **FR-25** [Aspirational] — Per-query hard cost/latency budget abort,
  independent of `max_iterations` — kills the loop early if projected spend
  exceeds a dollar/token ceiling even before the iteration cap is hit.
- **FR-26** [Aspirational] — Semantic caching of grading decisions for
  repeated or near-duplicate queries.
- **FR-27** [Aspirational] — LangSmith tracing integration, matching
  researchpilot-ai's `@traceable` approach, as a richer view on top of the
  FR-16 audit log.

---

## 2. Non-Functional Requirements

- **NFR-1** [DoD-gate] — Fast-path (FR-18) end-to-end latency target:
  proposed same envelope as Project 1's routed fast path. Unvalidated until
  measured — flag as a starting target, not a guarantee.
- **NFR-2** [DoD-gate] — Correction-path end-to-end latency target,
  worst case (full `max_iterations` exhausted, including one fallback):
  documented explicitly as a multiple of single-shot latency (see §4
  scalability note) rather than left implicit. Unvalidated default; tune
  against eval-harness measurements.
- **NFR-3** [DoD-gate] — Free-tier cost ceiling: document the maximum
  Groq/Gemini/Tavily calls a single worst-case query can generate (see §4),
  and confirm it stays inside free-tier rate limits at the target demo
  concurrency.
- **NFR-4** [DoD-gate] — Grader accuracy: document-grader and answer-grader
  agreement with the eval gold set (FR-20) must be measured and reported
  before the loop's outputs are treated as trustworthy; no fixed target
  claimed yet since the grading prompts don't exist yet — measure first,
  set a bar second.
- **NFR-5** [DoD-gate] — Escalation precision/recall: the escalation queue
  must catch planted hallucination/unanswerable cases from the gold set
  (recall) without flagging a large share of genuinely-good answers
  (precision) — both measured against FR-20's gold set, not asserted.
- **NFR-6** [DoD-gate] — Observability completeness: 100% of correction
  decisions for every `correction_path` query are logged and retrievable
  by query ID (FR-16) — no sampling, no best-effort logging.
- **NFR-7** [DoD-gate] — Fully standalone Docker deployment: own
  dedicated Qdrant instance/collection, own docker-compose — no shared
  container or network dependency on Project 1 (§0.3 Q3), so this project
  deploys and demos entirely on its own.
- **NFR-8** [Aspirational] — Grading reproducibility: same input + same
  config produces the same grade/branch decision, to make eval runs
  comparable across code changes.
- **NFR-9** [DoD-gate] — Concurrency safety: correction-loop state is
  thread-scoped in LangGraph; concurrent queries must not leak or overwrite
  each other's iteration/grade state.
- **NFR-10** [DoD-gate] — Rate-limit resilience: retry/backoff on Groq TPM
  429s. This matters more here than in Project 1 — the correction loop
  multiplies LLM calls per query several-fold (grading + rewriting +
  generation + re-grading), so the known TPM-throttling gotcha from prior
  builds is a materially higher risk here.
- **NFR-11** [DoD-gate] — Prompt-injection coverage extends to all fallback
  web content entering the correction loop (FR-8) with zero exceptions —
  no code path generates from ungated web content.

---

## 3. Correction-Loop Decision Logic

State fields threaded through the LangGraph run: `iteration_count`,
`max_iterations` (default **3**, unvalidated — see §0.3 Q1),
`rewrite_history: list[str]`, `fallback_used: bool`,
`generation_attempts`, `max_generation_attempts` (default **2**, shares the
`max_iterations` budget per §0.2 gap #1), `low_confidence: bool`.

### 3.1 Entry: routing gate (FR-3)

```
query → routing_gate
  ├─ fast_path        → retrieve (FR-1,2) → generate (FR-9) → stream tokens → done
  └─ correction_path   → retrieve (FR-1,2) → grade_documents (§3.2)
```

### 3.2 Document grading & aggregation (FR-4, FR-5)

Each reranked chunk is graded `CORRECT` / `AMBIGUOUS` / `INCORRECT`.
Aggregation over the chunk set (thresholds below are **unvalidated
starting defaults**, to be tuned against the FR-20 gold set):

```
let p_correct = proportion of chunks graded CORRECT

if p_correct >= 0.5:
    → drop INCORRECT chunks, keep CORRECT + AMBIGUOUS
    → proceed to generate (§3.4)

elif 0 < p_correct < 0.5:
    if iteration_count < max_iterations:
        → rewrite_query (§3.3) [DoD-gate FR-6]
    else:
        → fallback_retrieval (§3.3) as last resort before terminal (§3.5)

elif p_correct == 0:
    if not fallback_used:
        → fallback_retrieval (§3.3) directly — corpus judged insufficient,
          skip further rewrite attempts (§0.2 gap #2: fallback is single-shot)
    else:
        → terminal: low_confidence (§3.5)
```

### 3.3 Rewrite and fallback branches (FR-6, FR-7, FR-8)

```
rewrite_query:
    increment iteration_count
    rewrite ← LLM(cheap-mid tier, query, rewrite_history)   # FR-6
    append rewrite to rewrite_history
    → retrieve (FR-1,2) with rewritten query → grade_documents (§3.2)

fallback_retrieval:
    fallback_used = true
    results ← Tavily(query)                                  # FR-7
    results ← injection_guard(results)                       # FR-8, regex gate
    → grade_documents(results) (§3.2 grading logic, same grader)
    if p_correct == 0 after fallback:
        → terminal: low_confidence (§3.5)
```

### 3.4 Generation & answer-level grading (FR-9, FR-10, FR-11)

```
generate:
    answer ← LLM(strong tier, accepted_context)               # FR-9
    groundedness_score ← grade(answer, accepted_context)       # FR-10
    relevance_score    ← grade(answer, original_query)         # FR-11

    if groundedness_score >= 0.7 and relevance_score >= 0.7:   # unvalidated defaults
        → accept: stream final answer, log trail, done

    elif relevance_score < 0.7:
        # answer doesn't address the question — likely a retrieval-adequacy
        # problem the document grader missed, not a generation problem
        if iteration_count < max_iterations:
            → rewrite_query (§3.3)
        else:
            → terminal: low_confidence (§3.5)

    elif groundedness_score < 0.7:
        # possible hallucination
        if generation_attempts < max_generation_attempts:
            generation_attempts += 1
            → regenerate with a stricter "stick to context" instruction
        else:
            → terminal: low_confidence (§3.5)   # see §0.2 gap #3 —
              v1 does not re-open document grading on groundedness failure
```

### 3.5 Terminal / escalation (FR-13, FR-14, FR-15, FR-16)

```
terminal: low_confidence:
    low_confidence = true
    return best-effort last-generated answer, flagged and reasoned
    push full trace to escalation queue (async, non-blocking — §0.2 gap #5)
    log full audit trail (FR-16), regardless of outcome
```

---

## 4. Scope Boundary

**In v1**

- Hybrid retrieval substrate rebuilt fresh, standalone (FR-1, FR-2)
- CRAG-style per-chunk document grading (FR-4, FR-5)
- Bounded query-rewrite loop (FR-6)
- Single-shot Tavily fallback with injection guard (FR-7, FR-8)
- Dual groundedness + relevance answer grading (FR-10, FR-11)
- Shared iteration cap + async escalation queue (FR-13–FR-15)
- Full per-query audit trail (FR-16)
- Cost/latency-aware pre-routing gate + fast-path bypass (FR-3, FR-18)
- Cheap/expensive model tiering (FR-19)
- Correction-focused eval harness with planted trap cases (FR-20)
- Explicit streaming strategy: status events during correction, tokens only
  on accept/terminal (FR-17)
- Config-driven thresholds (FR-21)
- Fully standalone deployment — own Qdrant, own docker-compose (NFR-7)

**Out of v1 — Suggested v2**

- Numeric calibrated confidence exposed to callers (FR-22)
- Rewrite-diversity enforcement (FR-23)
- Provisional-stream-with-retraction UX (FR-24)
- Independent per-query cost/latency budget abort, separate from iteration
  cap (FR-25)
- Semantic caching of grading decisions (FR-26)
- LangSmith tracing integration (FR-27)
- Reopening document-grading on a downstream groundedness failure (§0.2
  gap #3) — v1 treats grading and answer-checking as one-directional
- Split retrieval/generation iteration caps if eval data shows the shared
  cap (§0.2 gap #1) is starving one phase
- Retryable (multi-attempt) fallback retrieval (§0.3 Q2)
- Multi-tenant / horizontal rate-limit pooling for concurrent-query scale
  beyond single-instance free-tier limits
- Active-learning loop that feeds escalated cases back into the eval gold
  set automatically
- Multi-hop / multi-document synthesis-aware grading (v1 assumes largely
  single-hop factual queries — a chunk is graded relevant to the literal
  query, not to an intermediate reasoning step)

**Scalability & cost boundary (documented up front, not discovered later)**

A single `correction_path` query can, at worst case (`max_iterations = 3`,
one fallback, one regeneration), issue on the order of ~10–12 LLM calls
(3× retrieval-grading passes, up to 2 rewrite calls, 1 fallback-content
grading pass, up to 2 generation calls, 2× answer-grading calls) versus a
single-shot fast-path query's ~1–2 calls. This is the binding cost driver,
not query volume alone: a corpus with a high natural rewrite-trigger rate
(ambiguous or sparse documents) will run `correction_path` far more often
than a well-indexed corpus, multiplying free-tier Groq TPM/RPD consumption
accordingly. At demo-level traffic (single user, live call) this stays
comfortably inside free tiers; at any sustained multi-user concurrency,
NFR-3's per-query call budget becomes the limiting factor before compute or
Qdrant does. Horizontal scaling and multi-tenant rate-limit pooling are
explicitly out of v1 (see Suggested v2 above).

---

## 5. Risk Register

| # | Failure mode | Mitigation |
|---|---|---|
| 1 | Grader model itself hallucinates a grade (marks an `INCORRECT` chunk `CORRECT`), silently corrupting the entire downstream loop | Validate grader accuracy against the FR-20 gold set (NFR-4) before trusting it in production paths; log low-agreement cases for manual review; where feasible, add deterministic keyword pre-filters as a sanity check alongside the LLM grade |
| 2 | Unbounded cost blowup from the correction loop against adversarial or naturally ambiguous corpora at scale | Hard shared iteration cap (FR-13) + rate-limit backoff (NFR-10); independent cost-budget abort deferred to v2 (FR-25) but flagged now, not discovered later |
| 3 | Prompt injection via Tavily fallback content bypasses the regex gate (novel pattern not covered by the marker list) | Reuse and extend CorpMind's gate patterns; treat all fallback content as strictly lower-trust than corpus content (never permitted to override system instructions); include planted-injection cases in the FR-20 gold set |
| 4 | Streaming/UX mismatch — correction-path queries take materially longer than fast-path with no client feedback, reading as broken rather than thorough | Explicit intermediate status events during correction (FR-17); NFR-2 documents the expected worst-case latency envelope so the client UI can set expectations up front |
| 5 | Miscalibrated thresholds (§3) cause false-escalation flooding — legitimate answers get flagged low-confidence, defeating the loop's purpose and swamping the review queue | Thresholds are config-driven and explicitly labeled unvalidated defaults (FR-21); NFR-5 sets escalation precision/recall targets measured against the eval gold set before thresholds are considered demo-ready |

---

## 6. Definition of Done

The system must handle the following three planted adversarial cases
**end-to-end**, with a correct, complete audit trail (FR-16) logged for
each — this is the acceptance bar, not a nice-to-have:

1. **Planted bad-retrieval case.** Corpus seeded so the correct answer
   requires disambiguating a term that collides with an unrelated domain
   (e.g. a term with two meanings, only one document set relevant). Initial
   retrieval must be graded `INCORRECT`/`AMBIGUOUS`, must trigger a rewrite
   (§3.2–3.3), and the rewrite must recover a `CORRECT`-graded chunk on
   retry — proving the rewrite loop actually improves retrieval rather than
   just retrying the same query.
2. **Planted hallucination case.** Context is graded `CORRECT` (it's
   topically relevant) but does not actually contain the specific fact the
   query asks for — a naive generator would hallucinate a plausible-sounding
   answer. The groundedness check (FR-10) must catch this and either force
   a stricter regeneration or route to terminal low-confidence — it must
   **not** be returned as a confident answer.
3. **Planted unanswerable-from-corpus case.** Neither the corpus nor the
   Tavily fallback contains the answer. The system must not hallucinate a
   confident-sounding response — it must exhaust the loop, return
   `low_confidence: true` with a reason code, and push the case to the
   escalation queue (FR-14, FR-15).

All three cases must produce a retrievable, complete decision trail:
which grade fired at each step, what was rewritten (if anything), whether
fallback triggered, and the final groundedness/relevance scores.
