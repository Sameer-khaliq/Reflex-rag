from typing import TypedDict

from schemas.chunk_grade import ChunkGrade


class GraphState(TypedDict):
    original_query: str
    query: str
    retrieved_chunks: list[dict]
    chunk_grades: list[ChunkGrade]
    accepted_context: list[str]
    answer: str | None

    iteration_count: int
    max_iterations: int
    rewrite_history: list[str]
    fallback_used: bool
    generation_attempts: int
    max_generation_attempts: int
    low_confidence: bool
    low_confidence_reason: str | None

    # --- Added for Phase 7 orchestration ---
    # LangGraph's conditional edges can only route on *state*, never on a
    # node's raw return value directly — and unlike aggregation's
    # p_correct (a pure, deterministic function of already-stored
    # chunk_grades), groundedness/relevance scores come from a live LLM
    # call with no cheap way to re-derive them from other state fields.
    # They need to be persisted here so route_after_answer_grading() can
    # read them without re-calling the grader (which would double the
    # LLM cost and risk a different result on re-grade). This also
    # directly serves FR-16: groundedness/relevance scores are audit
    # fields that need to be logged at each step, not just used
    # transiently.
    groundedness_score: float | None
    relevance_score: float | None