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