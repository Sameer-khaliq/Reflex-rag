"""
Full LangGraph state machine wiring Phases 2-6 into the correction loop
(FR-3, FR-12-FR-15), per REQUIREMENTS.md §3's decision trees.

fast_path and correction_path share the same retrieve_node function,
registered under two different node names ("retrieve_fast" and
"retrieve_correction") rather than storing the routing decision in
state — this avoids adding a field to GraphState just to remember which
path a query is on. The graph topology itself encodes it: whichever
node START's conditional edge lands on determines what happens next.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from config import get_config
from orchestration import nodes as N
from orchestration.routing_gate import routing_gate_selector
from schemas.state import GraphState


def build_initial_state(query: str) -> GraphState:
    cfg = get_config()
    return GraphState(
        original_query=query,
        query=query,
        retrieved_chunks=[],
        chunk_grades=[],
        accepted_context=[],
        answer=None,
        iteration_count=0,
        max_iterations=cfg.thresholds.max_iterations,
        rewrite_history=[],
        fallback_used=False,
        generation_attempts=0,
        max_generation_attempts=cfg.thresholds.max_generation_attempts,
        low_confidence=False,
        low_confidence_reason=None,
        groundedness_score=None,
        relevance_score=None,
    )


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve_fast", N.retrieve_node)
    graph.add_node("generate_fast", N.fast_generate_node)

    graph.add_node("retrieve_correction", N.retrieve_node)
    graph.add_node("grade_documents", N.grade_documents_node)
    graph.add_node("rewrite", N.rewrite_node)
    graph.add_node("fallback", N.fallback_node)
    graph.add_node("generate", N.generate_node)
    graph.add_node("grade_answer", N.grade_answer_node)
    graph.add_node("terminal", N.terminal_node)

    graph.add_conditional_edges(
        START,
        routing_gate_selector,
        {"fast_path": "retrieve_fast", "correction_path": "retrieve_correction"},
    )

    graph.add_edge("retrieve_fast", "generate_fast")
    graph.add_edge("generate_fast", END)

    graph.add_edge("retrieve_correction", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        N.route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "fallback": "fallback",
            "terminal": "terminal",
        },
    )

    graph.add_conditional_edges(
        "rewrite",
        N.route_after_rewrite,
        {"retry_retrieval": "retrieve_correction", "terminal": "terminal"},
    )

    graph.add_edge("fallback", "grade_documents")

    graph.add_edge("generate", "grade_answer")

    graph.add_conditional_edges(
        "grade_answer",
        N.route_after_answer_grading,
        {
            "accept": END,
            "rewrite": "rewrite",
            "regenerate": "generate",
            "terminal": "terminal",
        },
    )

    graph.add_edge("terminal", END)

    return graph.compile()