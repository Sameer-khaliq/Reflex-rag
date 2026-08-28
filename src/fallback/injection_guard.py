"""
Prompt-injection guard for Tavily fallback content (FR-8).

Deterministic, regex-based marker gate — not an LLM judge. This mirrors
the design choice flagged in REQUIREMENTS.md: an LLM faithfulness judge
alone proved unreliable against a planted injection in earlier work, so
detection here is pattern-based and auditable, not a model call that can
itself be talked out of flagging something.

Zero-exception policy (NFR-11): every item returned by
fallback.tavily_client.fetch_tavily_results() MUST pass through
injection_guard() before it reaches generation context. There is no
code path where that's optional. Flagged items are dropped, not
sanitized and kept — a chunk containing an injection marker is treated
as untrustworthy in its entirety, not surgically edited.

Risk Register #3 (REQUIREMENTS.md §5) is explicit that this marker list
won't catch every possible phrasing — that's a known, accepted limit of
a deterministic gate, not an oversight. Extend INJECTION_PATTERNS as new
patterns are found in the eval gold set (FR-20's planted-injection cases)
or in real fallback traffic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from logging_config import get_logger

INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all|any)?\s*(the\s+)?(previous|prior|above|preceding)\s+instructions",
        r"disregard\s+(all|any)?\s*(the\s+)?(previous|prior|above|preceding)",
        r"forget\s+(everything|all)\s+(you\s+(were|have\s+been)\s+told|previous\s+instructions)",
        r"new\s+instructions\s*:",
        r"system\s+prompt",
        r"you\s+are\s+now\s+(a|an|the)\b",
        r"act\s+as\s+(if\s+you\s+(are|were)|a)\b",
        r"reveal\s+(your\s+)?(system\s+prompt|instructions|hidden\s+prompt)",
        r"print\s+(the\s+)?(system\s+prompt|instructions)",
        r"\[\s*system\s*\]",
        r"<\s*system\s*>",
        r"#{2,}\s*system\b",
        r"end\s+of\s+(document|context)\.?\s*(new\s+task|new\s+instructions)",
        r"do\s+not\s+(follow|obey|listen\s+to)\s+(the|your)\s+(previous|original)\s+instructions",
        r"override\s+(your|the)\s+(previous|original|system)\s+instructions",
        r"this\s+is\s+(a\s+)?(test|drill)\s*[:\-]\s*ignore",
        r"\bstop\s+being\s+(an?\s+)?assistant\b",
        r"respond\s+only\s+with\s+['\"]?yes['\"]?\s+to\s+everything",
    ]
]


@dataclass
class ScreenResult:
    is_flagged: bool
    matched_patterns: list[str] = field(default_factory=list)


def screen_text(text: str) -> ScreenResult:
    matched = [p.pattern for p in INJECTION_PATTERNS if p.search(text)]
    return ScreenResult(is_flagged=bool(matched), matched_patterns=matched)


def injection_guard(
    items: list[dict],
    text_key: str = "content",
    trace_id: str = "injection_guard",
) -> dict:
    """
    Screens fetched content items for injection markers.

    items: list of dicts, each expected to carry the fetched text under
    text_key (Tavily results use "content"; adjust if a different
    fallback source is added later).

    Returns:
        {
            "clean": [...],        # items that passed, unmodified
            "flagged": [...],      # items dropped, each carrying
                                    # "injection_matched_patterns" for
                                    # the audit trail (FR-16)
            "all_flagged": bool,   # True if items was non-empty and
                                    # every item was flagged — the
                                    # orchestration node should treat
                                    # this identically to the
                                    # Tavily-empty-results case (§3 of
                                    # the implementation plan's error
                                    # taxonomy), not retry against the
                                    # same flagged content
        }
    """
    logger = get_logger(trace_id=trace_id)
    clean: list[dict] = []
    flagged: list[dict] = []

    for item in items:
        text = str(item.get(text_key, "") or "")
        result = screen_text(text)
        if result.is_flagged:
            flagged.append({**item, "injection_matched_patterns": result.matched_patterns})
            logger.warning(
                "injection_marker_detected",
                stage="injection_guard",
                url=item.get("url"),
                matched_patterns=result.matched_patterns,
            )
        else:
            clean.append(item)

    logger.info(
        "injection_guard_complete",
        stage="injection_guard",
        total=len(items),
        clean=len(clean),
        flagged=len(flagged),
    )

    return {
        "clean": clean,
        "flagged": flagged,
        "all_flagged": bool(items) and not clean,
    }