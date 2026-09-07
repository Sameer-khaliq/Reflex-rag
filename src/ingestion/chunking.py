"""
Structure-aware + Token-budget chunking for Markdown documents.
Preserves H1/H2 boundaries, injects document & section breadcrumbs,
and computes precise char/token offsets.
breadcrumbs mean adding some context of chunk at top
"""
from __future__ import annotations

import re
from typing import Any
import tiktoken
from config import get_config

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def _extract_sections(markdown_text: str) -> list[dict[str, Any]]:
    """
    Splits markdown into logical sections using H1 (#) and H2 (##) headings.
    Returns list of sections with title, heading, and body text.
    """
    lines = markdown_text.splitlines()
    doc_title = "General"
    sections: list[dict[str, Any]] = []

    # 1. Identify H1 as the Document Title
    for line in lines:
        if line.startswith("# ") and not line.startswith("## "):
            doc_title = line.replace("# ", "").strip()
            break

    # 2. Split by H2 boundaries
    h2_pattern = re.compile(r"^(##\s+.+)$", re.MULTILINE)
    parts = h2_pattern.split(markdown_text)

    # Preamble (Text before any H2)
    preamble = parts[0].strip()
    if preamble:
        # Remove doc title line from preamble body to avoid duplication
        body = re.sub(r"^#\s+.*$", "", preamble, flags=re.MULTILINE).strip()
        if body:
            sections.append({
                "doc_title": doc_title,
                "section_title": "Overview",
                "body": body,
            })

    # Paired H2 heading and section body
    for i in range(1, len(parts), 2):
        raw_heading = parts[i].strip()
        section_title = raw_heading.replace("##", "").strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            sections.append({
                "doc_title": doc_title,
                "section_title": section_title,
                "body": body,
            })

    return sections


def chunk_text(
    text: str,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> list[dict]:
    """
    Generates contextualized chunks. Each chunk gets a breadcrumb header:
    "Document: <Title> > Section: <Section>\n<Content>"
    Falls back to token-window splitting only if a single section exceeds max_tokens.
    """
    chunking_cfg = get_config().retrieval.chunking
    min_tokens = min_tokens if min_tokens is not None else chunking_cfg.min_tokens
    max_tokens = max_tokens if max_tokens is not None else chunking_cfg.max_tokens

    sections = _extract_sections(text)
    chunks = []

    # If document has no markdown headers, treat as single body
    if not sections:
        sections = [{"doc_title": "Document", "section_title": "General", "body": text.strip()}]

    for sec in sections:
        breadcrumb = f"Document: {sec['doc_title']} > Section: {sec['section_title']}\n"
        full_content = breadcrumb + sec["body"]
        token_count = count_tokens(full_content)

        # Case A: Fits within target chunk budget
        if token_count <= max_tokens:
            start_char = text.find(sec["body"][:40]) if len(sec["body"]) >= 40 else 0
            chunks.append({
                "text": full_content,
                "doc_title": sec["doc_title"],
                "section_title": sec["section_title"],
                "start_char": max(0, start_char),
                "end_char": max(0, start_char) + len(sec["body"]),
                "token_count": token_count,
            })
        else:
            # Case B: Large section — split paragraph by paragraph with breadcrumbs
            paragraphs = sec["body"].split("\n\n")
            current_chunk_body = ""

            for p in paragraphs:
                candidate = (current_chunk_body + "\n\n" + p).strip() if current_chunk_body else p
                if count_tokens(breadcrumb + candidate) > max_tokens and current_chunk_body:
                    c_text = breadcrumb + current_chunk_body
                    chunks.append({
                        "text": c_text,
                        "doc_title": sec["doc_title"],
                        "section_title": sec["section_title"],
                        "start_char": 0,
                        "end_char": len(current_chunk_body),
                        "token_count": count_tokens(c_text),
                    })
                    current_chunk_body = p
                else:
                    current_chunk_body = candidate

            if current_chunk_body:
                c_text = breadcrumb + current_chunk_body
                chunks.append({
                    "text": c_text,
                    "doc_title": sec["doc_title"],
                    "section_title": sec["section_title"],
                    "start_char": 0,
                    "end_char": len(current_chunk_body),
                    "token_count": count_tokens(c_text),
                })

    return chunks


