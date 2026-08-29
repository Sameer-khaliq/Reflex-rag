"""
Token-based chunking.

Splits raw text into chunks of chunk_min_tokens-chunk_max_tokens tokens,
with chunk_overlap_min_pct-chunk_overlap_max_pct overlap between
adjacent chunks, both bounds pulled from Settings.

Tokenizer: tiktoken's cl100k_base encoding. This is byte-level BPE, so
decode(encode(text)) == text exactly - decoding any contiguous slice of
token ids reconstructs the exact substring of the original text. That
lets us compute char offsets precisely instead of approximating them.
"""

import tiktoken

from config import get_config

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def chunk_text(
    text: str,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    overlap_min_pct: float | None = None,
    overlap_max_pct: float | None = None,
) -> list[dict]:
    """
    Returns a list of chunk dicts, each with:
        text          - the chunk's text (exact substring of the input)
        start_char    - inclusive char offset into the original text
        end_char      - exclusive char offset into the original text
        token_count   - token count of this chunk
        start_token   - inclusive token-index offset (internal use / tests)
        end_token     - exclusive token-index offset (internal use / tests)

    All bound params default to Settings if not passed, so callers never
    need to hardcode a value.
    """
    chunking_cfg = get_config().retrieval.chunking
    min_tokens = min_tokens if min_tokens is not None else chunking_cfg.min_tokens
    max_tokens = max_tokens if max_tokens is not None else chunking_cfg.max_tokens
    overlap_min_pct = (
        overlap_min_pct if overlap_min_pct is not None else chunking_cfg.overlap_min_pct
    )
    overlap_max_pct = (
        overlap_max_pct if overlap_max_pct is not None else chunking_cfg.overlap_max_pct
    )

    target_overlap_pct = (overlap_min_pct + overlap_max_pct) / 2
    overlap_tokens = round(max_tokens * target_overlap_pct)
    step = max_tokens - overlap_tokens
    if step <= 0:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) >= max_tokens ({max_tokens}); "
            f"chunks would never advance. Lower the overlap pct or raise max_tokens."
        )

    token_ids = _encoding.encode(text)
    total = len(token_ids)
    if total == 0:
        return []

    chunks = []
    start = 0
    while start < total:
        end = min(start + max_tokens, total)

        if end - start < min_tokens and start != 0:
            start = max(0, end - min_tokens)

        chunk_token_ids = token_ids[start:end]
        prefix_text = _encoding.decode(token_ids[:start]) if start > 0 else ""
        chunk_text_str = _encoding.decode(chunk_token_ids)

        start_char = len(prefix_text)
        end_char = start_char + len(chunk_text_str)

        chunks.append({
            "text": chunk_text_str,
            "start_char": start_char,
            "end_char": end_char,
            "token_count": len(chunk_token_ids),
            "start_token": start,
            "end_token": end,
        })

        if end == total:
            break
        start += step

    return chunks